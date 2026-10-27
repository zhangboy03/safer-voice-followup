"""FlowRuntime — core business policy engine for the flow layer.

Manages: current node, transitions, tool policy (mutable), epoch/inference_id
validation, context strategy delegation, deferred post_actions.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import TYPE_CHECKING, Any, cast

from src.core.config import settings
from src.core.logging import get_logger
from src.flows.ablation_switches import is_state_machine_disabled
from src.flows.fast_judge_context import FastJudgeContextBuilder
from src.flows.fast_judge_runner import hash_text as hash_fast_judge_text
from src.flows.models import (
    TERMINAL_ROUTE,
    Action,
    AgentConfig,
    FlowError,
    FlowState,
    JudgeResult,
    NodeConfig,
    NodeState,
    NodeStatus,
    NodeTransition,
    SideEffectLevel,
    ToolOutcome,
    ToolPolicy,
    ToolSpec,
    TransferCommand,
    TransitionKind,
    TurnInjections,
    TurnPlan,
)
from src.flows.note_log import safe_call_id
from src.flows.sop_decision_models import CallWorkingMemory, FastJudgeInternalIdentity, MemoryPatch
from src.flows.tool_executor import ToolExecutor
from src.flows.weighted_context import WeightedContextAssembler, WeightedContextLane, WeightedContextView
from src.handoff.models import TransferPriority, TransferReason
from src.handoff.transfer_service import get_transfer_service
from src.pipeline.events import SopTransitionEvent

if TYPE_CHECKING:
    from src.flows.context_builder import ContextBuilder
    from src.flows.sop_decision_models import JudgeNoteDTO
    from src.flows.state_committer import CommitDecision, FieldCommit, NodeActionResult
    from src.prompts.prompt_pack import PromptRegistry
    from src.session.conversation_history import ConversationHistory

logger = get_logger(__name__)

# Maximum tool call rounds before force-transferring to human
MAX_TOOL_ROUNDS = 10

KNOWLEDGE_QA_NODE = "knowledge_qa"

_FAST_JUDGE_FORBIDDEN_MESSAGE_MARKERS = ("[用户刚说]", "[WeightedContextHash]")


def _sanitize_fast_judge_model_visible_value(value: Any, *, current_user_text: str) -> Any:
    if isinstance(value, str):
        text = value
        if current_user_text:
            text = text.replace(current_user_text, "")
        for marker in _FAST_JUDGE_FORBIDDEN_MESSAGE_MARKERS:
            text = text.replace(marker, "")
        return text.replace("sha256:", "")
    if isinstance(value, list):
        return [_sanitize_fast_judge_model_visible_value(item, current_user_text=current_user_text) for item in value]
    if isinstance(value, tuple):
        return tuple(
            _sanitize_fast_judge_model_visible_value(item, current_user_text=current_user_text) for item in value
        )
    if isinstance(value, dict):
        return {
            key: _sanitize_fast_judge_model_visible_value(item, current_user_text=current_user_text)
            for key, item in value.items()
        }
    return value


def _hash_fast_judge_messages(messages: list[dict]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash_fast_judge_semantic_context(context: Any) -> str:
    payload = json.dumps(context.model_dump(mode="json", exclude_none=True), ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _make_cacheable_system_message(content: str) -> dict:
    if settings.enable_explicit_cache:
        return {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    return {"role": "system", "content": content}


class FlowRuntime:
    """Core business policy engine held by LLMProcessor as a collaborator."""

    def __init__(
        self,
        agent_config: AgentConfig,
        context_builder: ContextBuilder,
        tool_executor: ToolExecutor,
        conversation_history: ConversationHistory,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        self._agent = agent_config
        self._context_builder = context_builder
        self._tool_executor = tool_executor
        self._history = conversation_history
        self._prompt_registry = prompt_registry
        self._state = FlowState(current_node=agent_config.initial_node)
        self._state.call_working_memory = CallWorkingMemory()
        self._weighted_context = WeightedContextAssembler()
        self._fast_judge_context_builder = FastJudgeContextBuilder()
        self._tool_registry: dict[str, ToolSpec] = {}  # tool_name -> ToolSpec
        self._current_tool_policy: ToolPolicy | None = None
        self._deferred_post_actions: list[Action] = []
        self._context_to_node: dict[str, str] = {}  # context_id -> node_id
        self._bus = None  # EventBus, injected externally after construction

        # Initialize node states
        for name in agent_config.nodes:
            self._state.node_states[name] = NodeState()

        # Mark the initial node as IN_PROGRESS
        initial_ns = self._state.node_states.get(agent_config.initial_node)
        if initial_ns is not None:
            initial_ns.status = NodeStatus.IN_PROGRESS
            initial_ns.entered_turn = 0

        self._register_all_tools()
        self.call_id: str = ""
        self._call_start_iso: str = ""

    def set_call_id(self, *, room_name: str, call_start_iso: str) -> None:
        self._call_start_iso = call_start_iso
        self.call_id = safe_call_id(room_name=room_name, call_start_iso=call_start_iso)

    # ── Core methods ──────────────────────────────────────────────────────

    @property
    def call_memory(self) -> CallWorkingMemory:
        return cast(CallWorkingMemory, self._state.call_working_memory)

    def replace_call_memory(self, memory: CallWorkingMemory) -> None:
        self._state.call_working_memory = memory

    def build_weighted_context_view(
        self,
        user_text: str = "",
        *,
        lane: WeightedContextLane = "speech",
        exclude_current_turn_from_recent: bool = False,
    ) -> WeightedContextView:
        return self._weighted_context.build(
            memory=self.call_memory,
            history=self._history,
            current_user_text=user_text,
            lane=lane,
            flow_state=self._state if lane in {"speech", "fast_judge", "short_term_memory", "sop_progress"} else None,
            exclude_current_turn_from_recent=exclude_current_turn_from_recent,
        )

    def build_fast_judge_context(self, *, user_text: str, assistant_text: str, speech_result_id: str):
        context = self._fast_judge_context_builder.build(
            flow_state=self._state,
            node_config=self._current_node_config(),
            node_catalog=self._agent.nodes,
            user_text=user_text,
            speech_text=assistant_text,
            speech_result_id=speech_result_id,
            conversation_history=self._history,
        )
        context.short_term_memory = self._select_judge_visible_notes()
        return context

    def _select_judge_visible_notes(self) -> list[JudgeNoteDTO]:
        """Filter + clip short_term_memory items into JudgeNoteDTO list (spec §7.2)."""
        from src.flows.ablation_switches import is_note_judge_injection_enabled
        from src.flows.sop_decision_models import JudgeNoteDTO

        if not is_note_judge_injection_enabled():
            return []

        JUDGE_VISIBLE = {"correction_hint", "future_node_hint", "salient_fact"}
        items = list(self._state.short_term_memory.items)

        # Secondary: longer/later turn_ids sort first when expiry ties (proxy for recency).
        def _sort_key(it):
            expires = it.expires_after_turn_index if it.expires_after_turn_index is not None else 1_000_000
            latest_turn = it.source_turn_ids[-1] if it.source_turn_ids else ""
            return (expires, -len(latest_turn))

        visible = [it for it in items if it.category in JUDGE_VISIBLE]
        visible.sort(key=_sort_key)

        out = []
        char_budget = 400
        for it in visible:
            if len(out) >= 6:
                break
            if char_budget - len(it.text) < 0:
                continue
            char_budget -= len(it.text)
            out.append(JudgeNoteDTO(category=it.category, text=it.text))
        return out

    def build_fast_judge_messages(
        self,
        *,
        user_text: str,
        assistant_text: str,
        speech_result_id: str,
    ) -> tuple[list[dict], FastJudgeInternalIdentity]:
        if self._prompt_registry is None:
            raise RuntimeError("Fast Judge requires an active PromptRegistry")

        context = self.build_fast_judge_context(
            user_text=user_text,
            assistant_text=assistant_text,
            speech_result_id=speech_result_id,
        )
        semantic_context_hash = _hash_fast_judge_semantic_context(context)
        sanitized_context_payload = _sanitize_fast_judge_model_visible_value(
            context.model_dump(mode="python"),
            current_user_text=user_text,
        )
        sanitized_context = type(context).model_validate(sanitized_context_payload)
        if settings.fast_judge_prompt_v2_enabled and hasattr(
            self._prompt_registry, "render_fast_judge_static_prompt_v2"
        ):
            v2 = self._prompt_registry.render_fast_judge_static_prompt_v2(node_id=self._state.current_node)
            global_content = _sanitize_fast_judge_model_visible_value(v2.global_part.text, current_user_text=user_text)
            node_content = _sanitize_fast_judge_model_visible_value(v2.node_part.text, current_user_text=user_text)
            context_content = _sanitize_fast_judge_model_visible_value(
                self._prompt_registry.render_fast_judge_context_block(context=sanitized_context),
                current_user_text=user_text,
            )
            messages = [
                _make_cacheable_system_message(global_content),
                _make_cacheable_system_message(node_content),
                {"role": "system", "content": context_content},
                {"role": "user", "content": user_text},
            ]
        elif hasattr(self._prompt_registry, "render_fast_judge_static_prompt") and hasattr(
            self._prompt_registry,
            "render_fast_judge_context_block",
        ):
            static = self._prompt_registry.render_fast_judge_static_prompt(node_id=self._state.current_node)
            static_content = _sanitize_fast_judge_model_visible_value(static.text, current_user_text=user_text)
            context_content = _sanitize_fast_judge_model_visible_value(
                self._prompt_registry.render_fast_judge_context_block(context=sanitized_context),
                current_user_text=user_text,
            )
            messages = [
                _make_cacheable_system_message(static_content),
                {"role": "system", "content": context_content},
                {"role": "user", "content": user_text},
            ]
        else:
            rendered = self._prompt_registry.render_fast_judge_prompt(
                node_id=self._state.current_node,
                context=sanitized_context,
            )
            system_content = _sanitize_fast_judge_model_visible_value(
                rendered.text,
                current_user_text=user_text,
            )
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_text},
            ]
        prompt_hash = _hash_fast_judge_messages(messages)
        identity = FastJudgeInternalIdentity(
            turn_id=self._state.active_turn_id or "",
            turn_index=self._state.turn_count,
            node_id=self._state.current_node,
            flow_epoch=self._state.flow_epoch,
            context_snapshot_id=f"ctx:{prompt_hash[-12:]}",
            speech_result_id=speech_result_id,
            speech_text_hash=hash_fast_judge_text(assistant_text),
            fast_judge_prompt_hash=prompt_hash,
            weighted_context_hash=semantic_context_hash,
        )
        return messages, identity

    def begin_user_turn(self, turn_id: str) -> None:
        """New user turn. Increment counters. Call only once per turn."""
        self._state.turn_count += 1
        self._state.node_turn_count += 1
        self._state.active_turn_id = turn_id
        # Sync turn index to ConversationHistory for journal entries
        if hasattr(self._history, "set_turn_index"):
            self._history.set_turn_index(self._state.turn_count)

    async def prepare_user_turn(self, user_text: str, is_speculative: bool) -> TurnPlan:
        """Build TurnPlan. Can be called multiple times per turn
        (speculative/formal/restart).
        """
        inference_id = str(uuid.uuid4())
        self._state.active_inference_id = inference_id

        node = self._current_node_config()
        available_tools = self._merge_tools(node)

        self._current_tool_policy = ToolPolicy(
            max_side_effect=(SideEffectLevel.READ if is_speculative else SideEffectLevel.TRANSFER),
            is_speculative=is_speculative,
        )

        prompt_registry = self._prompt_registry
        injections = self._build_injections(user_text="" if prompt_registry is not None else user_text)
        weighted_context_view = (
            self.build_weighted_context_view("", lane="speech", exclude_current_turn_from_recent=True)
            if prompt_registry is not None
            else None
        )
        messages = await self._context_builder.build_turn_messages(
            agent_config=self._agent,
            node_config=node,
            flow_state=self._state,
            conversation_history=self._history,
            user_text=user_text,
            injections=injections,
            prompt_registry=prompt_registry,
            weighted_context_view=weighted_context_view,
        )

        # Register context→node mapping for PlaybackLedger
        self._context_to_node[inference_id] = self._state.current_node

        logger.info(
            "flow.prepare_user_turn",
            node=self._state.current_node,
            speculative=is_speculative,
            tools=len(available_tools),
        )

        return TurnPlan(
            messages=messages,
            available_tools=available_tools,
            tools_schema=self._build_tools_schema(available_tools),
            node_id=self._state.current_node,
            flow_epoch=self._state.flow_epoch,
            inference_id=inference_id,
            tool_policy=self._current_tool_policy,
        )

    async def prepare_node_entry(self, continuation_prefix: str = "") -> TurnPlan | None:
        """After transition, if respond_immediately=True, build TurnPlan for
        new node.

        Args:
            continuation_prefix: Stage 1 text to inject as assistant message
                so Stage 2 LLM can continue naturally.
        """
        node = self._current_node_config()
        if not node.respond_immediately:
            return None

        inference_id = str(uuid.uuid4())
        self._state.active_inference_id = inference_id

        available_tools = self._merge_tools(node)
        injections = self._build_injections()
        prompt_registry = self._prompt_registry
        weighted_context_view = (
            self.build_weighted_context_view("", lane="speech", exclude_current_turn_from_recent=True)
            if prompt_registry is not None
            else None
        )
        messages = await self._context_builder.build_node_entry_messages(
            agent_config=self._agent,
            node_config=node,
            flow_state=self._state,
            conversation_history=self._history,
            injections=injections,
            continuation_prefix=continuation_prefix,
            prompt_registry=prompt_registry,
            weighted_context_view=weighted_context_view,
        )

        # Register context→node mapping for PlaybackLedger
        self._context_to_node[inference_id] = self._state.current_node

        return TurnPlan(
            messages=messages,
            available_tools=available_tools,
            tools_schema=self._build_tools_schema(available_tools),
            node_id=self._state.current_node,
            flow_epoch=self._state.flow_epoch,
            inference_id=inference_id,
            tool_policy=ToolPolicy(max_side_effect=SideEffectLevel.TRANSFER, is_speculative=False),
        )

    async def prepare_stage2_entry(self, previous_speech_text: str = "", user_text: str = "") -> TurnPlan | None:
        """Build a dedicated Stage 2 entry-ask plan for the current node.

        Unlike prepare_node_entry(), this path does not render the full
        Stage-1 node speech prompt. It renders a short entry contract plus the
        first collectable missing field.
        """
        node = self._current_node_config()
        if not node.respond_immediately:
            return None

        missing = self.collectable_missing_fields()
        target_field = missing[0] if missing else ""
        target_field_description = node.field_definitions.get(target_field, target_field) if target_field else ""

        inference_id = str(uuid.uuid4())
        self._state.active_inference_id = inference_id

        available_tools = self._merge_tools(node)
        prompt_registry = self._prompt_registry
        weighted_context_view = self.build_weighted_context_view(
            "",
            lane="speech",
            exclude_current_turn_from_recent=True,
        )
        messages = await self._context_builder.build_stage2_entry_messages(
            agent_config=self._agent,
            node_config=node,
            flow_state=self._state,
            conversation_history=self._history,
            continuation_prefix=previous_speech_text,
            target_field=target_field,
            target_field_description=target_field_description,
            user_text=user_text,
            prompt_registry=prompt_registry,
            weighted_context_view=weighted_context_view,
        )

        self._context_to_node[inference_id] = self._state.current_node

        return TurnPlan(
            messages=messages,
            available_tools=available_tools,
            tools_schema=self._build_tools_schema(available_tools),
            node_id=self._state.current_node,
            flow_epoch=self._state.flow_epoch,
            inference_id=inference_id,
            tool_policy=ToolPolicy(max_side_effect=SideEffectLevel.TRANSFER, is_speculative=False),
        )

    async def prepare_silent_rescue_plan(self, node_id: str) -> TurnPlan | None:
        """ADR 0006 §4.2: build a Stage 1 plan with current node prompt + history.

        Re-uses build_node_entry_messages to render the unified followup-aware
        Stage 1 prompt; persona §应对持续沉默 soft constraint guides the model
        to choose between more empathy / pivoting to the missing field /
        suggesting the patient take their time.

        Bypasses respond_immediately (silent-rescue always re-runs Speech).
        """
        node = self._current_node_config()
        if node is None:
            return None

        inference_id = str(uuid.uuid4())
        self._state.active_inference_id = inference_id

        available_tools = self._merge_tools(node)
        injections = self._build_injections()
        prompt_registry = self._prompt_registry
        weighted_context_view = (
            self.build_weighted_context_view("", lane="speech", exclude_current_turn_from_recent=False)
            if prompt_registry is not None
            else None
        )
        messages = await self._context_builder.build_node_entry_messages(
            agent_config=self._agent,
            node_config=node,
            flow_state=self._state,
            conversation_history=self._history,
            injections=injections,
            continuation_prefix="",  # silent-rescue does not chain a Stage 2 continuation
            prompt_registry=prompt_registry,
            weighted_context_view=weighted_context_view,
            silent_rescue=True,
            include_current_turn_in_history=True,
        )

        self._context_to_node[inference_id] = self._state.current_node

        return TurnPlan(
            messages=messages,
            available_tools=available_tools,
            tools_schema=self._build_tools_schema(available_tools),
            node_id=self._state.current_node,
            flow_epoch=self._state.flow_epoch,
            inference_id=inference_id,
            tool_policy=ToolPolicy(max_side_effect=SideEffectLevel.TRANSFER, is_speculative=False),
        )

    async def prepare_stage1_followup(
        self,
        missing_fields: list[str],
        directive: str,
        continuation_prefix: str,
    ) -> TurnPlan | None:
        """Build an in-turn followup Speech plan (spec §4.2).

        Reuses build_node_entry_messages with the §stage1-followup section so
        the model bridges off its own Stage 1 closing line and asks the
        still-missing field(s). continuation_prefix = the Stage 1 wording.
        """
        node = self._current_node_config()
        if node is None or not missing_fields:
            return None

        # Inject human-readable field descriptions (e.g. "程控满意度（枚举：满意/
        # 不满意）"), not raw IDs — the Speech model reads these in a Chinese
        # prompt. Falls back to the ID if no description is defined.
        field_descriptions = [node.field_definitions.get(f, f) for f in missing_fields]

        inference_id = str(uuid.uuid4())
        self._state.active_inference_id = inference_id

        available_tools = self._merge_tools(node)
        injections = self._build_injections()
        prompt_registry = self._prompt_registry
        weighted_context_view = (
            self.build_weighted_context_view("", lane="speech", exclude_current_turn_from_recent=True)
            if prompt_registry is not None
            else None
        )
        messages = await self._context_builder.build_node_entry_messages(
            agent_config=self._agent,
            node_config=node,
            flow_state=self._state,
            conversation_history=self._history,
            injections=injections,
            continuation_prefix=continuation_prefix,
            prompt_registry=prompt_registry,
            weighted_context_view=weighted_context_view,
            stage1_followup_fields=field_descriptions,
            stage1_followup_directive=directive,
        )

        self._context_to_node[inference_id] = self._state.current_node

        return TurnPlan(
            messages=messages,
            available_tools=available_tools,
            tools_schema=self._build_tools_schema(available_tools),
            node_id=self._state.current_node,
            flow_epoch=self._state.flow_epoch,
            inference_id=inference_id,
            tool_policy=ToolPolicy(max_side_effect=SideEffectLevel.TRANSFER, is_speculative=False),
        )

    async def prepare_divert_entry(self, user_text: str) -> TurnPlan | None:
        """After DIVERT, build TurnPlan for knowledge_qa with user's question.

        Unlike prepare_node_entry, this uses build_divert_entry_messages
        (system prompt + user question, no Stage 2 continuation prefix).
        """
        node = self._current_node_config()
        if not node.respond_immediately:
            return None

        inference_id = str(uuid.uuid4())
        self._state.active_inference_id = inference_id

        available_tools = self._merge_tools(node)
        injections = self._build_injections()
        prompt_registry = self._prompt_registry
        weighted_context_view = (
            self.build_weighted_context_view("", lane="speech", exclude_current_turn_from_recent=True)
            if prompt_registry is not None
            else None
        )
        messages = await self._context_builder.build_divert_entry_messages(
            agent_config=self._agent,
            node_config=node,
            flow_state=self._state,
            conversation_history=self._history,
            user_text=user_text,
            injections=injections,
            prompt_registry=prompt_registry,
            weighted_context_view=weighted_context_view,
        )

        self._context_to_node[inference_id] = self._state.current_node

        return TurnPlan(
            messages=messages,
            available_tools=available_tools,
            tools_schema=self._build_tools_schema(available_tools),
            node_id=self._state.current_node,
            flow_epoch=self._state.flow_epoch,
            inference_id=inference_id,
            tool_policy=ToolPolicy(max_side_effect=SideEffectLevel.TRANSFER, is_speculative=False),
        )

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict,
        flow_epoch: int,
        inference_id: str,
    ) -> ToolOutcome:
        """Synchronous tool execution with safety checks.
        Reads _current_tool_policy.
        """
        # Stale checks
        if flow_epoch != self._state.flow_epoch:
            return self._stale_outcome(f"epoch {flow_epoch} != {self._state.flow_epoch}")
        if inference_id != self._state.active_inference_id:
            return self._stale_outcome("inference_id mismatch")

        # Tool lookup
        tool_spec = self._tool_registry.get(tool_name)
        if tool_spec is None:
            return ToolOutcome(
                tool_payload=None,
                result_text=f"[unknown tool: {tool_name}]",
                transition=None,
                side_effect_level=SideEffectLevel.NONE,
            )

        # Side-effect check
        if self._current_tool_policy and tool_spec.side_effect > self._current_tool_policy.max_side_effect:
            return ToolOutcome(
                tool_payload=None,
                result_text=(
                    f"[blocked: {tool_name} needs {tool_spec.side_effect.name}, "
                    f"max allowed {self._current_tool_policy.max_side_effect.name}]"
                ),
                transition=None,
                side_effect_level=SideEffectLevel.NONE,
            )

        # Execute
        node_config = self._current_node_config()
        outcome = await self._tool_executor.execute(tool_spec, arguments, self._state, node_config)

        # Handle transition
        if outcome.transition is not None:
            await self._commit_transition(outcome.transition)

        return outcome

    def upgrade_to_formal(self) -> None:
        """Called by LLMProcessor on formal confirm.
        Upgrades tool policy in-place.
        """
        self._current_tool_policy = ToolPolicy(max_side_effect=SideEffectLevel.TRANSFER, is_speculative=False)

    def request_human_transfer(
        self,
        reason: TransferReason,
        priority: TransferPriority,
        trigger_text: str = "",
    ) -> TransferCommand:
        """Build transfer command without mutating FlowState.

        LLMProcessor is responsible for executing the command and calling
        commit_transfer_terminal() on success.
        """
        transfer_service = get_transfer_service()
        transition_message = transfer_service.get_transition_message(reason, priority)

        completed = [name for name, ns in self._state.node_states.items() if ns.status == NodeStatus.COMPLETED]
        collected = {k: v.value for k, v in self._state.collected_fields.items()}

        return TransferCommand(
            reason=reason.value,
            priority=priority.value,
            trigger_text=trigger_text,
            transition_message=transition_message,
            source_node=self._state.current_node,
            collected_fields=collected,
            patient_info=dict(self._state.patient_info),
            conversation_turns=self._state.turn_count,
            nodes_completed=completed,
        )

    async def force_transfer_to_human(self, reason: str) -> TransferCommand:
        """System error hard transfer. Returns TransferCommand (no FlowState mutation).

        Kept async for backward compatibility — existing callers await this.
        Caller (LLMProcessor) executes the command and calls commit_transfer_terminal().
        """
        return self.request_human_transfer(
            reason=TransferReason.OTHER,
            priority=TransferPriority.URGENT,
            trigger_text=f"system_error: {reason}",
        )

    def commit_transfer_terminal(self, reason: str, priority: str) -> None:
        """Commit TERMINAL transition after SIP REFER success.

        Unlike force_transfer_to_human, this is the normal transfer path.
        Does not run pre_actions or context_strategy (already executed by LLMProcessor).
        """
        old_ns = self._state.node_states.get(self._state.current_node)
        if old_ns and old_ns.status == NodeStatus.IN_PROGRESS:
            old_ns.status = NodeStatus.COMPLETED
            old_ns.completed_turn = self._state.turn_count
            old_ns.completion_reason = f"transfer:{reason}:{priority}"

        # Clear divert state if transferring from knowledge node
        self._state.divert_return_node = None
        self._state.divert_context_snapshot = None

        # ABLATION: see docs/superpowers/specs/2026-04-15-thesis-ch4-eval-roadmap.md §4.6
        if not is_state_machine_disabled():
            self._state.flow_epoch += 1
        self._state.current_node = "human_transfer"
        self._state.node_turn_count = 0
        self._state.node_history.append("human_transfer")

        new_ns = self._state.node_states.setdefault("human_transfer", NodeState())
        new_ns.status = NodeStatus.COMPLETED
        new_ns.entered_turn = self._state.turn_count
        new_ns.completed_turn = self._state.turn_count
        new_ns.completion_reason = f"transfer:{reason}:{priority}"

        logger.info(
            "flow.transfer_terminal_committed",
            reason=reason,
            priority=priority,
        )

    async def on_node_reply_finished(self) -> None:
        """Called after BotStoppedSpeakingFrame.
        Executes deferred post_actions.
        """
        if self._deferred_post_actions:
            await self._execute_actions(self._deferred_post_actions)
            self._deferred_post_actions.clear()

    def has_deferred_post_actions(self) -> bool:
        return bool(self._deferred_post_actions)

    def is_terminal_node(self, node_name: str | None = None) -> bool:
        """Check if a node is terminal (no successors)."""
        name = node_name or self._state.current_node
        if name == KNOWLEDGE_QA_NODE:
            return False
        node = self._agent.nodes.get(name)
        return node is not None and (node.successors is None or len(node.successors) == 0)

    def build_terminal_payload(self) -> dict:
        """Build the form_submit payload from current FlowState."""
        fields = {}
        for k, v in self._state.collected_fields.items():
            fields[k] = {
                "value": v.value,
                "status": v.status.value.lower(),
                "source_node": v.source_node,
            }
        return {
            "collected_fields": fields,
            "patient_info": self._state.patient_info,
            "conversation_turns": self._state.turn_count,
            "nodes_completed": list(self._state.node_history),
        }

    def resolve_context_node(self, context_id: str) -> str:
        """PlaybackLedger callback: context_id -> node_id."""
        return self._context_to_node.get(context_id, "")

    def remap_context_node(self, context_id: str) -> None:
        """Re-map a context_id to the current node (for chained Stage 2)."""
        self._context_to_node[context_id] = self._state.current_node

    # ── Fast-judge / commit methods ──────────────────────────────────────

    def apply_field_commits(self, decision_id: str, commits: list["FieldCommit"]) -> None:
        if decision_id in self._state.applied_sop_decision_ids:
            return
        for commit in commits:
            self._state.collected_fields[commit.field] = commit.flow_value
        self._state.applied_sop_decision_ids.add(decision_id)

    def apply_memory_patch(self, decision_id: str, patch: MemoryPatch) -> None:
        if decision_id in self._state.applied_sop_decision_ids:
            return
        from src.flows.state_committer import StateCommitter

        decision = StateCommitter(self).merge_memory_patch(patch)
        if decision.accepted:
            self._state.applied_sop_decision_ids.add(decision_id)

    async def apply_node_action(self, decision: "CommitDecision") -> "NodeActionResult":
        from src.flows.state_committer import NodeActionResult

        stage_complete = decision.legacy_judge_result.stage_complete
        knowledge_query = decision.legacy_judge_result.knowledge_query
        should_check_route = False
        if knowledge_query:
            should_check_route = True
        elif decision.node_action == "followup":
            should_check_route = False
        elif decision.node_action in {"stay", "advance", "terminal"}:
            should_check_route = stage_complete
        elif decision.node_action == "divert":
            should_check_route = True

        route_result = (
            await self.check_and_advance(judge_result=decision.legacy_judge_result, speech_advance=None)
            if should_check_route
            else False
        )
        if (
            route_result is True
            and decision.decision_id.startswith("fast_judge:")
            and decision.node_action == "advance"
        ):
            chained_result = await self._advance_while_current_policy_satisfied()
            if chained_result == TERMINAL_ROUTE:
                route_result = TERMINAL_ROUTE
        terminal_requested = route_result == TERMINAL_ROUTE
        followup_requested = decision.node_action == "followup"
        if terminal_requested:
            node_after = TERMINAL_ROUTE
        elif followup_requested or route_result is False:
            node_after = decision.node_before
        else:
            node_after = self._state.current_node

        if self._state.fast_judge_state is not None:
            self._state.fast_judge_state = self._state.fast_judge_state.model_copy(update={"node_after": node_after})

        strategy = self.call_memory.next_speech_strategy
        if strategy is not None:
            memory = self.call_memory.model_copy(deep=True)
            memory.next_speech_strategy = strategy.model_copy(
                update={
                    "node_after": node_after,
                    "node_action": decision.node_action,
                }
            )
            self.replace_call_memory(memory)

        return NodeActionResult(
            advanced=route_result is True,
            node_before=decision.node_before,
            node_after=node_after,
            action=decision.node_action,
            route_result=route_result,
            terminal_requested=terminal_requested,
            followup_requested=followup_requested,
            followup_missing_fields=list(decision.followup_missing_fields),
        )

    async def _advance_while_current_policy_satisfied(self) -> bool | str:
        advanced = False
        visited = set()
        while True:
            node = self._current_node_config()
            if node.name in visited:
                break
            visited.add(node.name)
            if not node.successors or node.completion_policy is None:
                break
            if not node.completion_policy.is_satisfied(self._state):
                break
            target = node.route_fn(self._state) if node.route_fn else node.successors[0]
            if target == TERMINAL_ROUTE:
                logger.info("flow.route_to_terminal", node=self._state.current_node)
                return TERMINAL_ROUTE
            await self._commit_transition(
                NodeTransition(
                    target_node=target,
                    kind=TransitionKind.ADVANCE,
                    reason="lookahead_policy_satisfied",
                )
            )
            advanced = True
        return advanced

    async def prepare_appended_segment(
        self,
        decision: "CommitDecision",
        post_commit_context: WeightedContextView,
        previous_speech_text: str = "",
    ) -> TurnPlan | None:
        # The Stage 2 Speech prompt needs the exact Stage 1 wording as the
        # final assistant message so the model continues instead of restarting
        # the new node from a generic opener.
        continuation_prefix = previous_speech_text or decision.continuation_prefix
        return await self.prepare_node_entry(continuation_prefix=continuation_prefix)

    def get_missing_fields(self) -> list[str]:
        """Return required_fields not yet satisfied per CompletionPolicy semantics."""
        node = self._current_node_config()
        if not node.completion_policy:
            return []
        return node.completion_policy.unsatisfied_fields(self._state)

    def collectable_missing_fields(self) -> list[str]:
        """Required fields still collectable NOW.

        Like get_missing_fields() but excludes fields the Judge marked
        NOT_APPLICABLE (e.g. satisfaction when the patient hasn't done
        programming). `unsatisfied_fields` treats NOT_APPLICABLE as missing;
        for the stage1-followup trigger that would mean asking a question that
        does not apply, so we drop them here.
        """
        from src.flows.models import FieldStatus

        node = self._current_node_config()
        if not node.completion_policy:
            return []
        out = []
        for f in node.completion_policy.unsatisfied_fields(self._state):
            fv = self._state.collected_fields.get(f)
            if fv is not None and fv.status == FieldStatus.NOT_APPLICABLE:
                continue
            out.append(f)
        return out

    async def prepare_followup(self, missing_fields: list[str]) -> TurnPlan | None:
        """Build a short supplementary question targeting missing fields.

        Legacy helper for explicit follow-up turns when required fields are missing.
        """
        if not missing_fields:
            return None

        node = self._current_node_config()
        descriptions = []
        for f in missing_fields:
            desc = node.field_definitions.get(f, f)
            descriptions.append(f"- {desc}")
        field_list = "\n".join(descriptions)

        followup_system = (
            "你刚刚结束了一段回应。现在需要自然追问以下还未确认的信息：\n"
            f"{field_list}\n"
            '用一句话自然追问，用过渡词（"对了""另外"）衔接。不要重复刚才说过的话。'
        )

        inference_id = str(uuid.uuid4())
        self._state.active_inference_id = inference_id
        self._context_to_node[inference_id] = self._state.current_node

        messages = [{"role": "system", "content": followup_system}]

        return TurnPlan(
            messages=messages,
            available_tools={},
            tools_schema=[],
            node_id=self._state.current_node,
            flow_epoch=self._state.flow_epoch,
            inference_id=inference_id,
            tool_policy=ToolPolicy(max_side_effect=SideEffectLevel.TRANSFER, is_speculative=False),
        )

    async def check_and_advance(
        self,
        judge_result: JudgeResult | None = None,
        judge_stage_complete: bool = False,
        speech_advance: bool | None = None,
    ) -> bool | str:
        """Check completion, auto-advance if satisfied.

        Priority:
          1. knowledge_query → DIVERT to knowledge_qa (not from knowledge_qa)
          2. knowledge_qa stage_complete → RESUME to return node
          3. CompletionPolicy + stage_complete → ADVANCE / SKIP
        """
        from src.flows.models import TERMINAL_ROUTE

        # Kept for legacy callers; the new path does not use Speech marker votes.
        del speech_advance

        # Unpack judge_result for backward compatibility
        if judge_result is not None:
            judge_stage_complete = judge_result.stage_complete
            knowledge_query = judge_result.knowledge_query
        else:
            knowledge_query = False

        node = self._current_node_config()

        # 1. DIVERT: knowledge_query detected (not from knowledge_qa itself)
        if knowledge_query and self._state.current_node != KNOWLEDGE_QA_NODE:
            if KNOWLEDGE_QA_NODE in self._agent.nodes:
                await self._commit_transition(
                    NodeTransition(
                        target_node=KNOWLEDGE_QA_NODE,
                        kind=TransitionKind.DIVERT,
                        reason="knowledge_question_detected",
                    )
                )
                return True

        # 2. RESUME: knowledge_qa stage_complete → return to suspended node
        if (
            self._state.current_node == KNOWLEDGE_QA_NODE
            and judge_stage_complete
            and self._state.divert_return_node is not None
        ):
            return_node = self._state.divert_return_node
            await self._commit_transition(
                NodeTransition(
                    target_node=return_node,
                    kind=TransitionKind.RESUME,
                    reason="knowledge_session_complete",
                )
            )
            # Re-check if the return node's policy is now satisfied → auto-advance
            return_node_config = self._agent.nodes.get(return_node)
            if return_node_config and return_node_config.completion_policy:
                if return_node_config.completion_policy.is_satisfied(self._state):
                    target = (
                        return_node_config.route_fn(self._state)
                        if return_node_config.route_fn
                        else (return_node_config.successors[0] if return_node_config.successors else None)
                    )
                    if target and target != TERMINAL_ROUTE:
                        await self._commit_transition(
                            NodeTransition(
                                target_node=target,
                                kind=TransitionKind.ADVANCE,
                                reason="policy_satisfied_after_resume",
                            )
                        )
            return True

        # 3. Existing logic: no successors → can't advance
        if not node.successors:
            return False

        # 4. CompletionPolicy
        policy_satisfied = False
        if node.completion_policy is not None:
            policy_satisfied = node.completion_policy.is_satisfied(self._state)

        satisfied = policy_satisfied or judge_stage_complete

        if not satisfied and node.completion_criteria is not None:
            satisfied = node.completion_criteria(self._state)

        if not satisfied:
            return False

        if node.route_fn is not None:
            target = node.route_fn(self._state)
        else:
            target = node.successors[0]

        if target == TERMINAL_ROUTE:
            logger.info("flow.route_to_terminal", node=self._state.current_node)
            return TERMINAL_ROUTE

        reason = "judge_stage_complete" if judge_stage_complete else "policy_satisfied"
        kind = TransitionKind.SKIP if judge_stage_complete and not policy_satisfied else TransitionKind.ADVANCE

        await self._commit_transition(NodeTransition(target_node=target, kind=kind, reason=reason))
        return True

    # ── Internal: transition ──────────────────────────────────────────────

    async def _commit_transition(self, transition: NodeTransition) -> None:
        old_node = self._state.current_node
        new_node = transition.target_node

        if new_node not in self._agent.nodes:
            raise FlowError(f"transition to unknown node: {new_node}")

        old_node_config = self._agent.nodes.get(old_node)

        # Old node state update by TransitionKind
        old_ns = self._state.node_states.get(old_node)
        if old_ns and old_ns.status == NodeStatus.IN_PROGRESS:
            match transition.kind:
                case TransitionKind.ADVANCE:
                    old_ns.status = NodeStatus.COMPLETED
                    old_ns.completed_turn = self._state.turn_count
                    old_ns.completion_reason = transition.reason
                case TransitionKind.SKIP:
                    old_ns.status = NodeStatus.SKIPPED
                    old_ns.completed_turn = self._state.turn_count
                    old_ns.completion_reason = transition.reason
                case TransitionKind.TERMINAL:
                    old_ns.status = NodeStatus.COMPLETED
                    old_ns.completed_turn = self._state.turn_count
                    old_ns.completion_reason = transition.reason
                case TransitionKind.DIVERT:
                    old_ns.status = NodeStatus.SUSPENDED
                    old_ns.suspended_node_turn_count = self._state.node_turn_count
                    self._state.divert_return_node = old_node
                    self._state.divert_context_snapshot = self._context_builder.snapshot_window()
                case TransitionKind.RESUME:
                    pass  # knowledge_qa node — handled below

        # RESUME: mark knowledge node as COMPLETED
        if transition.kind == TransitionKind.RESUME:
            if old_ns:
                old_ns.status = NodeStatus.COMPLETED
                old_ns.completed_turn = self._state.turn_count
                old_ns.completion_reason = transition.reason

        # New node state
        new_ns = self._state.node_states.setdefault(new_node, NodeState())
        if transition.kind == TransitionKind.RESUME:
            # Restore suspended node
            new_ns.status = NodeStatus.IN_PROGRESS
            self._state.node_turn_count = new_ns.suspended_node_turn_count or 0
            new_ns.suspended_node_turn_count = None
            # Restore context window
            if self._state.divert_context_snapshot is not None:
                self._context_builder.restore_window(self._state.divert_context_snapshot)
            # Clear divert state
            self._state.divert_return_node = None
            self._state.divert_context_snapshot = None
        else:
            if new_ns.status == NodeStatus.PENDING:
                new_ns.status = NodeStatus.IN_PROGRESS
                new_ns.entered_turn = self._state.turn_count
            # Reset node_turn_count for non-RESUME transitions
            self._state.node_turn_count = 0

        # Epoch + navigation
        # ABLATION: see docs/superpowers/specs/2026-04-15-thesis-ch4-eval-roadmap.md §4.6
        # When ABLATION_DISABLE_SM=1 the state-machine epoch counter is frozen —
        # the ablated variant relies on prompt instructions alone, so stale-epoch
        # guards around tool execution and playback routing become no-ops.
        if not is_state_machine_disabled():
            self._state.flow_epoch += 1
        self._state.current_node = new_node
        self._state.node_history.append(new_node)

        if self._bus:
            self._bus.emit(
                SopTransitionEvent(
                    turn_id=getattr(self._state, "active_turn_id", "") or "",
                    from_node=old_node,
                    to_node=new_node,
                    trigger=transition.reason,
                    extracted_fields={},
                    session_id="",
                )
            )

        # Context strategy (skip for RESUME — already restored above)
        if transition.kind != TransitionKind.RESUME:
            new_node_config = self._agent.nodes.get(new_node)
            if new_node_config:
                await self._context_builder.apply_context_strategy(
                    strategy_config=new_node_config.context_strategy,
                    conversation_history=self._history,
                    flow_state=self._state,
                    agent_config=self._agent,
                )

        # Deferred post_actions (old node) — skip for DIVERT (node is suspended, not done)
        if transition.kind != TransitionKind.DIVERT:
            if old_node_config and old_node_config.post_actions:
                self._deferred_post_actions = list(old_node_config.post_actions)

        # Pre_actions (new node) — skip for RESUME (already ran when first entered)
        if transition.kind != TransitionKind.RESUME:
            new_node_config = self._agent.nodes.get(new_node)
            if new_node_config and new_node_config.pre_actions:
                await self._execute_actions(new_node_config.pre_actions)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _build_injections(self, user_text: str = "") -> TurnInjections:
        """Build TurnInjections with SOP progress + patient memory + last exchange."""
        from src.flows.formatting import format_sop_progress
        from src.flows.pinchi_flow_config import NODE_TO_GROUP

        last_exchange: str | None = None
        last_assistant = self._history.get_last_assistant_text()
        if last_assistant and user_text:
            last_exchange = f'你上一句话："{last_assistant}"\n用户刚回复："{user_text}"'

        return TurnInjections(
            sop_progress=format_sop_progress(
                flow_state=self._state,
                current_node=self._state.current_node,
                node_to_group=NODE_TO_GROUP,
            ),
            patient_memory=self._state.patient_info.get("memory"),
            last_exchange=last_exchange,
        )

    def _current_node_config(self) -> NodeConfig:
        return self._agent.nodes[self._state.current_node]

    def _merge_tools(self, node: NodeConfig) -> dict[str, ToolSpec]:
        """Merge global + node tools, filter by visibility."""
        all_tool_ids = list(self._agent.global_tools) + list(node.tools)
        tools: dict[str, ToolSpec] = {}
        for tid in all_tool_ids:
            spec = self._tool_registry.get(tid)
            if spec:
                tools[spec.name] = spec
        return ToolExecutor.filter_visible_tools(tools, self._state.current_node)

    def _register_all_tools(self) -> None:
        """Build tool_name -> ToolSpec registry from all known tool_ids.
        Subclasses or setup code should call register_tool_spec().
        """

    def register_tool_spec(self, spec: ToolSpec) -> None:
        """Register a ToolSpec by name for runtime lookup."""
        self._tool_registry[spec.name] = spec

    def _build_tools_schema(self, tools: dict[str, ToolSpec]) -> list[dict]:
        """Convert ToolSpec dict to OpenAI function calling schema."""
        return [
            {
                "type": "function",
                "function": {
                    "name": s.name,
                    "description": s.description,
                    "parameters": s.parameters,
                },
            }
            for s in tools.values()
        ]

    def _stale_outcome(self, reason: str) -> ToolOutcome:
        return ToolOutcome(
            tool_payload=None,
            result_text=f"[stale: {reason}]",
            transition=None,
            side_effect_level=SideEffectLevel.NONE,
        )

    async def _execute_actions(self, actions: list[Action]) -> None:
        """Execute pre/post actions.
        v1: log only, actual execution in integration phase.
        """
        for action in actions:
            pass  # TODO: implement tts_speak and emit_event in integration phase
