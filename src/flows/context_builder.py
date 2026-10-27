"""ContextBuilder — ephemeral prompt view assembler.

Assembles turn-scoped ephemeral messages for the LLM.  Does NOT modify
ConversationHistory (that's PlaybackLedger's job).  Reads from history +
injections + node config to construct a single system message + history
window + user text.

Key principle: "history is durable truth, turn-scoped messages are ephemeral
prompt view."
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol

from src.core.config import settings
from src.core.logging import get_logger
from src.flows.models import (
    AgentConfig,
    ContextStrategy,
    ContextStrategyConfig,
    ContextWindow,
    ContextWindowSnapshot,
    FlowState,
    NodeConfig,
    TurnInjections,
)
from src.session.conversation_history import ConversationHistory

if TYPE_CHECKING:
    from src.flows.weighted_context import WeightedContextView
    from src.prompts.prompt_pack import PromptRegistry

logger = get_logger(__name__)


class SummarizerBackend(Protocol):
    async def summarize(self, context: str, _instruction: str, max_tokens: int = 200) -> str: ...


class ContextBuilder:
    """Builds ephemeral message lists for LLM inference.

    Each call to ``build_turn_messages`` / ``build_node_entry_messages``
    produces a fresh ``list[dict]`` without touching ConversationHistory.
    ``apply_context_strategy`` updates the internal ``ContextWindow`` on
    node transitions (anchor index + optional summary).
    """

    def __init__(self, summary_llm: SummarizerBackend | None = None) -> None:
        self._summary_llm = summary_llm
        self._window = ContextWindow()

    # ------------------------------------------------------------------
    # Public — message builders
    # ------------------------------------------------------------------

    async def build_turn_messages(
        self,
        agent_config: AgentConfig,
        node_config: NodeConfig,
        flow_state: FlowState,
        conversation_history: ConversationHistory,
        user_text: str,
        injections: TurnInjections | None = None,
        prompt_registry: PromptRegistry | None = None,
        weighted_context_view: WeightedContextView | None = None,
    ) -> list[dict]:
        """Build ephemeral messages for a user turn.

        Cache architecture: system prompt (persona + task + hint) is stable
        per node → cached by DashScope prefix cache. Injections go in a
        separate uncached system message so they don't bust the prefix.
        """
        messages: list[dict] = []

        # Cached system message: stable within a node (persona + task)
        if prompt_registry is not None and weighted_context_view is not None:
            rendered = prompt_registry.render_speech_prompt(
                node_id=node_config.name,
                weighted_context=weighted_context_view,
                variables={"user_text": ""},
            )
            stable_content = rendered.text
        else:
            stable_content = self._compose_stable_system(agent_config, node_config)
        messages.append(self._make_system_message(stable_content))

        history_messages = self._history_messages_for_prompt(
            conversation_history,
            flow_state,
            weighted_context_view=weighted_context_view,
            include_with_weighted_context=prompt_registry is not None and weighted_context_view is not None,
        )
        messages.extend(history_messages)
        # Mark last history message with cache_control so DashScope caches
        # the [system + history] prefix — only new user message needs prefill
        if settings.enable_history_cache and history_messages:
            self._mark_history_cache_point(messages)

        # Uncached system messages: volatile state belongs after cached prefix.
        for content in self._dynamic_context_sections(
            flow_state=flow_state,
            injections=injections,
            prompt_registry=prompt_registry,
            weighted_context_view=weighted_context_view,
        ):
            messages.append({"role": "system", "content": content})

        if weighted_context_view is not None:
            messages.append({"role": "user", "content": self._current_user_block(flow_state, user_text)})
        elif not conversation_history.has_committed_user_message(user_text, flow_state.turn_count):
            messages.append({"role": "user", "content": user_text})
        return messages

    async def build_node_entry_messages(
        self,
        agent_config: AgentConfig,
        node_config: NodeConfig,
        flow_state: FlowState,
        conversation_history: ConversationHistory,
        injections: TurnInjections | None = None,
        continuation_prefix: str = "",
        prompt_registry: PromptRegistry | None = None,
        weighted_context_view: WeightedContextView | None = None,
        silent_rescue: bool = False,
        include_current_turn_in_history: bool = False,
        stage1_followup_fields: list[str] | None = None,
        stage1_followup_directive: str = "",
    ) -> list[dict]:
        """Build messages for node entry (no user_text, no RAG).

        Args:
            continuation_prefix: Stage 1 text injected as assistant message
                so Stage 2 LLM continues naturally without repeating.
            silent_rescue: ADR 0006 — patient stayed silent past timeout and
                we're re-running Stage 1. Swaps the standard Stage 1 guidance
                for §沉默续说 (read the prior assistant line + continue
                naturally, never re-ack/re-ask).
            include_current_turn_in_history: For silent-rescue, include the
                just-finished user/assistant messages so "last assistant" means
                the latest heard sentence, not the previous turn's prompt.
            stage1_followup_fields: When set, injects §stage1-followup section
                guiding the model to ask the listed missing fields after its
                own Stage 1 closing line. Mutually exclusive with §沉默续说
                and §环节过渡 — returns early from the dynamic-sections helper.
            stage1_followup_directive: Optional Judge-direction hint appended
                inside the §stage1-followup section (not shown when empty).
        """
        entry_injections = TurnInjections(
            patient_memory=injections.patient_memory if injections else None,
            sop_progress=injections.sop_progress if injections else None,
        )
        messages: list[dict] = []

        # Cached stable system message
        if prompt_registry is not None and weighted_context_view is not None:
            if silent_rescue and hasattr(prompt_registry, "render_silent_rescue_prompt"):
                # ADR 0006: silent-rescue bypasses the node script (which would
                # force the model to re-deliver greeting's mandatory intro etc.).
                rendered = prompt_registry.render_silent_rescue_prompt({"user_text": ""})
            else:
                rendered = prompt_registry.render_speech_prompt(
                    node_id=node_config.name,
                    weighted_context=weighted_context_view,
                    variables={"user_text": ""},
                )
            stable_content = rendered.text
        else:
            stable_content = self._compose_stable_system(agent_config, node_config)
        messages.append(self._make_system_message(stable_content))

        history_messages = self._history_messages_for_prompt(
            conversation_history,
            flow_state,
            weighted_context_view=weighted_context_view,
            include_with_weighted_context=prompt_registry is not None and weighted_context_view is not None,
            include_current_turn=include_current_turn_in_history,
        )
        messages.extend(history_messages)
        if settings.enable_history_cache and history_messages:
            self._mark_history_cache_point(messages)

        for content in self._dynamic_context_sections(
            flow_state=flow_state,
            node_config=node_config,
            injections=entry_injections,
            prompt_registry=prompt_registry,
            weighted_context_view=weighted_context_view,
            continuation_prefix=continuation_prefix,
            silent_rescue=silent_rescue,
            stage1_followup_fields=stage1_followup_fields,
            stage1_followup_directive=stage1_followup_directive,
        ):
            messages.append({"role": "system", "content": content})

        if continuation_prefix:
            messages.append({"role": "assistant", "content": continuation_prefix})
        return messages

    async def build_stage2_entry_messages(
        self,
        agent_config: AgentConfig,
        node_config: NodeConfig,
        flow_state: FlowState,
        conversation_history: ConversationHistory,
        *,
        continuation_prefix: str,
        target_field: str,
        target_field_description: str,
        user_text: str = "",
        prompt_registry: PromptRegistry | None = None,
        weighted_context_view: WeightedContextView | None = None,
    ) -> list[dict]:
        """Build the dedicated Stage 2 node-entry prompt.

        This intentionally avoids the full node speech prompt's Stage-1
        "用户本句" A/B decision rules. Stage 2 already has a routing decision;
        its job is to naturally continue and ask the target field.

        ``user_text`` (the current turn's user utterance) is pinned right before
        the continuation prefix as plain grounding — NOT as an A/B decision input.
        Without it the model only sees its own Stage 1 line as the last message
        and, finding no user turn, can misread a clean advance as a garbled/
        non-answer turn and re-ask or ask the patient to repeat (study_2
        v2_001_t01). The current-turn
        user message is excluded from the journal-derived history window, and the
        eval replay journals it only at turn end, so we thread it explicitly here
        rather than relying on ``include_current_turn``.
        """
        messages: list[dict] = []
        has_stage2_entry_prompt = (
            prompt_registry is not None
            and hasattr(prompt_registry, "render_stage2_entry_prompt")
            and hasattr(prompt_registry, "has_node_prompt")
            and prompt_registry.has_node_prompt(node_config.name, "stage2_entry")
        )
        if has_stage2_entry_prompt:
            rendered = prompt_registry.render_stage2_entry_prompt(node_id=node_config.name, variables={})
            stable_content = rendered.text
        else:
            persona = "\n\n".join(str(msg.get("content", "")) for msg in agent_config.persona_messages)
            task = "\n\n".join(str(msg.get("content", "")) for msg in node_config.task_messages)
            stable_content = "\n\n".join(
                section
                for section in [
                    persona,
                    task,
                    "# Stage 2 Entry Ask Mode",
                    "自然衔接是包装，目标字段提问是动作；包装不能吞掉动作。",
                ]
                if section
            )
        messages.append(self._make_system_message(stable_content))

        history_messages = self._history_messages_for_prompt(
            conversation_history,
            flow_state,
            weighted_context_view=weighted_context_view,
            include_with_weighted_context=weighted_context_view is not None,
        )
        messages.extend(history_messages)
        if settings.enable_history_cache and history_messages:
            self._mark_history_cache_point(messages)

        if weighted_context_view is not None:
            if prompt_registry is not None and hasattr(prompt_registry, "render_speech_runtime_context"):
                runtime_context = prompt_registry.render_speech_runtime_context(
                    weighted_context=weighted_context_view,
                    variables={},
                )
            elif prompt_registry is None:
                runtime_context = self._render_speech_context(weighted_context_view)
            else:
                runtime_context = ""
            if runtime_context:
                messages.append({"role": "system", "content": runtime_context})

        messages.append(
            {
                "role": "system",
                "content": self._stage2_entry_target_section(
                    node_config=node_config,
                    target_field=target_field,
                    target_field_description=target_field_description,
                ),
            }
        )
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if continuation_prefix:
            messages.append({"role": "assistant", "content": continuation_prefix})
        return messages

    # ------------------------------------------------------------------
    # Public — context strategy
    # ------------------------------------------------------------------

    async def apply_context_strategy(
        self,
        strategy_config: ContextStrategyConfig,
        conversation_history: ConversationHistory,
        flow_state: FlowState,
        agent_config: AgentConfig,
    ) -> None:
        """Modify ContextWindow on node transition.

        Does NOT modify ConversationHistory.
        """
        match strategy_config.strategy:
            case ContextStrategy.APPEND:
                self._window.summary = None
                self._window.summary_epoch = None
            case ContextStrategy.RESET:
                self._window.anchor_turn_index = flow_state.turn_count
                self._window.summary = None
                self._window.summary_epoch = None
            case ContextStrategy.RESET_WITH_SUMMARY:
                summary = await self._generate_summary(conversation_history, flow_state, agent_config, strategy_config)
                self._window.anchor_turn_index = flow_state.turn_count
                self._window.summary = summary
                self._window.summary_epoch = flow_state.flow_epoch

    # ------------------------------------------------------------------
    # Public — snapshot / restore (DIVERT / RESUME)
    # ------------------------------------------------------------------

    def snapshot_window(self) -> ContextWindowSnapshot:
        """Capture current ContextWindow state for later restoration."""
        return ContextWindowSnapshot(
            anchor_turn_index=self._window.anchor_turn_index,
            summary=self._window.summary,
            summary_epoch=self._window.summary_epoch,
        )

    def restore_window(self, snapshot: ContextWindowSnapshot) -> None:
        """Restore ContextWindow from a snapshot."""
        self._window.anchor_turn_index = snapshot.anchor_turn_index
        self._window.summary = snapshot.summary
        self._window.summary_epoch = snapshot.summary_epoch

    # ------------------------------------------------------------------
    # Public — DIVERT entry (knowledge node)
    # ------------------------------------------------------------------

    async def build_divert_entry_messages(
        self,
        agent_config: AgentConfig,
        node_config: NodeConfig,
        flow_state: FlowState,
        conversation_history: ConversationHistory,
        user_text: str,
        injections: TurnInjections | None = None,
        prompt_registry: PromptRegistry | None = None,
        weighted_context_view: WeightedContextView | None = None,
    ) -> list[dict]:
        """Build messages for DIVERT entry — system prompt + user question.

        Unlike build_node_entry_messages, this does NOT use the Stage 2
        continuation prompt. The user's original question is passed directly
        so the knowledge LLM can answer it immediately.
        """
        messages: list[dict] = []
        if prompt_registry is not None and weighted_context_view is not None:
            rendered = prompt_registry.render_speech_prompt(
                node_id=node_config.name,
                weighted_context=weighted_context_view,
                variables={"user_text": ""},
            )
            stable_content = rendered.text
        else:
            stable_content = self._compose_stable_system(agent_config, node_config)
        messages.append(self._make_system_message(stable_content))
        for content in self._dynamic_context_sections(
            flow_state=flow_state,
            node_config=node_config,
            injections=injections,
            prompt_registry=prompt_registry,
            weighted_context_view=weighted_context_view,
        ):
            messages.append({"role": "system", "content": content})
        if weighted_context_view is not None:
            messages.append({"role": "user", "content": self._current_user_block(flow_state, user_text)})
        else:
            messages.append({"role": "user", "content": user_text})
        return messages

    # ------------------------------------------------------------------
    # Internal — system message with optional cache_control
    # ------------------------------------------------------------------

    @staticmethod
    def _make_system_message(content: str) -> dict:
        """Build system message, optionally with DashScope explicit cache.

        When enable_explicit_cache=True, content is wrapped in a content array
        with cache_control marker so DashScope caches the system prompt prefix.
        Cached tokens are billed at 10% of standard input price (5-min TTL).
        """
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

    # ------------------------------------------------------------------
    # Internal — history cache marker
    # ------------------------------------------------------------------

    @staticmethod
    def _mark_history_cache_point(messages: list[dict]) -> None:
        """Add cache_control to the last message so DashScope caches the prefix.

        Marker 1 (system prompt) is set by _make_system_message.
        Marker 2 (history) is set here on the last history message, so the
        entire [system + history] prefix is cached. Only the new user message
        (appended after this call) needs fresh prefill.
        """
        last = messages[-1]
        content = last.get("content", "")
        messages[-1] = {
            "role": last["role"],
            "content": [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }

    @staticmethod
    def _current_user_block(flow_state: FlowState, user_text: str) -> str:
        return (
            "[CURRENT_USER_TURN_AFTER_SNAPSHOT]\n"
            f"turn_id: {flow_state.active_turn_id or ''}\n"
            f"用户：{user_text}\n"
            "[/CURRENT_USER_TURN_AFTER_SNAPSHOT]"
        )

    @staticmethod
    def _render_speech_context(weighted_context_view: WeightedContextView) -> str:
        if hasattr(weighted_context_view, "render_for_model"):
            return weighted_context_view.render_for_model(lane="speech")
        return weighted_context_view.render_for_prompt()

    def _history_messages_for_prompt(
        self,
        conversation_history: ConversationHistory,
        flow_state: FlowState,
        *,
        weighted_context_view: WeightedContextView | None,
        include_with_weighted_context: bool,
        include_current_turn: bool = False,
    ) -> list[dict]:
        if weighted_context_view is not None and not include_with_weighted_context:
            return []
        if weighted_context_view is None:
            return conversation_history.get_messages_from(self._window.anchor_turn_index)
        return [
            entry.message
            for entry in conversation_history.get_journal_entries()
            if entry.turn_index >= self._window.anchor_turn_index
            and (include_current_turn or entry.turn_index != flow_state.turn_count)
        ]

    def _dynamic_context_sections(
        self,
        *,
        flow_state: FlowState,
        node_config: NodeConfig | None = None,
        injections: TurnInjections | None,
        prompt_registry: PromptRegistry | None,
        weighted_context_view: WeightedContextView | None,
        continuation_prefix: str = "",
        silent_rescue: bool = False,
        stage1_followup_fields: list[str] | None = None,
        stage1_followup_directive: str = "",
    ) -> list[str]:
        sections: list[str] = []
        injections_content = self._compose_injections(flow_state, injections)
        if injections_content:
            sections.append(injections_content)
        if weighted_context_view is not None:
            runtime_context = ""
            if prompt_registry is not None and hasattr(prompt_registry, "render_speech_runtime_context"):
                runtime_context = prompt_registry.render_speech_runtime_context(
                    weighted_context=weighted_context_view,
                    variables={},
                )
            elif prompt_registry is None:
                runtime_context = self._render_speech_context(weighted_context_view)
            if runtime_context:
                sections.append(runtime_context)
        if silent_rescue and node_config is not None:
            sections.append(self._silent_rescue_current_node_section(node_config, flow_state))
        # Stage 1 only (no continuation_prefix): the directive carried in the
        # rendered runtime context comes from the PRIOR turn's Judge and was
        # written before the user's current utterance existed. Pin Stage 1 to
        # the latest user message, then let the directive serve as direction.
        if not continuation_prefix and not silent_rescue:
            sections.append(
                "\n".join(
                    [
                        "## 用户最新一句话优先",
                        "Judge 的「下一步意图」是上一轮根据上一句用户话语写的方向建议，并不知道用户本轮说了什么。",
                        "先完整理解用户最新一条消息（包括语气、是否追问、是否反悔、是否岔题），再决定怎么承接。",
                        "若用户已直接回答 → 简短承接即可，不要照着 Judge 方向硬推；若用户提出新问题/反悔/打断 → 先回应用户，必要时让 Judge 方向作废。",
                    ]
                )
            )
        # Stage 1 only: if the prior turn's Stage 2 already asked this node's
        # question, the user's reply now is the answer to it — Stage 1 must
        # only acknowledge, never re-ask.
        if (
            not continuation_prefix
            and not silent_rescue
            and flow_state.node_question_already_asked.get(flow_state.current_node)
        ):
            sections.append(
                "\n".join(
                    [
                        "## 本节点的问题上一轮已经提出",
                        "最近一条 assistant 消息已经问出本节点要问的问题，用户本句通常是对该问题的回答。",
                        "如果回答直接清晰，本句只能用「好的」「嗯」类一句很短的承接，绝对不要复述或换说法重新问本节点的问题。",
                        "只有本句转录读不通、残缺、明显是语音识别出错时，才自然说一句「这边信号好像有点不清楚，您方便再说一遍吗？」。",
                        "若本句只是没有直接回答本节点问题（提了新问题、打断、岔题、反悔等），按「用户最新一句话优先」正常承接或交给对应处理，不要重问本节点的问题。",
                    ]
                )
            )
        # Silent-rescue: patient didn't reply after our last utterance, we're
        # speaking again on our own. Hard constraint: read the prior assistant
        # line, continue naturally (not re-ack, not re-ask).
        if silent_rescue:
            sections.append(
                "\n".join(
                    [
                        "## 沉默续说（患者没回应，你正在主动再说一句）",
                        "**先认真读最后一条 assistant 消息**——那是你刚才说的话。本句必须像真人接着自己刚说完的话那样自然延展，绝不能像在重新开场。",
                        "硬约束：",
                        "- 绝不复用上一句的开头词（「好的」「嗯」「对」「那」等承接词，上一句用过这一句就换掉）。",
                        "- 绝不复述上一句已经说过的完整短句或同义改写。",
                        "- 如果上一句已经问出了问题，本句不要重新问同一个问题；可以换说法简短补一句、或加一点共情解释、或明确说「您慢慢说，我等您」。",
                        "做法（按优先级，挑最合适的一种）：",
                        "1) 上一句是 ack/铺垫（如「好的，那打扰您几分钟」）→ 直接进入当前节点要问的正事，无需再 ack；",
                        "2) 上一句已经问过问题 → 用很短一句换个说法、或给出选项示例帮患者理解，再停下等；",
                        "3) 连续两次都没回应 → 明确说「您慢慢说，我等您」，本句不再追问。",
                        "反例：上句「好的，那打扰您几分钟。」+ 本句「好的，那我先确认一下，请问您是患者本人吗？」 ← 连续「好的，那」，生硬，像在重新开场。",
                        "正例：上句「好的，那打扰您几分钟。」+ 本句「请问您是患者本人吗？还是家属代接？」 ← 直接接正题，无重复 ack。",
                    ]
                )
            )
        if stage1_followup_fields:
            field_lines = "\n".join(f"- {f}" for f in stage1_followup_fields)
            directive_line = (
                f"\nJudge 的方向建议（仅供措辞参考，不要照搬）：{stage1_followup_directive}"
                if stage1_followup_directive
                else ""
            )
            sections.append(
                "\n".join(
                    [
                        "## stage1-followup（你刚收尾了上一句，但本节点还有信息没问到）",
                        "**先逐字读最后一条 assistant 消息**——那是你刚说的收尾话。"
                        "本句要像真人想起还有一件事没问那样自然补问，用「对了」「另外」衔接。",
                        "本节点还没确认的信息：",
                        field_lines,
                        "硬约束：",
                        "- 只问上面列出的信息，用一句话问出来，问完停下等用户。",
                        "- 不复述、不改写上一句已经说过的话；不要再说「好的」「我了解了」这类收尾词。",
                        "- 不重新打招呼、不报家门、不替用户回答。",
                        directive_line.strip(),
                    ]
                ).strip()
            )
            return sections
        if continuation_prefix:
            sections.append(
                "\n".join(
                    [
                        "## 环节过渡（你刚说完上一节点，正接着说下一节点的第一句）",
                        "最后一条 assistant 消息，是你自己在同一通电话里刚说出口的话。**先逐字读它**，再像真人接着自己的话往下说——把它当成你这句话的上半句，你现在只是顺着说下半句。",
                        "成功标准：上下两句连起来听，像一个人一口气说下来的；没有重新开场、不和上一句撞开头词、没有重新打招呼。",
                        "硬约束：",
                        "- 本节点该问的问题别因为衔接而省略——环节过渡时本节点还一次都没问过，你把开场换成自然承接即可，但要问的核心问题必须问出来。（如果本节点本身不需要提问——比如结束语/告别/转接——就把本节点该做的事说完，不要硬造一个问题。）",
                        "- 不要和上一句撞第一个承接词：上一句以「好的」开头，本句就别再用「好的」起头；换「那」「对了」「另外」或直接进入问题都行。（即使上一句里出现过「那」，本句用「那」起头也可以——只盯住别和上一句的第一个词撞。）",
                        "- 不复述上一句说过的内容，也不要换个说法把它再说一遍。",
                        "- 不重新打招呼、不说「您好」、不报家门。",
                        "- 节点提示里的「字段→追问句式表」给的是本节点要问的核心字段和安全措辞，不是固定开场白；环节过渡时不要照搬整句开场，只取要问的核心，顺着上一句的语气自然问出来。",
                        "做法：",
                        "- 上一句是铺垫/承接（如「好的，那打扰您几分钟」）→ 直接进入本节点正事，用「那」「对了」「另外」起头或直接问。",
                        "- 上一句已带出信息（如「想了解您术后恢复情况」）→ 顺着这条信息引出本节点的问题。",
                        "反例：上句「好的，那打扰您几分钟。主要是想了解一下您术后恢复情况。」+ 本句「好的，那我先确认一下，请问您是患者本人吗？」 ← 连着两个「好的，那」，像重新打了一遍电话。",
                        "正例：上句「好的，那打扰您几分钟。主要是想了解一下您术后恢复情况。」+ 本句「那我先跟您核对一下，您是患者本人，还是家属帮忙接的呀？」 ← 换掉撞车的「好的」、直接接正题，一口气说下来。",
                    ]
                )
            )
        return sections

    @staticmethod
    def _silent_rescue_current_node_section(node_config: NodeConfig, flow_state: FlowState) -> str:
        missing_fields = (
            node_config.completion_policy.unsatisfied_fields(flow_state) if node_config.completion_policy else []
        )
        lines = [
            "## 当前节点续说目标（沉默兜底）",
            f"当前节点：{node_config.name}",
            "成功标准：先自然承接最后一条 assistant 消息；如果它只是铺垫，就顺着补问当前节点最该问的一件事。",
            "不要回到已完成节点，不要追问已确认字段；不要为了补问牺牲自然衔接。",
        ]
        if missing_fields:
            lines.append("当前节点仍缺：")
            for field in missing_fields:
                lines.append(f"- {field}：{node_config.field_definitions.get(field, field)}")
            first = missing_fields[0]
            lines.append(f"下一步只问第一个缺失字段：{first}：{node_config.field_definitions.get(first, first)}")
            if len(missing_fields) > 1:
                lines.append("其余缺失字段等用户回答后再问，不要一次问多个问题。")
        else:
            lines.append("当前节点没有未完成必填字段：只轻声确认对方是否还在，或表达可以等一等，不新增流程问题。")
        return "\n".join(lines)

    @staticmethod
    def _stage2_entry_target_section(
        *,
        node_config: NodeConfig,
        target_field: str,
        target_field_description: str,
    ) -> str:
        lines = [
            "## Stage 2 Runtime Target",
            f"当前节点：{node_config.name}",
        ]
        if target_field:
            lines.extend(
                [
                    "本句必须完成：ASK_ONE_NEW_FIELD",
                    "必须问出 runtime 指定的目标字段。",
                    "只问这个字段：",
                    f"- {target_field}：{target_field_description or node_config.field_definitions.get(target_field, target_field)}",
                    "硬性输出要求：本句必须是问题，必须带问号。",
                    "不要一次问多个字段；其余字段等用户回答后再问。",
                    "不要只铺垫，不要只说接下来要了解什么。",
                ]
            )
        else:
            lines.extend(
                [
                    "本句必须完成：NO_FIELD_QUESTION_REQUIRED",
                    "当前节点不需要新的字段问题。",
                    "自然完成当前节点的任务，不要硬造问号。",
                    "不复述上一句（最后一条 assistant）说过的话，也不要换个说法再说一遍。",
                    "不要和上一句撞第一个承接词：上一句以「好的」开头，本句就别再用「好的」起头。",
                    "上一句若已道谢/已确认，本句不要再道谢一次；整段合起来只收一次尾。",
                ]
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal — prompt composition
    # ------------------------------------------------------------------

    def _compose_stable_system(
        self,
        agent_config: AgentConfig,
        node_config: NodeConfig,
    ) -> str:
        """Assemble STABLE system prompt (persona + task).

        This content is the same for every turn within a node, enabling
        DashScope prefix cache hits. Injections go in a separate message.
        """
        sections: list[str] = []
        for msg in agent_config.persona_messages:
            sections.append(msg["content"])
        for msg in node_config.task_messages:
            sections.append(msg["content"])
        return "\n\n".join(sections)

    def _compose_injections(
        self,
        flow_state: FlowState,
        injections: TurnInjections | None,
    ) -> str | None:
        """Assemble per-turn injections as a SEPARATE uncached message.

        Kept out of the stable system prompt to preserve prefix cache.
        """
        if not injections:
            return None

        sections: list[str] = []
        if injections.last_exchange:
            sections.append(f"[对话聚焦] {injections.last_exchange}")
        if injections.patient_memory:
            sections.append(f"[患者信息] {injections.patient_memory}")
        if injections.sop_progress:
            sections.append(f"[已收集] {injections.sop_progress}")
        if self._window.summary and self._window.summary_epoch == flow_state.flow_epoch:
            sections.append(f"[摘要] {self._window.summary}")
        if injections.rag_context:
            sections.append(f"[知识] {injections.rag_context}")
        if injections.custom:
            for key, value in injections.custom.items():
                sections.append(f"[{key}] {value}")

        return "\n".join(sections) if sections else None

    # ------------------------------------------------------------------
    # Internal — summary generation
    # ------------------------------------------------------------------

    async def _generate_summary(
        self,
        conversation_history: ConversationHistory,
        flow_state: FlowState,
        agent_config: AgentConfig,
        config: ContextStrategyConfig,
    ) -> str | None:
        if self._summary_llm is None:
            return None
        history_messages = conversation_history.get_messages_from(self._window.anchor_turn_index)
        if not history_messages:
            return None
        summary_prompt = config.summary_prompt or "请用3-5句话总结对话关键信息。"
        try:
            summary = await asyncio.wait_for(
                self._summary_llm.summarize(
                    self._format_for_summary(history_messages),
                    summary_prompt,
                    config.summary_max_tokens,
                ),
                timeout=config.summary_timeout_s,
            )
            return summary
        except asyncio.TimeoutError:
            return None

    @staticmethod
    def _format_for_summary(messages: list[dict]) -> str:
        lines: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)
