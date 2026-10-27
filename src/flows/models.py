from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Callable, ClassVar

from src.flows.knowledge_session import KnowledgeReviewState
from src.flows.sop_decision_models import FastJudgeState, ShortTermMemoryState, SopProgressState

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SideEffectLevel(IntEnum):
    NONE = 0
    READ = 1
    WRITE = 2
    TRANSFER = 3


class InterruptPolicy(Enum):
    CANCELLABLE = "CANCELLABLE"
    MUST_COMPLETE = "MUST_COMPLETE"
    BLOCK_INTERRUPTION = "BLOCK_INTERRUPTION"


class ContextStrategy(Enum):
    APPEND = "APPEND"
    RESET = "RESET"
    RESET_WITH_SUMMARY = "RESET_WITH_SUMMARY"


class TransitionKind(Enum):
    ADVANCE = "ADVANCE"
    SKIP = "SKIP"
    TERMINAL = "TERMINAL"
    DIVERT = "DIVERT"
    RESUME = "RESUME"


# Sentinel returned by route_fn to signal direct terminal hook (no node transition)
TERMINAL_ROUTE = "__terminal__"
# Sentinel returned by check_and_advance when Speech wants to advance but Judge disagrees
NEEDS_FOLLOWUP = "__needs_followup__"


class FieldStatus(Enum):
    CONFIRMED = "CONFIRMED"
    INFERRED = "INFERRED"
    REFUSED = "REFUSED"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class NodeStatus(Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    SKIPPED = "SKIPPED"
    SUSPENDED = "SUSPENDED"


class EntryType(Enum):
    USER = "USER"
    ASSISTANT_TEXT = "ASSISTANT_TEXT"
    ASSISTANT_TOOL_CALL = "ASSISTANT_TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FlowError(Exception):
    """Base error for flow layer."""


class ToolError(Exception):
    """Error raised by tool execution."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


# ---------------------------------------------------------------------------
# Dataclasses — config
# ---------------------------------------------------------------------------


@dataclass
class ContextStrategyConfig:
    strategy: ContextStrategy = ContextStrategy.APPEND
    summary_prompt: str | None = None
    summary_max_tokens: int = 200
    summary_timeout_s: float = 5.0


@dataclass
class Action:
    type: str
    params: dict = field(default_factory=dict)


@dataclass
class NodeConfig:
    name: str
    task_messages: list[dict]
    tools: list[str]  # handler_id refs
    context_strategy: ContextStrategyConfig = field(default_factory=ContextStrategyConfig)
    respond_immediately: bool = True
    pre_actions: list[Action] | None = None
    post_actions: list[Action] | None = None
    allowed_fields: set[str] | None = None
    successors: list[str] | None = None
    # Dual-LLM fields
    completion_policy: CompletionPolicy | None = None
    completion_criteria: Callable[[FlowState], bool] | None = None
    judge_task_description: str = ""
    next_stage_description: str = ""
    field_definitions: dict[str, str] = field(default_factory=dict)
    # Conditional routing: returns target node name or "__terminal__"
    route_fn: Callable[[FlowState], str] | None = None
    # Buffer phrase pushed to TTS before Stage 1 inference, covering
    # thinking-model latency for nodes with slow first-token times.
    # E.g. knowledge_qa uses "好的，" so the patient hears immediate feedback
    # while Flash+think512 spends 3-5s in the thinking phase.
    followup_buffer: str = ""
    # Natural-language completion condition injected into Speech prompt for dual-vote.
    # Empty = this node does not participate in dual-vote (speech_advance=None).
    speech_completion_hint: str = ""


@dataclass
class AgentConfig:
    name: str
    persona_messages: list[dict]
    global_tools: list[str]
    initial_node: str
    nodes: dict[str, NodeConfig] = field(default_factory=dict)


@dataclass
class ToolSpec:
    handler_id: str
    name: str
    description: str
    parameters: dict
    side_effect: SideEffectLevel = SideEffectLevel.NONE
    interrupt_policy: InterruptPolicy = InterruptPolicy.CANCELLABLE
    timeout_s: float | None = None
    may_transition: bool = False
    visible_in_nodes: set[str] | None = None
    hidden_in_nodes: set[str] | None = None


# ---------------------------------------------------------------------------
# Dataclasses — runtime
# ---------------------------------------------------------------------------


@dataclass
class NodeTransition:
    target_node: str
    kind: TransitionKind
    reason: str


@dataclass
class ToolPolicy:
    max_side_effect: SideEffectLevel
    is_speculative: bool


@dataclass
class TurnPlan:
    messages: list[dict]
    available_tools: dict[str, ToolSpec]
    tools_schema: list[dict]
    node_id: str
    flow_epoch: int
    inference_id: str
    tool_policy: ToolPolicy


@dataclass
class ToolOutcome:
    tool_payload: str | dict | None
    result_text: str
    transition: NodeTransition | None
    side_effect_level: SideEffectLevel


@dataclass
class ToolResult:
    output: str | dict | None = None
    transition_to: str | None = None
    transition_kind: TransitionKind | None = None
    transition_reason: str | None = None


@dataclass
class NodeState:
    status: NodeStatus = NodeStatus.PENDING
    entered_turn: int | None = None
    completed_turn: int | None = None
    completion_reason: str | None = None
    suspended_node_turn_count: int | None = None


@dataclass
class FieldValue:
    value: Any
    status: FieldStatus
    source_node: str
    source_turn: int


@dataclass
class FlowState:
    flow_epoch: int = 0
    current_node: str = ""
    node_history: list[str] = field(default_factory=list)
    node_states: dict[str, NodeState] = field(default_factory=dict)
    patient_info: dict = field(default_factory=dict)
    collected_fields: dict[str, FieldValue] = field(default_factory=dict)
    turn_count: int = 0
    node_turn_count: int = 0
    active_turn_id: str | None = None
    active_inference_id: str | None = None
    divert_return_node: str | None = None
    divert_context_snapshot: ContextWindowSnapshot | None = None
    call_working_memory: Any = None
    applied_sop_decision_ids: set[str] = field(default_factory=set)
    fast_judge_state: FastJudgeState | None = None
    short_term_memory: ShortTermMemoryState = field(default_factory=ShortTermMemoryState)
    sop_progress_state: SopProgressState = field(default_factory=SopProgressState)
    # Set when Stage 2 of the prior turn already asked a node's question while
    # entering it. The next turn's Stage 1 reads this to avoid re-asking the
    # same question (see _dynamic_context_sections / llm_processor stage2 path).
    node_question_already_asked: dict[str, bool] = field(default_factory=dict)
    knowledge_review_state: KnowledgeReviewState = field(default_factory=KnowledgeReviewState)


@dataclass
class TurnInjections:
    rag_context: str | None = None
    patient_memory: str | None = None
    sop_progress: str | None = None
    last_exchange: str | None = None
    custom: dict[str, str] | None = None


@dataclass
class ContextWindow:
    anchor_turn_index: int = 0
    summary: str | None = None
    summary_epoch: int | None = None


@dataclass(frozen=True)
class ContextWindowSnapshot:
    anchor_turn_index: int
    summary: str | None
    summary_epoch: int | None


@dataclass
class HistoryEntry:
    message: dict
    turn_index: int
    entry_type: EntryType
    node_id: str
    timestamp: float


# ---------------------------------------------------------------------------
# Dataclasses — dual-LLM judge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompletionPolicy:
    """Structured stage completion check."""

    required_fields: list[str]
    accept_refused: bool = True
    min_confirmed: int | None = None

    def unsatisfied_fields(self, state: FlowState) -> list[str]:
        """Return required_fields not yet resolved (CONFIRMED or accepted REFUSED)."""
        missing = []
        for f in self.required_fields:
            fv = state.collected_fields.get(f)
            if fv is None:
                missing.append(f)
            elif fv.status == FieldStatus.CONFIRMED:
                continue
            elif fv.status == FieldStatus.REFUSED and self.accept_refused:
                continue
            else:
                missing.append(f)
        return missing

    def is_satisfied(self, state: FlowState) -> bool:
        if self.unsatisfied_fields(state):
            return False
        if self.min_confirmed is not None:
            resolved = sum(
                1
                for f in self.required_fields
                if (fv := state.collected_fields.get(f)) and fv.status == FieldStatus.CONFIRMED
            )
            if resolved < self.min_confirmed:
                return False
        return True


@dataclass(frozen=True)
class ExtractedField:
    value: Any
    status: FieldStatus


@dataclass(frozen=True)
class JudgeResult:
    extracted_fields: dict[str, ExtractedField]
    stage_complete: bool
    transfer_to_human: bool
    reason: str
    transfer_reason: str = "none"  # user_request | emotional_escalation | medical_risk | none
    risk_level: str = "normal"  # normal | elevated | urgent
    knowledge_query: bool = False

    EMPTY: ClassVar[JudgeResult]

    @staticmethod
    def parse(raw_text: str) -> JudgeResult:
        """Parse Judge LLM JSON output. Returns EMPTY on failure."""
        try:
            data = json.loads(raw_text)
            fields = {}
            for k, v in data.get("extracted_fields", {}).items():
                status = FieldStatus(v["status"].upper())
                fields[k] = ExtractedField(value=v["value"], status=status)
            return JudgeResult(
                extracted_fields=fields,
                stage_complete=data.get("stage_complete", False),
                transfer_to_human=data.get("transfer_to_human", False),
                reason=data.get("reason", ""),
                transfer_reason=data.get("transfer_reason", "none"),
                risk_level=data.get("risk_level", "normal"),
                knowledge_query=data.get("knowledge_query", False),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return JudgeResult.EMPTY


JudgeResult.EMPTY = JudgeResult(
    extracted_fields={},
    stage_complete=False,
    transfer_to_human=False,
    reason="parse_error",
    transfer_reason="none",
    risk_level="normal",
    knowledge_query=False,
)


@dataclass(frozen=True)
class TransferCommand:
    """Transfer command produced by FlowRuntime, executed by LLMProcessor."""

    reason: str  # TransferReason.value
    priority: str  # TransferPriority.value
    trigger_text: str
    transition_message: str
    source_node: str
    collected_fields: dict[str, Any]
    patient_info: dict[str, Any]
    conversation_turns: int
    nodes_completed: list[str]
