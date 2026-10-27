from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

NodeAction = Literal["stay", "followup", "advance", "repair", "divert", "handoff", "terminal"]
FieldStatusText = Literal["confirmed", "inferred", "refused", "unknown", "not_applicable", "pending"]
Stage1Posture = Literal["asked", "ack_only", "closed_unasked"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FieldCandidate(StrictModel):
    field: str
    value: Any = None
    status: FieldStatusText
    evidence_span: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    is_correction: bool = False


class MissingField(StrictModel):
    field: str
    reason: str


class SpeechRisk(StrictModel):
    type: str
    severity: Literal["low", "medium", "high", "critical"] = "low"
    reason: str = ""


class FastJudgeAnswerAssessment(StrictModel):
    current_question_answered: Literal[
        "answered",
        "partially_answered",
        "not_answered",
        "refused",
        "needs_reconfirmation",
        "unclear",
        "not_applicable",
    ] = "unclear"
    needs_followup: bool = False
    followup_reason: str = ""
    missing_fields: list[str] = Field(default_factory=list)
    reason_source: Literal["model", "fallback", "absent"] = "absent"


class FastJudgeFieldDelta(StrictModel):
    field: str
    value: Any = None
    status: Literal["confirmed", "inferred", "refused", "unknown", "not_applicable"] = "confirmed"
    evidence_span: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    is_correction: bool = False


class FastJudgeLookaheadFieldDelta(FastJudgeFieldDelta):
    node_id: str


class FastJudgeHandoff(StrictModel):
    transfer_to_human: bool = False
    reason: Literal["none", "user_request", "emotional_escalation", "medical_risk", "other"] = "none"
    risk_level: Literal["normal", "elevated", "urgent"] = "normal"


class FastJudgeKnowledgeQuestion(StrictModel):
    detected: bool = False
    query: str = ""


class FastJudgeSpeechDirective(StrictModel):
    code: str = ""
    text: str = ""


class FastJudgeModelFieldDelta(FastJudgeFieldDelta):
    node_id: str = ""


class FastJudgeModelOutput(StrictModel):
    reason: str = Field(default="", max_length=200)
    field_delta: list[FastJudgeModelFieldDelta]
    missing_fields: list[str]
    node_action: NodeAction
    speech_directive: FastJudgeSpeechDirective
    stage1_posture: Stage1Posture = "asked"


class FastJudgeResult(StrictModel):
    schema_version: Literal["fast_judge.v1"]
    turn_id: str
    node_id: str
    speech_result_id: str
    evaluated_until: Literal["speech_text_done", "playout_done", "formal_confirmed"]
    field_delta: list[FastJudgeFieldDelta] = Field(default_factory=list)
    lookahead_field_delta: list[FastJudgeLookaheadFieldDelta] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    stage_complete: bool = False
    node_action: NodeAction = "stay"
    node_after_candidate: str = ""
    next_node_goal: str = ""
    stage2_should_start: bool = False
    stage2_target_node: str = ""
    stage2_reason: str = ""
    answer_assessment: FastJudgeAnswerAssessment = Field(default_factory=FastJudgeAnswerAssessment)
    handoff: FastJudgeHandoff = Field(default_factory=FastJudgeHandoff)
    knowledge_question: FastJudgeKnowledgeQuestion = Field(default_factory=FastJudgeKnowledgeQuestion)
    speech_directive: FastJudgeSpeechDirective = Field(default_factory=FastJudgeSpeechDirective)
    stage1_posture: Stage1Posture = "asked"


class FastJudgeInternalIdentity(StrictModel):
    turn_id: str
    turn_index: int = Field(ge=0)
    node_id: str
    flow_epoch: int = Field(ge=0)
    context_snapshot_id: str
    speech_result_id: str
    speech_text_hash: str
    fast_judge_prompt_hash: str
    weighted_context_hash: str


class FastJudgeEnvelope(StrictModel):
    result: FastJudgeResult
    identity: FastJudgeInternalIdentity


class FieldSnapshot(StrictModel):
    value: Any = None
    status: str = ""
    source_node: str = ""
    source_turn: int | None = None
    needs_reconfirmation: bool = False


class FastJudgeState(StrictModel):
    source_turn_id: str
    speech_result_id: str
    node_before: str
    node_after: str = ""
    evaluated_until: Literal["speech_text_done", "playout_done", "formal_confirmed"]
    committed_delta: list[FastJudgeFieldDelta] = Field(default_factory=list)
    lookahead_committed_delta: list[FastJudgeLookaheadFieldDelta] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    answer_assessment: FastJudgeAnswerAssessment = Field(default_factory=FastJudgeAnswerAssessment)
    node_action: NodeAction = "stay"
    next_node_goal: str = ""
    stage2_should_start: bool = False
    stage2_target_node: str = ""
    stage2_reason: str = ""
    handoff: FastJudgeHandoff = Field(default_factory=FastJudgeHandoff)
    knowledge_question: FastJudgeKnowledgeQuestion = Field(default_factory=FastJudgeKnowledgeQuestion)
    speech_directive: FastJudgeSpeechDirective = Field(default_factory=FastJudgeSpeechDirective)
    stage1_posture: Stage1Posture = "asked"


SopProgressCandidateStatus = Literal[
    "confirmed",
    "inferred",
    "refused",
    "needs_reconfirmation",
    "conflict",
    "unknown",
]


class SopProgressFieldCandidate(StrictModel):
    node_id: str
    field: str
    value: Any = None
    status: SopProgressCandidateStatus
    evidence_span: str = ""
    source_turn_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    conflicts_with_fast_judge: bool = False


class FastJudgeSemanticContext(StrictModel):
    turn_id: str
    turn_index: int = Field(ge=0)
    node_id: str
    flow_epoch: int = Field(ge=0)
    speech_result_id: str
    speech_text: str
    user_text: str
    speech_asked_this_turn: str
    intended_ask_fields: list[str] = Field(default_factory=list)
    required_fields: list[str] = Field(default_factory=list)
    eligible_optional_fields: list[str] = Field(default_factory=list)
    successor_path: list[str] = Field(default_factory=list)
    node_field_catalog: dict[str, dict[str, str]] = Field(default_factory=dict)
    pending_fields_before_turn: list[str] = Field(default_factory=list)
    known_fields_before_turn: dict[str, FieldSnapshot] = Field(default_factory=dict)
    candidate_fields_before_turn: list[SopProgressFieldCandidate] = Field(default_factory=list)
    recent_relevant_context: str = ""
    short_term_memory: list[JudgeNoteDTO] = Field(default_factory=list)


class JudgeNoteDTO(StrictModel):
    category: Literal["correction_hint", "future_node_hint", "salient_fact"]
    text: str


class ShortTermMemoryItem(StrictModel):
    key: str
    text: str
    category: Literal[
        "caller_identity",
        "resolved_topic",
        "future_node_hint",
        "correction_hint",
        "salient_fact",
    ] = "salient_fact"
    source_turn_ids: list[str] = Field(default_factory=list)
    available_before_turn_id: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    expires_after_turn_index: int | None = None


class ShortTermMemoryState(StrictModel):
    items: list[ShortTermMemoryItem] = Field(default_factory=list)
    last_accepted_turn_index: int = -1


class ShortTermMemoryPatch(StrictModel):
    schema_version: Literal["short_term_memory_patch.v1"]
    turn_id: str
    turn_index: int = Field(ge=0)
    flow_epoch: int = Field(ge=0)
    source: Literal["short_term_memory_analyzer"]
    analyzed_turn_ids: list[str] = Field(default_factory=list)
    available_before_turn_id: str
    upsert_items: list[ShortTermMemoryItem] = Field(default_factory=list)
    remove_keys: list[str] = Field(default_factory=list)


class SopProgressState(StrictModel):
    field_candidates: list[SopProgressFieldCandidate] = Field(default_factory=list)
    summary_for_future_judges: str = ""


class SopProgressPatch(StrictModel):
    schema_version: Literal["sop_progress_patch.v1"]
    turn_id: str
    source: Literal["sop_progress_analyzer"]
    flow_epoch: int = Field(ge=0)
    analyzed_turn_ids: list[str] = Field(default_factory=list)
    available_after_turn_id: str
    field_candidates: list[SopProgressFieldCandidate] = Field(default_factory=list)
    summary_for_future_judges: str = ""
    reconfirmation_notes: list[str] = Field(default_factory=list)


class SpeakingFrame(StrictModel):
    caller_role: Literal["patient", "proxy_family", "proxy_other", "unknown"] = "unknown"
    relationship: str = ""
    refer_to_patient_as: str = "患者"
    avoid_phrases: list[str] = Field(default_factory=list)
    preferred_phrases: list[str] = Field(default_factory=list)
    evidence: str = ""
    source_turn_id: str = ""


class CommittedSopField(StrictModel):
    value: Any = None
    status: FieldStatusText
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = ""
    source_turn_id: str
    source_node: str
    active: bool = True


class PendingClarification(StrictModel):
    field: str
    reason: str
    asked_count: int = Field(ge=0)
    last_asked_turn_id: str = ""


class FutureObservation(StrictModel):
    field: str
    value: Any = None
    target_node: str
    evidence: str
    source_turn_id: str
    usable_when_node_active: bool = True


class DoNotRepeatFact(StrictModel):
    key: str
    reason: str
    source_turn_id: str


class CorrectionRecord(StrictModel):
    field: str
    old_value: Any = None
    new_value: Any = None
    evidence: str
    source_turn_id: str


class UserInteractionProfile(StrictModel):
    answer_style: Literal["brief", "detailed", "digressive", "confused", "unknown"] = "unknown"
    emotion: Literal["neutral", "anxious", "impatient", "upset", "unknown"] = "unknown"
    hearing_or_comprehension: Literal["normal", "needs_rephrase", "possible_hearing_issue", "unknown"] = "unknown"
    sensitivity: list[str] = Field(default_factory=list)


class OpenLoop(StrictModel):
    type: Literal["patient_question", "knowledge_divert", "interruption_repair"]
    content: str
    status: Literal["open", "answered", "deferred", "resumed"]
    source_turn_id: str


class PlaybackTruth(StrictModel):
    last_assistant_generated_text: str = ""
    last_assistant_heard_text: str = ""
    last_interrupted: bool = False


class NextSpeechGoal(StrictModel):
    mode: Literal["ack_then_ask", "transition", "repair", "clarify", "answer_then_resume", "close"] = "ack_then_ask"
    ask_field: str = ""
    max_questions: int = 1
    must_acknowledge: list[str] = Field(default_factory=list)
    forbidden_moves: list[str] = Field(default_factory=list)


class NextSpeechStrategy(StrictModel):
    strategy_id: str
    source_turn_id: str
    node_before: str
    node_after: str
    node_action: NodeAction
    committed_delta: list[FieldCandidate] = Field(default_factory=list)
    open_slots: list[MissingField] = Field(default_factory=list)
    do_not_repeat: list[DoNotRepeatFact] = Field(default_factory=list)
    previous_speech_text: str = ""
    heard_status: Literal["full", "interrupted", "unknown"] = "unknown"
    next_speech_goal: NextSpeechGoal = Field(default_factory=NextSpeechGoal)
    suggested_wording: str = ""


class CallWorkingMemory(StrictModel):
    speaking_frame: SpeakingFrame = Field(default_factory=SpeakingFrame)
    sop_fields: dict[str, CommittedSopField] = Field(default_factory=dict)
    pending_clarifications: list[PendingClarification] = Field(default_factory=list)
    future_observations: list[FutureObservation] = Field(default_factory=list)
    do_not_repeat: list[DoNotRepeatFact] = Field(default_factory=list)
    corrections: list[CorrectionRecord] = Field(default_factory=list)
    user_interaction_profile: UserInteractionProfile = Field(default_factory=UserInteractionProfile)
    open_loops: list[OpenLoop] = Field(default_factory=list)
    playback_truth: PlaybackTruth = Field(default_factory=PlaybackTruth)
    next_speech_strategy: NextSpeechStrategy | None = None


class MemoryPatch(StrictModel):
    schema_version: Literal["call_memory_patch.v1"]
    turn_id: str
    turn_index: int = Field(ge=0)
    node_id: str
    flow_epoch: int = Field(ge=0)
    source: Literal["memory_extractor", "state_committer"]
    speaking_frame: SpeakingFrame | None = None
    sop_fields: dict[str, CommittedSopField] = Field(default_factory=dict)
    pending_clarifications: list[PendingClarification] = Field(default_factory=list)
    future_observations: list[FutureObservation] = Field(default_factory=list)
    do_not_repeat: list[DoNotRepeatFact] = Field(default_factory=list)
    corrections: list[CorrectionRecord] = Field(default_factory=list)
    user_interaction_profile: UserInteractionProfile | None = None
    open_loops: list[OpenLoop] = Field(default_factory=list)
    playback_truth: PlaybackTruth | None = None
