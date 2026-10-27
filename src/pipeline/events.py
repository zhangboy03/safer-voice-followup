"""Typed event bus for pipeline observation.

BaseEvent is the root of all events. EventBus provides synchronous,
reentrant fan-out to PipelineObserver subscribers.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal

from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.pipeline.observer import PipelineObserver

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BaseEvent:
    wire_topic: ClassVar[str]
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    mono_ms: float = field(default_factory=lambda: time.monotonic() * 1000)
    wall_ts: float = field(default_factory=time.time)
    session_id: str = ""
    turn_id: str = ""

    def to_wire(self) -> dict:
        wt = getattr(type(self), "wire_topic", None)
        if not wt:
            raise TypeError(f"{type(self).__name__} missing wire_topic")
        return {"topic": wt, "event_type": type(self).__name__, **asdict(self)}


class EventBus:
    def __init__(self) -> None:
        self._observers: list[PipelineObserver] = []
        self._queue: deque[BaseEvent] = deque()
        self._draining: bool = False

    def subscribe(self, observer: PipelineObserver) -> None:
        self._observers.append(observer)

    def emit(self, event: BaseEvent) -> None:
        self._queue.append(event)
        if self._draining:
            return
        self._draining = True
        try:
            while self._queue:
                e = self._queue.popleft()
                for obs in self._observers:
                    try:
                        obs.on_event(e)
                    except Exception as err:
                        logger.warning("observer_failed", observer=type(obs).__name__, error=str(err))
        finally:
            self._draining = False


# ── Lifecycle events ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PipelineStartedEvent(BaseEvent):
    wire_topic = "pipeline_started"
    pipeline_id: str = ""


@dataclass(frozen=True, slots=True)
class AgentReadyEvent(BaseEvent):
    wire_topic = "agent_ready"
    prewarm_duration_ms: float = -1.0


@dataclass(frozen=True, slots=True)
class RoomJoinedEvent(BaseEvent):
    wire_topic = "room_joined"
    identity: str = ""
    role: str = ""


@dataclass(frozen=True, slots=True)
class SessionClosedEvent(BaseEvent):
    wire_topic = "session_closed"
    reason: str = ""
    turn_count: int = 0


# ── Turn events ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class AssistantTextEvent(BaseEvent):
    """Generated TTS input caption; never proof that all this text was heard."""
    wire_topic = "assistant_text"
    context_id: str = ""
    text: str = ""
    is_final: bool = False
    source: str = "generated"


@dataclass(frozen=True, slots=True)
class UserStartedSpeakingEvent(BaseEvent):
    wire_topic = "user_started"


@dataclass(frozen=True, slots=True)
class UserStoppedSpeakingEvent(BaseEvent):
    wire_topic = "user_stopped"
    silence_duration_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class AsrInterimEvent(BaseEvent):
    wire_topic = "asr_interim"
    text: str = ""
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class AsrFinalEvent(BaseEvent):
    wire_topic = "asr_final"
    text: str = ""
    confidence: float = 0.0
    latency_ms: float | None = None


@dataclass(frozen=True, slots=True)
class AsrPackageLatencyEvent(BaseEvent):
    wire_topic = "asr_pkg_latency"
    first_package_ms: float = -1.0
    total_ms: float = -1.0


@dataclass(frozen=True, slots=True)
class SmartTurnEmitEvent(BaseEvent):
    wire_topic = "smart_turn_emit"
    text: str = ""
    text_type: str = ""
    inference_ms: float = -1.0
    source: str = ""


@dataclass(frozen=True, slots=True)
class TurnShadowEvent(BaseEvent):
    wire_topic = "turn_shadow"
    text: str = ""
    results: list[dict] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class LLMRequestStartEvent(BaseEvent):
    wire_topic = "llm_request_start"
    actual_text: str = ""


@dataclass(frozen=True, slots=True)
class LLMFirstTokenEvent(BaseEvent):
    wire_topic = "llm_first_token"
    ttft_ms: float = -1.0
    actual_text: str = ""


@dataclass(frozen=True, slots=True)
class TTSSessionCreatedEvent(BaseEvent):
    wire_topic = "tts_session_created"
    create_ms: float = -1.0
    pool_hit: bool | None = None


@dataclass(frozen=True, slots=True)
class TTSFirstAudioEvent(BaseEvent):
    wire_topic = "tts_first_audio"
    ttfa_ms: float = -1.0
    first_audio_source: str = ""


@dataclass(frozen=True, slots=True)
class TTSCompleteEvent(BaseEvent):
    wire_topic = "tts_complete"
    context_id: str = ""
    text_fed: str = ""
    audio_duration_ms: float = -1.0


@dataclass(frozen=True, slots=True)
class SopDecisionTraceEvent(BaseEvent):
    wire_topic = "sop_decision_trace"
    node_id: str = ""
    speech_first_token_ms: float = -1
    speech_text_done_ms: float = -1
    tts_first_audio_ms: float = -1
    current_playback_end_ms: float = -1
    judge_start_ms: float = -1
    judge_done_ms: float = -1
    commit_done_ms: float = -1
    same_playout_append_attempted: bool = False
    same_playout_append_skipped_reason: str = ""
    stage1_to_stage2_gap_ms: float = -1
    prompt_hashes: dict = field(default_factory=dict)
    context_snapshot_id: str = ""
    weighted_context_hash: str = ""


@dataclass(frozen=True, slots=True)
class TTSInterruptedEvent(BaseEvent):
    wire_topic = "tts_interrupted"
    context_id: str = ""
    text_fed: str = ""
    audio_bytes_sent: int = 0


@dataclass(frozen=True, slots=True)
class FirstAudioQueuedEvent(BaseEvent):
    wire_topic = "first_audio_queued"


@dataclass(frozen=True, slots=True)
class AgentTTSDoneEvent(BaseEvent):
    wire_topic = "agent_tts_done"
    context_id: str = ""
    kind: Literal["normal_reply", "transfer_transition"] = "normal_reply"
    total_duration_ms: float = -1.0


@dataclass(frozen=True, slots=True)
class TTSPlayoutDrainedEvent(BaseEvent):
    """The audio source has truly finished emitting every captured frame for a
    context (``rtc.AudioSource.wait_for_playout()`` returned) — i.e. when the
    patient actually stops hearing the audio.

    Distinct from ``AgentTTSDoneEvent`` / ``TTSCompleteEvent``, which fire at
    *generation*-done: under CosyVoice duplex streaming, generation completes
    several times faster than realtime, so those events land ~1s into a ~6s
    utterance. Barge-in measurement (Study 1) needs the *audible* end, not the
    generation end — see ``src/eval/study1_barge_in.py``.
    """

    wire_topic = "tts_playout_drained"
    context_id: str = ""
    audio_duration_ms: float = -1.0


@dataclass(frozen=True, slots=True)
class TurnMetricsEvent(BaseEvent):
    wire_topic = "turn_metrics"
    smart_turn_ms: float = -1.0
    vad_tail_wait_ms: float = -1.0
    user_stop_to_emit_ms: float = -1.0
    gate_wait_ms: float = -1.0
    asr_text_at_emit: str = ""
    asr_text_type: str = ""
    asr_final_text: str = ""
    asr_final_latency_ms: float = -1.0
    asr_diff: str = ""
    emit_to_llm_ms: float = -1.0
    llm_ttft_ms: float = -1.0
    llm_actual_text: str = ""
    tts_session_create_ms: float = -1.0
    tts_pool_hit: bool | None = None
    tts_ttfa_ms: float = -1.0
    shadow_results: list[dict] | None = None
    emit_to_audio_ms: float = -1.0
    output_queue_wait_ms: float = -1.0
    e2e_ms: float = -1.0
    speech_end_to_audio_ms: float = -1.0
    first_audio_source: str = ""
    first_audio_is_bridge: bool = False
    unexplained_gap_ms: float = -1.0


# ── Flow / Judge / RAG / Transfer events ─────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SopTransitionEvent(BaseEvent):
    wire_topic = "sop_transition"
    from_node: str = ""
    to_node: str = ""
    trigger: str = ""
    extracted_fields: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class JudgeDecisionEvent(BaseEvent):
    wire_topic = "dual_llm_debug"
    node: str = ""
    decision: str = ""
    fields: dict = field(default_factory=dict)
    confidence: float = 0.0
    latency_ms: float = -1.0


@dataclass(frozen=True, slots=True)
class RagDecisionEvent(BaseEvent):
    wire_topic = "dual_llm_debug"
    query: str = ""
    tier: str = ""
    retrieved_count: int = 0
    decision: str = ""
    latency_ms: float = -1.0


@dataclass(frozen=True, slots=True)
class TransferTriggeredEvent(BaseEvent):
    wire_topic = "transfer_triggered"
    reason: str = ""
    node: str = ""
    destination: str = ""


@dataclass(frozen=True, slots=True)
class ErrorEvent(BaseEvent):
    wire_topic = "error"
    processor: str = ""
    error_type: str = ""
    message: str = ""


@dataclass(frozen=True, slots=True)
class DriftReportEvent(BaseEvent):
    wire_topic = "drift_report"
    expected_mono_ms: float = 0.0
    actual_mono_ms: float = 0.0
    delta_ms: float = 0.0


# ── VAD shadow events (pVAD integration) ─────────────────────────────────────


@dataclass(frozen=True, slots=True)
class VADShadowFrameEvent(BaseEvent):
    """Per-frame sample from a shadow VAD backend (low-rate sampled)."""

    wire_topic = "vad_shadow_frame"
    backend: str = ""  # "pvad"
    state: str = ""  # waiting_greeting | ready_for_enroll | collecting | encoding | ready | pvad_unavailable | ...
    is_speech: bool = False
    speech_prob: float | None = None
    threshold: float | None = None
    primary_is_speech: bool = False
    enroll_progress_ms: int | None = None
    ts_mono_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class VADShadowSpeechEvent(BaseEvent):
    """Speech-boundary transition from a shadow VAD backend."""

    wire_topic = "vad_shadow_speech"
    backend: str = ""
    event_type_: str = ""  # "started" | "stopped" (suffix avoids BaseEvent wire collision)
    primary_event_type: str | None = None
    ts_mono_ms: float = 0.0
    speech_duration_ms: float | None = None


# ── Knowledge QA metrics ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class KnowledgeMetricEvent(BaseEvent):
    """Per-turn knowledge_qa metric snapshot.

    Routed by the downstream observer distinct from TurnMetricsEvent so
    TurnMetricsAggregator stays focused on speech-pipeline latencies.

    Convention: -1.0 = not measured.
    """

    wire_topic = "knowledge_metric"
    route: str = ""  # 'bridge' | 'b1' | 'b2' | 'd' | 'e1' | 'e2'
    bridge_ttfa_ms: float = -1.0
    review_wall_ms: float = -1.0
    exit_wall_ms: float = -1.0
    knowledge_first_audio_ms: float = -1.0
    review_cache_hit_rate: float = -1.0
    exit_cache_hit_rate: float = -1.0
    review_response_format_kind: str = ""
    exit_response_format_kind: str = ""
    failure_reason: str = "none"
