"""TurnMetricsAggregator — per-turn latency aggregation for real-time dashboard.

Subscribes to EventBus and collects timestamps from pipeline events,
computes deltas, and publishes a complete metrics summary via TurnMetricsEvent
when all data points are available.

Six checkpoints per turn:
  1. smart_turn_inference_ms: SmartTurn ONNX inference time
  2. asr_text_at_emit / asr_text_type: ASR state when SmartTurn decided
  3. asr_final_latency_ms: time from emit to ASR final arrival
  4. asr_final_text / asr_diff: eventual final text and diff from emitted
  5. llm_ttft_ms: LLM time to first token
  6. tts_ttfa_ms: TTS time to first audio byte
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TurnMetrics:
    """Accumulated metrics for a single turn."""

    turn_id: str = ""
    # Anchor: user stop
    turn_stop_mono_ms: float = 0.0
    vad_tail_wait_ms: float = 0.0
    # Checkpoint 1: SmartTurn
    smart_turn_inference_ms: float = -1.0
    smart_turn_emit_mono_ms: float = 0.0  # monotonic timestamp when turn was emitted
    user_stop_to_emit_ms: float = -1.0
    gate_wait_ms: float = -1.0
    # Checkpoint 2: ASR state at emit
    asr_text_at_emit: str = ""
    asr_text_type: str = ""  # "partial" | "final"
    # Checkpoint 3+4: ASR final
    asr_final_text: str = ""
    asr_final_latency_ms: float = -1.0
    asr_diff: str = ""
    # Checkpoint 5: LLM TTFT
    llm_request_start_mono_ms: float = 0.0
    emit_to_llm_ms: float = -1.0
    llm_ttft_ms: float = -1.0
    llm_actual_text: str = ""  # text the LLM actually processed (may differ from emit text after restart)
    # Checkpoint 6: TTS TTFA
    tts_session_create_ms: float = -1.0
    tts_pool_hit: bool | None = None
    tts_ttfa_ms: float = -1.0
    # Shadow comparison
    shadow_results: list[dict] | None = None
    # E2E
    emit_to_audio_ms: float = -1.0
    output_queue_wait_ms: float = -1.0
    e2e_ms: float = -1.0
    speech_end_to_audio_ms: float = -1.0
    first_audio_mono_ms: float = 0.0
    first_audio_queued_mono_ms: float = 0.0
    first_audio_source: str = ""
    unexplained_gap_ms: float = -1.0


class TurnMetricsAggregator:
    """Aggregates per-turn metrics from pipeline events.

    Implements the PipelineObserver interface via on_event(), subscribing to
    EventBus for automatic dispatch. Also supports direct record_* calls from
    processors and the legacy on_complete callback for backward compatibility.
    """

    def __init__(
        self,
        bus=None,  # EventBus | None
        on_complete: Callable[[dict], None] | None = None,
    ) -> None:
        self._bus = bus
        self._on_complete = on_complete  # legacy
        self._turns: dict[str, TurnMetrics] = {}
        # Per-turn publish milestones already reached. Each record_* call
        # advances at most one milestone per call, so a turn publishes at most
        # ~4 times (initial / asr_final / tts_ttfa / first_audio_queued /
        # shadow) instead of firing on every record_* call.
        self._pub_milestones: dict[str, set[str]] = {}

    # ── PipelineObserver interface ───────────────────────────────────────────

    def on_event(self, event) -> None:
        """PipelineObserver interface — dispatch to internal record methods."""
        from src.pipeline.events import (
            AsrFinalEvent,
            FirstAudioQueuedEvent,
            LLMFirstTokenEvent,
            LLMRequestStartEvent,
            SmartTurnEmitEvent,
            TTSFirstAudioEvent,
            TTSSessionCreatedEvent,
            TurnShadowEvent,
            UserStoppedSpeakingEvent,
        )

        if isinstance(event, UserStoppedSpeakingEvent):
            self._record_user_stop(event.turn_id, vad_tail_wait_ms=event.silence_duration_ms)
        elif isinstance(event, SmartTurnEmitEvent):
            self._record_smart_turn_emit(event.turn_id, event.inference_ms, event.text, event.text_type)
        elif isinstance(event, AsrFinalEvent):
            self._record_asr_final(event.turn_id, event.text)
        elif isinstance(event, LLMRequestStartEvent):
            self._record_llm_request_start(event.turn_id, event.actual_text)
        elif isinstance(event, LLMFirstTokenEvent):
            self._record_llm_ttft(event.turn_id, event.ttft_ms, event.actual_text)
        elif isinstance(event, TurnShadowEvent):
            self._record_shadow(event.turn_id, event.results)
        elif isinstance(event, TTSSessionCreatedEvent):
            self._record_tts_session_create(event.turn_id, event.create_ms, event.pool_hit)
        elif isinstance(event, TTSFirstAudioEvent):
            self._record_tts_ttfa(event.turn_id, event.ttfa_ms, event.first_audio_source)
        elif isinstance(event, FirstAudioQueuedEvent):
            self._record_first_audio_queued(event.turn_id)
        # TurnMetricsEvent from self → ignore (no re-aggregation)

    # ── Internal helper ──────────────────────────────────────────────────────

    def _get(self, turn_id: str) -> TurnMetrics:
        if turn_id not in self._turns:
            self._turns[turn_id] = TurnMetrics(turn_id=turn_id)
        return self._turns[turn_id]

    # ── Private recording methods (called from on_event dispatch) ────────────

    def _record_user_stop(self, turn_id: str, vad_tail_wait_ms: float = 0.0) -> None:
        m = self._get(turn_id)
        m.turn_stop_mono_ms = time.monotonic() * 1000
        m.vad_tail_wait_ms = max(0.0, vad_tail_wait_ms)
        self._recompute(m)

    def _record_smart_turn_emit(
        self,
        turn_id: str,
        inference_ms: float,
        text: str,
        text_type: str,
    ) -> None:
        m = self._get(turn_id)
        m.smart_turn_inference_ms = inference_ms
        m.smart_turn_emit_mono_ms = time.monotonic() * 1000
        m.asr_text_at_emit = text
        m.asr_text_type = text_type
        self._recompute(m)
        self._try_publish(turn_id)

    def _record_asr_final(self, turn_id: str, final_text: str) -> None:
        m = self._get(turn_id)
        m.asr_final_text = final_text
        if m.smart_turn_emit_mono_ms > 0:
            m.asr_final_latency_ms = time.monotonic() * 1000 - m.smart_turn_emit_mono_ms
        if m.asr_text_at_emit and final_text != m.asr_text_at_emit:
            m.asr_diff = _compute_diff(m.asr_text_at_emit, final_text)
        self._try_publish(turn_id)

    def _record_llm_request_start(self, turn_id: str, actual_text: str = "") -> None:
        m = self._get(turn_id)
        m.llm_request_start_mono_ms = time.monotonic() * 1000
        if actual_text:
            m.llm_actual_text = actual_text
        self._recompute(m)
        self._try_publish(turn_id)

    def _record_llm_ttft(self, turn_id: str, ttft_ms: float, actual_text: str = "") -> None:
        m = self._get(turn_id)
        m.llm_ttft_ms = ttft_ms
        if actual_text:
            m.llm_actual_text = actual_text
        self._recompute(m)
        self._try_publish(turn_id)

    def _record_shadow(self, turn_id: str, shadow_results: list[dict]) -> None:
        m = self._get(turn_id)
        m.shadow_results = shadow_results
        self._try_publish(turn_id)

    def _record_tts_session_create(self, turn_id: str, create_ms: float, pool_hit: bool | None = None) -> None:
        m = self._get(turn_id)
        m.tts_session_create_ms = create_ms
        m.tts_pool_hit = pool_hit
        self._recompute(m)
        self._try_publish(turn_id)

    def _record_tts_ttfa(
        self,
        turn_id: str,
        ttfa_ms: float,
        first_audio_source: str = "",
        *,
        latency_role: str = "speech",
    ) -> None:
        m = self._get(turn_id)
        now = time.monotonic() * 1000
        if latency_role in ("speech", "bridge"):
            if m.first_audio_mono_ms == 0:
                m.tts_ttfa_ms = ttfa_ms
                m.first_audio_mono_ms = now
                if first_audio_source:
                    m.first_audio_source = first_audio_source
        self._recompute(m)
        self._try_publish(turn_id)

    def _record_first_audio_queued(self, turn_id: str) -> None:
        m = self._get(turn_id)
        m.first_audio_queued_mono_ms = time.monotonic() * 1000
        self._recompute(m)
        self._try_publish(turn_id)

    # ── Public recording methods (backward compat — processors call directly) ─

    def record_user_stop(self, turn_id: str, vad_tail_wait_ms: float = 0.0) -> None:
        """Called when VAD emits UserStoppedSpeaking for this turn."""
        self._record_user_stop(turn_id, vad_tail_wait_ms)

    def record_smart_turn_emit(
        self,
        turn_id: str,
        inference_ms: float,
        text: str,
        text_type: str,
    ) -> None:
        """Called when TurnDetector emits via SmartTurn poll."""
        self._record_smart_turn_emit(turn_id, inference_ms, text, text_type)

    def record_asr_final(self, turn_id: str, final_text: str) -> None:
        """Called when ASR final arrives (may be after emit)."""
        self._record_asr_final(turn_id, final_text)

    def record_llm_request_start(self, turn_id: str, actual_text: str = "") -> None:
        """Called immediately before sending the Stage 1 LLM request."""
        self._record_llm_request_start(turn_id, actual_text)

    def record_llm_ttft(self, turn_id: str, ttft_ms: float, actual_text: str = "") -> None:
        """Called when LLM produces first token."""
        self._record_llm_ttft(turn_id, ttft_ms, actual_text)

    def record_shadow(self, turn_id: str, shadow_results: list[dict]) -> None:
        """Called with shadow comparison results."""
        self._record_shadow(turn_id, shadow_results)

    def record_tts_session_create(self, turn_id: str, create_ms: float, pool_hit: bool | None = None) -> None:
        """Called when a turn-scoped TTS session has been acquired."""
        self._record_tts_session_create(turn_id, create_ms, pool_hit)

    def record_tts_ttfa(
        self,
        turn_id: str,
        ttfa_ms: float,
        first_audio_source: str = "",
        *,
        latency_role: str = "speech",
    ) -> None:
        """Called when the first TTS audio chunk is produced."""
        self._record_tts_ttfa(turn_id, ttfa_ms, first_audio_source, latency_role=latency_role)

    def record_first_audio_queued(self, turn_id: str) -> None:
        """Called when the first audio frame is handed to the output transport."""
        self._record_first_audio_queued(turn_id)

    # ── Publishing ──────────────────────────────────────────────────────────

    def _try_publish(self, turn_id: str) -> None:
        """Publish when a new milestone is reached; each milestone fires at most once."""
        m = self._turns[turn_id]
        milestones = self._pub_milestones.setdefault(turn_id, set())

        new_milestone = False
        # Initial: SmartTurn + any downstream latency (LLM TTFT or TTS TTFA).
        if "initial" not in milestones and m.smart_turn_emit_mono_ms > 0 and (m.llm_ttft_ms > 0 or m.tts_ttfa_ms > 0):
            milestones.add("initial")
            new_milestone = True
        # ASR final text arrival (drives asr_final_latency_ms / asr_diff).
        if "asr_final" not in milestones and m.asr_final_text:
            milestones.add("asr_final")
            new_milestone = True
        # First TTS audio — provisional e2e via first_audio_mono_ms.
        if "tts_ttfa" not in milestones and m.tts_ttfa_ms > 0:
            milestones.add("tts_ttfa")
            new_milestone = True
        # First frame handed to output transport — definitive e2e.
        if "first_audio_queued" not in milestones and m.first_audio_queued_mono_ms > 0:
            milestones.add("first_audio_queued")
            new_milestone = True
        # Shadow comparator results (can arrive late).
        if "shadow" not in milestones and m.shadow_results:
            milestones.add("shadow")
            new_milestone = True

        if not new_milestone:
            return
        # Require at least the initial milestone before any publish.
        if "initial" not in milestones:
            return

        serialized = self._serialize(m)

        if self._bus is not None:
            from src.pipeline.events import TurnMetricsEvent

            # _serialize includes "topic" key which is not a field of TurnMetricsEvent
            event_kwargs = {k: v for k, v in serialized.items() if k != "topic"}
            self._bus.emit(TurnMetricsEvent(**event_kwargs))

        if self._on_complete is not None:
            self._on_complete(serialized)

    @staticmethod
    def _recompute(m: TurnMetrics) -> None:
        if m.turn_stop_mono_ms > 0 and m.smart_turn_emit_mono_ms > 0:
            m.user_stop_to_emit_ms = m.smart_turn_emit_mono_ms - m.turn_stop_mono_ms
            if m.smart_turn_inference_ms >= 0:
                m.gate_wait_ms = max(0.0, m.user_stop_to_emit_ms - m.smart_turn_inference_ms)

        if m.smart_turn_emit_mono_ms > 0 and m.llm_request_start_mono_ms > 0:
            m.emit_to_llm_ms = m.llm_request_start_mono_ms - m.smart_turn_emit_mono_ms

        if m.smart_turn_emit_mono_ms > 0 and m.first_audio_mono_ms > 0:
            m.emit_to_audio_ms = m.first_audio_mono_ms - m.smart_turn_emit_mono_ms

        if m.first_audio_mono_ms > 0 and m.first_audio_queued_mono_ms > 0:
            m.output_queue_wait_ms = m.first_audio_queued_mono_ms - m.first_audio_mono_ms

        first_audio_anchor = m.first_audio_queued_mono_ms or m.first_audio_mono_ms
        if m.turn_stop_mono_ms > 0 and first_audio_anchor > 0:
            m.e2e_ms = first_audio_anchor - m.turn_stop_mono_ms
            m.speech_end_to_audio_ms = m.e2e_ms + m.vad_tail_wait_ms

        known_parts = [
            m.vad_tail_wait_ms,
            m.smart_turn_inference_ms,
            m.gate_wait_ms,
            m.emit_to_llm_ms,
            m.llm_ttft_ms,
            m.tts_session_create_ms,
            m.tts_ttfa_ms,
            m.output_queue_wait_ms,
        ]
        known_ms = sum(part for part in known_parts if part >= 0)
        latency_total = m.speech_end_to_audio_ms if m.speech_end_to_audio_ms >= 0 else m.e2e_ms
        if latency_total >= 0 and known_ms > 0:
            m.unexplained_gap_ms = max(0.0, latency_total - known_ms)

    @staticmethod
    def _serialize(m: TurnMetrics) -> dict:
        return {
            "topic": "turn_metrics",
            "turn_id": m.turn_id,
            "smart_turn_ms": round(m.smart_turn_inference_ms, 1),
            "vad_tail_wait_ms": round(m.vad_tail_wait_ms, 1),
            "user_stop_to_emit_ms": round(m.user_stop_to_emit_ms, 1) if m.user_stop_to_emit_ms >= 0 else -1,
            "gate_wait_ms": round(m.gate_wait_ms, 1) if m.gate_wait_ms >= 0 else -1,
            "asr_text_at_emit": m.asr_text_at_emit,
            "asr_text_type": m.asr_text_type,
            "asr_final_text": m.asr_final_text,
            "asr_final_latency_ms": round(m.asr_final_latency_ms, 1) if m.asr_final_latency_ms >= 0 else -1,
            "asr_diff": m.asr_diff,
            "emit_to_llm_ms": round(m.emit_to_llm_ms, 1) if m.emit_to_llm_ms >= 0 else -1,
            "llm_ttft_ms": round(m.llm_ttft_ms, 1) if m.llm_ttft_ms >= 0 else -1,
            "llm_actual_text": m.llm_actual_text,
            "tts_session_create_ms": round(m.tts_session_create_ms, 1) if m.tts_session_create_ms >= 0 else -1,
            "tts_pool_hit": m.tts_pool_hit,
            "tts_ttfa_ms": round(m.tts_ttfa_ms, 1) if m.tts_ttfa_ms >= 0 else -1,
            "shadow_results": m.shadow_results,
            "emit_to_audio_ms": round(m.emit_to_audio_ms, 1) if m.emit_to_audio_ms >= 0 else -1,
            "output_queue_wait_ms": round(m.output_queue_wait_ms, 1) if m.output_queue_wait_ms >= 0 else -1,
            "e2e_ms": round(m.e2e_ms, 1) if m.e2e_ms >= 0 else -1,
            "speech_end_to_audio_ms": round(m.speech_end_to_audio_ms, 1) if m.speech_end_to_audio_ms >= 0 else -1,
            "first_audio_source": m.first_audio_source,
            "first_audio_is_bridge": m.first_audio_source == "bridge",
            "unexplained_gap_ms": round(m.unexplained_gap_ms, 1) if m.unexplained_gap_ms >= 0 else -1,
        }

    def cleanup_turn(self, turn_id: str) -> None:
        """Remove stale turn data."""
        self._turns.pop(turn_id, None)
        self._pub_milestones.pop(turn_id, None)


# Deprecated alias for backward compatibility
TurnMetricsCollector = TurnMetricsAggregator


def _compute_diff(emitted: str, final: str) -> str:
    """Simple character-level diff description."""
    if emitted == final:
        return ""
    # Find common prefix
    i = 0
    while i < len(emitted) and i < len(final) and emitted[i] == final[i]:
        i += 1
    removed = emitted[i:]
    added = final[i:]
    parts = []
    if removed:
        parts.append(f"-{removed}")
    if added:
        parts.append(f"+{added}")
    return "".join(parts)
