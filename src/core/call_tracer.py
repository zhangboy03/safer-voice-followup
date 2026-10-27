"""Call-level diagnostic tracer.

Every call produces a structured trace that captures latencies,
SOP progress, RAG hits/misses, interruptions, and safety events.
The ``finalize()`` method emits a ``call.summary`` log and returns
a dict suitable for dashboards or post-call analytics.
"""

from __future__ import annotations

import time

import structlog

from src.core.logging import get_logger

logger: structlog.stdlib.BoundLogger = get_logger("call_tracer")


class CallTracer:
    """Accumulates per-call metrics and emits structured log events."""

    def __init__(self, session_id: str, patient_id: str) -> None:
        self._session_id = session_id
        self._patient_id = patient_id

        # Latency accumulators
        self._asr_latencies: list[float] = []
        self._llm_latencies: list[float] = []
        self._ttfa_latencies: list[float] = []

        # RAG tracking
        self._rag_result_count: int = 0
        self._rag_miss_queries: list[str] = []

        # SOP tracking
        self._sop_steps_visited: list[str] = []
        self._sop_completed: bool = False

        # Counters
        self._turn_count: int = 0
        self._interruption_count: int = 0
        self._tts_underrun_count: int = 0

        # Timing
        self._start_time: float | None = None
        self._total_duration: float = 0.0

    # ── Call lifecycle ──────────────────────────────────────────────

    def trace_call_started(self) -> None:
        """Record call start time and emit ``call.started``."""
        self._start_time = time.monotonic()
        logger.info(
            "call.started",
            session_id=self._session_id,
            patient_id=self._patient_id,
        )

    # ── Turn events ─────────────────────────────────────────────────

    def trace_turn_user_ended(
        self,
        turn_id: int,
        asr_text: str,
        asr_latency_ms: float,
        vad_method: str,
    ) -> None:
        """Log ``turn.user_ended`` and accumulate ASR latency."""
        self._asr_latencies.append(asr_latency_ms)
        self._turn_count += 1
        logger.info(
            "turn.user_ended",
            session_id=self._session_id,
            turn_id=turn_id,
            asr_text=asr_text,
            asr_latency_ms=asr_latency_ms,
            vad_method=vad_method,
        )

    def trace_turn_agent_ended(
        self,
        turn_id: int,
        tts_ttfa_ms: float,
        tts_total_ms: float,
        text_length: int,
    ) -> None:
        """Log ``turn.agent_ended`` and accumulate TTFA."""
        self._ttfa_latencies.append(tts_ttfa_ms)
        logger.info(
            "turn.agent_ended",
            session_id=self._session_id,
            turn_id=turn_id,
            tts_ttfa_ms=tts_ttfa_ms,
            tts_total_ms=tts_total_ms,
            text_length=text_length,
        )

    # ── LLM events ──────────────────────────────────────────────────

    def trace_llm_response(
        self,
        turn_id: int,
        model: str,
        completion_tokens: int,
        latency_ms: float,
        fields_extracted: list[str],
    ) -> None:
        """Log ``llm.response`` and accumulate LLM latency."""
        self._llm_latencies.append(latency_ms)
        logger.info(
            "llm.response",
            session_id=self._session_id,
            turn_id=turn_id,
            model=model,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            fields_extracted=fields_extracted,
        )

    # ── RAG events ──────────────────────────────────────────────────

    def trace_rag_result(
        self,
        turn_id: int,
        query: str,
        confidence: float,
        source_count: int,
        latency_ms: float,
    ) -> None:
        """Log ``rag.result`` and count successful retrievals."""
        self._rag_result_count += 1
        logger.info(
            "rag.result",
            session_id=self._session_id,
            turn_id=turn_id,
            query=query,
            confidence=confidence,
            source_count=source_count,
            latency_ms=latency_ms,
        )

    def trace_rag_miss(
        self,
        turn_id: int,
        query: str,
        reason: str,
    ) -> None:
        """Log ``rag.miss`` and record the failed query."""
        self._rag_miss_queries.append(query)
        logger.info(
            "rag.miss",
            session_id=self._session_id,
            turn_id=turn_id,
            query=query,
            reason=reason,
        )

    # ── SOP events ──────────────────────────────────────────────────

    def trace_sop_step_changed(
        self,
        from_step: str,
        to_step: str,
    ) -> None:
        """Log ``sop.step_changed`` and track visited steps."""
        self._sop_steps_visited.append(to_step)
        logger.info(
            "sop.step_changed",
            session_id=self._session_id,
            from_step=from_step,
            to_step=to_step,
        )

    def trace_sop_completed(
        self,
        total_turns: int,
        duration_sec: float,
    ) -> None:
        """Log ``sop.completed`` and mark SOP as finished."""
        self._sop_completed = True
        logger.info(
            "sop.completed",
            session_id=self._session_id,
            total_turns=total_turns,
            duration_sec=duration_sec,
        )

    def trace_form_submitted(self, payload: dict[str, object]) -> None:
        """Log ``form.submitted`` with terminal payload shape."""
        collected_fields = payload.get("collected_fields")
        field_count = len(collected_fields) if isinstance(collected_fields, dict) else 0
        logger.info(
            "form.submitted",
            session_id=self._session_id,
            field_count=field_count,
            node=payload.get("node", ""),
        )

    # ── TTS / interruption events ───────────────────────────────────

    def trace_tts_buffer_underrun(self, gap_ms: float) -> None:
        """Log ``tts.buffer_underrun`` and increment counter."""
        self._tts_underrun_count += 1
        logger.info(
            "tts.buffer_underrun",
            session_id=self._session_id,
            gap_ms=gap_ms,
        )

    def trace_interruption(
        self,
        turn_id: int,
        sentence_index: int,
    ) -> None:
        """Log ``tts.interrupted`` and increment counter."""
        self._interruption_count += 1
        logger.info(
            "tts.interrupted",
            session_id=self._session_id,
            turn_id=turn_id,
            sentence_index=sentence_index,
        )

    # ── Safety / handoff events ─────────────────────────────────────

    def trace_safety_triggered(
        self,
        turn_id: int,
        risk_type: str,
        action_taken: str,
    ) -> None:
        """Log ``safety.triggered``."""
        logger.info(
            "safety.triggered",
            session_id=self._session_id,
            turn_id=turn_id,
            risk_type=risk_type,
            action_taken=action_taken,
        )

    def trace_handoff_requested(
        self,
        reason: str,
        priority: str,
    ) -> None:
        """Log ``handoff.requested``."""
        logger.info(
            "handoff.requested",
            session_id=self._session_id,
            reason=reason,
            priority=priority,
        )

    # ── Finalize ────────────────────────────────────────────────────

    def finalize(self) -> dict[str, object]:
        """Compute averages, emit ``call.summary``, and return metrics dict."""
        if self._start_time is not None:
            self._total_duration = time.monotonic() - self._start_time

        avg_asr = sum(self._asr_latencies) / len(self._asr_latencies) if self._asr_latencies else 0.0
        avg_llm = sum(self._llm_latencies) / len(self._llm_latencies) if self._llm_latencies else 0.0
        avg_ttfa = sum(self._ttfa_latencies) / len(self._ttfa_latencies) if self._ttfa_latencies else 0.0

        summary: dict[str, object] = {
            "session_id": self._session_id,
            "patient_id": self._patient_id,
            "duration_sec": round(self._total_duration, 2),
            "total_turns": self._turn_count,
            "sop_completed": self._sop_completed,
            "sop_steps_visited": list(self._sop_steps_visited),
            "avg_asr_latency_ms": round(avg_asr, 2),
            "avg_llm_latency_ms": round(avg_llm, 2),
            "avg_ttfa_ms": round(avg_ttfa, 2),
            "rag_queries": self._rag_result_count,
            "rag_misses": len(self._rag_miss_queries),
            "rag_miss_queries": list(self._rag_miss_queries),
            "tts_underruns": self._tts_underrun_count,
            "interruptions": self._interruption_count,
        }

        logger.info(
            "call.summary",
            **{k: v for k, v in summary.items()},
        )

        return summary
