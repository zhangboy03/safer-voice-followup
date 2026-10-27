"""Session lifecycle helper — once-only post-call scheduling.

Shared across all three exit paths (completed / callback-required / disconnected)
to ensure post-call pipeline runs exactly once.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable

from src.core.logging import get_logger
from src.core.pilot_guardrails import uses_callback_followup
from src.memory.models import PostCallSnapshot

if TYPE_CHECKING:
    from src.core.call_tracer import CallTracer

logger = get_logger(__name__)

_CALLBACK_REQUIRED_OUTCOMES = {
    "transfer",
    "transferred",
    "transferred_safety",
    "human_transfer",
}


class SessionLifecycleHelper:
    """Once-only post-call scheduling for all exit paths."""

    def __init__(
        self,
        *,
        session_id: str,
        patient_id: str,
        history: Any,
        call_tracer: CallTracer | None,
        flow_state_fn: Callable[[], Any],
        turn_log: Any | None = None,
        post_call_pipeline: Callable[..., Any] | None = None,
    ) -> None:
        self._session_id = session_id
        self._patient_id = patient_id
        self._history = history
        self._call_tracer = call_tracer
        self._flow_state_fn = flow_state_fn
        self._turn_log = turn_log
        self._post_call_pipeline = post_call_pipeline
        self._post_call_scheduled = False
        self._last_snapshot: PostCallSnapshot | None = None  # exposed for testing

    def schedule_post_call_once(
        self,
        outcome: str,
        transfer_reason: str | None = None,
    ) -> None:
        """Idempotent: only the first call schedules the pipeline."""
        normalized_outcome = self._normalize_outcome(outcome, transfer_reason)
        if self._post_call_scheduled:
            logger.info("post_call.already_scheduled", outcome=normalized_outcome)
            return
        self._post_call_scheduled = True

        snapshot = self._build_snapshot(normalized_outcome, transfer_reason)
        self._last_snapshot = snapshot

        turn_log_entries = self._turn_log.snapshot() if self._turn_log else []

        if self._call_tracer:
            self._call_tracer.finalize()

        if self._post_call_pipeline:
            asyncio.create_task(self._post_call_pipeline(snapshot, turn_log_entries))

        logger.info(
            "post_call.scheduled",
            outcome=normalized_outcome,
            transfer_reason=transfer_reason,
            session_id=self._session_id,
        )

    def _normalize_outcome(self, outcome: str, transfer_reason: str | None) -> str:
        """Map live-transfer outcomes to callback-required during the pilot."""
        if uses_callback_followup() and (outcome in _CALLBACK_REQUIRED_OUTCOMES or transfer_reason):
            return "callback_required"
        return outcome

    def _build_snapshot(self, outcome: str, transfer_reason: str | None) -> PostCallSnapshot:
        flow_state = self._flow_state_fn()
        patient_info = flow_state.patient_info if flow_state else {}
        turn_count = flow_state.turn_count if flow_state else 0

        transcript_messages = (
            self._history.get_journal_messages() if hasattr(self._history, "get_journal_messages") else []
        )

        collected = None
        if flow_state and flow_state.collected_fields:
            collected = {k: v.value for k, v in flow_state.collected_fields.items()}

        return PostCallSnapshot(
            session_id=self._session_id,
            patient_id=self._patient_id,
            patient_name=patient_info.get("name", ""),
            product_line=patient_info.get("product_line", ""),
            months_since_surgery=0,
            transcript=tuple(transcript_messages),
            plus_analysis=None,
            turn_count=turn_count,
            outcome=outcome,
            transfer_reason=transfer_reason,
            collected_fields=collected,
        )
