"""Mid-call callback queue — JSONL persistence with idempotency.

Persists callback intents for deferred human follow-up to a JSONL file.
Idempotent on f"{call_id}:{session_id}" — duplicate enqueues within the same
writer instance are no-ops that return the prior record.

Usage:
    writer = get_callback_queue_writer()
    record = writer.enqueue(
        flow_state=flow_state,
        call_id=call_id,
        kind=RefuseKind.LEGITIMATE,
        out_of_scope_class="medical_judgment",
        priority=CallbackPriority.NORMAL,
        user_text=user_text,
        reason_text=reason_text,
    )
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from src.flows.knowledge_review_models import (
    CallbackPriority,
    EscalationKind,
    RefuseKind,
)
from src.handoff.models import CallbackRecord, TransferReason  # re-export so callers can import from here
from src.handoff.transfer_service import get_transfer_service

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(os.environ.get("CALLBACK_QUEUE_PATH", "data/callbacks/pending.jsonl"))


def _map_to_transfer_reason(
    kind: Union[RefuseKind, EscalationKind],
    out_of_scope_class: Optional[str],
) -> TransferReason:
    if isinstance(kind, RefuseKind):
        if out_of_scope_class == "medical_judgment":
            return TransferReason.MEDICAL_COMPLEX
        if out_of_scope_class == "safety_risk":
            return TransferReason.SAFETY_EMERGENCY
    if kind == EscalationKind.DEPTH_CAP:
        return TransferReason.MEDICAL_COMPLEX
    if kind == EscalationKind.EXIT_ESCALATE:
        return TransferReason.USER_REQUEST
    return TransferReason.OTHER


class CallbackQueueWriter:
    """Append-only JSONL writer with in-memory idempotency dedup.

    One instance per call is the expected usage pattern. The lazy singleton
    returned by get_callback_queue_writer() is process-wide (suitable for
    production where call_id + session_id uniqueness guarantees safety).
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path if path is not None else _DEFAULT_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._dedup: dict[str, CallbackRecord] = {}

    def enqueue(
        self,
        *,
        flow_state: Any,
        call_id: str,
        kind: Union[RefuseKind, EscalationKind],
        out_of_scope_class: Optional[str],
        priority: CallbackPriority,
        user_text: str,
        reason_text: str,
    ) -> CallbackRecord:
        """Enqueue a callback record, idempotent on call_id:session_id.

        Args:
            flow_state: FlowState — used to read patient_info and
                knowledge_review_state.current_session.
            call_id: Caller-supplied call identifier (lives on
                SessionLifecycleHelper, NOT on FlowState).
            kind: RefuseKind or EscalationKind enum value.
            out_of_scope_class: Optional OOS class string from KnowledgeReviewFacts.
            priority: CallbackPriority enum value.
            user_text: The patient utterance that triggered the callback.
            reason_text: Human-readable reason string (passed as trigger_text to
                TransferService for context packaging).

        Returns:
            The persisted (or previously-persisted) CallbackRecord.

        Raises:
            RuntimeError: If flow_state.knowledge_review_state.current_session is
                None — this is an orchestrator bug; the session must be opened
                before enqueueing.
        """
        krs = flow_state.knowledge_review_state
        session = krs.current_session
        if session is None:
            raise RuntimeError(
                "callback_queue.enqueue called with no current_session — orchestrator bug. "
                "Open a KnowledgeSession before enqueueing a callback."
            )

        idempotency_key = f"{call_id}:{session.session_id}"

        with self._lock:
            if idempotency_key in self._dedup:
                logger.debug(
                    "callback_queue: dedup hit for key=%s — returning prior record",
                    idempotency_key,
                )
                return self._dedup[idempotency_key]

            transfer_service = get_transfer_service()
            context_text = transfer_service.prepare_handoff_from_flow_state(
                flow_state,
                reason=_map_to_transfer_reason(kind, out_of_scope_class),
                trigger_text=reason_text,
            )

            patient_info = flow_state.patient_info if flow_state.patient_info else {}
            patient_id = patient_info.get("patient_id", "unknown")

            record = CallbackRecord(
                call_id=call_id,
                session_id=session.session_id,
                patient_id=patient_id,
                reason=kind.value,
                out_of_scope_class=out_of_scope_class,
                priority=priority.value,
                context_text=context_text,
                user_text=user_text,
                timestamp=datetime.utcnow().isoformat() + "Z",
                idempotency_key=idempotency_key,
            )

            self._dedup[idempotency_key] = record
            self._append(record)
            return record

    def _append(self, record: CallbackRecord) -> None:
        """Append a single JSON line to the JSONL file and fsync."""
        line = record.to_json()
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        logger.info(
            "callback_queue: persisted idempotency_key=%s priority=%s path=%s",
            record.idempotency_key,
            record.priority,
            self._path,
        )


# ---------------------------------------------------------------------------
# Lazy process-wide singleton
# ---------------------------------------------------------------------------

_callback_queue_writer: Optional[CallbackQueueWriter] = None


def get_callback_queue_writer() -> CallbackQueueWriter:
    """Return the process-wide lazy singleton CallbackQueueWriter."""
    global _callback_queue_writer
    if _callback_queue_writer is None:
        _callback_queue_writer = CallbackQueueWriter()
    return _callback_queue_writer
