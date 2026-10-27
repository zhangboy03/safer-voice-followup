"""Off-path survey capture for the solo driver: fire-and-forget extraction per
turn, a latest snapshot, ★-coverage for the end_call backstop, and a dict-shaped
terminal payload for trace_form_submitted + the lifecycle flow_state_fn shim."""

from __future__ import annotations

import asyncio

from src.playground.form_extractor import extract_form
from src.playground.form_schema import field_catalog


class CaptureStore:
    def __init__(self, *, backend=None, catalog=None, on_update=None):
        self._backend = backend
        self._catalog = catalog if catalog is not None else field_catalog()
        self._on_update = on_update
        self._snapshot: list[dict] = []
        self._tasks: set[asyncio.Task] = set()
        # Monotonic sequencing (B2): every extraction — scheduled or awaited —
        # takes its seq when ISSUED; a result is applied only if its seq is
        # newer than the last APPLIED one. Invariant: the snapshot only moves
        # forward in issue order, so an older, slower extraction can never
        # clobber a newer result. 0 = "nothing issued/applied yet";
        # _next_seq() pre-increments, so the first extraction gets seq 1 and
        # beats the initial empty snapshot.
        self._seq: int = 0
        self._applied_seq: int = 0

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _apply(self, seq: int, snapshot: list[dict]) -> None:
        if seq > self._applied_seq:  # stale out-of-order result → dropped
            self._applied_seq = seq
            self._snapshot = snapshot
            self._publish_update()

    def _publish_update(self) -> None:
        if self._on_update is None:
            return
        collected = {
            r["name"]: {
                "value": r.get("value", ""),
                "status": r.get("status", "pending"),
                "evidence": r.get("evidence", ""),
                "source_node": r.get("group", ""),
            }
            for r in self._snapshot
            if r.get("status") in ("confirmed", "refused")
        }
        missing = [
            r["name"] for r in self._snapshot if r.get("required") and r.get("status") not in ("confirmed", "refused")
        ]
        node = next(
            (r.get("group", "") for r in reversed(self._snapshot) if r.get("status") in ("confirmed", "refused")), ""
        )
        try:
            self._on_update(
                {
                    "topic": "dual_llm_debug",
                    "event": "advance_decision",
                    "capture_revision": self._applied_seq,
                    "record_source": "solo_async_extraction",
                    "node": node,
                    "collected_fields": collected,
                    "policy_missing": missing,
                }
            )
        except Exception:  # noqa: BLE001 — telemetry must never affect the call
            return

    def schedule(self, transcript: str) -> None:
        """Fire-and-forget extraction; updates the snapshot on completion
        (only if no newer extraction has been applied meanwhile)."""
        if self._backend is None:
            return
        task = asyncio.create_task(self._run(self._next_seq(), transcript))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(self, seq: int, transcript: str) -> None:
        try:
            snapshot = await extract_form(transcript, backend=self._backend, catalog=self._catalog)
        except Exception:  # noqa: BLE001 — capture must never crash the call
            return  # failure applies nothing (and never advances _applied_seq)
        self._apply(seq, snapshot)

    async def extract_now(self, transcript: str) -> None:
        """Awaited extraction (terminal hook / pre-reject refresh): input
        includes the last turn. Takes its seq at CALL time — being the most
        recently issued it wins over any older in-flight scheduled task, and
        an older task completing later cannot clobber this result."""
        if self._backend is None:
            return
        seq = self._next_seq()
        self._apply(seq, await extract_form(transcript, backend=self._backend, catalog=self._catalog))

    def missing_required(self) -> list[str]:
        by_name = {r["name"]: r for r in self._snapshot}
        return [f.name for f in self._catalog if f.required and by_name.get(f.name, {}).get("status") not in ("confirmed", "refused")]

    def all_required_covered(self) -> bool:
        return not self.missing_required()

    def terminal_payload(self, *, patient_info: dict, turn_count: int) -> dict:
        collected = {
            r["name"]: {
                "value": r.get("value", ""),
                "status": r.get("status", "pending"),
                "source_node": r.get("group", ""),
            }
            for r in self._snapshot
            if r.get("status") in ("confirmed", "refused")
        }
        return {
            "collected_fields": collected,
            "patient_info": patient_info,
            "conversation_turns": turn_count,
            "nodes_completed": [],
        }

    def flow_state_view(self, *, patient_info: dict, turn_count: int):
        """A FlowState-shaped view for SessionLifecycleHelper._build_snapshot, which
        reads .patient_info / .turn_count / .collected_fields (dict of objs with .value)."""
        from dataclasses import dataclass

        @dataclass
        class _FV:
            value: object

        collected = {
            r["name"]: _FV(value=r.get("value", ""))
            for r in self._snapshot
            if r.get("status") in ("confirmed", "refused")
        }

        @dataclass
        class _FS:
            patient_info: dict
            turn_count: int
            collected_fields: dict

        return _FS(patient_info=patient_info, turn_count=turn_count, collected_fields=collected)
