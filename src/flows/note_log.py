"""Best-effort persistence of analyzer patch events to JSONL.

One file per call. Disk failure does NOT block in-memory commit; the
caller is expected to catch exceptions if it wants telemetry, otherwise
this module logs errors and returns silently.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path

import structlog

from src.core.logging import get_logger
from src.flows.sop_decision_models import ShortTermMemoryItem, ShortTermMemoryPatch

logger: structlog.stdlib.BoundLogger = get_logger("note_log")

_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9_\-]")
_ISO_UNSAFE_RE = re.compile(r"[:.]")


def _notes_dir() -> Path:
    return Path(os.environ.get("PINCHI_NOTES_DIR", "data/feedback/notes"))


def safe_call_id(room_name: str, call_start_iso: str) -> str:
    """Build a filesystem-safe `<safe_room_id>__<iso>` identifier (spec §5.0)."""
    had_leading_dot = room_name.startswith(".")
    base = _SAFE_CHARS_RE.sub("_", room_name)
    if not base:
        base = "unknown"
    if had_leading_dot:
        base = "r_" + base
    base = base[:64]
    safe_iso = _ISO_UNSAFE_RE.sub("-", call_start_iso)
    return f"{base}__{safe_iso}"


class NoteLogWriter:
    """Append-only JSONL writer for analyzer patch events.

    Best-effort: any IO error is logged but not re-raised — main path
    (in-memory commit) must not block on disk.
    """

    def append_patch_event(
        self,
        *,
        call_id: str,
        sop_node_id: str,
        before_items: list[ShortTermMemoryItem],
        kept_patch: ShortTermMemoryPatch,
        filtered_out: list[tuple[dict | str, str]],
    ) -> None:
        # Skip legitimate empty (no upsert, no remove, no filtered_out).
        if not kept_patch.upsert_items and not kept_patch.remove_keys and not filtered_out:
            return

        record = {
            "event_id": f"ne_{uuid.uuid4().hex[:12]}",
            "call_id": call_id,
            "turn_id": kept_patch.turn_id,
            "turn_index": kept_patch.turn_index,
            "flow_epoch": kept_patch.flow_epoch,
            "sop_node_id": sop_node_id,
            "occurred_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "before_items": [it.model_dump() for it in before_items],
            "patch": {
                "upsert_items": [it.model_dump() for it in kept_patch.upsert_items],
                "remove_keys": list(kept_patch.remove_keys),
            },
            "filtered_out": [{"item": item, "reason": reason} for item, reason in filtered_out],
        }
        path = _notes_dir() / f"{call_id}.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        except OSError as exc:
            logger.error("notes_jsonl_write_failed", call_id=call_id, error=str(exc))
