"""Local JSONL-backed CSR edit store for incremental notes."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

DEFAULT_EDITS_FILE = Path("data/feedback/notes_edits.jsonl")


def edits_file_path() -> Path:
    return Path(os.environ.get("PINCHI_NOTES_EDITS_FILE", str(DEFAULT_EDITS_FILE)))


def append_edit(
    *,
    call_id: str,
    item_key: str,
    csr_id: str,
    analyzer_source_text_at_edit_time: str,
    before_text: str,
    after_text: str,
    based_on_event_id: str,
) -> dict | None:
    """Append an edit record. Returns the record, or None if it's a noop."""
    if before_text == after_text:
        return None
    record = {
        "edit_id": f"ned_{uuid.uuid4().hex[:12]}",
        "call_id": call_id,
        "item_key": item_key,
        "csr_id": csr_id,
        "analyzer_source_text_at_edit_time": analyzer_source_text_at_edit_time,
        "before_text": before_text,
        "after_text": after_text,
        "based_on_event_id": based_on_event_id,
        "edited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = edits_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return record


def list_edits_for_call(call_id: str) -> list[dict]:
    path = edits_file_path()
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("call_id") == call_id:
                out.append(rec)
    return out
