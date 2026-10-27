"""HTTP API for incremental notes (GET + PATCH)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel

from server.notes_edits_store import append_edit, list_edits_for_call
from src.flows.note_log import safe_call_id
from src.flows.note_replay import NoteReplay


def _notes_dir() -> Path:
    return Path(os.environ.get("PINCHI_NOTES_DIR", "data/feedback/notes"))


def _read_events(call_id: str) -> list[dict]:
    path = _notes_dir() / f"{call_id}.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


class PatchPayload(BaseModel):
    before_text: str
    after_text: str
    based_on_event_id: str
    csr_id: str


def register_notes_routes(app: FastAPI) -> None:
    router = APIRouter()

    @router.get("/api/notes/{call_id}")
    async def get_notes(call_id: str):
        events = _read_events(call_id)
        if not events:
            raise HTTPException(status_code=404, detail="call_not_found")
        edits = list_edits_for_call(call_id)
        current_items, last_event_id = NoteReplay.events_to_current_items(events, edits)
        return {
            "call_id": call_id,
            "events": events,
            "current_items": current_items,
            "last_event_id": last_event_id,
        }

    @router.patch("/api/notes/{call_id}/items/{item_key}")
    async def patch_note_item(call_id: str, item_key: str, payload: PatchPayload):
        events = _read_events(call_id)
        if not events:
            raise HTTPException(status_code=404, detail="call_not_found")
        edits = list_edits_for_call(call_id)
        current_items, _ = NoteReplay.events_to_current_items(events, edits)
        by_key = {item["key"]: item for item in current_items}
        if item_key not in by_key:
            raise HTTPException(status_code=404, detail="item_not_found")
        current = by_key[item_key]
        if current["latest_event_id"] != payload.based_on_event_id:
            raise HTTPException(
                status_code=409,
                detail={
                    "reason": "stale_event_id",
                    "current_event_id": current["latest_event_id"],
                    "current_text": current["text"],
                },
            )
        # Belt-and-braces: catches a concurrent CSR edit landing for the same event_id
        # between this request's replay and append_edit.
        if current["text"] != payload.before_text:
            raise HTTPException(
                status_code=409,
                detail={
                    "reason": "stale_text",
                    "current_event_id": current["latest_event_id"],
                    "current_text": current["text"],
                },
            )
        # analyzer_source is the raw text the model last wrote for this key —
        # preserved so future regret-mining can compare what the analyzer said vs what the CSR changed.
        analyzer_source = ""
        for event in events:
            for upsert in event.get("patch", {}).get("upsert_items", []):
                if upsert["key"] == item_key:
                    analyzer_source = upsert["text"]
        rec = append_edit(
            call_id=call_id,
            item_key=item_key,
            csr_id=payload.csr_id,
            analyzer_source_text_at_edit_time=analyzer_source,
            before_text=payload.before_text,
            after_text=payload.after_text,
            based_on_event_id=payload.based_on_event_id,
        )
        return {"ok": True, "edit": rec}

    @router.get("/api/notes/by-room/{room_name}")
    async def list_call_ids_by_room(room_name: str):
        # Reuse safe_call_id's sanitization so the lookup prefix matches the writer side.
        sanitized_prefix = safe_call_id(room_name, "x").split("__", 1)[0]
        notes_dir = _notes_dir()
        if not notes_dir.exists():
            return {"call_ids": []}
        matches: list[str] = []
        for path in notes_dir.iterdir():
            if not path.name.endswith(".jsonl"):
                continue
            stem = path.stem
            prefix, sep, _ = stem.partition("__")
            if sep and prefix == sanitized_prefix:
                matches.append(stem)
        matches.sort(reverse=True)
        return {"call_ids": matches}

    app.include_router(router)
