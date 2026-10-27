"""Replay analyzer events + CSR edits to compute current_items for GET /api/notes."""

from __future__ import annotations


class NoteReplay:
    @staticmethod
    def events_to_current_items(events: list[dict], edits: list[dict]) -> tuple[list[dict], str | None]:
        """Return (current_items, last_event_id).

        current_items[i] keys:
            key, text, category, source_turn_ids, confidence,
            expires_after_turn_index, expired, edit_history, latest_event_id
        """
        items_by_key: dict[str, dict] = {}
        latest_event_id_per_key: dict[str, str] = {}
        max_turn_index = -1
        last_event_id: str | None = None

        for event in events:
            event_id = event.get("event_id")
            last_event_id = event_id
            turn_index = event.get("turn_index", -1)
            if turn_index > max_turn_index:
                max_turn_index = turn_index

            patch = event.get("patch", {})
            for raw_item in patch.get("upsert_items", []):
                key = raw_item["key"]
                items_by_key[key] = dict(raw_item)
                latest_event_id_per_key[key] = event_id

            for remove_key in patch.get("remove_keys", []):
                items_by_key.pop(remove_key, None)
                latest_event_id_per_key.pop(remove_key, None)

        edits_by_key: dict[str, list[dict]] = {}
        for edit in sorted(edits, key=lambda e: e.get("edited_at", "")):
            edits_by_key.setdefault(edit["item_key"], []).append(edit)

        out: list[dict] = []
        for key, item in items_by_key.items():
            expires_idx = item.get("expires_after_turn_index")
            expired = expires_idx is not None and expires_idx <= max_turn_index

            history = edits_by_key.get(key, [])
            # Edit overlay (spec §5.3 — analyzer-latest wins on text).
            text = item["text"]
            if history:
                last_edit = history[-1]
                if last_edit.get("based_on_event_id") == latest_event_id_per_key.get(key):
                    text = last_edit["after_text"]

            out.append(
                {
                    "key": key,
                    "text": text,
                    "category": item.get("category", "salient_fact"),
                    "source_turn_ids": item.get("source_turn_ids", []),
                    "confidence": item.get("confidence", 1.0),
                    "expires_after_turn_index": expires_idx,
                    "expired": expired,
                    "edit_history": history,
                    "latest_event_id": latest_event_id_per_key.get(key),
                }
            )
        return out, last_event_id
