"""Local JSONL-backed CSR feedback store."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

DEFAULT_FEEDBACK_FILE = Path("data/feedback/feedback.jsonl")


class FeedbackTurn(BaseModel):
    turn_id: str = ""
    user_text: str = ""
    assistant_text: str = ""
    node_id: str = ""


class FeedbackEvidence(BaseModel):
    turn_metrics: dict[str, Any] | None = None
    debug: dict[str, Any] | None = None
    recent_events: list[dict[str, Any]] = Field(default_factory=list)


class FeedbackCreate(BaseModel):
    csr_id: str = Field(min_length=1, max_length=80)
    room_name: str = Field(min_length=1, max_length=120)
    source: Literal["csr_simple", "debug"]
    note: str = Field(min_length=1, max_length=2000)
    turn: FeedbackTurn = Field(default_factory=FeedbackTurn)
    evidence: FeedbackEvidence = Field(default_factory=FeedbackEvidence)

    @field_validator("csr_id", "room_name", "note")
    @classmethod
    def _strip_non_empty(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("must not be empty")
        return trimmed


class FeedbackRecord(FeedbackCreate):
    id: str
    created_at: str


def feedback_file_path() -> Path:
    return Path(os.environ.get("PINCHI_FEEDBACK_FILE", str(DEFAULT_FEEDBACK_FILE)))


def create_feedback(payload: FeedbackCreate) -> FeedbackRecord:
    record = FeedbackRecord(
        **payload.model_dump(),
        id=f"fb_{uuid.uuid4().hex[:12]}",
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    path = feedback_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.model_dump(), ensure_ascii=False, separators=(",", ":")) + "\n")
    return record


def list_feedback(limit: int = 100) -> list[FeedbackRecord]:
    path = feedback_file_path()
    if not path.exists():
        return []
    records: list[FeedbackRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(FeedbackRecord.model_validate_json(line))
    records.sort(key=lambda item: item.created_at, reverse=True)
    return records[: max(1, min(limit, 500))]


def get_feedback(feedback_id: str) -> FeedbackRecord | None:
    for item in list_feedback(limit=500):
        if item.id == feedback_id:
            return item
    return None
