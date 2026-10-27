"""JSONL trace writer for VAD shadow events.

Subscribed via EventBus. Each VADShadowFrameEvent / VADShadowSpeechEvent
becomes one JSON line. Other event types are ignored.

The trailing-underscore `event_type_` field on VADShadowSpeechEvent is
written as `event_type` for trace consumers (jq-friendly).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.core.logging import get_logger
from src.pipeline.events import (
    BaseEvent,
    VADShadowFrameEvent,
    VADShadowSpeechEvent,
)

logger = get_logger(__name__)


class PVADTraceWriter:
    """Append shadow VAD events to a JSONL file. Call close() on session end."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._fp = path.open("wb")
        self._lines = 0

    def on_event(self, event: BaseEvent) -> None:
        if isinstance(event, VADShadowFrameEvent):
            record = {
                "ts_mono_ms": event.ts_mono_ms,
                "kind": "frame",
                "backend": event.backend,
                "state": event.state,
                "is_speech": event.is_speech,
                "speech_prob": event.speech_prob,
                "threshold": event.threshold,
                "primary_is_speech": event.primary_is_speech,
                "enroll_progress_ms": event.enroll_progress_ms,
            }
        elif isinstance(event, VADShadowSpeechEvent):
            record = {
                "ts_mono_ms": event.ts_mono_ms,
                "kind": "speech",
                "backend": event.backend,
                "event_type": event.event_type_,
                "primary_event_type": event.primary_event_type,
                "speech_duration_ms": event.speech_duration_ms,
            }
        else:
            return
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            self._fp.write(line)
            self._lines += 1
        except Exception as exc:
            logger.warning("pvad_trace.write_failed", error=str(exc), path=str(self._path))

    def close(self) -> int:
        try:
            self._fp.close()
        except Exception:
            pass
        return self._lines
