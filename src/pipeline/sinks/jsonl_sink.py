"""JsonlSinkObserver — writes events as JSON lines to a file."""

from __future__ import annotations

import json
from pathlib import Path

from src.pipeline.events import BaseEvent


class JsonlSinkObserver:
    """Writes each event as one JSON line. Call close() when done."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = path.open("wb")
        self._bytes = 0

    def on_event(self, event: BaseEvent) -> None:
        line = (json.dumps(event.to_wire(), ensure_ascii=False) + "\n").encode("utf-8")
        self._fp.write(line)
        self._bytes += len(line)

    def close(self) -> int:
        self._fp.close()
        return self._bytes
