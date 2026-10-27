"""PipelineObserver protocol (converged) and LoggingObserver.

PipelineObserver now has a single method: on_event(BaseEvent).
LoggingObserver forwards every event to the structured logger synchronously,
and also retains the last N wire dicts in a bounded deque for inspection/tests.
"""

from __future__ import annotations

from collections import deque
from typing import Protocol, runtime_checkable

from src.core.logging import get_logger
from src.pipeline.events import BaseEvent

logger = get_logger(__name__)


@runtime_checkable
class PipelineObserver(Protocol):
    def on_event(self, event: BaseEvent) -> None: ...


class LoggingObserver:
    """Logs every pipeline event via structlog. Never blocks the pipeline.

    The bounded deque (`_queue`) is a small inspection tap for tests / ad-hoc
    debugging; `drain()` empties it. Logging to `logger.info` is the actual
    sink — it always runs, regardless of whether anyone calls drain().
    """

    def __init__(self, maxsize: int = 4096) -> None:
        self._queue: deque[dict] = deque(maxlen=maxsize)

    def on_event(self, event: BaseEvent) -> None:
        wire = event.to_wire()
        self._queue.append(wire)
        event_type = wire.get("event_type", type(event).__name__)
        log_kwargs = {k: v for k, v in wire.items() if k not in ("event_type", "topic")}
        try:
            logger.info("pipeline_event", event_type=event_type, **log_kwargs)
        except Exception as err:
            logger.warning("pipeline_event_log_failed", event_type=event_type, error=str(err))

    def drain(self) -> list[dict]:
        events = list(self._queue)
        self._queue.clear()
        return events
