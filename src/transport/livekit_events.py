"""DataChannelObserver: publishes pipeline events to LiveKit data channel."""

from __future__ import annotations

import asyncio
import json
import os

from livekit import rtc

from src.core.logging import get_logger
from src.pipeline.events import BaseEvent

logger = get_logger(__name__)

_MAX_PACKET_BYTES = 14_000
_WIRE_EXCLUDE = frozenset({"DriftReportEvent"})


def _resolve_reliable_default() -> bool:
    """EVAL_DATA_CHANNEL_UNRELIABLE=1 flips the default to unreliable.

    Eval VC subprocess path sets this to dodge Docker Desktop macOS SCTP/DTLS
    stalls on the reliable data channel (bulk ~20s timeouts once NAT table
    drops an ACK). Unreliable datagrams are each independent UDP packets —
    losing one event is cheap compared to losing the whole channel.
    """
    return os.environ.get("EVAL_DATA_CHANNEL_UNRELIABLE", "").lower() not in ("", "0", "false")


class DataChannelObserver:
    """PipelineObserver that forwards events to LiveKit data channel.

    Excluded events (DriftReportEvent) are silently dropped.
    Oversized payloads have long fields truncated.
    """

    def __init__(self, room: rtc.Room, *, reliable: bool | None = None) -> None:
        self._room = room
        # Retain pending tasks so they're not GC'd mid-flight (which would
        # silently drop the event).
        self._pending: set[asyncio.Task] = set()
        # Default: reliable=True (production); EVAL_DATA_CHANNEL_UNRELIABLE=1
        # flips to unreliable for eval VC runs.
        self._reliable = reliable if reliable is not None else (not _resolve_reliable_default())

    def on_event(self, event: BaseEvent) -> None:
        if type(event).__name__ in _WIRE_EXCLUDE:
            return
        try:
            wire = event.to_wire()
            data = json.dumps(wire, ensure_ascii=False).encode("utf-8")
            if len(data) > _MAX_PACKET_BYTES:
                # Truncate long fields
                for field in ("asr_diff", "llm_actual_text", "text_fed"):
                    if field in wire and isinstance(wire[field], str) and len(wire[field]) > 200:
                        wire[field] = wire[field][:200] + "..."
                data = json.dumps(wire, ensure_ascii=False).encode("utf-8")
                if len(data) > _MAX_PACKET_BYTES:
                    logger.warning("event_too_large", event_type=type(event).__name__, size=len(data))
                    return
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.warning("data_channel_no_running_loop", event_type=type(event).__name__)
                return
            task = loop.create_task(
                self._room.local_participant.publish_data(
                    payload=data,
                    reliable=self._reliable,
                )
            )
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)
        except Exception as e:
            logger.warning("data_channel_publish_failed", error=str(e))
