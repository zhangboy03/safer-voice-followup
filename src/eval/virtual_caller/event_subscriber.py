from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable

# Keys we care about from `pipeline_event` stderr lines. These are the signal
# events the VC blocks on — if the LiveKit data channel stalls (Docker Desktop
# macOS publisher PC failure ~20s in), the log fallback keeps drive_turns alive.
_LOG_SCRAPE_TYPES = frozenset(
    {
        "AgentTTSDoneEvent",
        "TTSCompleteEvent",
        "TTSPlayoutDrainedEvent",
        "TTSFirstAudioEvent",
        "TTSInterruptedEvent",
        "TurnMetricsEvent",
        "SopTransitionEvent",
        "FirstAudioQueuedEvent",
        # VAD-level events needed by §4.2.5-C interrupt recovery timing:
        "UserStartedSpeakingEvent",
        "UserStoppedSpeakingEvent",
        "VADShadowFrameEvent",
        "VADShadowSpeechEvent",
    }
)

# Match: `pipeline_event    key=value key2=value2 event_type=XxxEvent ...`
_KV_RE = re.compile(r"(\w+)=((?:'[^']*'|\"[^\"]*\"|\S+))")


def _parse_pipeline_event_line(line: str) -> dict | None:
    """Parse a stderr `pipeline_event` line into a dict (best-effort)."""
    idx = line.find("pipeline_event")
    if idx < 0:
        return None
    kv_blob = line[idx + len("pipeline_event") :]
    parsed: dict = {}
    for m in _KV_RE.finditer(kv_blob):
        key = m.group(1)
        raw = m.group(2)
        if raw.startswith("'") and raw.endswith("'"):
            val: object = raw[1:-1]
        elif raw.startswith('"') and raw.endswith('"'):
            val = raw[1:-1]
        else:
            # Try numeric coercion
            try:
                val = int(raw)
            except ValueError:
                try:
                    val = float(raw)
                except ValueError:
                    val = raw
        parsed[key] = val
    if "event_type" not in parsed:
        return None
    # mirror data-channel wire: include a topic field (some callers inspect it)
    parsed.setdefault("topic", parsed["event_type"])
    return parsed


class EventSubscriber:
    """Bridges LiveKit data channel events to asyncio waits and queries.

    Events are indexed by `event_type` (from to_wire()), NOT by `wire_topic`,
    to distinguish events that share the same wire_topic (e.g. JudgeDecisionEvent
    vs RagDecisionEvent both use "dual_llm_debug").

    Also supports a log-scrape fallback for key pipeline events — needed because
    the LiveKit publisher PeerConnection on Docker Desktop macOS stalls ~20s
    into sessions, causing AgentTTSDoneEvent (and other late events) to never
    reach the data channel. The fallback tails `agent_stderr.log` for
    `pipeline_event event_type=...` lines and synthesizes identical payloads,
    de-duped by (event_type, event_id or turn_id).
    """

    def __init__(self) -> None:
        self._by_type: dict[str, deque[dict]] = defaultdict(lambda: deque(maxlen=1000))
        self._all_events: list[dict] = []
        self._waiters: dict[str, list[asyncio.Future]] = defaultdict(list)
        self._raw_handlers: list[Callable[[dict], None]] = []
        self._seen_event_ids: set[str] = set()
        self._log_tail_task: asyncio.Task | None = None

    @property
    def current_cursor(self) -> int:
        return len(self._all_events)

    def attach(self, room) -> None:
        """Subscribe to room data_received events."""
        room.on("data_received", self._on_data_packet)

    def register_raw_handler(self, fn: Callable[[dict], None]) -> None:
        """Register a handler that sees every raw event dict (for ScenarioReporter)."""
        self._raw_handlers.append(fn)

    def _on_data_packet(self, packet) -> None:
        self._on_data_bytes(packet.data)

    def _on_data_bytes(self, data: bytes) -> None:
        try:
            payload = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        self._ingest(payload)

    def _ingest(self, payload: dict) -> None:
        """Shared ingestion path: data-channel packet OR log-scrape synthesis."""
        event_id = payload.get("event_id")
        if event_id and event_id in self._seen_event_ids:
            return
        if event_id:
            self._seen_event_ids.add(event_id)
        event_type = payload.get("event_type", "")
        self._by_type[event_type].append(payload)
        self._all_events.append(payload)
        for handler in self._raw_handlers:
            try:
                handler(payload)
            except Exception:
                pass
        # Resolve any waiters for this event_type
        for f in self._waiters.pop(event_type, []):
            if not f.done():
                f.set_result(payload)

    # ── Log scrape fallback ──

    def attach_log_fallback(self, stderr_log: Path | str) -> None:
        """Start a background task that tails stderr_log for pipeline_event lines.

        Synthesizes wire-format payloads for key event types (AgentTTSDone,
        TurnMetrics, TTSComplete, etc.) so drive_turns still advances even
        when the LiveKit publisher PeerConnection has stalled.

        Must be called inside a running event loop. No-op if already attached.
        """
        if self._log_tail_task is not None:
            return
        path = Path(stderr_log)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._log_tail_task = loop.create_task(self._tail_log(path))

    async def _tail_log(self, path: Path) -> None:
        """Tail `path`, feeding matching pipeline_event lines into _ingest."""
        # Wait for the file to appear (agent subprocess opens it async).
        for _ in range(100):
            if path.exists():
                break
            await asyncio.sleep(0.1)
        else:
            return
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fp:
                # Start from the beginning — we rely on event_id dedupe to
                # avoid double-ingesting events that arrived via data channel.
                while True:
                    line = fp.readline()
                    if not line:
                        await asyncio.sleep(0.05)
                        continue
                    if "pipeline_event" not in line:
                        continue
                    parsed = _parse_pipeline_event_line(line)
                    if parsed is None:
                        continue
                    if parsed.get("event_type") not in _LOG_SCRAPE_TYPES:
                        continue
                    self._ingest(parsed)
        except asyncio.CancelledError:
            return
        except Exception:
            return

    async def stop_log_fallback(self) -> None:
        task = self._log_tail_task
        self._log_tail_task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def wait(
        self,
        event_type: str,
        *,
        timeout_s: float = 30.0,
        after_cursor: int | None = None,
    ) -> dict:
        """Wait for an event of the given type. Checks history first."""
        start = after_cursor if after_cursor is not None else 0
        for evt in self._all_events[start:]:
            if evt.get("event_type") == event_type:
                return evt
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._waiters[event_type].append(future)
        return await asyncio.wait_for(future, timeout=timeout_s)

    def saw_any(
        self,
        event_types: list[str],
        *,
        after_cursor: int | None = None,
    ) -> bool:
        """Check if any event of the given types has been received."""
        start = after_cursor if after_cursor is not None else 0
        return any(evt.get("event_type") in event_types for evt in self._all_events[start:])

    def snapshot_since(self, cursor: int) -> list[dict]:
        """Return all events received since the given cursor position."""
        return list(self._all_events[cursor:])
