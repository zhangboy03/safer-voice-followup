"""VirtualCaller — LiveKit participant driven by a PatientDriver."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from src.core.logging import get_logger
from src.eval.virtual_caller.actions import PatientAction
from src.eval.virtual_caller.audio_player import AudioPlayer
from src.eval.virtual_caller.drivers import AgentTurn, PatientDriver
from src.eval.virtual_caller.event_subscriber import EventSubscriber
from src.eval.virtual_caller.tts_cache import TTSFixtureCache

logger = get_logger(__name__)


class CallerConnectionError(Exception):
    pass


class AgentReadyTimeout(Exception):
    pass


class TurnTimeout(Exception):
    pass


class ScopeCursorNotLocked(Exception):
    pass


@dataclass
class CallerConfig:
    tts_cache: TTSFixtureCache | None = None
    default_voice: str = "doubao-female-30s"


@dataclass
class DriveResult:
    turns: int = 0
    stopped_by: str = ""
    history: list[AgentTurn] = field(default_factory=list)


class _TurnScope:
    """Context manager that locks a cursor on first audio frame push."""

    def __init__(self, subscriber: EventSubscriber) -> None:
        self._sub = subscriber
        self._cursor: int | None = None

    @property
    def cursor(self) -> int:
        if self._cursor is None:
            raise ScopeCursorNotLocked("cursor not yet locked — first frame hasn't been pushed")
        return self._cursor

    def lock_cursor(self) -> None:
        """Called by AudioPlayer on_first_frame callback."""
        if self._cursor is None:
            self._cursor = self._sub.current_cursor

    def __enter__(self) -> "_TurnScope":
        return self

    def __exit__(self, *args) -> None:
        pass


class VirtualCaller:
    """LiveKit participant driven by a PatientDriver. Lives for one scenario."""

    def __init__(
        self,
        room_url: str,
        room_name: str,
        token: str,
        config: CallerConfig | None = None,
    ) -> None:
        self._room_url = room_url
        self._room_name = room_name
        self._token = token
        self._config = config or CallerConfig()
        self._room = None
        self._audio_source = None
        self._audio_track = None
        self._audio_player = AudioPlayer()
        self._event_sub = EventSubscriber()
        self._history: list[AgentTurn] = []
        self._connected = False

    @property
    def event_subscriber(self) -> EventSubscriber:
        return self._event_sub

    # ── Lifecycle ──

    async def connect(self) -> None:
        """Join the LiveKit room, publish audio track, attach EventSubscriber."""
        try:
            from livekit import rtc

            self._room = rtc.Room()
            await asyncio.wait_for(
                self._room.connect(self._room_url, self._token),
                timeout=30.0,
            )
            self._event_sub.attach(self._room)

            self._audio_source = rtc.AudioSource(48000, 1)
            self._audio_track = rtc.LocalAudioTrack.create_audio_track("caller-audio", self._audio_source)
            # Publish with SOURCE_MICROPHONE so the agent's
            # LiveKitInputTransport (which filters by MICROPHONE track source)
            # actually subscribes to it. Without this the agent sits idle.
            publish_opts = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            await self._room.local_participant.publish_track(self._audio_track, publish_opts)
            self._audio_player.bind(self._audio_source)
            self._connected = True
            logger.info("caller_connected", room=self._room_name)
        except ImportError:
            # Test mode without livekit — use mock
            logger.warning("livekit_not_available", msg="running in mock mode")
            self._connected = True
        except asyncio.TimeoutError as e:
            raise CallerConnectionError(f"Failed to connect to room {self._room_name}") from e

    async def disconnect(self) -> None:
        """Leave the room."""
        if self._room:
            try:
                await self._room.disconnect()
            except Exception as e:
                logger.warning("disconnect_error", error=str(e))
        self._connected = False
        logger.info("caller_disconnected", room=self._room_name)

    # ── Main driving interface ──

    async def drive_turns(
        self,
        driver: PatientDriver,
        *,
        max_turns: int = 20,
        stop_on: str | list[str] | None = None,
        per_turn_timeout_s: float = 30.0,
    ) -> DriveResult:
        """Drive conversation turns using a PatientDriver."""
        stop_topics = _normalize_stop_on(stop_on)
        history: list[AgentTurn] = []

        for turn_idx in range(max_turns):
            action = await driver.next_utterance(history)

            if action.kind == "hangup":
                return DriveResult(turns=turn_idx, stopped_by="hangup", history=history)

            if action.kind == "silence":
                await asyncio.sleep((action.silence_ms or 1000) / 1000)
                continue

            # Execute turn with scope tracking
            scope = _TurnScope(self._event_sub)
            with scope:
                await self._execute_action(action, scope)

                # Wait for agent to finish speaking
                try:
                    agent_done = await self.wait_agent_tts_done(
                        after_cursor=scope.cursor if scope._cursor else None,
                        timeout_s=per_turn_timeout_s,
                    )
                except asyncio.TimeoutError:
                    raise TurnTimeout(f"Agent did not respond within {per_turn_timeout_s}s on turn {turn_idx}")

                # Build agent turn record
                events_this_turn = self._event_sub.snapshot_since(scope.cursor) if scope._cursor is not None else []
                agent_text = agent_done.get("context_id", "")  # or extract from events
                agent_turn = AgentTurn(text=agent_text, events=events_this_turn)
                history.append(agent_turn)
                self._history = history

            # Check stop conditions — optionally wait briefly for trailing events
            if stop_topics and scope._cursor is not None:
                matched = await self._check_stop_on(
                    stop_topics,
                    after_cursor=scope.cursor,
                    settle_s=0.1,
                )
                if matched:
                    return DriveResult(turns=turn_idx + 1, stopped_by=matched, history=history)

        return DriveResult(turns=max_turns, stopped_by="max_turns", history=history)

    # ── Low-level primitives ──

    async def wait_for_event(
        self,
        event_type_name: str,
        *,
        timeout_s: float = 30.0,
        after_cursor: int | None = None,
    ) -> dict:
        """Wait for a specific event type from the agent."""
        return await self._event_sub.wait(
            event_type_name,
            timeout_s=timeout_s,
            after_cursor=after_cursor,
        )

    async def wait_agent_tts_done(
        self,
        *,
        kind: str = "normal_reply",
        timeout_s: float = 30.0,
        after_cursor: int | None = None,
    ) -> dict:
        """Wait for AgentTTSDoneEvent."""
        return await self._event_sub.wait(
            "AgentTTSDoneEvent",
            timeout_s=timeout_s,
            after_cursor=after_cursor,
        )

    def turn_scope(self) -> _TurnScope:
        """Create a new turn scope for cursor tracking."""
        return _TurnScope(self._event_sub)

    # ── Internal ──

    async def _check_stop_on(
        self,
        stop_topics: list[str],
        *,
        after_cursor: int,
        settle_s: float = 0.1,
    ) -> str | None:
        """Check if any stop event arrived; wait up to settle_s for trailing events."""
        # Check immediately
        for topic in stop_topics:
            if self._event_sub.saw_any([topic], after_cursor=after_cursor):
                return topic
        # Wait briefly for trailing events
        try:
            for topic in stop_topics:
                event = await asyncio.wait_for(
                    self._event_sub.wait(topic, after_cursor=after_cursor, timeout_s=settle_s),
                    timeout=settle_s,
                )
                if event:
                    return topic
        except (asyncio.TimeoutError, TimeoutError):
            pass
        return None

    async def _execute_action(self, action: PatientAction, scope: _TurnScope) -> None:
        """Execute a single PatientAction (speak or backchannel)."""
        if action.kind in ("speak", "backchannel"):
            text = action.text or action.backchannel_label or ""
            voice = action.voice or self._config.default_voice

            if self._config.tts_cache:
                wav_path = await self._config.tts_cache.get_wav(text, voice)
                await self._audio_player.play_wav(
                    wav_path,
                    on_first_frame=scope.lock_cursor,
                )
            else:
                # No TTS cache — lock cursor immediately
                scope.lock_cursor()
                logger.warning("no_tts_cache", msg="playing without audio")


def _normalize_stop_on(stop_on: str | list[str] | None) -> list[str]:
    if stop_on is None:
        return []
    if isinstance(stop_on, str):
        return [stop_on]
    return list(stop_on)
