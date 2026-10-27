"""LiveKitOutputTransport: pipeline exit via LiveKit WebRTC audio track."""

from __future__ import annotations

import asyncio
import time

from livekit import rtc

from src.core.logging import get_logger
from src.pipeline.events import AgentTTSDoneEvent, FirstAudioQueuedEvent, TTSInterruptedEvent, TTSPlayoutDrainedEvent
from src.pipeline.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    FrameDirection,
    InterruptionFrame,
    OutputAudioFrame,
    PlaybackQueuedFrame,
    TTSAudioFrame,
    TTSPlayoutDoneFrame,
)
from src.session.playback_ledger import PlaybackLedger
from src.session.playout_gate import PlayoutGate
from src.transport.base import OutputTransport

logger = get_logger(__name__)


class LiveKitOutputTransport(OutputTransport):
    """Pipeline exit. Publishes TTS audio to LiveKit room via WebRTC track."""

    def __init__(
        self,
        name: str,
        *,
        room: rtc.Room,
        sample_rate: int = 16000,
        channels: int = 1,
        playback_ledger: PlaybackLedger | None = None,
        playout_gate: PlayoutGate | None = None,
        turn_metrics=None,
    ) -> None:
        super().__init__(name)
        self._room = room
        self._sample_rate = sample_rate
        self._channels = channels
        self._audio_source: rtc.AudioSource | None = None
        self._track: rtc.LocalAudioTrack | None = None
        self._track_sid: str = ""
        self._is_speaking: bool = False
        self._current_context_id: str = ""
        self._current_turn_id: str = ""
        self._queued_text: str = ""
        self._queued_media_duration_ms: float = 0.0
        self._playback_ledger: PlaybackLedger | None = playback_ledger
        self._playout_gate = playout_gate
        self._turn_metrics = turn_metrics
        # Retain in-flight wait_for_playout tasks so they aren't GC'd mid-await.
        self._drain_tasks: set[asyncio.Task] = set()

    async def setup(self) -> None:
        self._audio_source = rtc.AudioSource(
            sample_rate=self._sample_rate,
            num_channels=self._channels,
        )
        self._track = rtc.LocalAudioTrack.create_audio_track("agent-audio", self._audio_source)
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        publication = await self._room.local_participant.publish_track(self._track, options)
        self._track_sid = publication.sid
        logger.info("lk_output.track_published", track_sid=self._track_sid, sample_rate=self._sample_rate)

    async def process_frame(self, frame: object, direction: FrameDirection) -> None:
        if isinstance(frame, TTSAudioFrame):
            await self._handle_tts_audio(frame)
        elif isinstance(frame, TTSPlayoutDoneFrame):
            await self._handle_playout_done(frame)
        elif isinstance(frame, OutputAudioFrame):
            await self._publish_audio(frame.audio, frame.sample_rate)
            await self.push_frame(frame, direction)  # type: ignore[arg-type]
        else:
            await self.push_frame(frame, direction)  # type: ignore[arg-type]

    async def _handle_tts_audio(self, frame: TTSAudioFrame) -> None:
        if frame.context_id and frame.context_id != self._current_context_id:
            if self._is_speaking:
                if self._playback_ledger and self._current_context_id:
                    self._playback_ledger.commit_to_history(self._current_context_id, interrupted=False)
                await self._emit_queued_and_stop(frame.session_id)
            self._current_context_id = frame.context_id
            self._queued_text = ""
            self._queued_media_duration_ms = 0.0

        if not self._is_speaking:
            self._is_speaking = True
            if self._playout_gate and self._current_context_id:
                self._playout_gate.mark_started(
                    self._current_context_id,
                    turn_id=frame.turn_id,
                    mono_ms=time.monotonic() * 1000,
                )
            await self.push_frame(
                BotStartedSpeakingFrame(session_id=frame.session_id),
                FrameDirection.UPSTREAM,
            )

        # Track first audio queued per turn for E2E latency (C5)
        if frame.turn_id and frame.turn_id != self._current_turn_id:
            self._current_turn_id = frame.turn_id
            mono_ts_ms = time.monotonic() * 1000
            self._emit(
                "emit_latency",
                self.name,
                "first_audio_queued",
                0.0,
                frame.turn_id,
                mono_ts_ms,
            )
            if self._bus:
                self._bus.emit(
                    FirstAudioQueuedEvent(
                        turn_id=frame.turn_id,
                        session_id=frame.session_id,
                    )
                )
            if self._turn_metrics is not None:
                self._turn_metrics.record_first_audio_queued(frame.turn_id)

        if frame.text_span:
            self._queued_text += frame.text_span

        samples_per_channel = len(frame.audio) // (2 * self._channels)
        frame_duration_ms = (samples_per_channel / frame.sample_rate) * 1000
        self._queued_media_duration_ms += frame_duration_ms

        await self._publish_audio(frame.audio, frame.sample_rate)

        if self._playback_ledger and self._current_context_id:
            self._playback_ledger.register_audio_played(
                self._current_context_id,
                len(frame.audio),
                frame_duration_ms,
            )

    async def _handle_playout_done(self, frame: TTSPlayoutDoneFrame) -> None:
        """Natural TTS completion — commit and signal BotStoppedSpeaking."""
        if self._is_speaking and self._current_context_id == frame.context_id:
            if self._playback_ledger and self._current_context_id:
                self._playback_ledger.commit_to_history(self._current_context_id, interrupted=False)
            now_ms = time.monotonic() * 1000
            if self._playout_gate and self._current_context_id:
                self._playout_gate.mark_playout_done(
                    self._current_context_id,
                    mono_ms=now_ms,
                    audio_duration_ms=self._queued_media_duration_ms,
                )
                # mark_playout_done = generation done (faster than realtime under
                # CosyVoice duplex). The patient keeps hearing audio until the
                # AudioSource queue actually drains — track that separately so
                # silent-rescue (ADR 0006) doesn't talk over still-playing audio.
                self._spawn_drain_watch(
                    self._current_context_id,
                    turn_id=self._current_turn_id,
                    audio_duration_ms=self._queued_media_duration_ms,
                )
            session_id = frame.session_id or ""
            if self._bus:
                self._bus.emit(
                    AgentTTSDoneEvent(
                        turn_id=self._current_turn_id,
                        context_id=self._current_context_id,
                        kind="normal_reply",
                        session_id=self._room.name if self._room else "",
                    )
                )
            await self._emit_queued_and_stop(session_id)
        elif not self._is_speaking:
            if self._playout_gate and frame.context_id:
                now_ms = time.monotonic() * 1000
                self._playout_gate.mark_playout_done(
                    frame.context_id,
                    mono_ms=now_ms,
                    audio_duration_ms=0.0,
                )
                # No audio to drain — mark drained at once so silent-rescue
                # isn't stuck waiting 30s for a wait_for_playout that won't come.
                self._playout_gate.mark_drained(frame.context_id, mono_ms=now_ms)
            # Zero audio produced (e.g., dead TTS session) — still signal
            # BotStoppedSpeaking so upstream pipeline (deferred judge, post
            # actions) doesn't get stuck waiting for a frame that never comes.
            logger.warning(
                "lk_output.playout_done_zero_audio",
                context_id=frame.context_id,
            )
            await self.push_frame(
                BotStoppedSpeakingFrame(session_id="", context_id=frame.context_id),
                FrameDirection.UPSTREAM,
            )

    def _spawn_drain_watch(self, context_id: str, *, turn_id: str = "", audio_duration_ms: float = -1.0) -> None:
        """Await the AudioSource's real playout end, then mark the gate drained.

        ``rtc.AudioSource.wait_for_playout()`` returns when every captured frame
        has actually been emitted (it honours backpressure and TTS underruns,
        unlike a ``playback_start + audio_duration`` estimate). Runs as a retained
        background task so it doesn't block the post-turn pipeline.

        On drain it also emits ``TTSPlayoutDrainedEvent`` — the only bus signal of
        *audible* end (vs. generation-done ``AgentTTSDoneEvent``); Study 1 barge-in
        anchors its qualification window on it.
        """
        source = self._audio_source
        gate = self._playout_gate
        if source is None or gate is None:
            return

        async def _watch() -> None:
            try:
                await source.wait_for_playout()
            except Exception as e:  # never let a drain-watch failure wedge a turn
                logger.warning("lk_output.wait_for_playout_failed", error=str(e), context_id=context_id)
            now_ms = time.monotonic() * 1000
            gate.mark_drained(context_id, mono_ms=now_ms)
            if self._bus:
                self._bus.emit(
                    TTSPlayoutDrainedEvent(
                        turn_id=turn_id,
                        context_id=context_id,
                        audio_duration_ms=audio_duration_ms,
                        session_id=self._room.name if self._room else "",
                    )
                )

        task = asyncio.create_task(_watch())
        self._drain_tasks.add(task)
        task.add_done_callback(self._drain_tasks.discard)

    async def _publish_audio(self, audio: bytes, sample_rate: int) -> None:
        assert self._audio_source is not None
        samples = len(audio) // (2 * self._channels)
        audio_frame = rtc.AudioFrame(
            data=audio,
            sample_rate=sample_rate,
            num_channels=self._channels,
            samples_per_channel=samples,
        )
        try:
            await self._audio_source.capture_frame(audio_frame)
            self._capture_fail_count = 0
            logger.info("lk_output.audio_published", bytes=len(audio), samples=samples, sample_rate=sample_rate)
        except Exception as e:
            self._capture_fail_count = getattr(self, "_capture_fail_count", 0) + 1
            logger.exception(
                "lk_output.capture_frame_failed",
                error=str(e),
                consecutive_failures=self._capture_fail_count,
            )
            if self._capture_fail_count >= 10:
                logger.error(
                    "lk_output.audio_track_dead: %d consecutive failures, user hears nothing", self._capture_fail_count
                )

    async def _emit_queued_and_stop(self, session_id: str = "") -> None:
        # Snapshot then reset — prevents duplicate frames on re-entrant flush
        context_id = self._current_context_id
        queued_text = self._queued_text
        queued_media_duration_ms = self._queued_media_duration_ms
        was_speaking = self._is_speaking

        self._is_speaking = False
        self._current_context_id = ""
        self._current_turn_id = ""
        self._queued_text = ""
        self._queued_media_duration_ms = 0.0

        if was_speaking and context_id:
            if self._playback_ledger:
                text_queued = self._playback_ledger.get_full_text(context_id)
            else:
                text_queued = queued_text
            await self.push_frame(
                PlaybackQueuedFrame(
                    context_id=context_id,
                    text_queued=text_queued,
                    chars_queued=len(text_queued),
                    audio_duration_ms=queued_media_duration_ms,
                    session_id=session_id,
                ),
                FrameDirection.UPSTREAM,
            )
        if was_speaking:
            await self.push_frame(
                BotStoppedSpeakingFrame(session_id=session_id, context_id=context_id),
                FrameDirection.UPSTREAM,
            )

    async def on_flush(self, frame: object) -> None:
        session_id = getattr(frame, "session_id", "")
        interrupted = isinstance(frame, InterruptionFrame)
        cut_while_audible = interrupted and bool(self._current_context_id) and self._is_speaking
        if cut_while_audible:
            context_id = self._current_context_id
            turn_id = self._current_turn_id or ""
            text_fed = self._playback_ledger.get_full_text(context_id) if self._playback_ledger else ""
            if self._playback_ledger:
                self._playback_ledger.commit_to_history(context_id, interrupted=True)
            if self._playout_gate:
                self._playout_gate.mark_interrupted(context_id, mono_ms=time.monotonic() * 1000)
            # Audio cut while still audible (not drained) → honest interruption record.
            # SOLE emission site: tts_duplex no longer emits, so exactly one event per cut.
            self._emit("emit_tts_interrupted", self.name, context_id, text_fed, 0, turn_id)
            if self._bus:
                self._bus.emit(TTSInterruptedEvent(turn_id=turn_id, context_id=context_id, text_fed=text_fed))
        await self._emit_queued_and_stop(session_id)
        self._flush_mailbox()
        if self._audio_source is not None:
            self._audio_source.clear_queue()

    async def cleanup(self) -> None:
        if self._track_sid and self._room.local_participant:
            await self._room.local_participant.unpublish_track(self._track_sid)
