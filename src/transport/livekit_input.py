"""LiveKitInputTransport: pipeline entry via LiveKit WebRTC audio stream."""

from __future__ import annotations

import asyncio
import contextlib

from livekit import rtc

from src.core.logging import get_logger
from src.pipeline.frames import (
    EndFrame,
    ErrorFrame,
    FrameDirection,
    InputAudioFrame,
    StartFrame,
)
from src.transport.base import InputTransport

logger = get_logger(__name__)


class LiveKitInputTransport(InputTransport):
    """Pipeline entry. Subscribes to primary participant's audio via LiveKit WebRTC."""

    def __init__(
        self,
        name: str,
        *,
        room: rtc.Room,
        primary_participant: rtc.RemoteParticipant,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        super().__init__(name)
        self._room = room
        self._primary = primary_participant
        self._sample_rate = sample_rate
        self._channels = channels
        self._audio_stream: rtc.AudioStream | None = None
        self._source_task: asyncio.Task | None = None  # type: ignore[type-arg]

    async def setup(self) -> None:
        # Create AudioStream from participant (handles existing + future tracks)
        self._audio_stream = rtc.AudioStream.from_participant(
            participant=self._primary,
            track_source=rtc.TrackSource.SOURCE_MICROPHONE,
            sample_rate=self._sample_rate,
            num_channels=self._channels,
        )
        # Register disconnect handler
        self._room.on("participant_disconnected", self._on_participant_disconnected)

    async def process_frame(self, frame: object, direction: FrameDirection) -> None:
        # StartFrame triggers source task start (ControlFrame arrives via process_frame)
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            self._source_task = asyncio.create_task(self._recv_audio_loop())
            return
        await self.push_frame(frame, direction)  # type: ignore[arg-type]

    async def _recv_audio_loop(self) -> None:
        assert self._audio_stream is not None
        frame_count = 0
        try:
            async for audio_event in self._audio_stream:
                if not self._running:
                    break
                frame_count += 1
                if frame_count <= 3 or frame_count % 500 == 0:
                    logger.info(
                        "lk_input.audio_frame",
                        frame_count=frame_count,
                        bytes=len(audio_event.frame.data.tobytes()),
                        sample_rate=audio_event.frame.sample_rate,
                        channels=audio_event.frame.num_channels,
                    )
                frame = InputAudioFrame(
                    audio=audio_event.frame.data.tobytes(),
                    sample_rate=audio_event.frame.sample_rate,
                    channels=audio_event.frame.num_channels,
                    session_id=self._room.name or "",
                )
                self.submit_frame(frame, FrameDirection.DOWNSTREAM)
        except Exception as e:  # noqa: BLE001
            self.submit_frame(
                ErrorFrame(error_type=type(e).__name__, message=str(e)),
                FrameDirection.DOWNSTREAM,
            )

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        if participant.identity != self._primary.identity:
            return
        logger.info("lk_input.participant_disconnected", identity=participant.identity)
        # Don't immediately EndFrame — user might be refreshing the browser.
        # Schedule a delayed check instead.
        if self._loop:
            self._loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self._delayed_disconnect_check()))

    async def _delayed_disconnect_check(self) -> None:
        await asyncio.sleep(5)
        # Check if primary participant has reconnected
        new_participant = None
        for p in self._room.remote_participants.values():
            if p.identity == self._primary.identity:
                new_participant = p
                break
        if new_participant is not None:
            logger.info("lk_input.participant_reconnected", identity=self._primary.identity)
            # Stop old source task + audio stream
            if self._source_task and not self._source_task.done():
                self._source_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._source_task
            if self._audio_stream is not None:
                await self._audio_stream.aclose()
            # Rebuild with new participant
            self._primary = new_participant
            self._audio_stream = rtc.AudioStream.from_participant(
                participant=self._primary,
                track_source=rtc.TrackSource.SOURCE_MICROPHONE,
                sample_rate=self._sample_rate,
                num_channels=self._channels,
            )
            self._source_task = asyncio.create_task(self._recv_audio_loop())
            logger.info("lk_input.audio_stream_rebuilt")
            return
        logger.info("lk_input.participant_not_reconnected_sending_end_frame")
        self.submit_frame(
            EndFrame(session_id=self._room.name or ""),
            FrameDirection.DOWNSTREAM,
        )

    async def cleanup(self) -> None:
        if self._source_task and not self._source_task.done():
            self._source_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._source_task
        if self._audio_stream is not None:
            await self._audio_stream.aclose()
