"""Local transport test doubles for the voice pipeline.

LocalInputTransport  — wraps an async generator of raw audio bytes and feeds
                       InputAudioFrames into the pipeline, then emits EndFrame.
LocalOutputTransport — collects all frames in output_frames list; forwards
                       SystemFrames downstream but swallows DataFrames.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from src.pipeline.frames import (
    EndFrame,
    Frame,
    FrameDirection,
    InputAudioFrame,
    StartFrame,
)
from src.transport.base import InputTransport, OutputTransport


class LocalInputTransport(InputTransport):
    """Feeds audio from an async iterator into the pipeline.

    On StartFrame arrival (ControlFrame, handled via process_frame), starts
    _feed_loop as a background task.  All other frames are forwarded as-is.
    """

    def __init__(
        self,
        name: str,
        audio_source: AsyncIterator[bytes],
        sample_rate: int = 16000,
    ) -> None:
        super().__init__(name)
        self._audio_source = audio_source
        self._sample_rate = sample_rate
        self._feed_task: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, StartFrame):
            # StartFrame is a ControlFrame (priority 1), arrives here.
            # Forward it downstream first, then kick off audio feeding.
            await self.push_frame(frame, direction)
            self._feed_task = asyncio.create_task(self._feed_loop())
            return
        await self.push_frame(frame, direction)

    async def _feed_loop(self) -> None:
        """Consume audio_source, emit InputAudioFrames, then EndFrame."""
        try:
            async for chunk in self._audio_source:
                frame = InputAudioFrame(audio=chunk, sample_rate=self._sample_rate)
                await self.push_frame(frame, FrameDirection.DOWNSTREAM)
            # Audio exhausted — signal end of stream.
            await self.push_frame(EndFrame(), FrameDirection.DOWNSTREAM)
        except asyncio.CancelledError:
            pass

    async def cleanup(self) -> None:
        if self._feed_task and not self._feed_task.done():
            self._feed_task.cancel()
            try:
                await self._feed_task
            except asyncio.CancelledError:
                pass
        self._feed_task = None


class LocalOutputTransport(OutputTransport):
    """Collects all frames that arrive at the pipeline sink.

    - DataFrames: appended to output_frames, NOT forwarded.
    - SystemFrames: appended to output_frames AND forwarded downstream
      (so EndFrame reaches _TerminalSink and terminates the runner).
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.output_frames: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Handle DataFrames and ControlFrames: collect but do not forward."""
        self.output_frames.append(frame)

    async def handle_system_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Handle SystemFrames: collect AND forward so EndFrame reaches terminal."""
        self.output_frames.append(frame)
        await self.push_frame(frame, direction)

    async def on_flush(self, frame: Frame) -> None:
        self._flush_mailbox()

    async def cleanup(self) -> None:
        pass
