"""STTProcessor: converts audio frames to transcription frames via an async STT backend.

Stream lifecycle:
  UserStartedSpeaking → cancel + close old stream → create new stream (async)
  InputAudioFrame     → push to stream; if stream not ready yet, buffer in _audio_buffer
  UserStoppedSpeaking → schedule async flush_and_close (non-blocking on system path)

Generation guard: each new stream increments _stream_generation; stale callbacks are
discarded, preventing out-of-order transcription frames from previous turns.

InputAudioFrame is processed AND shadow-forwarded downstream (for TurnDetector audio ring buffer).
TranscriptionFrame is produced via submit_frame (thread-safe callback path).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Protocol

from src.core.config import settings
from src.core.logging import get_logger
from src.pipeline.events import AsrFinalEvent, AsrInterimEvent, AsrPackageLatencyEvent
from src.pipeline.frames import (
    Frame,
    FrameDirection,
    InputAudioFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from src.pipeline.processor import FrameProcessor

logger = get_logger(__name__)


# ── Protocols ─────────────────────────────────────────────────────────────────


class STTStream(Protocol):
    session_id: str
    turn_id: str

    def push_audio(self, audio: bytes) -> None: ...
    async def flush(self) -> None: ...
    async def cancel(self) -> None: ...
    async def aclose(self) -> None: ...


class STTBackend(Protocol):
    async def create_stream(
        self,
        on_transcript,
        session_id: str,
        turn_id: str,
    ) -> STTStream: ...


# ── Processor ─────────────────────────────────────────────────────────────────


class STTProcessor(FrameProcessor):
    """Manages a per-turn STT stream and emits TranscriptionFrames.

    Parameters
    ----------
    name:
        Processor name (used in tracing / logs).
    stt_backend:
        Any object conforming to the STTBackend protocol.
    """

    def __init__(self, name: str, *, stt_backend: STTBackend) -> None:
        super().__init__(name)
        self._backend = stt_backend
        self._stream: STTStream | None = None
        self._stream_id: int | None = None
        self._stream_generation: int = 0
        self._pending_stop_generation: int | None = None
        self._stream_ready: bool = False
        self._audio_buffer: list[bytes] = []
        self._create_task: asyncio.Task | None = None
        self._standby_stream: STTStream | None = None
        self._standby_stream_id: int | None = None
        self._standby_ready_at: float = 0.0
        self._standby_task: asyncio.Task | None = None
        self._next_stream_id: int = 1
        self._stream_metadata: dict[int, dict[str, object]] = {}
        # session_id for the current turn (used when creating a new stream)
        self._current_session_id: str = ""
        self._current_turn_id: str = ""

    async def setup(self) -> None:
        self._schedule_standby_stream()

    # ── System frame handling ─────────────────────────────────────────────────

    async def handle_system_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, UserStartedSpeakingFrame):
            await self._on_user_started(frame)
            # Do NOT forward UserStartedSpeakingFrame; let base handle other types
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, UserStoppedSpeakingFrame):
            # Forward system frame first (non-blocking for system path)
            await self.push_frame(frame, direction)
            generation = self._stream_generation
            self._pending_stop_generation = generation
            # Then schedule async flush+close without blocking
            if self._stream is not None:
                asyncio.create_task(self._flush_and_close(self._stream, generation, self._stream_id))
            return

        await super().handle_system_frame(frame, direction)

    # ── Data frame handling ───────────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, InputAudioFrame):
            # Process audio for STT, then shadow-forward downstream for TurnDetector ring buffer
            self._handle_audio(frame)
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _on_user_started(self, frame: UserStartedSpeakingFrame) -> None:
        """Close old stream (if any) and begin creating a new one."""
        # Cancel any in-flight stream creation
        if self._create_task and not self._create_task.done():
            self._create_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._create_task

        # Close old active stream
        old_stream = self._stream
        old_stream_id = self._stream_id
        if old_stream is not None:
            asyncio.create_task(self._cancel_stream_and_forget(old_stream, old_stream_id))

        # Bump generation so stale callbacks from old stream are ignored
        self._stream_generation += 1
        generation = self._stream_generation

        # Reset state for new turn
        self._stream = None
        self._stream_id = None
        self._stream_ready = False
        self._audio_buffer = []
        self._pending_stop_generation = None
        self._current_session_id = frame.session_id
        self._current_turn_id = frame.turn_id

        logger.info(
            "stt.stream.creating",
            turn_id=frame.turn_id,
            session_id=frame.session_id,
            generation=generation,
        )

        standby_stream = self._standby_stream
        standby_stream_id = self._standby_stream_id
        if self._standby_task and not self._standby_task.done():
            self._standby_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._standby_task
        self._standby_task = None
        self._standby_stream = None
        self._standby_stream_id = None

        if standby_stream is not None and standby_stream_id is not None:
            # The provider can stop an ASR task after an idle opening. A warm
            # object is not proof that its recognition session is still alive.
            # Reconnect during the utterance; the existing audio buffer retains
            # all onset frames while the replacement is being created.
            age = time.monotonic() - self._standby_ready_at
            if age > 15.0 or not getattr(standby_stream, "is_healthy", True):
                logger.info("stt.standby_stream.expired", age_s=round(age, 1))
                asyncio.create_task(self._cancel_stream_and_forget(standby_stream, standby_stream_id))
                standby_stream = None
                standby_stream_id = None
        self._standby_ready_at = 0.0
        if standby_stream is not None and standby_stream_id is not None:
            self._promote_standby_stream(
                stream=standby_stream,
                stream_id=standby_stream_id,
                generation=generation,
                session_id=frame.session_id,
                turn_id=frame.turn_id,
            )
            return

        # Create stream asynchronously; audio arriving before it's ready gets buffered.
        self._create_task = asyncio.create_task(self._create_stream(generation, frame.session_id, frame.turn_id))

    def _allocate_stream_id(self, *, generation: int, session_id: str, turn_id: str) -> int:
        stream_id = self._next_stream_id
        self._next_stream_id += 1
        self._stream_metadata[stream_id] = {
            "generation": generation,
            "session_id": session_id,
            "turn_id": turn_id,
        }
        return stream_id

    async def _create_backend_stream(self, *, stream_id: int) -> STTStream:
        metadata = self._stream_metadata[stream_id]

        def on_transcript(text: str, is_final: bool, confidence: float) -> None:
            current = self._stream_metadata.get(stream_id)
            if current is None:
                logger.debug("stt.transcript.stale_stream", stream_id=stream_id)
                return
            self._handle_transcript(
                text,
                is_final,
                confidence,
                turn_id=str(current.get("turn_id", "")),
                generation=int(current.get("generation", -1)),
                session_id=str(current.get("session_id", "")),
            )

        return await self._backend.create_stream(
            on_transcript,
            str(metadata["session_id"]),
            str(metadata["turn_id"]),
        )

    def _update_stream_metadata(self, *, stream_id: int, generation: int, session_id: str, turn_id: str) -> None:
        self._stream_metadata[stream_id] = {
            "generation": generation,
            "session_id": session_id,
            "turn_id": turn_id,
        }

    def _promote_standby_stream(
        self,
        *,
        stream: STTStream,
        stream_id: int,
        generation: int,
        session_id: str,
        turn_id: str,
    ) -> None:
        self._update_stream_metadata(
            stream_id=stream_id,
            generation=generation,
            session_id=session_id,
            turn_id=turn_id,
        )
        with contextlib.suppress(Exception):
            stream.session_id = session_id
        with contextlib.suppress(Exception):
            stream.turn_id = turn_id
        self._stream = stream
        self._stream_id = stream_id
        self._stream_ready = True

        logger.info(
            "stt.stream.ready",
            turn_id=turn_id,
            generation=generation,
            buffered_chunks=0,
            standby=True,
        )

    async def _create_stream(self, generation: int, session_id: str, turn_id: str) -> None:
        """Async task: call backend.create_stream, then flush buffered audio."""
        stream_id = self._allocate_stream_id(generation=generation, session_id=session_id, turn_id=turn_id)
        try:
            stream = await self._create_backend_stream(stream_id=stream_id)
        except asyncio.CancelledError:
            self._stream_metadata.pop(stream_id, None)
            raise
        except Exception:
            self._stream_metadata.pop(stream_id, None)
            logger.exception("stt.stream.create_failed", turn_id=turn_id, generation=generation)
            # Clear stale buffer so it doesn't grow unbounded
            self._audio_buffer = []
            # Emit an empty final transcript so downstream (TurnDetector) doesn't
            # hang waiting for ASR text that will never arrive
            fallback = TranscriptionFrame(
                text="",
                is_final=True,
                confidence=0.0,
                session_id=session_id,
                turn_id=turn_id,
            )
            self.submit_frame(fallback, FrameDirection.DOWNSTREAM)
            return

        # Guard: if generation changed while we were waiting, discard this stream
        if generation != self._stream_generation:
            asyncio.create_task(self._cancel_stream_and_forget(stream, stream_id))
            return

        self._stream = stream
        self._stream_id = stream_id

        # Flush buffered audio into the now-ready stream
        for chunk in self._audio_buffer:
            stream.push_audio(chunk)
        buffered_chunks = len(self._audio_buffer)
        self._audio_buffer = []
        self._stream_ready = True

        logger.info(
            "stt.stream.ready",
            turn_id=turn_id,
            generation=generation,
            buffered_chunks=buffered_chunks,
        )

        if self._pending_stop_generation == generation:
            asyncio.create_task(self._flush_and_close(stream, generation, stream_id))

    def _schedule_standby_stream(self) -> None:
        if not settings.stt_standby_enabled:
            return
        if self._stream is not None or self._standby_stream is not None:
            return
        if self._standby_task is not None and not self._standby_task.done():
            return
        self._standby_task = asyncio.create_task(self._create_standby_stream())

    async def _create_standby_stream(self) -> None:
        stream_id = self._allocate_stream_id(generation=-1, session_id="", turn_id="")
        try:
            stream = await self._create_backend_stream(stream_id=stream_id)
        except asyncio.CancelledError:
            self._stream_metadata.pop(stream_id, None)
            raise
        except Exception:
            self._stream_metadata.pop(stream_id, None)
            logger.exception("stt.standby_stream.create_failed")
            return

        if self._stream is not None or self._standby_stream is not None:
            asyncio.create_task(self._cancel_stream_and_forget(stream, stream_id))
            return

        self._standby_stream = stream
        self._standby_stream_id = stream_id
        self._standby_ready_at = time.monotonic()
        logger.info("stt.standby_stream.ready", stream_id=stream_id)

    def _handle_audio(self, frame: InputAudioFrame) -> None:
        """Route audio to the stream or buffer it if the stream is not yet ready."""
        if self._stream_ready and self._stream is not None:
            self._stream.push_audio(frame.audio)
        else:
            self._audio_buffer.append(frame.audio)
            if settings.polaris_asr_audit_log and len(self._audio_buffer) in {1, 5, 10, 20}:
                logger.info(
                    "stt.audio_buffering",
                    turn_id=frame.turn_id,
                    buffered_chunks=len(self._audio_buffer),
                    sample_rate=frame.sample_rate,
                )

    def _handle_transcript(
        self,
        text: str,
        is_final: bool,
        confidence: float,
        turn_id: str,
        generation: int,
        session_id: str,
    ) -> None:
        """Called from the STT backend callback (may be from any context).

        Stale callbacks (generation mismatch) are discarded.
        """
        if generation != self._stream_generation:
            logger.debug(
                "stt.transcript.stale",
                generation=generation,
                current_generation=self._stream_generation,
                turn_id=turn_id,
            )
            return

        t_frame = TranscriptionFrame(
            text=text,
            is_final=is_final,
            confidence=confidence,
            turn_id=turn_id,
            session_id=session_id,
        )
        if self._bus:
            if is_final:
                self._bus.emit(AsrFinalEvent(turn_id=turn_id, text=text, confidence=confidence))
            else:
                self._bus.emit(AsrInterimEvent(turn_id=turn_id, text=text, confidence=confidence))
        # submit_frame is thread-safe
        self.submit_frame(t_frame, FrameDirection.DOWNSTREAM)

    async def _flush_and_close(self, stream: STTStream | None, generation: int, stream_id: int | None) -> None:
        """Flush the stream (triggers final transcript callback) then close it."""
        if stream is None:
            return
        try:
            await asyncio.wait_for(stream.flush(), timeout=5.0)
            # Emit computed durations (not raw epoch timestamps)
            turn_id = getattr(stream, "turn_id", "")
            first_pkg_ms = getattr(stream, "first_package_latency_ms", -1)
            total_ms = getattr(stream, "total_latency_ms", -1)
            if first_pkg_ms > 0:
                self._emit(
                    "emit_latency", self.name, "stt_first_package", first_pkg_ms, turn_id, time.monotonic() * 1000
                )
            if total_ms > 0:
                self._emit("emit_latency", self.name, "stt_last_package", total_ms, turn_id, time.monotonic() * 1000)
            if self._bus and (first_pkg_ms > 0 or total_ms > 0):
                self._bus.emit(
                    AsrPackageLatencyEvent(
                        turn_id=turn_id,
                        first_package_ms=first_pkg_ms if first_pkg_ms > 0 else -1.0,
                        total_ms=total_ms if total_ms > 0 else -1.0,
                    )
                )
        except Exception:
            logger.exception("stt.stream.flush_failed", generation=generation)
        finally:
            if self._pending_stop_generation == generation:
                self._pending_stop_generation = None
            if generation == self._stream_generation and self._stream is stream:
                self._stream = None
                self._stream_id = None
                self._stream_ready = False
            if stream_id is not None:
                self._stream_metadata.pop(stream_id, None)
            await self._cancel_stream(stream)
            if generation == self._stream_generation and self._stream is None:
                self._schedule_standby_stream()

    @staticmethod
    async def _cancel_stream(stream: STTStream) -> None:
        with contextlib.suppress(Exception):
            await stream.cancel()
        with contextlib.suppress(Exception):
            await stream.aclose()

    async def _cancel_stream_and_forget(self, stream: STTStream, stream_id: int | None) -> None:
        await self._cancel_stream(stream)
        if stream_id is not None:
            self._stream_metadata.pop(stream_id, None)

    # ── Flush / reset ─────────────────────────────────────────────────────────

    async def on_flush(self, frame: Frame) -> None:
        """Called on InterruptionFrame / CancelFrame — discard current turn."""
        old_stream = self._stream
        old_stream_id = self._stream_id
        self._stream = None
        self._stream_id = None
        self._stream_ready = False
        self._audio_buffer = []
        self._pending_stop_generation = None
        self._stream_generation += 1

        if self._create_task and not self._create_task.done():
            self._create_task.cancel()

        if old_stream is not None:
            asyncio.create_task(self._cancel_stream_and_forget(old_stream, old_stream_id))
            self._schedule_standby_stream()

        self._flush_mailbox()

    async def cleanup(self) -> None:
        stream = self._stream
        stream_id = self._stream_id
        self._stream = None
        self._stream_id = None
        if self._create_task and not self._create_task.done():
            self._create_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._create_task
        if self._standby_task and not self._standby_task.done():
            self._standby_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._standby_task
        standby_stream = self._standby_stream
        standby_stream_id = self._standby_stream_id
        self._standby_stream = None
        self._standby_stream_id = None
        if stream is not None:
            await self._cancel_stream(stream)
        if standby_stream is not None:
            await self._cancel_stream(standby_stream)
        if stream_id is not None:
            self._stream_metadata.pop(stream_id, None)
        if standby_stream_id is not None:
            self._stream_metadata.pop(standby_stream_id, None)
