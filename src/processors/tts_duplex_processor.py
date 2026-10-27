"""TTSDuplexProcessor: token-level duplex TTS with writer/recv task model.

Execution model:

  process_frame(LLMTextFrame)       -> text_queue.put_nowait(token)      [zero blocking]
  process_frame(LLMResponseEndFrame)-> text_queue.put_nowait(SENTINEL)   [zero blocking]

  _writer_loop(ctx_id, session):  text_queue -> session.send_text() -> ledger.register_text()
  _recv_loop(ctx_id, turn_id, session):  session.iter_audio() -> ledger.register_audio_received() -> push TTSAudioFrame

Three concurrent tasks (_start_session spawns writer + recv) ensure that token
ingestion never blocks on TTS network I/O.

Cancellation: InterruptionFrame / CancelFrame -> _cancel_session() tears down
writer, queue, session, recv in that order.

BotStarted/StoppedSpeaking are NOT emitted here — OutputTransport handles those.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass

from src.core.logging import get_logger
from src.pipeline.events import TTSCompleteEvent, TTSFirstAudioEvent, TTSSessionCreatedEvent
from src.pipeline.frames import (
    Frame,
    FrameDirection,
    LLMResponseEndFrame,
    LLMTextFrame,
    TTSAudioFrame,
    TTSPlayoutDoneFrame,
)
from src.pipeline.processor import FrameProcessor

logger = get_logger(__name__)


# ── Shared data type ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TTSAudioChunk:
    """A single audio chunk produced by a TTS backend session."""

    data: bytes
    sample_rate: int


# Sentinel object to signal end-of-text in the queue
_FINISH_SENTINEL = object()


class TTSDuplexProcessor(FrameProcessor):
    """Duplex TTS processor: concurrent writer + recv tasks with PlaybackLedger.

    Parameters
    ----------
    name:
        Processor name (used in tracing / logs).
    tts_backend:
        CosyVoiceDuplexBackend (or any duck-typed equivalent).
    playback_ledger:
        PlaybackLedger for text/audio accounting.
    """

    def __init__(
        self,
        name: str,
        *,
        tts_backend,
        playback_ledger,
        turn_metrics=None,
    ) -> None:
        super().__init__(name)
        self._backend = tts_backend
        self._playback_ledger = playback_ledger
        self._turn_metrics = turn_metrics

        # Per-turn state
        self._session = None
        self._writer_task: asyncio.Task | None = None
        self._recv_task: asyncio.Task | None = None
        self._text_queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._context_id: str = ""
        self._active_turn_id: str = ""
        self._audio_seq: int = 0
        self._session_lock = asyncio.Lock()
        self._first_text_latency_role: str = ""
        # R0005 §6 PR3=B: per-role TTFA tracking within the same TTS session.
        # `_current_text_latency_role` updates on every text frame so the recv
        # loop knows which role each chunk belongs to; `_role_first_text_mono`
        # records the monotonic time the first text for each role was enqueued
        # so we can compute a per-role ttfb relative to its own text-side start.
        self._current_text_latency_role: str = ""
        self._role_first_text_mono: dict[str, float] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────────

    async def setup(self) -> None:
        """Warm TTS pool at pipeline startup."""
        if hasattr(self._backend, "setup"):
            await self._backend.setup()

    async def cleanup(self) -> None:
        """Tear down active session + pool."""
        await self._cancel_session()
        if hasattr(self._backend, "cleanup"):
            await self._backend.cleanup()

    # ── Helpers ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _drain_queue(q: asyncio.Queue) -> None:
        """Drain all items from an asyncio.Queue without blocking."""
        while not q.empty():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                break

    # ── Subclass hooks ─────────────────────────────────────────────────────────

    def _is_tts_related(self) -> bool:
        return True

    async def cancel_inflight(self) -> None:
        await self._cancel_session()
        await super().cancel_inflight()

    async def on_flush(self, frame: Frame) -> None:
        await self._cancel_session()
        await super().on_flush(frame)

    # ── Data frame handling ────────────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, LLMTextFrame):
            ctx_id = frame.metadata.get("context_id", "")
            turn_id = frame.turn_id

            # Start session on first token of a new turn, or after a
            # mid-turn flush (Stage 1 → Stage 2 transition sends an
            # intermediate LLMResponseEndFrame that finishes the writer).
            writer_done = self._writer_task is not None and self._writer_task.done()
            if not self._session or turn_id != self._active_turn_id or writer_done:
                await self._start_session(ctx_id, turn_id)

            role = frame.metadata.get("latency_role", "")
            role = role if isinstance(role, str) and role else "speech"
            if not self._first_text_latency_role:
                self._first_text_latency_role = role
            if role != self._current_text_latency_role:
                self._current_text_latency_role = role
                if role not in self._role_first_text_mono:
                    self._role_first_text_mono[role] = time.monotonic()

            # Enqueue token — zero blocking path
            try:
                self._text_queue.put_nowait(frame.text)
            except asyncio.QueueFull:
                await self._text_queue.put(frame.text)
            return

        if isinstance(frame, LLMResponseEndFrame):
            # Signal end of text input
            try:
                self._text_queue.put_nowait(_FINISH_SENTINEL)
            except asyncio.QueueFull:
                await self._text_queue.put(_FINISH_SENTINEL)
            return

        # All other frames: pass through
        await self.push_frame(frame, direction)

    # ── Session lifecycle ──────────────────────────────────────────────────────

    async def _start_session(self, ctx_id: str, turn_id: str) -> None:
        """Create a duplex TTS session and spawn writer + recv tasks."""
        async with self._session_lock:
            await self._start_session_inner(ctx_id, turn_id)

    async def _start_session_inner(self, ctx_id: str, turn_id: str) -> None:
        """Actual session creation, must be called under _session_lock."""
        # Handle existing session
        if self._session:
            t0 = time.monotonic()
            writer_naturally_done = self._writer_task and self._writer_task.done()
            if writer_naturally_done and self._recv_task and not self._recv_task.done():
                # Stage 1→2 transition: writer finished (send_finish sent),
                # but recv may still have audio chunks in flight.
                # Wait for Stage 1 audio to drain instead of killing it.
                try:
                    await asyncio.wait_for(self._recv_task, timeout=2.0)
                    logger.info("tts.duplex.stage1_drained", turn_id=turn_id)
                except asyncio.TimeoutError:
                    logger.warning("tts.duplex.stage1_drain_timeout", turn_id=turn_id)
            await self._cancel_session()
            cancel_ms = (time.monotonic() - t0) * 1000
            logger.info("tts.duplex.session.cancel_ms", cancel_ms=round(cancel_ms, 1), turn_id=turn_id)
            self._emit("emit_latency", self.name, "tts_session_cancel", cancel_ms, turn_id, time.monotonic() * 1000)

        self._context_id = ctx_id
        self._active_turn_id = turn_id
        self._audio_seq = 0
        self._first_text_latency_role = ""
        self._current_text_latency_role = ""
        self._role_first_text_mono = {}

        # Drain any stale items from the text queue
        self._drain_queue(self._text_queue)

        session = await self._backend.create_session()
        self._session = session

        connect_ms = getattr(session, "connect_duration_ms", 0.0)
        acquire_ms = getattr(session, "acquire_duration_ms", connect_ms)
        pool_hit = getattr(session, "pool_hit", None)
        logger.info(
            "tts.duplex.session.started",
            context_id=ctx_id,
            turn_id=turn_id,
            connect_ms=connect_ms,
            acquire_ms=acquire_ms,
            pool_hit=pool_hit,
        )
        self._emit("emit_latency", self.name, "tts_session_create", acquire_ms, turn_id, time.monotonic() * 1000)
        if self._bus:
            self._bus.emit(
                TTSSessionCreatedEvent(
                    turn_id=turn_id,
                    create_ms=acquire_ms,
                    pool_hit=pool_hit,
                )
            )
        if self._turn_metrics:
            self._turn_metrics.record_tts_session_create(turn_id, acquire_ms, pool_hit)

        # Snapshot closure parameters — tasks never read self._context_id / self._active_turn_id
        self._writer_task = asyncio.create_task(self._writer_loop(ctx_id, session))
        self._recv_task = asyncio.create_task(self._recv_loop(ctx_id, turn_id, session))

    async def _cancel_session(self) -> None:
        """Tear down writer -> queue -> session -> recv, in order."""
        # 1. Cancel writer
        if self._writer_task and not self._writer_task.done():
            self._writer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._writer_task

        # 2. Drain text queue
        self._drain_queue(self._text_queue)

        # 3. Cancel session (closes WebSocket, unblocks iter_audio)
        if self._session:
            await self._session.cancel()

        # 4. Cancel recv
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._recv_task

        # 5. Reset state
        # Interrupt labeling is owned solely by LiveKitOutputTransport.on_flush
        # (it knows the drain state); _cancel_session no longer emits.
        self._session = None
        self._writer_task = None
        self._recv_task = None
        self._context_id = ""

    # ── Background tasks ───────────────────────────────────────────────────────

    async def _writer_loop(self, ctx_id: str, session) -> None:
        """Read tokens from text_queue, send to TTS session, register in ledger."""
        try:
            while True:
                item = await self._text_queue.get()
                if item is _FINISH_SENTINEL:
                    await session.send_finish()
                    logger.debug("tts.duplex.writer.finish_sent", context_id=ctx_id)
                    break
                await session.send_text(item)
                self._playback_ledger.register_text(ctx_id, item)
                logger.debug(
                    "tts.duplex.writer.sent",
                    context_id=ctx_id,
                    token_len=len(item),
                )
        except asyncio.CancelledError:
            logger.info("tts.duplex.writer.cancelled", context_id=ctx_id)
            raise
        except Exception:
            logger.exception("tts.duplex.writer.error", context_id=ctx_id)
            # Cancel session so recv loop exits cleanly instead of hanging
            with contextlib.suppress(Exception):
                await session.cancel()
            # If session.cancel() didn't unblock recv, force-cancel the recv task
            if self._recv_task and not self._recv_task.done():
                self._recv_task.cancel()

    async def _recv_loop(self, ctx_id: str, turn_id: str, session) -> None:
        """Read audio chunks from TTS session, register in ledger, push downstream.

        Per R0005 §6 PR3=B: fires TTFA per latency_role observed within the
        same session, using `_current_text_latency_role` as the role hint at
        chunk-arrival time. The legacy `emit_tts_ttfb` observer dispatch and
        `tts_ttfb` latency emit still fire only on the very first role per
        session — frontend timing UI depends on that single-emission contract.
        """
        start_mono = time.monotonic()
        seen_roles: set[str] = set()
        try:
            async for chunk in session.iter_audio():
                role = self._current_text_latency_role or self._first_text_latency_role or "speech"
                if role not in seen_roles:
                    role_text_mono = self._role_first_text_mono.get(role, start_mono)
                    ttfb_ms = (time.monotonic() - role_text_mono) * 1000
                    seen_roles.add(role)
                    is_first_role_in_session = len(seen_roles) == 1
                    if is_first_role_in_session:
                        self._emit("emit_latency", self.name, "tts_ttfb", ttfb_ms, turn_id, time.monotonic() * 1000)
                        self._emit("emit_tts_ttfb", self.name, ttfb_ms, turn_id, "")
                    if self._bus:
                        self._bus.emit(
                            TTSFirstAudioEvent(
                                turn_id=turn_id,
                                ttfa_ms=ttfb_ms,
                                first_audio_source=role,
                            )
                        )
                    if self._turn_metrics:
                        self._turn_metrics.record_tts_ttfa(
                            turn_id,
                            ttfb_ms,
                            first_audio_source=role,
                            latency_role=role,
                        )

                seq = self._audio_seq
                self._audio_seq += 1

                self._playback_ledger.register_audio_received(ctx_id, len(chunk.data))

                audio_frame = TTSAudioFrame(
                    audio=chunk.data,
                    sample_rate=chunk.sample_rate,
                    text_span="",
                    context_id=ctx_id,
                    sentence_id="",
                    seq=seq,
                    turn_id=turn_id,
                )
                await self.push_frame(audio_frame, FrameDirection.DOWNSTREAM)

                logger.debug(
                    "tts.duplex.recv.chunk",
                    context_id=ctx_id,
                    seq=seq,
                    bytes=len(chunk.data),
                )

            # All audio chunks delivered — natural completion
            if not seen_roles:
                logger.warning(
                    "tts.duplex.recv.zero_audio",
                    context_id=ctx_id,
                    turn_id=turn_id,
                )
            total_duration_ms = (time.monotonic() - start_mono) * 1000
            await self.push_frame(
                TTSPlayoutDoneFrame(context_id=ctx_id),
                FrameDirection.DOWNSTREAM,
            )
            self._emit(
                "emit_tts_complete",
                self.name,
                ctx_id,
                self._playback_ledger.get_full_text(ctx_id) if self._playback_ledger else "",
                total_duration_ms,
                turn_id,
            )
            if self._bus:
                self._bus.emit(
                    TTSCompleteEvent(
                        turn_id=turn_id,
                        context_id=ctx_id,
                        text_fed=self._playback_ledger.get_full_text(ctx_id) if self._playback_ledger else "",
                        audio_duration_ms=total_duration_ms,
                    )
                )
        except asyncio.CancelledError:
            logger.info("tts.duplex.recv.cancelled", context_id=ctx_id)
            raise
        except Exception:
            logger.exception("tts.duplex.recv.error", context_id=ctx_id)
