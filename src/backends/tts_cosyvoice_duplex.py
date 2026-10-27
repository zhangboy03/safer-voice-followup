"""CosyVoice duplex TTS backend — one WebSocket session per turn.

Unlike CosyVoiceBackend (one session per sentence), this backend keeps a single
WebSocket open for the entire turn and accepts multiple send_text() calls
(one per LLM token / sentence fragment) followed by a single send_finish().
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import AsyncIterator
from uuid import uuid4

import websockets
from websockets.protocol import State as _WsState

from src.core.config import settings
from src.core.logging import get_logger
from src.core.voice_selection import resolve_current_tts_profile
from src.processors.tts_duplex_processor import TTSAudioChunk

logger = get_logger(__name__)


class CosyVoiceDuplexSession:
    """Per-turn duplex session.

    Typical usage::

        session = await backend.create_session()
        await session.send_text("hello ")
        await session.send_text("world")
        await session.send_finish()
        async for chunk in session.iter_audio():
            ...   # process PCM bytes
    """

    def __init__(self, ws, task_id: str, sample_rate: int, model: str = "", voice: str = "") -> None:
        self._ws = ws
        self._task_id = task_id
        self._sample_rate = sample_rate
        self._audio_queue: asyncio.Queue[TTSAudioChunk | None] = asyncio.Queue()
        self._recv_task: asyncio.Task | None = None
        self.connect_duration_ms: float = 0.0  # populated by create_session()
        self.acquire_duration_ms: float = 0.0  # actual wait paid by this turn
        self.pool_hit: bool = False
        self.model = model
        self.voice = voice
        self.created_mono: float = time.monotonic()  # for proactive age-based eviction

    # ── Public API ────────────────────────────────────────────────────────────

    async def send_text(self, text: str) -> None:
        """Send one continue-task message with a text fragment."""
        msg = {
            "header": {
                "action": "continue-task",
                "task_id": self._task_id,
                "streaming": "duplex",
            },
            "payload": {"input": {"text": text}},
        }
        await self._ws.send(json.dumps(msg))

    async def send_finish(self) -> None:
        """Signal end of text input for this turn."""
        msg = {
            "header": {
                "action": "finish-task",
                "task_id": self._task_id,
                "streaming": "duplex",
            },
            "payload": {"input": {}},
        }
        await self._ws.send(json.dumps(msg))

    async def cancel(self) -> None:
        """Immediately close the WebSocket and terminate audio iteration.

        The WebSocket close handshake is fire-and-forget: DashScope takes ~1s
        to respond to a Close frame, and blocking on that adds 1s to every
        turn transition.  We send the Close frame then move on.
        """
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
        # Fire-and-forget: send close frame but don't wait for the server's reply
        asyncio.ensure_future(self._safe_ws_close())
        await self._audio_queue.put(None)  # unblock iter_audio

    async def _safe_ws_close(self) -> None:
        """Background WebSocket close — never raises, never blocks callers."""
        try:
            await self._ws.close()
        except Exception:
            pass

    async def iter_audio(self) -> AsyncIterator[TTSAudioChunk]:
        """Yield PCM audio chunks until the session ends or is cancelled."""
        while True:
            chunk = await self._audio_queue.get()
            if chunk is None:
                break
            yield chunk

    def start_recv(self) -> None:
        """Start the background recv task. Called by create_session()."""
        self._recv_task = asyncio.create_task(self._recv_loop())

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _recv_loop(self) -> None:
        """Read WebSocket messages: binary frames → audio queue; JSON → status."""
        chunk_count = 0
        total_bytes = 0
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    chunk_count += 1
                    total_bytes += len(message)
                    self._audio_queue.put_nowait(TTSAudioChunk(data=message, sample_rate=self._sample_rate))
                elif isinstance(message, str):
                    data = json.loads(message)
                    event = data.get("header", {}).get("event", "")
                    if event in ("task-finished", "task-failed"):
                        logger.info(
                            "tts.duplex.recv.done",
                            task_event=event,
                            chunk_count=chunk_count,
                            total_bytes=total_bytes,
                        )
                        break
        except asyncio.CancelledError:
            logger.warning(
                "tts.duplex.recv.cancelled",
                chunk_count=chunk_count,
                total_bytes=total_bytes,
            )
            raise
        except websockets.exceptions.ConnectionClosed:
            logger.warning("tts.duplex.recv.connection_closed", chunk_count=chunk_count)
        except Exception:
            logger.exception("tts.duplex.recv.error", chunk_count=chunk_count)
        finally:
            await self._audio_queue.put(None)  # signal end to iter_audio


class CosyVoiceDuplexBackend:
    """TTS backend with optional pre-warmed session pool.

    When pool_size > 0 (set via setup()), maintains ready-to-use sessions
    (WebSocket connected + task-started). acquire_session() returns from pool
    on hit; falls back to on-demand creation on miss. Background task refills.
    Stale sessions (closed WebSocket) are detected and discarded.
    """

    def __init__(
        self,
        model: str | None = None,
        voice: str | None = None,
        sample_rate: int | None = None,
    ) -> None:
        self._default_model = model or settings.tts_model
        self._default_voice = voice or settings.tts_voice
        self._sample_rate = sample_rate or settings.tts_sample_rate

        # Pool state (initialized by setup())
        self._pool: asyncio.Queue[CosyVoiceDuplexSession] = asyncio.Queue()
        self._target_pool_size: int = 0
        self._refill_event = asyncio.Event()
        self._refill_task: asyncio.Task | None = None
        self._closed = False
        self._setup_lock = asyncio.Lock()
        self._setup_done = False
        self._pool_profile_key: str | None = None

        # Metrics
        self._pool_hits = 0
        self._pool_misses = 0
        self._pool_stale = 0

    @property
    def pool_size(self) -> int:
        return self._pool.qsize()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def setup(self, pool_size: int | None = None) -> None:
        """Pre-warm the session pool. pool_size=0 disables pooling."""
        if self._setup_done:
            return

        async with self._setup_lock:
            if self._setup_done:
                return

            self._closed = False
            model, voice = self._resolve_profile()
            self._pool_profile_key = self._make_profile_key(model, voice)
            self._target_pool_size = pool_size if pool_size is not None else settings.tts_pool_size
            if self._target_pool_size <= 0:
                self._setup_done = True
                logger.info("tts.pool.disabled")
                return

            for i in range(self._target_pool_size):
                try:
                    session = await self._create_raw_session()
                    self._pool.put_nowait(session)
                except Exception:
                    logger.warning("tts.pool.warmup_failed", index=i, exc_info=True)
            logger.info("tts.pool.warmed", size=self._pool.qsize(), target=self._target_pool_size)
            self._refill_task = asyncio.create_task(self._refill_loop())
            self._setup_done = True

    async def cleanup(self) -> None:
        """Close all pooled sessions and stop refill."""
        self._closed = True
        self._refill_event.set()  # wake refill so it exits
        if self._refill_task and not self._refill_task.done():
            self._refill_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._refill_task
        while not self._pool.empty():
            try:
                session = self._pool.get_nowait()
                await session.cancel()
            except Exception:
                pass
        logger.info(
            "tts.pool.shutdown",
            hits=self._pool_hits,
            misses=self._pool_misses,
            stale=self._pool_stale,
        )
        self._pool_profile_key = None
        self._setup_done = False

    # ── Session acquisition ──────────────────────────────────────────────────

    async def acquire_session(self) -> CosyVoiceDuplexSession:
        """Get a session: pool hit (O(1)) or on-demand fallback."""
        await self._refresh_pool_if_profile_changed()
        acquire_start = time.monotonic()
        while not self._pool.empty():
            try:
                session = self._pool.get_nowait()
            except asyncio.QueueEmpty:
                break
            if self._is_session_healthy(session):
                self._pool_hits += 1
                session.acquire_duration_ms = (time.monotonic() - acquire_start) * 1000
                session.pool_hit = True
                logger.info(
                    "tts.pool.hit",
                    pool_remaining=self._pool.qsize(),
                    task_id=session._task_id,
                    model=session.model,
                    voice=session.voice,
                )
                session.start_recv()
                self._refill_event.set()
                return session
            else:
                self._pool_stale += 1
                logger.warning("tts.pool.stale", task_id=session._task_id)
                await session.cancel()

        # Pool empty or all stale — create on demand
        self._pool_misses += 1
        logger.info("tts.pool.miss")
        session = await self._create_raw_session()
        session.acquire_duration_ms = (time.monotonic() - acquire_start) * 1000
        session.pool_hit = False
        session.start_recv()
        self._refill_event.set()
        return session

    async def create_session(self) -> CosyVoiceDuplexSession:
        """Backwards-compatible alias for acquire_session()."""
        return await self.acquire_session()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _resolve_profile(self) -> tuple[str, str]:
        return resolve_current_tts_profile(self._default_model, self._default_voice)

    @staticmethod
    def _make_profile_key(model: str, voice: str) -> str:
        return f"{model}|{voice}"

    async def _refresh_pool_if_profile_changed(self) -> None:
        model, voice = self._resolve_profile()
        profile_key = self._make_profile_key(model, voice)
        if profile_key == self._pool_profile_key:
            return

        async with self._setup_lock:
            model, voice = self._resolve_profile()
            profile_key = self._make_profile_key(model, voice)
            if profile_key == self._pool_profile_key:
                return

            previous = self._pool_profile_key
            self._pool_profile_key = profile_key
            if previous is None:
                logger.info("tts.voice.selected", model=model, voice=voice)
                return

            logger.info(
                "tts.voice.changed",
                previous_profile=previous,
                model=model,
                voice=voice,
                pooled_sessions=self._pool.qsize(),
            )
            await self._drain_pool()
            self._refill_event.set()

    async def _drain_pool(self) -> None:
        while not self._pool.empty():
            try:
                session = self._pool.get_nowait()
            except asyncio.QueueEmpty:
                break
            with contextlib.suppress(Exception):
                await session.cancel()

    @staticmethod
    def _is_session_healthy(session: CosyVoiceDuplexSession) -> bool:
        """Check if session's WebSocket is still usable.

        Three failure modes:
        1. Server closed WebSocket (idle timeout) → state != OPEN
        2. Server sent task-finished while idle → recv_task done (if started)
        3. Session too old → DashScope duplex task will timeout soon
        """
        ws = session._ws
        if getattr(ws, "state", _WsState.CLOSED) != _WsState.OPEN:
            return False
        # If recv loop was somehow started and already exited, session is dead
        recv_task = session._recv_task
        if recv_task is not None and recv_task.done():
            return False
        # Proactive age check — evict before DashScope times out the task
        age_s = time.monotonic() - session.created_mono
        if age_s > settings.tts_pool_max_age_s:
            return False
        return True

    async def _create_raw_session(self) -> CosyVoiceDuplexSession:
        """Connect WebSocket, run-task handshake, return ready session."""
        model, voice = self._resolve_profile()
        url = settings.active_dashscope_ws_api_url
        headers = {
            "Authorization": f"bearer {settings.dashscope_api_key}",
            "X-DashScope-DataInspection": "enable",
        }

        t0 = time.monotonic()
        ws = await websockets.connect(url, additional_headers=headers, close_timeout=0.1)
        connect_ms = (time.monotonic() - t0) * 1000

        task_id = uuid4().hex
        run_task = {
            "header": {
                "action": "run-task",
                "task_id": task_id,
                "streaming": "duplex",
            },
            "payload": {
                "task_group": "audio",
                "task": "tts",
                "function": "SpeechSynthesizer",
                "model": model,
                "parameters": {
                    "text_type": "PlainText",
                    "voice": voice,
                    "format": "pcm",
                    "sample_rate": self._sample_rate,
                    "volume": 50,
                    "rate": 1,
                    "pitch": 1,
                },
                "input": {},
            },
        }
        await ws.send(json.dumps(run_task))

        response = await ws.recv()
        data = json.loads(response)
        if data.get("header", {}).get("event") != "task-started":
            await ws.close()
            raise RuntimeError(f"TTS duplex task failed to start: {data}")

        logger.info("tts.duplex.session.created", task_id=task_id, connect_ms=connect_ms, model=model, voice=voice)

        session = CosyVoiceDuplexSession(ws, task_id, self._sample_rate, model=model, voice=voice)
        session.connect_duration_ms = connect_ms
        # NOTE: start_recv() is NOT called here — pooled sessions must stay
        # idle until acquired, otherwise DashScope times out the task and
        # the recv loop exits prematurely. acquire_session() calls start_recv().
        return session

    async def _refill_loop(self) -> None:
        """Background: keep pool at target size and evict aged sessions.

        Runs on two triggers:
        1. _refill_event (after session acquired) — immediate refill
        2. Periodic sweep every tts_pool_max_age_s/2 — evict sessions
           approaching DashScope idle timeout before they go stale
        """
        sweep_interval = max(settings.tts_pool_max_age_s / 2, 5.0)
        while not self._closed:
            try:
                await asyncio.wait_for(self._refill_event.wait(), timeout=sweep_interval)
                self._refill_event.clear()
            except asyncio.TimeoutError:
                # Periodic sweep: evict aged sessions proactively
                await self._evict_aged_sessions()
            if self._closed:
                break
            while self._pool.qsize() < self._target_pool_size and not self._closed:
                try:
                    session = await self._create_raw_session()
                    self._pool.put_nowait(session)
                    logger.debug("tts.pool.refilled", pool_size=self._pool.qsize())
                except asyncio.CancelledError:
                    return
                except Exception:
                    logger.warning("tts.pool.refill_failed", exc_info=True)
                    await asyncio.sleep(1.0)
                    break

    async def _evict_aged_sessions(self) -> None:
        """Remove sessions older than tts_pool_max_age_s from the pool."""
        kept: list[CosyVoiceDuplexSession] = []
        evicted = 0
        while not self._pool.empty():
            try:
                session = self._pool.get_nowait()
            except asyncio.QueueEmpty:
                break
            if self._is_session_healthy(session):
                kept.append(session)
            else:
                evicted += 1
                self._pool_stale += 1
                with contextlib.suppress(Exception):
                    await session.cancel()
        for s in kept:
            self._pool.put_nowait(s)
        if evicted:
            logger.info("tts.pool.evict_aged", evicted=evicted, remaining=len(kept))
