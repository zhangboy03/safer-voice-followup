"""AudioPlayer — pushes 48kHz/20ms PCM16 frames to LiveKit AudioSource."""

from __future__ import annotations

import asyncio
import struct
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from src.core.logging import get_logger
from src.pipeline.events import DriftReportEvent, EventBus

logger = get_logger(__name__)

SAMPLE_RATE = 48000
FRAME_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000  # 960
BYTES_PER_SAMPLE = 2  # int16
FRAME_BYTES = SAMPLES_PER_FRAME * BYTES_PER_SAMPLE  # 1920
DRIFT_CHECK_INTERVAL = 50  # frames
DRIFT_WARN_MS = 50.0
DRIFT_EVENT_MS = 200.0


@dataclass
class PlaybackHandle:
    started: asyncio.Event
    done: asyncio.Event


def load_wav_frames(path: Path) -> list[bytes]:
    """Load a wav file and return list of 960-sample PCM16 frames.

    Assumes input is 48kHz mono PCM16. If not, raises ValueError.
    """
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(f"Expected mono, got {wf.getnchannels()} channels: {path}")
        if wf.getframerate() != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE}Hz, got {wf.getframerate()}Hz: {path}")
        if wf.getsampwidth() != BYTES_PER_SAMPLE:
            raise ValueError(f"Expected {BYTES_PER_SAMPLE} bytes/sample, got {wf.getsampwidth()}: {path}")
        frames = []
        while True:
            chunk = wf.readframes(SAMPLES_PER_FRAME)
            if not chunk:
                break
            if len(chunk) < FRAME_BYTES:
                chunk += b"\x00" * (FRAME_BYTES - len(chunk))
            frames.append(chunk)
    return frames


def mix_frames(a: bytes, b: bytes) -> bytes:
    """Mix two PCM16 frames by adding samples and clipping."""
    samples_a = struct.unpack(f"<{SAMPLES_PER_FRAME}h", a)
    samples_b = struct.unpack(f"<{SAMPLES_PER_FRAME}h", b)
    mixed = [max(-32768, min(32767, sa + sb)) for sa, sb in zip(samples_a, samples_b)]
    return struct.pack(f"<{SAMPLES_PER_FRAME}h", *mixed)


def _split_into_frames(audio: bytes) -> list[bytes]:
    """Split a 20ms-aligned PCM16 blob into FRAME_BYTES chunks. Raises on misalignment."""
    if not audio or len(audio) % FRAME_BYTES != 0:
        raise ValueError(
            f"inject audio must be a multiple of {FRAME_BYTES} bytes (20ms @ 48kHz mono PCM16), got {len(audio)}"
        )
    return [audio[i : i + FRAME_BYTES] for i in range(0, len(audio), FRAME_BYTES)]


class AudioPlayer:
    """Pushes 48kHz/20ms PCM16 frames to a LiveKit AudioSource."""

    def __init__(self, bus: EventBus | None = None) -> None:
        self._bus = bus
        self._source = None  # rtc.AudioSource, set via bind()
        self._inject_queue: asyncio.Queue[tuple[bytes, Literal["overlap", "replace"]]] = asyncio.Queue()
        self._playback_start_mono_ms: float = 0.0
        self._frame_count: int = 0
        self._on_first_frame: Callable[[], None] | None = None
        self._first_frame_fired: bool = False
        # Per-playback pending overlap frames (queued across iterations).
        self._overlap_pending: deque[bytes] = deque()

    def bind(self, source) -> None:
        """Bind to an rtc.AudioSource. Must be called before play_wav."""
        self._source = source

    async def play_wav(
        self,
        wav_path: str | Path,
        on_first_frame: Callable[[], None] | None = None,
    ) -> PlaybackHandle:
        """Enqueue a wav file for playback. Returns handle with started/done events."""
        frames = load_wav_frames(Path(wav_path))
        handle = PlaybackHandle(started=asyncio.Event(), done=asyncio.Event())
        self._on_first_frame = on_first_frame
        self._first_frame_fired = False
        self._frame_count = 0
        self._playback_start_mono_ms = time.monotonic() * 1000
        self._overlap_pending.clear()

        if self._source is None:
            raise RuntimeError("AudioPlayer not bound to AudioSource — call bind() first")

        # Deque so a `replace` inject can flush remaining frames and prepend new audio.
        play_queue: deque[bytes] = deque(frames)

        while play_queue:
            frame_data = play_queue.popleft()

            # Drain inject queue before playing this frame.
            while not self._inject_queue.empty():
                inject_data, inject_mode = self._inject_queue.get_nowait()
                try:
                    inject_frames = _split_into_frames(inject_data)
                except ValueError as err:
                    logger.warning("inject_audio_invalid_size", mode=inject_mode, error=str(err))
                    continue
                if not inject_frames:
                    continue
                if inject_mode == "replace":
                    # Flush remaining original frames; prepend injected frames;
                    # current frame becomes the first injected frame.
                    play_queue.clear()
                    play_queue.extend(inject_frames[1:])
                    frame_data = inject_frames[0]
                    # A `replace` supersedes any queued overlaps.
                    self._overlap_pending.clear()
                    break
                if inject_mode == "overlap":
                    # Mix first inject frame into current; queue the rest to mix
                    # into the next N frames.
                    frame_data = mix_frames(frame_data, inject_frames[0])
                    self._overlap_pending.extend(inject_frames[1:])

            # Apply any in-flight overlap carried over from prior iterations.
            if self._overlap_pending:
                frame_data = mix_frames(frame_data, self._overlap_pending.popleft())

            if not self._first_frame_fired:
                handle.started.set()
                self._first_frame_fired = True
                if self._on_first_frame:
                    self._on_first_frame()

            try:
                from livekit.rtc import AudioFrame

                audio_frame = AudioFrame(
                    data=frame_data,
                    sample_rate=SAMPLE_RATE,
                    num_channels=1,
                    samples_per_channel=SAMPLES_PER_FRAME,
                )
                await self._source.capture_frame(audio_frame)
            except ImportError:
                # In test mode without livekit — just sleep to simulate pacing
                await asyncio.sleep(FRAME_MS / 1000)

            self._frame_count += 1
            self._check_drift()

        handle.done.set()
        return handle

    def schedule_inject(self, audio_data: bytes, mode: Literal["overlap", "replace"]) -> None:
        """Schedule audio injection during current playback.

        audio_data must be 48kHz mono PCM16 and a multiple of FRAME_BYTES (20ms).
        Raises ValueError on misalignment — caller is responsible for encoding.
        """
        # Validate up-front so the caller sees the error, not the playback loop.
        _split_into_frames(audio_data)
        self._inject_queue.put_nowait((audio_data, mode))

    def _check_drift(self) -> None:
        if self._frame_count % DRIFT_CHECK_INTERVAL != 0:
            return
        expected_ms = self._frame_count * FRAME_MS
        actual_ms = time.monotonic() * 1000 - self._playback_start_mono_ms
        delta = abs(actual_ms - expected_ms)
        if delta > DRIFT_WARN_MS:
            logger.warning(
                "audio_drift", expected_ms=expected_ms, actual_ms=round(actual_ms, 1), delta_ms=round(delta, 1)
            )
        if delta > DRIFT_EVENT_MS and self._bus:
            self._bus.emit(
                DriftReportEvent(
                    expected_mono_ms=self._playback_start_mono_ms + expected_ms,
                    actual_mono_ms=time.monotonic() * 1000,
                    delta_ms=round(delta, 1),
                )
            )
