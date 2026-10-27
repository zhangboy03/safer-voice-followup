"""Audio fixture prewarm helpers for Study 1 live evaluation."""

from __future__ import annotations

import asyncio
import wave
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Protocol

import numpy as np

_SAMPLE_WIDTH_BYTES = 2


class _TTSCache(Protocol):
    async def get_wav(self, text: str, voice: str): ...


def write_wav_pcm16(path: Path, pcm: bytes, sample_rate: int) -> None:
    """Write mono PCM16 audio to a WAV file, creating parent dirs as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(_SAMPLE_WIDTH_BYTES)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def build_background_noise_wav(
    path: Path,
    *,
    sample_rate: int = 48000,
    duration_ms: int = 1000,
    seed: int = 0,
    amplitude: int = 1200,
) -> Path:
    """Generate a deterministic background-noise WAV for false-barge-in probes.

    This is intentionally a simple, seeded broadband noise burst — enough to drive the
    user channel with non-speech energy and check that it does not falsely advance a
    business turn. SNR-controlled mixed-noise fixtures are a pre-Main upgrade and live in
    ``src/eval/virtual_caller/noise_mixing.py``; this MVP/pilot probe does not need them.
    """

    rng = np.random.default_rng(seed)
    sample_count = sample_rate * duration_ms // 1000
    noise = rng.integers(-amplitude, amplitude, size=sample_count, dtype=np.int16)
    write_wav_pcm16(path, noise.tobytes(), sample_rate)
    return path


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "throttling.ratequota" in message or "rate limit" in message


async def prewarm_tts_cache(
    tts_cache: _TTSCache,
    texts: Iterable[str],
    voice: str,
    *,
    pause_s: float = 1.0,
    retry_delays_s: tuple[float, ...] = (3.0, 8.0, 15.0),
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Pre-generate unique patient audio fixtures without timing the cases."""

    seen: set[str] = set()
    for text in texts:
        if text in seen:
            continue
        seen.add(text)

        attempts = len(retry_delays_s) + 1
        for attempt in range(attempts):
            try:
                await tts_cache.get_wav(text, voice)
                if pause_s > 0:
                    await sleep(pause_s)
                break
            except Exception as exc:
                if attempt >= len(retry_delays_s) or not _is_rate_limit_error(exc):
                    raise
                await sleep(retry_delays_s[attempt])
