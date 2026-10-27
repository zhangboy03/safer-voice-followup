"""AudioRingBuffer: 8-second rolling audio buffer at 16kHz 16-bit mono."""

from __future__ import annotations


class AudioRingBuffer:
    """Rolling PCM audio buffer capped at max_duration_s seconds.

    Stores raw PCM bytes. When the buffer exceeds the capacity, the oldest
    audio is discarded (keeping only the most recent max_duration_s seconds).

    Parameters
    ----------
    max_duration_s:
        Maximum duration in seconds to retain. Default 8.0 s.
    sample_rate:
        Audio sample rate in Hz. Default 16000.
    sample_width:
        Bytes per sample. Default 2 (16-bit).
    """

    def __init__(
        self,
        max_duration_s: float = 8.0,
        sample_rate: int = 16000,
        sample_width: int = 2,
    ) -> None:
        self._max_bytes = int(max_duration_s * sample_rate * sample_width)
        self._buffer = bytearray()

    # ── Public API ────────────────────────────────────────────────────────────

    def append(self, audio: bytes) -> None:
        """Append PCM bytes. Trimming is deferred to get_snapshot()."""
        self._buffer.extend(audio)

    def get_snapshot(self) -> bytes:
        """Return an immutable copy of the most recent max_duration_s audio."""
        if len(self._buffer) > self._max_bytes:
            del self._buffer[: len(self._buffer) - self._max_bytes]
        return bytes(self._buffer)

    def reset(self) -> None:
        """Clear all buffered audio."""
        self._buffer.clear()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def max_bytes(self) -> int:
        """Maximum capacity in bytes."""
        return self._max_bytes

    @property
    def current_bytes(self) -> int:
        """Current number of bytes stored."""
        return len(self._buffer)
