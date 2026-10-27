"""TTS cache: pre-synthesize fixed phrases at startup for zero-latency playback."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CachedAudio:
    chunks: list[bytes]
    sample_rate: int
    text: str


class TTSCache:
    def __init__(self) -> None:
        self._cache: dict[str, CachedAudio] = {}

    async def preload(self, phrases: dict[str, str], backend) -> None:
        """Pre-synthesize fixed phrases at startup.

        Parameters
        ----------
        phrases:
            Mapping of cache key -> text to synthesize.
        backend:
            CosyVoiceDuplexBackend (or any duck-typed equivalent with
            create_session() -> session that supports send_text, send_finish,
            start_recv, and iter_audio).
        """
        for key, text in phrases.items():
            session = await backend.create_session()
            session.start_recv()
            await session.send_text(text)
            await session.send_finish()
            chunks: list[bytes] = []
            sample_rate = 16000
            async for chunk in session.iter_audio():
                chunks.append(chunk.data)
                sample_rate = chunk.sample_rate
            self._cache[key] = CachedAudio(chunks=chunks, sample_rate=sample_rate, text=text)

    def get(self, key: str) -> CachedAudio | None:
        return self._cache.get(key)
