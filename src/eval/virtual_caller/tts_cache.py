"""TTSFixtureCache — synthesize + cache audio files for Virtual Caller."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import uuid
import wave
from abc import ABC, abstractmethod
from pathlib import Path

import httpx

from src.core.logging import get_logger

logger = get_logger(__name__)

CACHE_DIR = Path("fixtures/tts_cache")


class TTSProvider(ABC):
    """Abstract TTS provider interface.

    Subclasses override `audio_format` / `sample_rate` if they emit something
    other than the default 48kHz mono PCM WAV (which AudioPlayer can play
    directly).
    """

    @property
    def audio_format(self) -> str:
        """File extension / format identifier (e.g. 'wav', 'mp3')."""
        return "wav"

    @property
    def sample_rate(self) -> int:
        return 48000

    @abstractmethod
    async def synthesize(self, text: str, voice: str) -> bytes:
        """Return raw audio bytes in the provider's configured format."""
        ...


class DoubaoTTSProvider(TTSProvider):
    """Volcengine Doubao TTS — V3 SSE API with X-Api-Key auth.

    Endpoint: https://openspeech.bytedance.com/api/v3/tts/unidirectional/sse
    Auth: X-Api-Key header (from 新版豆包语音控制台 → API Key管理)
    Resource: seed-tts-2.0 (大模型语音合成 2.0)
    Response: SSE events with base64-encoded audio chunks.

    Defaults (wav/48000/mono PCM16) match AudioPlayer's strict frame requirements
    so cached files can be played directly without resampling.
    """

    SSE_URL = "https://openspeech.bytedance.com/api/v3/tts/unidirectional/sse"

    def __init__(
        self,
        api_key: str | None = None,
        # volc.service_type.10029 = Doubao 大模型语音合成（复刻音色库），which is
        # what the VOLCENGINE_TTS_ACCESS_TOKEN on this project grants.
        # The previous "seed-tts-2.0" string returns code=55000000 "resource ID
        # is mismatched with speaker related resource". See scripts/bench/_tts_common.py.
        resource_id: str = "volc.service_type.10029",
        audio_format: str = "wav",
        sample_rate: int = 48000,
    ) -> None:
        self._api_key = api_key or os.environ.get("VOLCENGINE_TTS_ACCESS_TOKEN", "")
        self._resource_id = resource_id
        self._audio_format = audio_format
        self._sample_rate = sample_rate
        if not self._api_key:
            raise ValueError("VOLCENGINE_TTS_ACCESS_TOKEN (API Key) required (env or constructor)")

    @property
    def audio_format(self) -> str:
        return self._audio_format

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def synthesize(self, text: str, voice: str) -> bytes:
        """Synthesize text to audio bytes via Doubao TTS V3 SSE API."""
        headers = {
            "X-Api-Key": self._api_key,
            "X-Api-Resource-Id": self._resource_id,
            "X-Api-Connect-Id": uuid.uuid4().hex,
            "Content-Type": "application/json",
        }
        payload = {
            "user": {"uid": "virtual-caller"},
            "req_params": {
                "speaker": voice,
                "text": text,
                "audio_params": {
                    "format": self._audio_format,
                    "sample_rate": self._sample_rate,
                },
            },
        }

        audio_chunks: list[bytes] = []
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(self.SSE_URL, json=payload, headers=headers)
            resp.raise_for_status()

            for line in resp.text.split("\n"):
                if not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                code = data.get("code")
                # 0 = streaming chunk; 20000000 = terminal success ("OK").
                # Anything else is a real error.
                if code is not None and code not in (0, 20000000):
                    raise RuntimeError(f"Doubao TTS failed: code={code}, message={data.get('message', 'unknown')}")
                if code == 20000000:
                    break
                audio_b64 = data.get("data", "")
                if audio_b64:
                    audio_chunks.append(base64.b64decode(audio_b64))

        if not audio_chunks:
            raise RuntimeError("Doubao TTS returned no audio data")

        return b"".join(audio_chunks)


class CosyVoiceTTSProvider(TTSProvider):
    """DashScope CosyVoice provider for Virtual Caller patient fixtures.

    The live agent path uses CosyVoice duplex PCM. Virtual Caller still plays
    WAV files through its existing 48kHz AudioSource, so this provider reuses
    the duplex backend and wraps the resulting PCM bytes in a WAV container.
    """

    def __init__(self, model: str | None = None, voice: str | None = None, sample_rate: int = 48000) -> None:
        from src.core.config import settings

        self._model = model or settings.tts_model
        self._voice = voice or settings.tts_voice
        self._sample_rate = sample_rate

    @property
    def audio_format(self) -> str:
        return "wav"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def synthesize(self, text: str, voice: str) -> bytes:
        from src.backends.tts_cosyvoice_duplex import CosyVoiceDuplexBackend

        resolved_voice = self._resolve_voice(voice)
        backend = CosyVoiceDuplexBackend(model=self._model, voice=resolved_voice, sample_rate=self._sample_rate)
        session = await backend.create_session()

        async def _send_text() -> None:
            await session.send_text(text)
            await session.send_finish()

        send_task = asyncio.create_task(_send_text())
        chunks: list[bytes] = []
        try:
            async for chunk in session.iter_audio():
                chunks.append(chunk.data)
            await send_task
        finally:
            if not send_task.done():
                send_task.cancel()
            await session.cancel()

        if not chunks:
            raise RuntimeError("CosyVoice TTS returned no audio data")
        return self._pcm_to_wav(b"".join(chunks), self._sample_rate)

    def _resolve_voice(self, voice: str) -> str:
        # Legacy ch4 manifests used Doubao voice ids. Keep those rows runnable
        # while routing synthesis through the current production CosyVoice voice.
        if voice and "bigtts" not in voice:
            return voice
        return self._voice

    @staticmethod
    def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)
        return buf.getvalue()


class TTSFixtureCache:
    """TTS synthesis with local audio caching.

    Cache key encodes text + voice + audio_format + sample_rate so a format
    change invalidates the cache rather than returning stale wrong-format bytes.
    File extension on disk matches the provider's audio_format.
    """

    def __init__(self, provider: TTSProvider, cache_dir: Path = CACHE_DIR) -> None:
        self._provider = provider
        self._cache_dir = cache_dir

    def _key_parts(self, text: str, voice: str) -> tuple[str, str]:
        fmt = self._provider.audio_format
        sr = self._provider.sample_rate
        text_hash = hashlib.sha256(f"{text}|{fmt}|{sr}".encode()).hexdigest()[:16]
        return text_hash, fmt

    async def get_wav(self, text: str, voice: str) -> Path:
        """Get cached audio path, synthesizing if needed.

        Name is historical — the returned file is whatever format the provider
        is configured to emit (default: 48kHz mono PCM WAV, which AudioPlayer
        can decode directly).
        """
        text_hash, fmt = self._key_parts(text, voice)
        key = f"{voice}/{text_hash}.{fmt}"
        path = self._cache_dir / key
        if path.exists():
            logger.debug("tts_cache_hit", voice=voice, text_hash=text_hash, format=fmt)
            return path
        logger.info("tts_cache_miss", voice=voice, text=text[:50], format=fmt)
        path.parent.mkdir(parents=True, exist_ok=True)
        audio_bytes = await self._provider.synthesize(text, voice)
        path.write_bytes(audio_bytes)
        return path

    def cache_key(self, text: str, voice: str) -> str:
        text_hash, fmt = self._key_parts(text, voice)
        return f"{voice}/{text_hash}.{fmt}"
