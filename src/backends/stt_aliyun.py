"""Aliyun streaming ASR backend using DashScope Recognition."""

from __future__ import annotations

import asyncio
import time
from typing import Callable

import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class _DashScopeSTTStream:
    """Wraps DashScope Recognition into STTStream protocol."""

    def __init__(self, on_transcript: Callable, session_id: str, turn_id: str, model: str | None = None):
        self.session_id = session_id
        self.turn_id = turn_id
        self._on_transcript = on_transcript
        self._loop = asyncio.get_running_loop()
        self._recognition: Recognition | None = None
        self._started = False
        self._provider_closed = False
        self._model = model or settings.asr_model
        self._start_epoch_ms: float = 0.0
        self._start_mono_ms: float = 0.0

        # Use 16000 to match LiveKit AudioStream output (not settings.asr_sample_rate=8000)
        self._actual_sample_rate = 16000
        callback = _Callback(self)
        # Keep English recognition explicit across supported streaming model families.
        recognition_kwargs: dict = {
            "model": self._model,
            "format": "pcm",
            "sample_rate": self._actual_sample_rate,
            "callback": callback,
            "max_sentence_silence": settings.asr_max_sentence_silence_ms,
        }
        if (
            self._model.startswith(("fun-asr-realtime", "qwen-audio-3."))
            and settings.agent_brain == "solo"
            and settings.solo_language == "en"
        ):
            recognition_kwargs["language_hints"] = ["en"]
        elif self._model.startswith("paraformer"):
            recognition_kwargs["language_hints"] = settings.asr_language_hints
        if settings.asr_vocabulary_id:
            recognition_kwargs["vocabulary_id"] = settings.asr_vocabulary_id
        self._recognition = Recognition(**recognition_kwargs)
        logger.info(
            "asr.stream.starting",
            model=self._model,
            sample_rate=self._actual_sample_rate,
            language_hints=recognition_kwargs.get("language_hints"),
            max_sentence_silence=settings.asr_max_sentence_silence_ms,
        )
        self._recognition.start()
        self._started = True
        self._start_epoch_ms = time.time() * 1000
        self._start_mono_ms = time.monotonic() * 1000

    @property
    def is_healthy(self) -> bool:
        return self._started and self._recognition is not None and not self._provider_closed

    def push_audio(self, audio: bytes) -> None:
        if self._recognition and self._started:
            self._recognition.send_audio_frame(audio)

    @property
    def first_package_timestamp(self) -> float:
        """Milliseconds since epoch for first ASR result package, or -1 if not set."""
        if self._recognition is None:
            return -1.0
        return float(getattr(self._recognition, "_first_package_timestamp", -1))

    @property
    def stop_stream_timestamp(self) -> float:
        """Milliseconds since epoch when stream was stopped, or -1 if not set."""
        if self._recognition is None:
            return -1.0
        return float(getattr(self._recognition, "_stop_stream_timestamp", -1))

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def start_mono_ms(self) -> float:
        """Monotonic start time in ms for cross-processor correlation."""
        return self._start_mono_ms

    @property
    def first_package_latency_ms(self) -> float:
        """Duration in ms from stream start to first ASR result, or -1."""
        fp = self.first_package_timestamp
        return fp - self._start_epoch_ms if fp > 0 and self._start_epoch_ms > 0 else -1.0

    @property
    def total_latency_ms(self) -> float:
        """Duration in ms from stream start to stream stop, or -1."""
        st = self.stop_stream_timestamp
        return st - self._start_epoch_ms if st > 0 and self._start_epoch_ms > 0 else -1.0

    async def flush(self) -> None:
        if self._recognition and self._started:
            await asyncio.get_running_loop().run_in_executor(None, self._recognition.stop)
            self._started = False

    async def cancel(self) -> None:
        if self._recognition and self._started:
            try:
                await asyncio.get_running_loop().run_in_executor(None, self._recognition.stop)
            except Exception:
                pass
            self._started = False

    async def aclose(self) -> None:
        self._recognition = None

    def _handle_event(self, text: str, is_final: bool, confidence: float) -> None:
        """Called from DashScope callback thread."""
        self._loop.call_soon_threadsafe(self._on_transcript, text, is_final, confidence)


class _Callback(RecognitionCallback):
    def __init__(self, stream: _DashScopeSTTStream):
        self._stream = stream

    def on_event(self, result) -> None:
        try:
            # get_sentence() returns a single dict, NOT a list of dicts
            sentence = result.get_sentence()
            if not sentence or not isinstance(sentence, dict):
                return
            text = sentence.get("text", "")
            if not text:
                return
            is_final = bool(sentence.get("sentence_end"))
            confidence = sentence.get("confidence", 0.0)
            logger.info("asr.transcript", text=text, is_final=is_final, confidence=confidence)
            self._stream._handle_event(text, is_final, confidence)
        except Exception as e:
            logger.exception("asr.on_event.error", error=str(e))

    def on_close(self) -> None:
        self._stream._provider_closed = True
        logger.info("asr.stream.closed")

    def on_error(self, result) -> None:
        self._stream._provider_closed = True
        logger.error("asr.stream.error", result=str(result))


class AliyunSTTBackend:
    """STT backend using Aliyun DashScope Paraformer real-time ASR."""

    def __init__(self) -> None:
        # DashScope SDK reads api_key from this global
        dashscope.api_key = settings.dashscope_api_key
        dashscope.base_websocket_api_url = settings.active_dashscope_ws_api_url

    async def create_stream(
        self,
        on_transcript: Callable,
        session_id: str,
        turn_id: str,
        model: str | None = None,
    ) -> _DashScopeSTTStream:
        return _DashScopeSTTStream(on_transcript, session_id, turn_id, model=model)
