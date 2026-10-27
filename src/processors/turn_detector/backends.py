"""TurnDetectorBackend protocol and concrete backend implementations.

Backends
--------
TurnDetectorBackend  — structural Protocol (duck-typing)
SmartTurnV3Backend   — ONNX inference on pipecat-ai/smart-turn-v3, fallback-safe
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


# ── Shared data types ─────────────────────────────────────────────────────────


@dataclass
class TurnContext:
    """All inputs a backend may need to predict whether a user turn is complete."""

    audio_buffer: bytes
    transcript: str
    conversation_history: list[dict]
    speech_duration_ms: float
    silence_duration_ms: float


@dataclass
class TurnPrediction:
    """Output from a TurnDetectorBackend."""

    is_complete: bool
    confidence: float
    source: str  # e.g. "smart_turn_v3", "smart_turn_v3_fallback"


# ── Protocol ──────────────────────────────────────────────────────────────────


class TurnDetectorBackend(Protocol):
    """Structural protocol for all turn-detection backends."""

    async def predict(self, context: TurnContext) -> TurnPrediction:
        """Return a TurnPrediction given the current TurnContext."""
        ...


# ── SmartTurnV3Backend ────────────────────────────────────────────────────────

_SMART_TURN_REPO = "pipecat-ai/smart-turn-v3"
_SMART_TURN_FILENAME = "model.onnx"
_SMART_TURN_SAMPLE_RATE = 16000
_SMART_TURN_SAMPLE_WIDTH = 2  # bytes (16-bit)


class SmartTurnV3Backend:
    """ONNX-based turn detector using pipecat-ai/smart-turn-v3.

    Attempts to load the model at construction time in a background thread.
    If the model is unavailable (missing file, onnxruntime not installed, etc.)
    the backend falls back to a safe default prediction so that the pipeline
    can run without the model present.

    Parameters
    ----------
    model_path:
        Optional path to a local ``model.onnx`` file.  When *None* the backend
        attempts to download from HuggingFace Hub.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model = None  # onnxruntime.InferenceSession, loaded lazily
        self._feature_extractor = None
        self._model_path = model_path
        self._thread_pool = ThreadPoolExecutor(max_workers=1)
        self._loading_done = False
        # Pre-import on main thread — transformers lazy loading can fail in worker threads
        try:
            from transformers import WhisperFeatureExtractor  # type: ignore[import]

            self._whisper_fe_cls = WhisperFeatureExtractor
        except ImportError:
            self._whisper_fe_cls = None

        # Attempt non-blocking load — failure is logged, not raised.
        try:
            future = self._thread_pool.submit(self._load_model_sync)
            future.add_done_callback(self._on_load_done)
        except Exception as exc:
            logger.warning("[SmartTurnV3] Could not schedule model load: %s", exc)
            self._loading_done = True

    # ── Model loading ─────────────────────────────────────────────────────────

    def _on_load_done(self, future) -> None:
        self._loading_done = True
        exc = future.exception()
        if exc:
            logger.warning("[SmartTurnV3] Model unavailable, using fallback: %s", exc)

    def _load_model_sync(self) -> None:
        """Runs in thread pool. Sets self._model on success."""
        import onnxruntime as ort  # type: ignore[import]

        # Use pre-imported class (thread-safe); fall back to direct import
        WhisperFeatureExtractor = self._whisper_fe_cls
        if WhisperFeatureExtractor is None:
            from transformers import WhisperFeatureExtractor  # type: ignore[import]

        model_path = self._model_path
        if model_path is None:
            from huggingface_hub import hf_hub_download  # type: ignore[import]

            model_path = hf_hub_download(_SMART_TURN_REPO, _SMART_TURN_FILENAME)

        cpu_count = os.cpu_count() or 2
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = min(cpu_count // 2, 4)
        opts.inter_op_num_threads = 1

        self._model = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self._feature_extractor = WhisperFeatureExtractor(
            chunk_length=8,
            sampling_rate=_SMART_TURN_SAMPLE_RATE,
        )
        logger.info("[SmartTurnV3] Model loaded from %s", model_path)

    # ── Inference ─────────────────────────────────────────────────────────────

    async def predict(self, context: TurnContext) -> TurnPrediction:
        """Return a TurnPrediction.  Falls back gracefully if model not ready."""
        if self._model is None:
            return TurnPrediction(
                is_complete=True,
                confidence=0.5,
                source="smart_turn_v3_fallback",
            )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._thread_pool, self._infer_sync, context)

    def _infer_sync(self, context: TurnContext) -> TurnPrediction:
        """Synchronous ONNX inference (executed in thread pool)."""
        try:
            # Convert raw PCM bytes to float32 numpy array in [-1, 1]
            audio_np = np.frombuffer(context.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0

            # WhisperFeatureExtractor: produces log-mel spectrogram as input_features
            inputs = self._feature_extractor(
                audio_np,
                sampling_rate=_SMART_TURN_SAMPLE_RATE,
                return_tensors="np",
            )
            input_features = inputs["input_features"].astype(np.float32)

            outputs = self._model.run(None, {"input_features": input_features})
            # Expect logits or probability in outputs[0]; last value is completion prob
            raw = float(outputs[0].flatten()[-1])
            # Interpret as probability if in [0,1], else apply sigmoid
            if 0.0 <= raw <= 1.0:
                prob = raw
            else:
                prob = float(1.0 / (1.0 + np.exp(-raw)))

            return TurnPrediction(
                is_complete=prob >= 0.5,
                confidence=prob,
                source="smart_turn_v3",
            )
        except Exception as exc:
            logger.warning("[SmartTurnV3] Inference error, using fallback: %s", exc)
            return TurnPrediction(is_complete=True, confidence=0.5, source="smart_turn_v3_fallback")

    def shutdown(self) -> None:
        self._thread_pool.shutdown(wait=False)
