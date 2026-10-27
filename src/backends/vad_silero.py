"""Silero VAD backend with energy-based fallback."""

from __future__ import annotations

import math

import numpy as np

from src.core.config import settings
from src.core.logging import get_logger
from src.processors.vad_processor import VADResult

logger = get_logger(__name__)


class SileroHybridVAD:
    """
    VAD with Silero as the primary speech detector.

    Energy is used only as a temporary bootstrap before enough samples are
    buffered for Silero, and as a fallback when Silero is unavailable or
    throws at runtime.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        speech_threshold: float | None = None,
        energy_threshold: float | None = None,
        shadow: bool = False,
    ):
        self._sample_rate = sample_rate
        self._speech_threshold = settings.silero_speech_threshold if speech_threshold is None else speech_threshold
        self._energy_threshold = settings.noise_min_threshold if energy_threshold is None else energy_threshold
        self._speech_hysteresis = max(settings.silero_speech_hysteresis, 0.0)
        self._smoothing_alpha = min(max(settings.vad_smoothing_alpha, 0.0), 1.0)
        self._shadow = shadow
        self._model = None
        self._runtime_fallback_count = 0
        self._speech_duration_ms = 0.0
        self._silence_duration_ms = 0.0
        self._smoothed_speech_prob: float | None = None
        self._silero_in_speech = False
        self._min_chunk_samples = max(int(math.ceil(sample_rate / 31.25)), 1)
        self._window_samples = max(self._min_chunk_samples, 512 if sample_rate >= 16000 else 256)
        self._sample_buffer = np.zeros(0, dtype=np.float32)
        if not shadow:
            self._load_model()
        else:
            logger.info("vad.backend.shadow_mode", reason="silero_disabled_energy_only")
        logger.info(
            "vad.backend.initialized",
            backend="energy_only" if shadow else "silero_hybrid",
            model_loaded=self._model is not None,
            shadow=shadow,
            speech_threshold=self._speech_threshold,
            energy_threshold=self._energy_threshold,
            speech_hysteresis=self._speech_hysteresis,
            smoothing_alpha=self._smoothing_alpha,
            min_chunk_samples=self._min_chunk_samples,
            window_samples=self._window_samples,
        )

    def _load_model(self) -> None:
        try:
            import torch
            from silero_vad import load_silero_vad

            self._model = load_silero_vad()
            self._model.eval()
            self._torch = torch
        except ImportError:
            self._model = None  # Falls back to energy-only
            logger.warning("vad.backend.energy_only", reason="silero_import_failed")

    def process(self, audio: bytes, sample_rate: int) -> VADResult:
        pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
        energy = float(np.sqrt(np.mean(pcm**2)) / 32768.0)
        frame_ms = (pcm.size / sample_rate) * 1000.0 if sample_rate > 0 else 0.0
        normalized = pcm / 32768.0 if pcm.size else pcm
        if normalized.size:
            self._sample_buffer = np.concatenate((self._sample_buffer, normalized))
            max_buffer = self._window_samples * 4
            if self._sample_buffer.size > max_buffer:
                self._sample_buffer = self._sample_buffer[-max_buffer:]

        energy_is_speech = energy > self._energy_threshold
        silero_is_speech = False
        silero_ready = False
        is_speech = False
        speech_prob: float | None = None
        source = "energy"
        threshold = self._energy_threshold
        if self._model is not None and self._sample_buffer.size >= self._min_chunk_samples:
            try:
                model_input = self._sample_buffer[-self._window_samples :]
                tensor = self._torch.from_numpy(model_input)
                with self._torch.no_grad():
                    speech_prob = float(self._model(tensor, sample_rate).item())
                if self._smoothed_speech_prob is None:
                    self._smoothed_speech_prob = speech_prob
                else:
                    alpha = self._smoothing_alpha
                    self._smoothed_speech_prob = alpha * speech_prob + (1.0 - alpha) * self._smoothed_speech_prob

                continue_threshold = max(self._speech_threshold - self._speech_hysteresis, 0.0)
                threshold = continue_threshold if self._silero_in_speech else self._speech_threshold
                silero_is_speech = (self._smoothed_speech_prob or 0.0) > threshold
                self._silero_in_speech = silero_is_speech
                silero_ready = True
            except Exception as exc:
                self._runtime_fallback_count += 1
                if settings.polaris_asr_audit_log and (
                    self._runtime_fallback_count <= 5 or self._runtime_fallback_count % 100 == 0
                ):
                    logger.warning(
                        "vad.backend.fallback",
                        error=str(exc),
                        fallback_count=self._runtime_fallback_count,
                        energy_threshold=self._energy_threshold,
                    )
                is_speech = energy_is_speech
                source = "energy_fallback"
                self._silero_in_speech = False
                self._smoothed_speech_prob = None
        else:
            is_speech = energy_is_speech

        if silero_ready:
            is_speech = silero_is_speech
            source = "silero"
        else:
            self._silero_in_speech = False
            self._smoothed_speech_prob = None

        if is_speech:
            self._speech_duration_ms += frame_ms
            self._silence_duration_ms = 0.0
        else:
            self._silence_duration_ms += frame_ms
            self._speech_duration_ms = 0.0

        return VADResult(
            is_speech=is_speech,
            energy=energy,
            speech_duration_ms=self._speech_duration_ms,
            silence_duration_ms=self._silence_duration_ms,
            source=source,
            speech_prob=speech_prob,
            threshold=threshold,
        )

    def reset(self) -> None:
        self._speech_duration_ms = 0.0
        self._silence_duration_ms = 0.0
        self._smoothed_speech_prob = None
        self._silero_in_speech = False
        self._sample_buffer = np.zeros(0, dtype=np.float32)
        if self._model is not None:
            self._model.reset_states()
