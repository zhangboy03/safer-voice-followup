"""Pure energy-threshold VAD.

Tracks per-frame silence/speech duration accumulation that feeds
VADProcessor's end-of-speech gate.
"""

from __future__ import annotations

import numpy as np

from src.core.config import settings
from src.core.logging import get_logger
from src.processors.vad_processor import VADResult

logger = get_logger(__name__)


class EnergyVAD:
    """Pure energy-threshold VAD."""

    def __init__(
        self,
        energy_threshold: float | None = None,
    ) -> None:
        self._energy_threshold = settings.noise_min_threshold if energy_threshold is None else energy_threshold
        self._speech_duration_ms = 0.0
        self._silence_duration_ms = 0.0
        logger.info(
            "vad.energy.initialized",
            energy_threshold=self._energy_threshold,
        )

    def process(self, audio: bytes, sample_rate: int) -> VADResult:
        pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
        energy = float(np.sqrt(np.mean(pcm**2)) / 32768.0) if pcm.size else 0.0
        frame_ms = (pcm.size / sample_rate) * 1000.0 if sample_rate > 0 else 0.0
        is_speech = energy > self._energy_threshold
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
            source="energy",
            speech_prob=None,
            threshold=self._energy_threshold,
        )

    def reset(self) -> None:
        self._speech_duration_ms = 0.0
        self._silence_duration_ms = 0.0
