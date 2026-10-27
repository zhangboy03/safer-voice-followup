# src/backends/ecapa_encoder.py
"""SpeechBrain ECAPA-TDNN wrapper (process-level singleton).

Loads VoxCeleb-trained spkrec-ecapa-voxceleb on construction and exposes
a single `encode(pcm_bytes, sample_rate=16000) -> np.ndarray (192,)`
that returns an L2-normalized embedding.

The class is meant to be constructed exactly once per process (via
`get_singleton()`) and shared across PVADBackend instances.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
_singleton: "ECAPAEncoder | None" = None


def _default_loader(source: str):
    """Real loader. Tested via integration, not unit."""
    from speechbrain.inference.speaker import EncoderClassifier

    return EncoderClassifier.from_hparams(source=source, savedir=f"./.cache/{source}")


class ECAPAEncoder:
    """Lazy-loaded singleton wrapper around SpeechBrain ECAPA-TDNN."""

    def __init__(
        self,
        source: str | None = None,
        _model=None,
        _loader: Callable[[str], object] | None = None,
    ) -> None:
        self._source = source or getattr(settings, "vad_pvad_ecapa_source", _DEFAULT_MODEL)
        self._model = _model
        self._load_error: str | None = None
        if self._model is None:
            loader = _loader or _default_loader
            try:
                self._model = loader(self._source)
            except Exception as exc:
                self._load_error = str(exc)
                logger.warning("ecapa.load_failed", error=str(exc), source=self._source)

    @property
    def available(self) -> bool:
        return self._model is not None

    def encode(self, audio: bytes, sample_rate: int) -> np.ndarray:
        if not self._model:
            raise RuntimeError(f"ECAPA encoder unavailable: {self._load_error}")
        if sample_rate != 16000:
            raise ValueError(f"ECAPA expects sample_rate=16000, got {sample_rate}")
        if not audio:
            raise ValueError("ECAPA encode received empty audio")
        import torch

        pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        signal = torch.from_numpy(pcm).unsqueeze(0)
        with torch.no_grad():
            raw = self._model.encode_batch(signal)
        emb = raw.squeeze().cpu().numpy().astype(np.float32)
        if emb.ndim != 1 or emb.shape[0] != 192:
            raise RuntimeError(f"ECAPA returned unexpected shape {emb.shape}")
        norm = float(np.linalg.norm(emb))
        if norm < 1e-6:
            raise ValueError("ECAPA produced degenerate (zero-magnitude) embedding")
        return (emb / norm).astype(np.float32)


def get_singleton() -> ECAPAEncoder:
    # Single-caller contract: invoked once from server lifespan during startup.
    # Not thread-safe — concurrent first-callers could each load a ~400MB model.
    global _singleton
    if _singleton is None:
        _singleton = ECAPAEncoder()
    return _singleton


def reset_singleton_for_tests() -> None:
    global _singleton
    _singleton = None
