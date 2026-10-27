"""PCM16 helpers for Study 1 mixed-noise audio fixtures."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

_INT16_MIN = -32768
_INT16_MAX = 32767


@dataclass(frozen=True, slots=True)
class SNRMix:
    audio: bytes
    snr_db: float
    signal_rms: float
    input_interference_rms: float
    target_interference_rms: float
    interference_gain: float


def _validate_pcm16(audio: bytes, *, name: str) -> None:
    if len(audio) % 2 != 0:
        raise ValueError(f"{name} must be 16-bit PCM with an even byte length")


def _unpack_pcm16(audio: bytes, *, name: str) -> tuple[int, ...]:
    _validate_pcm16(audio, name=name)
    return struct.unpack(f"<{len(audio) // 2}h", audio)


def _clip_int16(value: float) -> int:
    return max(_INT16_MIN, min(_INT16_MAX, int(round(value))))


def pcm16_rms(audio: bytes) -> float:
    """Return RMS amplitude for little-endian mono PCM16 bytes."""

    samples = _unpack_pcm16(audio, name="audio")
    if not samples:
        return 0.0
    return math.sqrt(sum(float(sample) * float(sample) for sample in samples) / len(samples))


def scale_pcm16(audio: bytes, gain: float) -> bytes:
    """Scale little-endian PCM16 samples by gain and clip to int16."""

    samples = _unpack_pcm16(audio, name="audio")
    scaled = [_clip_int16(sample * gain) for sample in samples]
    return struct.pack(f"<{len(scaled)}h", *scaled)


def fit_interference_length(interference: bytes, target_bytes: int) -> bytes:
    """Repeat/truncate interference PCM so it matches a target byte length."""

    if target_bytes < 0 or target_bytes % 2 != 0:
        raise ValueError("target_bytes must be a non-negative even byte length")
    _validate_pcm16(interference, name="interference")
    if target_bytes == 0:
        return b""
    if not interference:
        raise ValueError("interference must be non-empty when target_bytes is positive")

    repeats = math.ceil(target_bytes / len(interference))
    return (interference * repeats)[:target_bytes]


def mix_pcm16_at_snr(signal: bytes, interference: bytes, *, snr_db: float) -> SNRMix:
    """Mix PCM16 signal and interference after scaling interference to SNR.

    ``snr_db`` is signal RMS over interference RMS in dB. The interference is
    repeated or truncated to the signal length before gain is applied.
    """

    _validate_pcm16(signal, name="signal")
    fitted_interference = fit_interference_length(interference, len(signal))

    signal_rms = pcm16_rms(signal)
    input_interference_rms = pcm16_rms(fitted_interference)
    if signal_rms <= 0:
        raise ValueError("signal must be a non-silent signal to define SNR")
    if input_interference_rms <= 0:
        raise ValueError("interference must be non-silent to define SNR")

    target_interference_rms = signal_rms / (10 ** (snr_db / 20.0))
    interference_gain = target_interference_rms / input_interference_rms
    scaled_interference = scale_pcm16(fitted_interference, interference_gain)

    signal_samples = _unpack_pcm16(signal, name="signal")
    interference_samples = _unpack_pcm16(scaled_interference, name="interference")
    mixed = [
        _clip_int16(signal_sample + interference_sample)
        for signal_sample, interference_sample in zip(signal_samples, interference_samples)
    ]

    return SNRMix(
        audio=struct.pack(f"<{len(mixed)}h", *mixed),
        snr_db=snr_db,
        signal_rms=signal_rms,
        input_interference_rms=input_interference_rms,
        target_interference_rms=target_interference_rms,
        interference_gain=interference_gain,
    )
