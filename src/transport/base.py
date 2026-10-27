"""Abstract base classes for pipeline transports."""

from abc import ABC

from src.pipeline.processor import FrameProcessor


class InputTransport(FrameProcessor, ABC):
    """Input transport base. Not TTS-related."""

    def _is_tts_related(self) -> bool:
        return False


class OutputTransport(FrameProcessor, ABC):
    """Output transport base. TTS-related (flushed on TTS_ONLY interruption)."""

    def _is_tts_related(self) -> bool:
        return True
