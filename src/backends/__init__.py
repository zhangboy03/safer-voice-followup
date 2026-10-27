"""Backend implementations for VAD, STT, LLM, and TTS providers."""

from src.backends.ecapa_encoder import ECAPAEncoder
from src.backends.ecapa_encoder import get_singleton as get_ecapa_singleton
from src.backends.llm_base import LLMBackend
from src.backends.llm_dashscope import DashScopeLLMBackend, QwenDashScopeBackend
from src.backends.llm_doubao import DoubaoLLMBackend
from src.backends.llm_factory import get_speech_llm_backend
from src.backends.llm_zhipu import ZhipuLLMBackend
from src.backends.stt_aliyun import AliyunSTTBackend
from src.backends.tts_cosyvoice_duplex import CosyVoiceDuplexBackend
from src.backends.vad_energy import EnergyVAD
from src.backends.vad_pvad import PVADBackend, PVADState
from src.backends.vad_silero import SileroHybridVAD

__all__ = [
    "AliyunSTTBackend",
    "CosyVoiceDuplexBackend",
    "DashScopeLLMBackend",
    "DoubaoLLMBackend",
    "ECAPAEncoder",
    "EnergyVAD",
    "LLMBackend",
    "PVADBackend",
    "PVADState",
    "QwenDashScopeBackend",
    "SileroHybridVAD",
    "ZhipuLLMBackend",
    "get_ecapa_singleton",
    "get_speech_llm_backend",
]
