"""Speech LLM backend factory.

Reads :attr:`Settings.speech_llm_provider` and returns the configured Speech
lane backend.  ``SPEECH_LLM_PROVIDER`` defaults to ``qwen`` so the production
path is unchanged unless the user (or an eval harness) explicitly opts into
``doubao`` or ``glm``.

Judge LLM is *not* routed through this factory — ``server/debug_full_pipeline.py``
continues to construct ``QwenDashScopeBackend`` directly for Judge, ensuring the
§4.3.1 "same SOP, swap Speech only" control variable holds.
"""

from __future__ import annotations

from src.backends.llm_base import LLMBackend
from src.backends.llm_dashscope import QwenDashScopeBackend
from src.backends.llm_doubao import DoubaoLLMBackend
from src.backends.llm_zhipu import ZhipuLLMBackend
from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


def get_speech_llm_backend(
    *,
    provider: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> LLMBackend:
    """Construct the Speech lane LLM backend for the active provider.

    Args:
        provider: Override ``settings.speech_llm_provider`` (primarily for tests
            / eval runners).  Accepted values: ``qwen``, ``doubao``, ``glm``.
        max_tokens: Forwarded to the concrete backend (None → backend default).
        temperature: Forwarded to the concrete backend (None → backend default).

    Returns:
        A backend implementing :class:`LLMBackend`.

    Raises:
        ValueError: If the provider string is not recognised.
    """
    chosen = (provider or settings.speech_llm_provider).strip().lower()

    if chosen == "qwen":
        logger.info("speech_llm.backend", provider="qwen", model=settings.conversation_model)
        return QwenDashScopeBackend(
            model=settings.conversation_model,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=settings.conversation_enable_thinking,
        )

    if chosen == "doubao":
        model_id = settings.doubao_speech_endpoint_id or settings.doubao_speech_model
        logger.info("speech_llm.backend", provider="doubao", model=model_id)
        return DoubaoLLMBackend(
            model=None,  # backend resolves from settings
            max_tokens=max_tokens,
            temperature=temperature,
        )

    if chosen == "glm":
        logger.info("speech_llm.backend", provider="glm", model=settings.zhipu_speech_model)
        return ZhipuLLMBackend(
            model=None,  # backend resolves from settings
            max_tokens=max_tokens,
            temperature=temperature,
        )

    raise ValueError(
        f"Unknown SPEECH_LLM_PROVIDER={chosen!r}. Valid values: 'qwen', 'doubao', 'glm'.",
    )


__all__ = ["get_speech_llm_backend"]
