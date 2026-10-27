"""Zhipu GLM (智谱 bigmodel) LLM backend — OpenAI-compatible endpoint.

Used for the §4.3.1 "same SOP, swap Speech LLM" ablation experiment (peer to
``llm_doubao.py``).  Only the Speech lane is swapped; Judge stays on DashScope
qwen — see ``src/backends/llm_factory.py`` + ``src/processors/llm_processor.py``.

Endpoint reference (bigmodel.cn, April 2026):
  base_url: https://open.bigmodel.cn/api/paas/v4/
  auth:     Bearer $ZHIPU_API_KEY
  SDK:      OpenAI-compatible (``AsyncOpenAI`` with the above base_url)
  docs:     https://docs.bigmodel.cn/cn/api/llm/llm-chat

Default Speech model alias: ``glm-4.5-airx`` (latency-optimized flash-tier).
Settle this via smoke test — if 404, fall back to ``glm-4-flashx`` or
``glm-4.6-air`` per docs.  Configurable through ``ZHIPU_SPEECH_MODEL`` env.
"""

from __future__ import annotations

from src.backends.llm_base import OpenAICompatibleStreamingBackend
from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


# Public constant — bigmodel OpenAI-compatible base URL.
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"


class ZhipuLLMBackend(OpenAICompatibleStreamingBackend):
    """Speech LLM backend for Zhipu GLM via OpenAI-compatible SDK."""

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        resolved_model = model or settings.zhipu_speech_model
        if not resolved_model:
            raise ValueError("ZhipuLLMBackend: model (or ZHIPU_SPEECH_MODEL) must be set")
        resolved_api_key = api_key or settings.zhipu_api_key
        if not resolved_api_key:
            raise ValueError("ZhipuLLMBackend: ZHIPU_API_KEY is required")
        super().__init__(
            api_key=resolved_api_key,
            base_url=base_url or ZHIPU_BASE_URL,
            model=resolved_model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=30.0,
        )


__all__ = ["ZHIPU_BASE_URL", "ZhipuLLMBackend"]
