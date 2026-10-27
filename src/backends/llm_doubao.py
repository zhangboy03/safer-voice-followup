"""Doubao (火山方舟) LLM backend — OpenAI-compatible endpoint.

Used for the §4.3.1 "same SOP, swap Speech LLM" ablation experiment.
Only the Speech lane is swapped; Judge stays on DashScope qwen — see
``src/backends/llm_factory.py`` + ``src/processors/llm_processor.py``.

Endpoint reference (volcengine, April 2026):
  base_url: https://ark.cn-beijing.volces.com/api/v3
  auth:     Bearer $ARK_API_KEY
  SDK:      OpenAI-compatible (``AsyncOpenAI`` with the above base_url)

Model IDs — per the volcengine console some models must be "接入" before they
can be called (they then expose a stable ``ep-xxx`` endpoint ID).  To support
both regimes the factory reads ``doubao_speech_model`` *and* optionally
``doubao_speech_endpoint_id``; if the endpoint-id is set it takes precedence.
Default model alias tried first: ``doubao-seed-1-6-flash-250615``.
"""

from __future__ import annotations

from src.backends.llm_base import OpenAICompatibleStreamingBackend
from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


# Public constant — volcengine Ark OpenAI-compatible base URL.
# (Also mirrored in settings.ark_base_url so .env can override, e.g. for a
# different region.)
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


class DoubaoLLMBackend(OpenAICompatibleStreamingBackend):
    """Speech LLM backend for Doubao (字节跳动 火山方舟) via OpenAI-compatible SDK."""

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        resolved_model = model or settings.doubao_speech_endpoint_id or settings.doubao_speech_model
        if not resolved_model:
            raise ValueError("DoubaoLLMBackend: model (or DOUBAO_SPEECH_MODEL / DOUBAO_SPEECH_ENDPOINT_ID) must be set")
        resolved_api_key = api_key or settings.ark_api_key
        if not resolved_api_key:
            raise ValueError("DoubaoLLMBackend: ARK_API_KEY is required")
        super().__init__(
            api_key=resolved_api_key,
            base_url=base_url or settings.ark_base_url or ARK_BASE_URL,
            model=resolved_model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=30.0,
        )


__all__ = ["ARK_BASE_URL", "DoubaoLLMBackend"]
