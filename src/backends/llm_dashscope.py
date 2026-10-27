"""Qwen LLM backend using DashScope OpenAI-compatible API.

Thin subclass of :class:`OpenAICompatibleStreamingBackend` — the heavy lifting
(streaming, tool-call accumulation, warmup) lives in ``llm_base.py`` so that
the Doubao and Zhipu backends share exactly the same streaming contract.

Historical imports (``StreamEvent``, ``StreamEventType``, ``CompletedToolCall``,
``QwenDashScopeBackend``) are re-exported here so existing consumers continue
to work without modification — see tests/processors/*, tests/integration/*,
src/processors/llm_processor.py.
"""

from __future__ import annotations

from typing import Any

from src.backends.llm_base import (
    CompletedToolCall,
    OpenAICompatibleStreamingBackend,
    StreamEvent,
    StreamEventType,
    StructuredCompletionResult,
)
from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class QwenDashScopeBackend(OpenAICompatibleStreamingBackend):
    """LLM backend streaming tokens from DashScope's OpenAI-compatible API."""

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        enable_thinking: bool | None = None,
    ):
        super().__init__(
            api_key=settings.dashscope_api_key,
            base_url=settings.active_dashscope_compatible_base_url,
            model=model or settings.conversation_model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=30.0,
        )
        self._enable_thinking = enable_thinking

    def _build_extra_body(self) -> dict[str, Any] | None:
        """DashScope Qwen3.6/Qwen3.5 thinking-mode toggle (non-standard OpenAI field)."""
        if self._enable_thinking is None:
            return None
        return {"enable_thinking": self._enable_thinking}


# Public alias used by the LLM factory / tests.
DashScopeLLMBackend = QwenDashScopeBackend


__all__ = [
    "CompletedToolCall",
    "DashScopeLLMBackend",
    "QwenDashScopeBackend",
    "StreamEvent",
    "StreamEventType",
    "StructuredCompletionResult",
]
