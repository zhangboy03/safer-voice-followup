"""LLMBackend Protocol + shared OpenAI-compatible streaming implementation.

The production Speech/Judge LLM lanes (see src/processors/llm_processor.py) only
interact with a backend through two methods:

    async def warmup(self) -> None
    def stream_with_tools(...) -> AsyncIterator[StreamEvent]

This module formalises that contract as a ``typing.Protocol`` (:class:`LLMBackend`)
so that the ``SPEECH_LLM_PROVIDER`` env switch can swap Qwen / Doubao / GLM at
runtime without LLMProcessor caring about provider-specific details.

A concrete helper class :class:`OpenAICompatibleStreamingBackend` captures the
shared OpenAI-compatible streaming + tool-call accumulation logic that all three
providers implement.  Each provider subclass (see ``llm_dashscope.py``,
``llm_doubao.py``, ``llm_zhipu.py``) only needs to supply base_url + api_key +
model and, if relevant, a provider-specific ``extra_body`` payload.

Design parity constraint: the *behavior* of the Qwen path when
``SPEECH_LLM_PROVIDER=qwen`` (the default) must remain bit-for-bit identical to
the pre-refactor ``QwenDashScopeBackend``.  That is why the shared base encodes
exactly the same ``max_tokens=500``, ``temperature=0.7`` defaults and the same
tool-delta accumulator format.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import httpx
from openai import AsyncOpenAI

from src.core.logging import get_logger

logger = get_logger(__name__)


# ---------- StreamEvent protocol ----------


class StreamEventType(Enum):
    TEXT = "text"
    TOOL_CALL = "tool_call"
    DONE = "done"


@dataclass
class CompletedToolCall:
    """A complete tool call after accumulating streaming deltas."""

    id: str
    name: str
    arguments: dict


@dataclass
class StreamEvent:
    """Event yielded by stream_with_tools()."""

    type: StreamEventType
    text: str | None = None
    tool_call: CompletedToolCall | None = None


@dataclass
class StructuredCompletionResult:
    raw_text: str
    parsed_json: Any | None
    usage: dict[str, Any] | None = None


@dataclass
class _ToolAccumulator:
    """Internal: accumulates tool call deltas across streaming chunks."""

    id: str = ""
    name: str = ""
    arguments_buffer: str = ""


# ---------- Protocol ----------


@runtime_checkable
class LLMBackend(Protocol):
    """Runtime-checkable protocol for Speech / Judge LLM backends.

    Anything exposing ``stream_with_tools`` with the StreamEvent signature can
    be plugged in.  ``warmup`` is optional — the pipeline checks for
    ``hasattr(b, 'warmup')`` before calling.
    """

    def stream_with_tools(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        session_id: str = "",
        turn_id: str = "",
        model: str | None = None,
    ) -> AsyncIterator[StreamEvent]: ...

    async def complete_structured(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        response_format: dict,
        max_tokens: int | None = None,
        temperature: float | None = None,
        session_id: str = "",
        turn_id: str = "",
    ) -> StructuredCompletionResult: ...


# ---------- Shared OpenAI-compatible implementation ----------


class OpenAICompatibleStreamingBackend:
    """Shared base for OpenAI-compatible streaming LLM backends.

    Concrete providers (DashScope / Doubao / Zhipu) subclass this and pass
    provider-specific ``api_key`` + ``base_url`` + ``model`` through
    ``__init__``.  Overriding ``_build_extra_body`` lets a provider inject
    non-standard kwargs (e.g. ``enable_thinking`` for DashScope Qwen3).

    All providers share the same streaming contract that ``LLMProcessor``
    depends on (``stream_with_tools`` yielding ``TEXT`` / ``TOOL_CALL`` / ``DONE``
    events).
    """

    # Default sampling parameters — historical values from QwenDashScopeBackend.
    DEFAULT_MAX_TOKENS = 500
    DEFAULT_TEMPERATURE = 0.7

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._model = model
        self._max_tokens_override = max_tokens
        self._temperature_override = temperature
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(max_connections=64, max_keepalive_connections=16, keepalive_expiry=60.0),
            ),
        )
        self._warmup_lock = asyncio.Lock()
        self._warmed_up = False

    # ------------- provider hook points -------------

    def _build_extra_body(self) -> dict[str, Any] | None:
        """Return provider-specific ``extra_body`` kwargs or None.

        DashScope Qwen passes ``enable_thinking`` here.  Doubao / GLM override
        (or don't) as needed.
        """
        return None

    # ------------- shared streaming logic -------------

    async def warmup(self) -> None:
        """Pre-warm httpx connection pool with a minimal streaming request.

        The AsyncOpenAI httpx pool is lazy — the first real request pays
        TCP + TLS handshake cost.  Issuing a tiny throwaway streaming call
        during pipeline setup moves that cost out of the user-facing path.
        """
        if self._warmed_up:
            return

        async with self._warmup_lock:
            if self._warmed_up:
                return
            try:
                kwargs = {
                    "model": self._model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                    "max_tokens": 1,
                }
                extra = self._build_extra_body()
                if extra is not None:
                    kwargs["extra_body"] = extra
                resp = await self._client.chat.completions.create(**kwargs)
                try:
                    # Read the tiny response through DONE. Stopping at the role
                    # delta left its HTTP connection borrowed instead of warm.
                    async with asyncio.timeout(5.0):
                        async for _ in resp:
                            pass
                finally:
                    close = getattr(resp, "close", None)
                    if close is not None:
                        with contextlib.suppress(Exception):
                            await close()
                self._warmed_up = True
                logger.info("llm.warmup.ok", model=self._model)
            except Exception:
                logger.warning("llm.warmup.failed", model=self._model, exc_info=True)

    @staticmethod
    def _strip_internal_metadata(messages: list[dict]) -> list[dict]:
        """Remove _turn_index and other internal fields before sending to API."""
        return [{k: v for k, v in msg.items() if not k.startswith("_")} for msg in messages]

    async def stream_with_tools(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        session_id: str = "",
        turn_id: str = "",
        model: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream with tool support. Yields StreamEvent objects.

        - TEXT events for content tokens
        - TOOL_CALL events for completed tool calls (accumulated from deltas)
        - DONE event at the end

        ``model`` optionally overrides the construction-time model for this one
        call (e.g. a per-node speech model). None → use ``self._model`` (default,
        fully backward compatible).
        """
        clean_messages = self._strip_internal_metadata(messages)

        create_kwargs: dict = dict(
            model=model or self._model,
            messages=clean_messages,
            stream=True,
            max_tokens=(
                self._max_tokens_override if self._max_tokens_override is not None else self.DEFAULT_MAX_TOKENS
            ),
            temperature=(
                self._temperature_override if self._temperature_override is not None else self.DEFAULT_TEMPERATURE
            ),
        )
        if tools:
            create_kwargs["tools"] = tools

        extra_body = self._build_extra_body()
        if extra_body:
            create_kwargs["extra_body"] = extra_body

        response = await self._client.chat.completions.create(**create_kwargs)

        accumulators: dict[int, _ToolAccumulator] = {}

        async for chunk in response:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta
            finish_reason = choice.finish_reason

            # Text content
            if delta.content:
                yield StreamEvent(type=StreamEventType.TEXT, text=delta.content)

            # Tool call deltas
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in accumulators:
                        accumulators[idx] = _ToolAccumulator()

                    acc = accumulators[idx]
                    if tc_delta.id:
                        acc.id = tc_delta.id
                    if tc_delta.function and tc_delta.function.name:
                        acc.name = tc_delta.function.name
                    if tc_delta.function and tc_delta.function.arguments:
                        acc.arguments_buffer += tc_delta.function.arguments

            # Finish: emit accumulated tool calls, then DONE
            if finish_reason == "tool_calls":
                for _idx in sorted(accumulators):
                    acc = accumulators[_idx]
                    try:
                        parsed_args = json.loads(acc.arguments_buffer) if acc.arguments_buffer else {}
                    except json.JSONDecodeError:
                        parsed_args = {"_raw": acc.arguments_buffer}
                    yield StreamEvent(
                        type=StreamEventType.TOOL_CALL,
                        tool_call=CompletedToolCall(
                            id=acc.id,
                            name=acc.name,
                            arguments=parsed_args,
                        ),
                    )
                yield StreamEvent(type=StreamEventType.DONE)
            elif finish_reason is not None:
                yield StreamEvent(type=StreamEventType.DONE)

    async def complete_structured(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        response_format: dict,
        max_tokens: int | None = None,
        temperature: float | None = None,
        session_id: str = "",
        turn_id: str = "",
    ) -> StructuredCompletionResult:
        clean_messages = self._strip_internal_metadata(messages)
        kwargs: dict[str, Any] = {
            "model": model or self._model,
            "messages": clean_messages,
            "stream": False,
            "max_tokens": max_tokens if max_tokens is not None else self.DEFAULT_MAX_TOKENS,
            "temperature": temperature if temperature is not None else self.DEFAULT_TEMPERATURE,
            "response_format": response_format,
        }
        extra_body = self._build_extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body
        response = await self._client.chat.completions.create(**kwargs)
        raw_text = response.choices[0].message.content or ""
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            parsed = None
        usage = getattr(response, "usage", None)
        if hasattr(usage, "model_dump"):
            usage_payload = usage.model_dump()
        elif isinstance(usage, dict):
            usage_payload = usage
        else:
            usage_payload = None
        return StructuredCompletionResult(raw_text=raw_text, parsed_json=parsed, usage=usage_payload)
