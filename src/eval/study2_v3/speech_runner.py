"""Streaming speech_runner for Study 2 eval, with TTFT instrumentation.

Both arms (our-system and SPA) call this so TTFT is measured identically:
TTFT = wall-clock from request start to the FIRST streamed TEXT event.
This is LLM inference latency only — NOT end-to-end voice latency.
"""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable

from src.backends.llm_base import StreamEventType


def make_speech_runner(backend: Any) -> Callable[..., Awaitable[dict]]:
    """Return an async speech_runner(messages, *, session_id, turn_id) -> dict.

    The returned dict matches what driver.run_journey expects
    (``raw_text`` / ``clean_text``) plus ``ttft_ms``.
    """

    async def _run(messages: list[dict], *, session_id: str = "", turn_id: str = "") -> dict:
        started = time.monotonic()
        ttft_ms: float | None = None
        parts: list[str] = []
        async for event in backend.stream_with_tools(messages, session_id=session_id, turn_id=turn_id):
            if event.type is StreamEventType.TEXT and event.text:
                if ttft_ms is None:
                    ttft_ms = (time.monotonic() - started) * 1000.0
                parts.append(event.text)
            elif event.type is StreamEventType.DONE:
                break
        text = "".join(parts)
        return {
            "raw_text": text,
            "clean_text": text,
            "ttft_ms": ttft_ms if ttft_ms is not None else (time.monotonic() - started) * 1000.0,
        }

    return _run
