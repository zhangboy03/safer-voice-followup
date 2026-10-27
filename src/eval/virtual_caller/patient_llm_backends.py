"""Patient LLM backends for Virtual Caller (thesis §4.6 A4 ablation).

Provides pluggable LLM backends for ``LLMPatientDriver``. Default backend
is Qwen (DashScope) to match production family. When
``PATIENT_LLM_PROVIDER=doubao-pro`` (方舟/Volcengine Ark), the driver
switches to Doubao-Pro — an异族 model — enabling the §4.6 A4 ablation
that quantifies "Judge reads its own family's output more fluently" bias
described in Virtual Caller spec §11.2.

This module is **evaluation-only** — it is not imported from
``src/flows/`` or ``src/backends/`` production paths.
"""

from __future__ import annotations

import json
from typing import Protocol

from openai import AsyncOpenAI

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


# JSON schema used by function-calling / tool to force a PatientAction-shaped
# response from the model. Kept minimal on purpose — the driver's
# ``_parse_response`` is tolerant to extra / missing fields.
PATIENT_ACTION_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "emit_patient_action",
        "description": "输出患者下一步动作。仅在你确认要结束通话时使用 hangup；不确定时用 silence。",
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["speak", "silence", "hangup", "backchannel"],
                    "description": "动作类型",
                },
                "text": {
                    "type": "string",
                    "description": "speak/backchannel 时的文本；silence/hangup 时可留空",
                },
                "silence_ms": {
                    "type": ["integer", "null"],
                    "description": "silence 时的静默时长（毫秒）",
                },
            },
            "required": ["kind"],
        },
    },
}


class PatientLLMBackend(Protocol):
    """Protocol for patient-side LLM backends.

    Implementations must return a dict matching the PatientAction schema:
    ``{"kind": "speak"|"silence"|"hangup"|"backchannel", "text": str | None,
    "silence_ms": int | None}``.
    """

    provider_name: str

    async def generate_structured(self, prompt: str) -> dict: ...


# ────────────────────────────────────────────────────────────
# Qwen backend — DashScope OpenAI-compatible (default analyzer family)
# ────────────────────────────────────────────────────────────


class QwenPatientBackend:
    """Patient LLM driven by Qwen (settings.patient_qwen_model) via DashScope.

    Same family as the Judge LLM — this is the default and matches the
    main evaluation configuration. Use ``DoubaoPatientBackend`` for the
    §4.6 A4 cross-family control.

    Note: Qwen3.6-plus defaults to *thinking mode* on DashScope, which
    disallows ``tool_choice={"type":"function"}``. We pass
    ``extra_body={"enable_thinking": False}`` so tool-choice is honoured
    and latency stays predictable for patient generation.
    """

    provider_name = "qwen"

    def __init__(
        self,
        *,
        model: str | None = None,
        temperature: float = 0.8,
        max_tokens: int = 400,
        enable_thinking: bool = False,
    ) -> None:
        self._model = model or settings.patient_qwen_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._enable_thinking = enable_thinking
        self._client = AsyncOpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.active_dashscope_compatible_base_url,
            timeout=30.0,
        )

    async def generate_structured(self, prompt: str) -> dict:
        # DashScope Qwen3.x requires extra_body to disable thinking mode
        # when using forced tool_choice.
        return await _chat_tool_call(
            client=self._client,
            model=self._model,
            prompt=prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            provider=self.provider_name,
            extra_body={"enable_thinking": self._enable_thinking},
        )


# ────────────────────────────────────────────────────────────
# Doubao-Pro backend — Volcengine Ark (OpenAI-compatible, 异族)
# ────────────────────────────────────────────────────────────


class DoubaoPatientBackend:
    """Patient LLM driven by Doubao-Pro via Volcengine Ark.

    Ark exposes an OpenAI-compatible endpoint so tool-calling works the
    same as DashScope. ``settings.patient_doubao_model`` accepts either
    a snapshot id (e.g. ``doubao-1-5-pro-32k-250115``) or an endpoint id
    (``ep-xxxxxxxx``) — both are accepted by the Ark chat/completions
    endpoint as the ``model`` field.

    If ``settings.doubao_patient_endpoint_id`` is set, it overrides the
    snapshot id. This is convenient when the snapshot must first be
    "接入" in the Ark console before it can be called.
    """

    provider_name = "doubao-pro"

    def __init__(
        self,
        *,
        model: str | None = None,
        temperature: float = 0.8,
        max_tokens: int = 400,
    ) -> None:
        resolved_model = model or (settings.doubao_patient_endpoint_id or None) or settings.patient_doubao_model
        self._model = resolved_model
        self._temperature = temperature
        self._max_tokens = max_tokens

        if not settings.ark_api_key:
            raise RuntimeError(
                "DoubaoPatientBackend requires ARK_API_KEY in environment. "
                "Set ARK_API_KEY in .env to use PATIENT_LLM_PROVIDER=doubao-pro."
            )

        self._client = AsyncOpenAI(
            api_key=settings.ark_api_key,
            base_url=settings.ark_base_url,
            timeout=30.0,
        )

    async def generate_structured(self, prompt: str) -> dict:
        return await _chat_tool_call(
            client=self._client,
            model=self._model,
            prompt=prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            provider=self.provider_name,
        )


# ────────────────────────────────────────────────────────────
# Shared tool-calling helper (Ark's tool-calling is identical to OpenAI)
# ────────────────────────────────────────────────────────────


async def _chat_tool_call(
    *,
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    provider: str,
    extra_body: dict | None = None,
) -> dict:
    """Invoke chat.completions forcing the ``emit_patient_action`` tool.

    Returns the parsed tool-call arguments as a dict. Falls back to
    assistant ``content`` parsing if the model didn't emit a tool call
    (some snapshots under-deliver on ``tool_choice`` compliance).
    """
    create_kwargs: dict = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        tools=[PATIENT_ACTION_TOOL_SCHEMA],
        tool_choice={"type": "function", "function": {"name": "emit_patient_action"}},
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if extra_body:
        create_kwargs["extra_body"] = extra_body
    resp = await client.chat.completions.create(**create_kwargs)

    if not resp.choices:
        raise RuntimeError(f"patient_llm.{provider}.empty_choices")

    msg = resp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None) or []
    if tool_calls:
        tc = tool_calls[0]
        # Ark + DashScope both mirror OpenAI's shape: tc.function.arguments is JSON str.
        raw_args = tc.function.arguments or "{}"
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            logger.warning(
                "patient_llm.tool_args_parse_fail",
                provider=provider,
                model=model,
                raw=raw_args[:200],
            )
            return {"kind": "speak", "text": raw_args.strip() or None}

    # Fallback: model replied in plain content — attempt JSON parse, else speak.
    content = (msg.content or "").strip()
    if not content:
        raise RuntimeError(f"patient_llm.{provider}.empty_response")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.info(
            "patient_llm.content_fallback",
            provider=provider,
            model=model,
            content_preview=content[:120],
        )
        return {"kind": "speak", "text": content}


# ────────────────────────────────────────────────────────────
# Factory
# ────────────────────────────────────────────────────────────


def get_patient_llm_backend(provider: str | None = None) -> PatientLLMBackend:
    """Return a patient LLM backend based on provider name.

    Args:
        provider: Override for ``settings.patient_llm_provider``. Useful
            in tests. If ``None``, read from settings (which in turn
            reads the ``PATIENT_LLM_PROVIDER`` env var).

    Raises:
        ValueError: if the provider name is not recognized.
    """
    name = (provider or settings.patient_llm_provider).strip().lower()
    if name == "qwen":
        return QwenPatientBackend()
    if name in ("doubao-pro", "doubao_pro", "doubao"):
        return DoubaoPatientBackend()
    raise ValueError(f"Unknown PATIENT_LLM_PROVIDER={name!r}; expected one of: qwen, doubao-pro")
