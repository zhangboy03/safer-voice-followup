from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from src.eval.virtual_caller.actions import PatientAction


@dataclass
class AgentTurn:
    """One agent response in conversation history."""

    text: str = ""
    events: list[dict] = field(default_factory=list)


class PatientDriver(Protocol):
    async def next_utterance(self, history: list[AgentTurn]) -> PatientAction: ...


class StaticScriptDriver:
    """Plays back a fixed sequence of utterances. Returns hangup when exhausted."""

    def __init__(self, utterances: list[str | PatientAction]) -> None:
        self._items = [u if isinstance(u, PatientAction) else PatientAction(kind="speak", text=u) for u in utterances]
        self._idx = 0

    async def next_utterance(self, history: list[AgentTurn]) -> PatientAction:
        if self._idx >= len(self._items):
            return PatientAction(kind="hangup")
        action = self._items[self._idx]
        self._idx += 1
        return action


class BackchannelInjectDriver:
    """Wraps another driver, injecting backchannel at specified turns."""

    def __init__(self, base: PatientDriver, *, at_turns: list[int], labels: list[str]) -> None:
        self._base = base
        self._schedule = dict(zip(at_turns, labels))
        self._turn = 0
        self._pending_base = False

    async def next_utterance(self, history: list[AgentTurn]) -> PatientAction:
        if self._pending_base:
            self._pending_base = False
            return await self._base.next_utterance(history)
        if self._turn in self._schedule:
            label = self._schedule[self._turn]
            self._turn += 1
            self._pending_base = True
            return PatientAction(kind="backchannel", mode="overlap", backchannel_label=label)
        self._turn += 1
        return await self._base.next_utterance(history)


class HangupAfterDriver:
    """Wraps another driver, forcing hangup after N turns."""

    def __init__(self, base: PatientDriver, *, after_turns: int) -> None:
        self._base = base
        self._limit = after_turns
        self._count = 0

    async def next_utterance(self, history: list[AgentTurn]) -> PatientAction:
        if self._count >= self._limit:
            return PatientAction(kind="hangup")
        self._count += 1
        return await self._base.next_utterance(history)


class LLMPatientDriver:
    """LLM-driven patient. Uses an LLM client + persona prompt + function-calling.

    Supports seed-based replay caching: first run calls LLM and saves responses
    to a JSONL file; subsequent runs with the same seed read from cache.

    The ``llm_client`` is any object with ``async generate_structured(prompt) -> dict``.
    If ``None`` is passed, a backend is instantiated via
    :func:`src.eval.virtual_caller.patient_llm_backends.get_patient_llm_backend`
    which honours the ``PATIENT_LLM_PROVIDER`` env switch (§4.6 A4).
    """

    def __init__(
        self,
        persona_prompt: str,
        llm_client=None,  # any async client with generate_structured(prompt) -> dict
        *,
        seed: str | None = None,
        replay_cache_dir: Path | None = None,
        default_voice: str = "doubao-female-30s",
        provider: str | None = None,
    ) -> None:
        self._persona_prompt = persona_prompt
        if llm_client is None:
            # Lazy import so unit tests that inject a Mock client don't pay
            # the openai/dashscope import cost or require API env vars.
            from src.eval.virtual_caller.patient_llm_backends import get_patient_llm_backend

            llm_client = get_patient_llm_backend(provider=provider)
        self._llm = llm_client
        # Provider tag is used to partition the replay cache so swapping
        # PATIENT_LLM_PROVIDER doesn't pollute prior runs' cached dialogues.
        # Prefer explicit override, then client's own ``provider_name`` attr,
        # then fall back to the settings-level value, then "mock".
        from src.core.config import settings as _settings

        self._provider = (
            provider
            or getattr(llm_client, "provider_name", None)
            or (_settings.patient_llm_provider if llm_client is not None else None)
            or "mock"
        )
        self._seed = seed
        self._cache_dir = replay_cache_dir
        self._default_voice = default_voice
        self._turn_idx = 0
        self._cache: list[PatientAction] | None = None
        if seed and replay_cache_dir:
            self._cache = self._load_cache()

    def _cache_path(self) -> Path | None:
        if self._cache_dir and self._seed:
            # Provider-suffixed so qwen and doubao runs never collide.
            # e.g. fixtures/recorded_dialogues/<scenario>/s42-qwen.jsonl
            return self._cache_dir / f"{self._seed}-{self._provider}.jsonl"
        return None

    def _load_cache(self) -> list[PatientAction]:
        path = self._cache_path()
        if not path or not path.exists():
            return []
        actions = []
        for line in path.read_text().strip().split("\n"):
            if line:
                d = json.loads(line)
                actions.append(PatientAction(**d))
        return actions

    def _save_to_cache(self, action: PatientAction) -> None:
        path = self._cache_path()
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            d = {
                "kind": action.kind,
                "mode": action.mode,
                "text": action.text,
                "silence_ms": action.silence_ms,
                "backchannel_label": action.backchannel_label,
                "voice": action.voice,
                "relative_ms": action.relative_ms,
            }
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    async def next_utterance(self, history: list[AgentTurn]) -> PatientAction:
        # Check cache first
        if self._cache and self._turn_idx < len(self._cache):
            action = self._cache[self._turn_idx]
            self._turn_idx += 1
            return action

        # Call LLM
        prompt = self._build_prompt(history)
        try:
            response = await self._llm.generate_structured(prompt)
            action = self._parse_response(response)
        except Exception:
            # LLM failure — fail honestly so callers can handle it
            raise

        if self._seed:
            self._save_to_cache(action)
        self._turn_idx += 1
        return action

    def _build_prompt(self, history: list[AgentTurn]) -> str:
        lines = [self._persona_prompt, "", "## 对话历史", ""]
        for i, turn in enumerate(history):
            lines.append(f"客服第{i + 1}轮: {turn.text}")
        lines.append("")
        lines.append("请根据你的角色设定，生成下一句患者回复。")
        lines.append('输出 JSON: {"kind": "speak"|"silence"|"hangup", "text": "...", "silence_ms": null}')
        return "\n".join(lines)

    def _parse_response(self, response: dict) -> PatientAction:
        kind = response.get("kind", "speak")
        if kind not in ("speak", "silence", "hangup", "backchannel"):
            kind = "speak"
        return PatientAction(
            kind=kind,
            text=response.get("text"),
            silence_ms=response.get("silence_ms"),
            voice=response.get("voice", self._default_voice),
        )
