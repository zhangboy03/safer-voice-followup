"""Our-system arm: real wiring of the v2 被测链路 for Study 2 eval.

First production-style assembly of run_journey's collaborators (FlowRuntime +
FastJudgeRunner + StateCommitter + streaming speech_runner). Mirrors the
construction in server/debug_full_pipeline.py but without the voice transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from src.backends.llm_dashscope import QwenDashScopeBackend
from src.core.config import settings
from src.eval.study2_v3.cards import Card
from src.eval.study2_v3.driver import render_history, run_journey
from src.eval.study2_v3.speech_runner import make_speech_runner


def journey_to_transcript(result: dict[str, Any]) -> str:
    """压成判官要的扁平 transcript（两臂同形）。our-arm 用 combined_assistant_text，
    spa-arm 用 assistant_text。"""
    items = []
    for it in result.get("actual_items", []):
        items.append(
            {
                "user_text": it.get("user_text", ""),
                "combined_assistant_text": it.get("combined_assistant_text") or it.get("assistant_text") or "",
            }
        )
    return render_history(items, opening_text=result.get("opening_text", ""))


def _build_flow_runtime():
    from src.flows.context_builder import ContextBuilder
    from src.flows.pinchi_flow_config import build_agent_config
    from src.flows.runtime import FlowRuntime
    from src.flows.tool_executor import ToolExecutor
    from src.flows.tools.global_tools import transfer_to_human
    from src.prompts.prompt_pack import PromptPackLoader, PromptRegistry
    from src.session.conversation_history import ConversationHistory

    tool_executor = ToolExecutor()
    tool_executor.register_many({"transfer_to_human": transfer_to_human})
    flow = FlowRuntime(
        agent_config=build_agent_config(),
        context_builder=ContextBuilder(),
        tool_executor=tool_executor,
        conversation_history=ConversationHistory(system_prompt=settings.system_prompt),
        prompt_registry=PromptRegistry(PromptPackLoader().load_active()),
    )
    flow.set_call_id(
        room_name="study2-eval",
        call_start_iso=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    return flow


def build_our_arm(*, agent_model: str) -> Callable[..., Awaitable[dict]]:
    """Return ``run(card, *, run_id, max_turns) -> journey result`` with the full
    被测链路 wired on ``agent_model`` (speech + fast judge both on this model)."""
    from src.flows.fast_judge_runner import FastJudgeRunner
    from src.flows.state_committer import StateCommitter

    speech_backend = QwenDashScopeBackend(model=agent_model, enable_thinking=False)
    judge_backend = QwenDashScopeBackend(
        model=agent_model,
        max_tokens=settings.fast_judge_max_tokens,
        temperature=settings.fast_judge_temperature,
        enable_thinking=False,
    )
    speech_runner = make_speech_runner(speech_backend)

    async def _run(card: Card, *, run_id: str, max_turns: int = 18) -> dict[str, Any]:
        from src.eval.study2_v3.runner import build_patient  # late import (defined in Task 6)

        flow = _build_flow_runtime()
        fast_judge_runner = FastJudgeRunner(
            backend=judge_backend,
            model=agent_model,
            max_tokens=settings.fast_judge_max_tokens,
            temperature=settings.fast_judge_temperature,
        )
        committer = StateCommitter(flow)
        result = await run_journey(
            card,
            flow=flow,
            fast_judge_runner=fast_judge_runner,
            committer=committer,
            speech_runner=speech_runner,
            patient=build_patient(),
            run_id=run_id,
            max_turns=max_turns,
        )
        result["arm"] = "ours"
        return result

    return _run
