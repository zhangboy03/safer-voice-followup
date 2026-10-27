"""One interactive brain session per WS connection. Runs an Arm's journey as a
background task, emits structured events, handles end_call (cancel) and errors
(terminal). Arm is injected so the session is unit-tested without network."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable

from src.playground.human_patient import HumanPatient

Emit = Callable[[dict], Awaitable[None]]


@dataclass
class Arm:
    # run(patient, on_opening, on_turn, on_form) -> journey result dict
    run: Callable[..., Awaitable[dict]]
    timeline_for: Callable[[str], dict]


class PlaygroundSession:
    def __init__(self, *, emit: Emit, arm: Arm) -> None:
        self._emit = emit
        self._arm = arm
        self._patient = HumanPatient()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.ensure_future(self._run())

    async def submit(self, text: str) -> None:
        await self._patient.submit(text)

    async def _run(self) -> None:
        async def on_opening(text: str) -> None:
            await self._emit({"type": "opening", "text": text})

        async def on_turn(item: dict) -> None:
            timeline = self._arm.timeline_for(item.get("turn_id", ""))
            await self._emit({"type": "assistant_turn", **item, "timeline": timeline})

        async def on_form(form: dict) -> None:
            await self._emit({"type": "form_update", **form})

        try:
            await self._arm.run(self._patient, on_opening, on_turn, on_form)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # backend/API failure -> terminal (spec §3.2)
            await self._emit({"type": "error", "message": str(exc)})
        await self._emit({"type": "ended"})

    async def end_call(self) -> None:
        # Emit `ended` only if we actually cancelled a running journey; if the journey
        # already completed naturally, _run already emitted `ended` (avoid a double).
        cancelled = await self._cancel_task()
        if cancelled:
            await self._emit({"type": "ended"})

    async def aclose(self) -> None:
        await self._cancel_task()

    async def _cancel_task(self) -> bool:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            return True
        self._task = None
        return False


def build_arm(*, arch: str, model: str) -> Arm:
    """Production wiring. Mirrors src/eval/study2_v3/our_arm.build_our_arm and the spa
    wiring in runner.default_arm_factory, but injects a HumanPatient + on_opening/on_turn
    and captures timing. The journey loop itself (run_journey/run_spa_journey) is unchanged.
    Backend construction is deferred into run() so build_arm needs no network/API key."""
    if arch not in ("ours", "spa", "solo"):
        raise ValueError(f"unknown arch: {arch}")

    from src.playground.timing import TimingCollector

    timing = TimingCollector()

    if arch == "ours":

        async def _run_ours(patient, on_opening, on_turn, on_form):  # on_form unused (ours has no live form)
            from src.backends.llm_dashscope import QwenDashScopeBackend
            from src.core.config import settings
            from src.eval.study2_v3.cards import Card
            from src.eval.study2_v3.driver import run_journey
            from src.eval.study2_v3.our_arm import _build_flow_runtime
            from src.eval.study2_v3.speech_runner import make_speech_runner
            from src.flows.fast_judge_runner import FastJudgeRunner
            from src.flows.state_committer import StateCommitter
            from src.playground.timing import wrap_judge_runner, wrap_speech_runner

            flow = _build_flow_runtime()
            speech_backend = QwenDashScopeBackend(model=model, enable_thinking=False)
            judge_backend = QwenDashScopeBackend(
                model=model,
                max_tokens=settings.fast_judge_max_tokens,
                temperature=settings.fast_judge_temperature,
                enable_thinking=False,
            )
            speech_runner = wrap_speech_runner(make_speech_runner(speech_backend), timing)
            fast_judge_runner = wrap_judge_runner(
                FastJudgeRunner(
                    backend=judge_backend,
                    model=model,
                    max_tokens=settings.fast_judge_max_tokens,
                    temperature=settings.fast_judge_temperature,
                ),
                timing,
            )
            committer = StateCommitter(flow)
            placeholder = Card(case_id="playground", case_type="playground", persona={}, node_facts={}, fields=[])
            return await run_journey(
                placeholder,
                flow=flow,
                fast_judge_runner=fast_judge_runner,
                committer=committer,
                speech_runner=speech_runner,
                patient=patient,
                run_id="playground",
                on_opening=on_opening,
                on_turn=on_turn,
            )

        return Arm(run=_run_ours, timeline_for=timing.timeline_for)

    if arch == "solo":

        async def _run_solo(patient, on_opening, on_turn, on_form):
            from src.backends.llm_dashscope import QwenDashScopeBackend
            from src.playground.form_extractor import extract_form
            from src.playground.form_schema import field_catalog
            from src.playground.kb_tool import make_kb_tool
            from src.playground.solo_agent import run_solo_journey
            from src.playground.solo_prompt import build_solo_system_prompt

            speech_backend = QwenDashScopeBackend(model=model, enable_thinking=False)
            kb_backend = QwenDashScopeBackend(model=model, enable_thinking=False)
            extractor_backend = QwenDashScopeBackend(
                model=model, max_tokens=1200, temperature=0.0, enable_thinking=False
            )
            catalog = field_catalog()
            kb_tool = make_kb_tool(kb_backend)

            async def _extract(transcript: str):
                return await extract_form(transcript, backend=extractor_backend, catalog=catalog)

            return await run_solo_journey(
                system_prompt=build_solo_system_prompt(),
                backend=speech_backend,
                kb_tool=kb_tool,
                extract_form=_extract,
                catalog=catalog,
                patient=patient,
                run_id="playground",
                on_opening=on_opening,
                on_turn=on_turn,
                on_form=on_form,
            )

        return Arm(run=_run_solo, timeline_for=lambda tid: {})

    async def _run_spa(patient, on_opening, on_turn, on_form):  # on_form unused (spa has no live form)
        from src.backends.llm_dashscope import QwenDashScopeBackend
        from src.eval.study2_v3.cards import Card
        from src.eval.study2_v3.spa_agent import run_spa_journey
        from src.eval.study2_v3.speech_runner import make_speech_runner
        from src.playground.timing import wrap_speech_runner

        speech_runner = wrap_speech_runner(
            make_speech_runner(QwenDashScopeBackend(model=model, enable_thinking=False)), timing
        )
        placeholder = Card(case_id="playground", case_type="playground", persona={}, node_facts={}, fields=[])
        return await run_spa_journey(
            placeholder, speech_runner=speech_runner, patient=patient, run_id="playground", on_turn=on_turn
        )

    return Arm(run=_run_spa, timeline_for=timing.timeline_for)
