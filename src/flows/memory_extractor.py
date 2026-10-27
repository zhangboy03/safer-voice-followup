from __future__ import annotations

import asyncio

from src.backends.structured_capabilities import build_response_format
from src.flows.ablation_switches import (
    can_legacy_memory_extractor_mutate,
    is_legacy_memory_extractor_shadow_enabled,
)
from src.flows.sop_decision_models import MemoryPatch
from src.flows.state_committer import StateCommitter


class CallMemoryExtractor:
    def __init__(self, *, backend, model: str, max_tokens: int = 500, temperature: float = 0.0) -> None:
        self._backend = backend
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    @staticmethod
    def build_messages(
        *,
        prompt_registry,
        weighted_context,
        transcript_delta: str,
        variables: dict | None = None,
    ) -> tuple[list[dict], str]:
        rendered = prompt_registry.render_memory_prompt(
            weighted_context=weighted_context,
            transcript_delta=transcript_delta,
            variables=variables or {},
        )
        return [{"role": "system", "content": rendered.text}], rendered.hash

    async def extract(self, *, messages: list[dict], session_id: str, turn_id: str) -> MemoryPatch:
        fmt = build_response_format(
            model=self._model,
            schema_name="call_memory_patch",
            schema=MemoryPatch.model_json_schema(),
        )
        completion = await self._backend.complete_structured(
            messages,
            response_format=fmt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            session_id=session_id,
            turn_id=turn_id,
        )
        if completion.parsed_json is None:
            raise ValueError("memory_patch_parse_failed")
        return MemoryPatch.model_validate(completion.parsed_json)


class MemoryExtractorScheduler:
    def __init__(self, *, flow, extractor, timeout_s: float = 3.0, max_concurrency: int = 1) -> None:
        self._flow = flow
        self._extractor = extractor
        self._timeout_s = timeout_s
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._tasks: set[asyncio.Task] = set()

    def on_playout_done(self, event) -> asyncio.Task:
        transcript_delta = getattr(event, "transcript_delta", None)
        if transcript_delta is None:
            transcript_delta = self._flow._history.get_summary_context(last_n=2)
        return self.schedule_after_playback(transcript_delta=transcript_delta)

    def on_node_transition(self, event) -> asyncio.Task:
        return self.schedule_on_node_transition(transcript_delta=getattr(event, "transcript_delta", []))

    def on_turn_committed(self, turn_id: str) -> asyncio.Task | None:
        return self.schedule_periodic_if_needed(transcript_delta=self._flow._history.get_summary_context(last_n=2))

    async def on_session_closed(self, event) -> None:
        task = self.schedule_after_playback(transcript_delta=self._flow._history.get_summary_context(last_n=4))
        await task

    def schedule_after_playback(self, *, transcript_delta: list[dict]) -> asyncio.Task:
        captured_turn_id = self._flow._state.active_turn_id or ""
        captured_turn_index = self._flow._state.turn_count
        captured_node_id = self._flow._state.current_node
        captured_flow_epoch = self._flow._state.flow_epoch
        task = asyncio.create_task(
            self._run(
                transcript_delta=transcript_delta,
                captured_turn_id=captured_turn_id,
                captured_turn_index=captured_turn_index,
                captured_node_id=captured_node_id,
                captured_flow_epoch=captured_flow_epoch,
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def schedule_on_node_transition(self, *, transcript_delta: list[dict]) -> asyncio.Task:
        return self.schedule_after_playback(transcript_delta=transcript_delta)

    def schedule_periodic_if_needed(self, *, transcript_delta: list[dict]) -> asyncio.Task | None:
        if self._flow._state.turn_count % 3 != 0:
            return None
        return self.schedule_after_playback(transcript_delta=transcript_delta)

    def _format_transcript_delta(self, transcript_delta: list[dict]) -> str:
        return "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in transcript_delta)

    def _maybe_merge_legacy_patch(self, patch: MemoryPatch) -> None:
        if is_legacy_memory_extractor_shadow_enabled():
            return
        if can_legacy_memory_extractor_mutate():
            StateCommitter(self._flow).merge_memory_patch(patch)

    async def _run(
        self,
        *,
        transcript_delta: list[dict],
        captured_turn_id: str,
        captured_turn_index: int,
        captured_node_id: str,
        captured_flow_epoch: int,
    ) -> None:
        transcript_text = self._format_transcript_delta(transcript_delta)
        weighted_context = self._flow.build_weighted_context_view(
            transcript_text,
            lane="memory",
        )
        prompt_registry = getattr(self._flow, "_prompt_registry", None)
        if prompt_registry is not None and hasattr(self._extractor, "build_messages"):
            messages, _prompt_hash = self._extractor.build_messages(
                prompt_registry=prompt_registry,
                weighted_context=weighted_context,
                transcript_delta=transcript_text,
                variables={
                    "turn_id": captured_turn_id,
                    "turn_index": captured_turn_index,
                    "node_id": captured_node_id,
                    "flow_epoch": captured_flow_epoch,
                },
            )
        else:
            messages = [{"role": "system", "content": transcript_text}]
        async with self._semaphore:
            patch = await asyncio.wait_for(
                self._extractor.extract(
                    messages=messages,
                    session_id="",
                    turn_id=captured_turn_id,
                ),
                timeout=self._timeout_s,
            )
        if patch.turn_id != captured_turn_id or patch.turn_index < captured_turn_index - 1:
            return
        self._maybe_merge_legacy_patch(patch)
