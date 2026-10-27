from __future__ import annotations

import asyncio
import hashlib
import json
from typing import TYPE_CHECKING

from src.backends.structured_capabilities import build_response_format
from src.flows.ablation_switches import is_sop_progress_analyzer_enabled
from src.flows.models import FieldStatus, FieldValue
from src.flows.sop_decision_models import SopProgressFieldCandidate, SopProgressPatch

if TYPE_CHECKING:
    from src.flows.runtime import FlowRuntime


def _format_transcript_delta(transcript_delta: list[dict]) -> str:
    return "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in transcript_delta)


class SopProgressAnalyzer:
    def __init__(self, *, backend, model: str, max_tokens: int = 500, temperature: float = 0.0) -> None:
        self._backend = backend
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    @staticmethod
    def build_messages(*, weighted_context, transcript_delta: str, variables: dict | None = None) -> list[dict]:
        del variables
        content = "\n\n".join(
            [
                "Extract SOP progress candidates. Output only JSON matching sop_progress_patch.v1.",
                "Write candidate ledger entries only; do not mutate collected_fields or CallWorkingMemory.sop_fields.",
                "[WeightedContext]",
                weighted_context.render_for_model(lane="sop_progress"),
                "[TranscriptDelta]",
                transcript_delta,
            ]
        )
        return [{"role": "system", "content": content}]

    async def extract(self, *, messages: list[dict], session_id: str, turn_id: str) -> SopProgressPatch:
        fmt = build_response_format(
            model=self._model,
            schema_name="sop_progress_patch",
            schema=SopProgressPatch.model_json_schema(),
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
            raise ValueError("sop_progress_patch_parse_failed")
        return SopProgressPatch.model_validate(completion.parsed_json)


class SopProgressScheduler:
    def __init__(self, *, flow: FlowRuntime, analyzer, timeout_s: float = 3.0, max_concurrency: int = 1) -> None:
        self._flow = flow
        self._analyzer = analyzer
        self._timeout_s = timeout_s
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._tasks: set[asyncio.Task] = set()

    def on_playout_done(self, event) -> asyncio.Task:
        transcript_delta = getattr(event, "transcript_delta", None)
        if transcript_delta is None:
            transcript_delta = self._flow._history.get_summary_context(last_n=2)
        return self.schedule_after_playback(transcript_delta=transcript_delta)

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

    async def _run(
        self,
        *,
        transcript_delta: list[dict],
        captured_turn_id: str,
        captured_turn_index: int,
        captured_node_id: str,
        captured_flow_epoch: int,
    ) -> None:
        if not is_sop_progress_analyzer_enabled():
            return
        transcript_text = _format_transcript_delta(transcript_delta)
        weighted_context = self._flow.build_weighted_context_view(transcript_text, lane="sop_progress")
        if hasattr(self._analyzer, "build_messages"):
            messages = self._analyzer.build_messages(
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
                self._analyzer.extract(messages=messages, session_id="", turn_id=captured_turn_id),
                timeout=self._timeout_s,
            )
        SopProgressCommitter(self._flow).merge_patch(patch, completed_turn_id=captured_turn_id)


class SopProgressCommitter:
    def __init__(self, flow: FlowRuntime) -> None:
        self._flow = flow

    def merge_patch(self, patch: SopProgressPatch, *, completed_turn_id: str | None = None) -> None:
        if self._stale_or_mismatched(patch, completed_turn_id=completed_turn_id):
            return

        decision_id = self._decision_id(patch)
        if decision_id in self._flow._state.applied_sop_decision_ids:
            return

        state = self._flow._state.sop_progress_state.model_copy(deep=True)
        state.field_candidates.extend(self._with_conflicts(patch.field_candidates))
        if patch.summary_for_future_judges:
            state.summary_for_future_judges = patch.summary_for_future_judges
        self._flow._state.sop_progress_state = state
        self._flow._state.applied_sop_decision_ids.add(decision_id)

    def _stale_or_mismatched(self, patch: SopProgressPatch, *, completed_turn_id: str | None = None) -> bool:
        state = self._flow._state
        if patch.flow_epoch != state.flow_epoch:
            return True
        if patch.available_after_turn_id not in patch.analyzed_turn_ids:
            return True
        active_turn_id = state.active_turn_id
        if active_turn_id is None:
            return False
        if active_turn_id == completed_turn_id:
            return False
        return active_turn_id in patch.analyzed_turn_ids

    def _decision_id(self, patch: SopProgressPatch) -> str:
        payload = json.dumps(patch.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        patch_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"sop_progress:{patch.available_after_turn_id}:{patch_hash}"

    def _with_conflicts(
        self,
        candidates: list[SopProgressFieldCandidate],
    ) -> list[SopProgressFieldCandidate]:
        return [self._mark_conflict(candidate) for candidate in candidates]

    def _mark_conflict(self, candidate: SopProgressFieldCandidate) -> SopProgressFieldCandidate:
        if self._conflicts_with_fast_judge(candidate):
            return self._needs_reconfirmation(candidate)

        collected = self._flow._state.collected_fields.get(candidate.field)
        if collected is None or not self._conflicts_with_collected(candidate, collected):
            return candidate
        return self._needs_reconfirmation(candidate)

    def _needs_reconfirmation(self, candidate: SopProgressFieldCandidate) -> SopProgressFieldCandidate:
        return candidate.model_copy(update={"status": "needs_reconfirmation", "conflicts_with_fast_judge": True})

    def _conflicts_with_fast_judge(self, candidate: SopProgressFieldCandidate) -> bool:
        fast_judge_state = self._flow._state.fast_judge_state
        if fast_judge_state is None:
            return False
        return candidate.field in fast_judge_state.missing_fields

    def _conflicts_with_collected(self, candidate: SopProgressFieldCandidate, collected: FieldValue) -> bool:
        if candidate.status in {"needs_reconfirmation", "unknown"}:
            return False
        if candidate.status == "refused":
            return collected.status != FieldStatus.REFUSED
        if candidate.status in {"confirmed", "inferred"}:
            return bool(collected.value != candidate.value)
        return False
