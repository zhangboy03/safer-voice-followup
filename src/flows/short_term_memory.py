from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from src.backends.structured_capabilities import build_response_format
from src.core.logging import get_logger
from src.flows.ablation_switches import (
    get_note_filter_blacklist,
    get_note_filter_confidence_threshold,
    is_short_term_memory_analyzer_enabled,
    is_stm_for_patient_calls_enabled,
)
from src.flows.note_filter import FilterConfig, filter_removes, filter_upserts
from src.flows.note_log import NoteLogWriter
from src.flows.sop_decision_models import ShortTermMemoryPatch

if TYPE_CHECKING:
    from src.flows.runtime import FlowRuntime


def _recent_turn_ids(flow_state) -> set[str]:
    """Return last 2 turn ids based on FlowState turn_count + history if available."""
    history = getattr(flow_state, "turn_history", None)
    if history:
        return {t.turn_id for t in history[-2:] if getattr(t, "turn_id", None)}
    active = getattr(flow_state, "active_turn_id", None)
    return {active} if active else set()


def _format_transcript_delta(transcript_delta: list[dict]) -> str:
    return "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in transcript_delta)


class ShortTermMemoryAnalyzer:
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
                "Extract short-lived analyzer memory. Output only JSON matching short_term_memory_patch.v1.",
                "硬约束（dedup）：如果最近 TranscriptDelta 中没有此前已采纳要点未覆盖的新信息，"
                "返回 upsert_items=[] 且 remove_keys=[]。不要把客套语、单字应答、已确认信息"
                "重新写成一条 item。"
                "——这些内容对 CSR 决策无增量价值，重复落库只会增加噪音、稀释关键要点。",
                "few-shot 反例（这些都应该返回空 patch）：\n"
                "  - user: 嗯 / 好的 / 谢谢 / 明白了 → 空\n"
                "  - assistant: 我帮你查一下 / 稍等 → 空\n"
                "  - 用户重复刚刚说过的话 → 空",
                "few-shot 正例（这些应该 upsert）：\n"
                "  - user: 我夜里头痛 → upsert salient_fact\n"
                "  - user: 我不是患者，是家属 → upsert correction_hint\n"
                "  - user: 等会儿能问一下用药吗 → upsert future_node_hint",
                "[WeightedContext]",
                weighted_context.render_for_model(lane="short_term_memory"),
                "[TranscriptDelta]",
                transcript_delta,
            ]
        )
        return [{"role": "system", "content": content}]

    async def extract(self, *, messages: list[dict], session_id: str, turn_id: str) -> ShortTermMemoryPatch:
        fmt = build_response_format(
            model=self._model,
            schema_name="short_term_memory_patch",
            schema=ShortTermMemoryPatch.model_json_schema(),
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
            raise ValueError("short_term_memory_patch_parse_failed")
        return ShortTermMemoryPatch.model_validate(completion.parsed_json)


class ShortTermMemoryScheduler:
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
        captured_flow_epoch = self._flow._state.flow_epoch
        task = asyncio.create_task(
            self._run(
                transcript_delta=transcript_delta,
                captured_turn_id=captured_turn_id,
                captured_turn_index=captured_turn_index,
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
        captured_flow_epoch: int,
    ) -> None:
        if not (is_short_term_memory_analyzer_enabled() or is_stm_for_patient_calls_enabled()):
            return
        try:
            transcript_text = _format_transcript_delta(transcript_delta)
            weighted_context = self._flow.build_weighted_context_view(transcript_text, lane="short_term_memory")
            if hasattr(self._analyzer, "build_messages"):
                messages = self._analyzer.build_messages(
                    weighted_context=weighted_context,
                    transcript_delta=transcript_text,
                    variables={
                        "turn_id": captured_turn_id,
                        "turn_index": captured_turn_index,
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
            ShortTermMemoryCommitter(self._flow).merge_patch(patch, completed_turn_id=captured_turn_id)
        except Exception as exc:
            get_logger("stm_scheduler").warning("stm_run_failed", error=str(exc))


class ShortTermMemoryCommitter:
    def __init__(self, flow: FlowRuntime) -> None:
        self._flow = flow

    def merge_patch(self, patch: ShortTermMemoryPatch, *, completed_turn_id: str | None = None) -> None:
        if self._stale_or_mismatched(patch, completed_turn_id=completed_turn_id):
            return

        pre_state = self._flow._state.short_term_memory.model_copy(deep=True)

        # Monotonic turn-index check (spec §3.3).
        if patch.turn_index <= pre_state.last_accepted_turn_index:
            return

        # Filter pass A (spec §6).
        cfg = FilterConfig(
            confidence_threshold=get_note_filter_confidence_threshold(),
            blacklist=get_note_filter_blacklist(),
        )
        kept_upserts, dropped_upserts = filter_upserts(patch.upsert_items, cfg)
        recent_turn_ids = _recent_turn_ids(self._flow._state)
        kept_removes, dropped_removes = filter_removes(
            patch.remove_keys,
            kept_upserts=kept_upserts,
            existing_items=pre_state.items,
            recent_turn_ids=recent_turn_ids,
        )

        # Apply filtered patch.
        new_state = pre_state.model_copy(deep=True)
        remove_set = set(kept_removes)
        items = {it.key: it for it in new_state.items if it.key not in remove_set}
        for it in kept_upserts:
            items[it.key] = it
        new_state.items = list(items.values())
        new_state.last_accepted_turn_index = patch.turn_index
        self._flow._state.short_term_memory = new_state

        # Best-effort persistence (spec §3.2). Must NOT raise into main path.
        try:
            call_id = getattr(self._flow, "call_id", "") or ""
            if call_id:
                sop_node_id = ""
                getter = getattr(self._flow, "current_sop_node", None)
                if callable(getter):
                    sop_node_id = getter() or ""
                kept_patch = patch.model_copy(
                    update={
                        "upsert_items": kept_upserts,
                        "remove_keys": kept_removes,
                    }
                )
                filtered_out_serialized = [(it.model_dump(), reason) for it, reason in dropped_upserts] + [
                    (key, reason) for key, reason in dropped_removes
                ]
                NoteLogWriter().append_patch_event(
                    call_id=call_id,
                    sop_node_id=sop_node_id,
                    before_items=pre_state.items,  # pre-merge snapshot
                    kept_patch=kept_patch,
                    filtered_out=filtered_out_serialized,
                )
        except Exception as exc:
            get_logger("stm_committer").error("note_log_writer_failed", error=str(exc))

    def _stale_or_mismatched(self, patch: ShortTermMemoryPatch, *, completed_turn_id: str | None = None) -> bool:
        state = self._flow._state
        if patch.flow_epoch != state.flow_epoch:
            return True
        if patch.available_before_turn_id in patch.analyzed_turn_ids:
            return True

        active_turn_id = state.active_turn_id
        if active_turn_id is None:
            return False
        if active_turn_id == completed_turn_id:
            return False
        return active_turn_id in patch.analyzed_turn_ids
