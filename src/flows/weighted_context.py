from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from src.flows.sop_decision_models import CallWorkingMemory
from src.session.conversation_history import ConversationHistory

if TYPE_CHECKING:
    from src.flows.models import FlowState

WeightedContextLane = Literal["speech", "judge", "memory", "fast_judge", "short_term_memory", "sop_progress"]
ModelVisibleLane = Literal["speech", "fast_judge", "short_term_memory", "sop_progress"]


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


@dataclass(frozen=True)
class WeightedContextView:
    context_snapshot_id: str
    weighted_context_hash: str
    lane: WeightedContextLane
    speaking_frame: str
    next_speech_strategy: str
    call_working_memory: str
    recent_verbatim_window: str
    rolling_call_summary: str
    current_user_text: str
    fast_judge_state: str = ""
    short_term_memory: str = ""
    sop_progress_candidates: str = ""
    # Set only for lane == "speech" so renderers can scope SOP fields to the
    # current node and avoid leaking other nodes' field names into Stage 1
    # prompts (e.g. greeting Stage 1 must not see "identity.is_patient").
    current_node: str = ""

    @property
    def snapshot_id(self) -> str:
        return self.context_snapshot_id

    @property
    def content_hash(self) -> str:
        return self.weighted_context_hash

    def render_canonical_payload(self) -> str:
        return "\n\n".join(
            [
                f"[Lane]\n{self.lane}",
                f"[通话视角]\n{self.speaking_frame}",
                f"[上一轮策略]\n{self.next_speech_strategy}",
                f"[工作记忆]\n{self.call_working_memory}",
                f"[最近听到的上下文]\n{self.recent_verbatim_window}",
                f"[摘要]\n{self.rolling_call_summary}",
                f"[用户刚说]\n{self.current_user_text}",
            ]
        )

    def render_for_model(self, *, lane: ModelVisibleLane) -> str:
        parts = [
            f"[Lane]\n{lane}",
            f"[通话视角]\n{self.speaking_frame}",
            f"[FAST_JUDGE_STATE]\n{self.fast_judge_state or '无'}",
            f"[SHORT_TERM_MEMORY]\n{self.short_term_memory or '无'}",
            f"[SOP_PROGRESS_CANDIDATES]\n{self.sop_progress_candidates or '无'}",
            f"[最近听到的上下文]\n{self.recent_verbatim_window}",
            f"[摘要]\n{self.rolling_call_summary}",
        ]
        return "\n\n".join(parts)

    def render_for_trace(self) -> str:
        return "\n\n".join(
            [
                f"[ContextSnapshot]\n{self.context_snapshot_id}",
                f"[WeightedContextHash]\n{self.weighted_context_hash}",
                self.render_canonical_payload(),
            ]
        )

    def render_for_prompt(self) -> str:
        return self.render_for_trace()


def _json(data) -> str:
    return json.dumps(data, ensure_ascii=False, default=str, sort_keys=True)


def _render_memory_sections(memory: CallWorkingMemory) -> str:
    never_drop_sections = [
        "[NEVER_DROP: speaking_frame]\n" + memory.speaking_frame.model_dump_json(exclude_none=True),
        "[NEVER_DROP: pending_clarifications]\n"
        + _json([item.model_dump(exclude_none=True) for item in memory.pending_clarifications]),
        "[NEVER_DROP: do_not_repeat]\n" + _json([item.model_dump(exclude_none=True) for item in memory.do_not_repeat]),
        "[NEVER_DROP: corrections]\n" + _json([item.model_dump(exclude_none=True) for item in memory.corrections]),
        "[NEVER_DROP: playback_truth]\n" + memory.playback_truth.model_dump_json(exclude_none=True),
    ]
    optional_sections = [
        "[OPTIONAL: active_fields_trimmed]\n"
        + _clip(_json({k: v.model_dump(exclude_none=True) for k, v in memory.sop_fields.items()}), 600),
        "[OPTIONAL: future_observations_trimmed]\n"
        + _clip(_json([item.model_dump(exclude_none=True) for item in memory.future_observations]), 300),
        "[OPTIONAL: open_loops_trimmed]\n"
        + _clip(_json([item.model_dump(exclude_none=True) for item in memory.open_loops]), 300),
        "[OPTIONAL: user_interaction_profile]\n" + memory.user_interaction_profile.model_dump_json(exclude_none=True),
    ]
    return "\n\n".join(never_drop_sections + optional_sections)


class WeightedContextAssembler:
    def build(
        self,
        *,
        memory: CallWorkingMemory,
        history: ConversationHistory,
        current_user_text: str,
        lane: WeightedContextLane = "speech",
        rolling_summary: str = "无",
        flow_state: FlowState | None = None,
        exclude_current_turn_from_recent: bool = False,
    ) -> WeightedContextView:
        if memory.speaking_frame.caller_role == "proxy_family":
            frame = (
                f"接听人不是患者本人，是患者{memory.speaking_frame.relationship or '家属'}。称呼时不要把接听人当患者。"
            )
        elif memory.speaking_frame.caller_role == "patient":
            frame = "接听人是患者本人。"
        else:
            # Entity-free: name no roles in the unknown case so the words
            # "患者本人" don't appear in every node's Stage 1 prompt before
            # identity has been determined.
            frame = "接听人身份尚未确认。用语保持中性，按当前节点的任务输出。"

        summary_context = history.get_summary_context(
            last_n=2,
            exclude_turn_index=(history.current_turn_index if exclude_current_turn_from_recent else None),
        )
        if inspect.isawaitable(summary_context):
            if hasattr(summary_context, "close"):
                summary_context.close()
            summary_context = []

        recent = []
        for msg in summary_context:
            role = "客服" if msg.get("role") == "assistant" else "用户"
            recent.append(f"{role}: {msg.get('content', '')}")

        memory_sections = _render_memory_sections(memory)
        fast_judge_state = ""
        short_term_memory = ""
        sop_progress_candidates = ""
        if flow_state is not None:
            if flow_state.fast_judge_state is not None:
                fast_judge_state = flow_state.fast_judge_state.model_dump_json(exclude_none=True)
            if flow_state.short_term_memory.items:
                short_term_memory = "\n".join(f"- {item.text}" for item in flow_state.short_term_memory.items)
            if flow_state.sop_progress_state.field_candidates:
                # Speech lane: keep only the current node's field candidates,
                # and drop the node_id prefix (it's always the current node).
                # Other lanes (fast_judge, etc.) keep the full cross-node view.
                if lane == "speech":
                    scoped = [
                        item
                        for item in flow_state.sop_progress_state.field_candidates
                        if item.node_id == flow_state.current_node
                    ]
                    sop_progress_candidates = "\n".join(
                        f"- {item.field}: {item.value} ({item.status})" for item in scoped
                    )
                else:
                    sop_progress_candidates = "\n".join(
                        f"- {item.node_id}.{item.field}: {item.value} ({item.status})"
                        for item in flow_state.sop_progress_state.field_candidates
                    )
        draft = WeightedContextView(
            context_snapshot_id="",
            weighted_context_hash="",
            lane=lane,
            speaking_frame=_clip(frame, 500),
            next_speech_strategy=_clip(
                memory.next_speech_strategy.model_dump_json() if memory.next_speech_strategy else "无上一轮策略。",
                800,
            ),
            call_working_memory=memory_sections,
            recent_verbatim_window=_clip("\n".join(recent) or "无", 1200),
            rolling_call_summary=_clip(rolling_summary or "无", 1000),
            current_user_text=current_user_text,
            fast_judge_state=_clip(fast_judge_state, 1000),
            short_term_memory=_clip(short_term_memory, 1000),
            sop_progress_candidates=_clip(sop_progress_candidates, 1000),
            current_node=flow_state.current_node if flow_state is not None else "",
        )
        canonical_payload = draft.render_canonical_payload()
        digest = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
        return WeightedContextView(
            **{
                **draft.__dict__,
                "context_snapshot_id": f"ctx:{digest[:12]}",
                "weighted_context_hash": f"sha256:{digest}",
            }
        )
