from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from src.core.logging import get_logger
from src.core.pilot_guardrails import kb_divert_nonclosing_enabled
from src.flows.ablation_switches import (
    can_legacy_memory_extractor_mutate,
    is_state_machine_disabled,
)
from src.flows.models import TERMINAL_ROUTE, FieldStatus, FieldValue, JudgeResult
from src.flows.sop_decision_models import (
    CommittedSopField,
    FastJudgeEnvelope,
    FastJudgeFieldDelta,
    FastJudgeInternalIdentity,
    FastJudgeResult,
    FastJudgeSpeechDirective,
    FastJudgeState,
    MemoryPatch,
    NextSpeechStrategy,
    SpeechRisk,
)

if TYPE_CHECKING:
    from src.flows.runtime import FlowRuntime

logger = get_logger(__name__)

# Per-node follow-up budget. A required field the Judge never confirms (e.g. a
# patient who stays vague — "还行吧" — and never upgrades to the exact value)
# would otherwise loop the node forever: node_action stays "followup", the field
# is never satisfied, and downstream nodes are starved (the live H1 service
# stall). Once a node has consumed this many user turns without satisfying its
# required fields, the follow-up is force-advanced carrying partial fields. Set
# to mirror the knowledge path's MAX_FOLLOWUP_ROUNDS=2 (initial ask + 2 retries).
MAX_NODE_FOLLOWUP_TURNS = 3

STATUS_MAP = {
    "confirmed": FieldStatus.CONFIRMED,
    "inferred": FieldStatus.INFERRED,
    "refused": FieldStatus.REFUSED,
    "unknown": FieldStatus.UNKNOWN,
    "not_applicable": FieldStatus.NOT_APPLICABLE,
}


@dataclass(frozen=True)
class FieldCommit:
    field: str
    flow_value: FieldValue
    memory_value: CommittedSopField


@dataclass(frozen=True)
class NodeActionResult:
    advanced: bool
    node_before: str
    node_after: str
    action: str
    route_result: bool | str
    terminal_requested: bool = False
    followup_requested: bool = False
    followup_missing_fields: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CommitDecision:
    decision_id: str
    accepted: bool
    node_before: str
    node_after: str
    node_action: str
    field_commits: list[FieldCommit] = field(default_factory=list)
    memory_patch: MemoryPatch | None = None
    next_speech_strategy: NextSpeechStrategy | None = None
    speech_risks: list[SpeechRisk] = field(default_factory=list)
    append_allowed_candidate: bool = False
    reject_reason: str = ""
    idempotent: bool = False
    continuation_prefix: str = ""
    legacy_judge_result: JudgeResult = JudgeResult.EMPTY
    followup_missing_fields: list[str] = field(default_factory=list)
    stage2_should_start: bool = False
    stage2_target_node: str = ""
    stage2_reason: str = ""
    stage1_posture: str = "asked"  # mirrors FastJudgeResult.stage1_posture for trigger routing
    speech_directive: "FastJudgeSpeechDirective | None" = None


class StateCommitter:
    def __init__(self, flow: FlowRuntime) -> None:
        self._flow = flow

    def commit_fast_judge_result(
        self,
        envelope: FastJudgeEnvelope,
        *,
        expected: FastJudgeInternalIdentity,
    ) -> CommitDecision:
        result = envelope.result
        decision_id = f"fast_judge:{result.turn_id}:{result.speech_result_id}:{result.node_id}"
        if decision_id in self._flow._state.applied_sop_decision_ids:
            return CommitDecision(
                decision_id=decision_id,
                accepted=True,
                idempotent=True,
                node_before=result.node_id,
                node_after=self._flow._state.current_node,
                node_action=result.node_action,
            )

        reject = self._reject_fast_judge_reason(envelope.identity, expected, result)
        if reject:
            return CommitDecision(
                decision_id=decision_id,
                accepted=False,
                reject_reason=reject,
                node_before=result.node_id,
                node_after=self._flow._state.current_node,
                node_action="stay",
            )

        node = self._flow._current_node_config()
        commits: list[FieldCommit] = []
        committed_delta = []
        lookahead_committed_delta = []

        for candidate in result.field_delta:
            commit = self._field_commit_from_fast_judge_candidate(candidate, node_id=node.name, turn_id=result.turn_id)
            if commit is not None:
                commits.append(commit)
                committed_delta.append(candidate)

        for candidate in result.lookahead_field_delta:
            commit = self._field_commit_from_fast_judge_candidate(
                candidate,
                node_id=candidate.node_id,
                turn_id=result.turn_id,
            )
            if commit is not None:
                commits.append(commit)
                lookahead_committed_delta.append(candidate)

        normalized = self._normalize_fast_judge_result(result, field_commits=commits)
        memory = self._flow.call_memory.model_copy(deep=True)
        for commit in commits:
            memory.sop_fields[commit.field] = commit.memory_value
        memory.next_speech_strategy = NextSpeechStrategy(
            strategy_id=f"{result.turn_id}:fast_strategy",
            source_turn_id=result.turn_id,
            node_before=self._flow._state.current_node,
            node_after=self._flow._state.current_node,
            node_action=normalized.node_action,
            committed_delta=[],
            open_slots=[],
        )
        self._flow.apply_field_commits(decision_id, commits)
        self._flow.replace_call_memory(memory)
        self._flow._state.fast_judge_state = FastJudgeState(
            source_turn_id=normalized.turn_id,
            speech_result_id=normalized.speech_result_id,
            node_before=normalized.node_id,
            node_after=self._flow._state.current_node,
            evaluated_until=normalized.evaluated_until,
            committed_delta=committed_delta,
            lookahead_committed_delta=lookahead_committed_delta,
            missing_fields=normalized.missing_fields,
            answer_assessment=normalized.answer_assessment,
            node_action=normalized.node_action,
            next_node_goal=normalized.next_node_goal,
            stage2_should_start=normalized.stage2_should_start,
            stage2_target_node=normalized.stage2_target_node,
            stage2_reason=normalized.stage2_reason,
            handoff=normalized.handoff,
            knowledge_question=normalized.knowledge_question,
            speech_directive=normalized.speech_directive,
        )

        return CommitDecision(
            decision_id=decision_id,
            accepted=True,
            node_action=normalized.node_action,
            node_before=normalized.node_id,
            node_after=self._flow._state.current_node,
            field_commits=commits,
            next_speech_strategy=memory.next_speech_strategy,
            append_allowed_candidate=normalized.node_action in {"advance", "followup"},
            legacy_judge_result=JudgeResult(
                extracted_fields={},
                stage_complete=normalized.stage_complete,
                transfer_to_human=normalized.handoff.transfer_to_human,
                reason=normalized.answer_assessment.followup_reason,
                transfer_reason=normalized.handoff.reason,
                risk_level=normalized.handoff.risk_level,
                knowledge_query=normalized.knowledge_question.detected,
            ),
            followup_missing_fields=normalized.missing_fields,
            stage2_should_start=normalized.stage2_should_start,
            stage2_target_node=normalized.stage2_target_node,
            stage2_reason=normalized.stage2_reason,
            stage1_posture=normalized.stage1_posture,
            speech_directive=normalized.speech_directive,
        )

    def merge_memory_patch(self, patch: MemoryPatch) -> CommitDecision:
        patch_payload = json.dumps(patch.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        patch_hash = hashlib.sha256(patch_payload.encode("utf-8")).hexdigest()
        decision_id = f"memory:{patch.turn_id}:{patch.node_id}:{patch.source}:{patch_hash}"
        if not can_legacy_memory_extractor_mutate():
            return CommitDecision(
                decision_id=decision_id,
                accepted=False,
                reject_reason="legacy_memory_mutation_disabled",
                node_before=self._flow._state.current_node,
                node_after=self._flow._state.current_node,
                node_action="stay",
            )
        if decision_id in self._flow._state.applied_sop_decision_ids:
            return CommitDecision(
                decision_id=decision_id,
                accepted=True,
                idempotent=True,
                node_before=self._flow._state.current_node,
                node_after=self._flow._state.current_node,
                node_action="stay",
            )
        reject = self._reject_patch_reason(patch)
        if reject:
            return CommitDecision(
                decision_id=decision_id,
                accepted=False,
                reject_reason=reject,
                node_before=self._flow._state.current_node,
                node_after=self._flow._state.current_node,
                node_action="stay",
            )
        memory = self._flow.call_memory.model_copy(deep=True)
        if patch.speaking_frame is not None:
            memory.speaking_frame = patch.speaking_frame
        memory.sop_fields.update(patch.sop_fields)
        memory.pending_clarifications.extend(patch.pending_clarifications)
        memory.future_observations.extend(patch.future_observations)
        memory.do_not_repeat.extend(patch.do_not_repeat)
        memory.corrections.extend(patch.corrections)
        if patch.user_interaction_profile is not None:
            memory.user_interaction_profile = patch.user_interaction_profile
        memory.open_loops.extend(patch.open_loops)
        if patch.playback_truth is not None:
            memory.playback_truth = patch.playback_truth
        self._flow.replace_call_memory(memory)
        self._flow._state.applied_sop_decision_ids.add(decision_id)
        return CommitDecision(
            decision_id=decision_id,
            accepted=True,
            node_before=self._flow._state.current_node,
            node_after=self._flow._state.current_node,
            node_action="stay",
            memory_patch=patch,
        )

    def _reject_fast_judge_reason(
        self,
        actual: FastJudgeInternalIdentity,
        expected: FastJudgeInternalIdentity,
        result: FastJudgeResult,
    ) -> str:
        state = self._flow._state
        if result.turn_id != state.active_turn_id:
            return "turn_id_mismatch"
        for field_name in (
            "turn_id",
            "turn_index",
            "node_id",
            "flow_epoch",
            "context_snapshot_id",
            "speech_result_id",
            "speech_text_hash",
            "fast_judge_prompt_hash",
            "weighted_context_hash",
        ):
            if getattr(actual, field_name) != getattr(expected, field_name):
                return f"stale_{field_name}"
        if result.turn_id != expected.turn_id:
            return "result_turn_mismatch"
        if result.node_id != expected.node_id:
            return "result_node_mismatch"
        if result.speech_result_id != expected.speech_result_id:
            return "result_speech_result_mismatch"
        if expected.node_id != state.current_node:
            return "node_id_mismatch"
        if expected.flow_epoch != state.flow_epoch:
            return "flow_epoch_mismatch"
        return ""

    def _normalize_fast_judge_result(
        self,
        result: FastJudgeResult,
        *,
        field_commits: list[FieldCommit] | None = None,
    ) -> FastJudgeResult:
        normalized = result.model_copy(deep=True)
        field_commits = field_commits or []
        if not normalized.missing_fields:
            normalized.missing_fields = list(normalized.answer_assessment.missing_fields)
        if normalized.handoff.transfer_to_human:
            normalized.node_action = "handoff"
            normalized.stage_complete = False
            normalized.stage2_should_start = False
            return normalized
        # Spec 2026-05-26 §3 Change 4: gate non-closing divert behind the
        # rollout flag. Clearing `detected` keeps the runtime divert path
        # (runtime.py:997) and the derived `knowledge_query` (line ~186)
        # consistent; the `divert→stay` revert below then resets node_action.
        # Closing-node divert is never gated.
        if (
            normalized.knowledge_question.detected
            and self._flow._state.current_node != "closing"
            and not kb_divert_nonclosing_enabled()
        ):
            normalized.knowledge_question.detected = False
        if normalized.knowledge_question.detected and normalized.node_action not in {"divert", "handoff"}:
            normalized.node_action = "divert"
            normalized.stage_complete = False
            normalized.stage2_should_start = False
            return normalized
        if normalized.node_action == "divert" and not normalized.knowledge_question.detected:
            normalized.node_action = "stay"
            normalized.stage_complete = False
            normalized.stage2_should_start = False
            return normalized
        if normalized.node_action == "followup":
            # db22c85 extension: a follow-up whose commits already satisfy the
            # node's required fields is moot — promote it to advance (mirrors the
            # stay→advance promotion below) instead of looping. Surfaced by the
            # live insurance→closing stall where the completing delta (self_pay)
            # landed on this followup-labelled turn and the node never advanced.
            if self._required_fields_satisfied_with(field_commits):
                normalized.node_action = "advance"
                normalized.stage_complete = True
                normalized.answer_assessment.needs_followup = False
            elif self._flow._state.node_turn_count >= MAX_NODE_FOLLOWUP_TURNS:
                # Budget exhausted: stop looping on an unconfirmable required
                # field and force-advance carrying whatever partial fields were
                # collected, so the journey reaches downstream nodes instead of
                # starving them. The unmet field stays genuinely uncollected.
                logger.warning(
                    "flow.node_followup_budget_exhausted",
                    node=self._flow._state.current_node,
                    node_turn_count=self._flow._state.node_turn_count,
                    missing_fields=list(normalized.answer_assessment.missing_fields),
                )
                normalized.node_action = "advance"
                normalized.stage_complete = True
                normalized.answer_assessment.needs_followup = False
            else:
                normalized.stage_complete = False
                normalized.answer_assessment.needs_followup = True
                normalized.stage2_should_start = False
                if not normalized.missing_fields:
                    normalized.missing_fields = list(normalized.answer_assessment.missing_fields)
                return normalized
        if normalized.answer_assessment.needs_followup and normalized.node_action == "advance":
            if not self._required_fields_satisfied_with(field_commits):
                normalized.node_action = "followup"
                normalized.stage_complete = False
            return normalized
        if normalized.node_action == "advance" and not normalized.stage_complete:
            if self._required_fields_satisfied_with(field_commits):
                normalized.stage_complete = True
            else:
                normalized.node_action = "stay"
                normalized.stage2_should_start = False
                return normalized
        if normalized.node_action == "stay" and self._required_fields_satisfied_with(field_commits):
            normalized.node_action = "advance"
            normalized.stage_complete = True
        if normalized.node_action == "terminal" and (
            not normalized.stage_complete or not self._terminal_action_allowed()
        ):
            normalized.node_action = "stay"
            normalized.stage_complete = False
            normalized.stage2_should_start = False
            return normalized
        if normalized.node_action != "advance":
            normalized.stage2_should_start = False
        if normalized.stage2_target_node and normalized.stage2_target_node not in self._flow._agent.nodes:
            normalized.stage2_should_start = False
            normalized.stage2_target_node = ""
        return normalized

    def _field_commit_from_fast_judge_candidate(
        self,
        candidate: FastJudgeFieldDelta,
        *,
        node_id: str,
        turn_id: str,
    ) -> FieldCommit | None:
        node = self._flow._agent.nodes.get(node_id)
        if node is None:
            return None
        if node.allowed_fields and candidate.field not in node.allowed_fields:
            return None
        if candidate.status not in STATUS_MAP:
            return None
        # ABLATION: see docs/superpowers/specs/2026-04-15-thesis-ch4-eval-roadmap.md §4.6
        # ABLATION_DISABLE_SM=1 collapses the 5-level status taxonomy into binary
        # EMPTY/FILLED — only non-empty values persist, tagged CONFIRMED.
        sm_disabled = is_state_machine_disabled()
        v = candidate.value
        is_empty = v is None or v == "" or v == "null" or v == "None"
        if sm_disabled and is_empty:
            return None
        evidence = candidate.evidence_span or str(candidate.value or "")
        if candidate.status in {"confirmed", "inferred"} and not evidence:
            return None
        status_text = "confirmed" if sm_disabled else candidate.status
        fv = FieldValue(
            value=candidate.value,
            status=FieldStatus.CONFIRMED if sm_disabled else STATUS_MAP[candidate.status],
            source_node=node_id,
            source_turn=self._flow._state.turn_count,
        )
        committed = CommittedSopField(
            value=candidate.value,
            status=status_text,
            confidence=candidate.confidence,
            evidence=evidence,
            source_turn_id=turn_id,
            source_node=node_id,
            active=True,
        )
        return FieldCommit(field=candidate.field, flow_value=fv, memory_value=committed)

    def _required_fields_satisfied_with(self, field_commits: list[FieldCommit]) -> bool:
        node = self._flow._current_node_config()
        if node.completion_policy is None:
            return False
        prospective_fields = dict(self._flow._state.collected_fields)
        for commit in field_commits:
            prospective_fields[commit.field] = commit.flow_value
        prospective_state = replace(self._flow._state, collected_fields=prospective_fields)
        return node.completion_policy.unsatisfied_fields(prospective_state) == []

    def _terminal_action_allowed(self) -> bool:
        if self._flow.is_terminal_node():
            return True
        node = self._flow._current_node_config()
        if node.route_fn is None:
            return False
        return node.route_fn(self._flow._state) == TERMINAL_ROUTE

    def _reject_patch_reason(self, patch: MemoryPatch) -> str:
        state = self._flow._state
        if patch.turn_index < state.turn_count - 1:
            return "memory_patch_stale"
        if patch.flow_epoch != state.flow_epoch:
            return "flow_epoch_mismatch"
        return ""
