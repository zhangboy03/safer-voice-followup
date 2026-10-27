"""LLMProcessor: Speech-first + FastJudge-after-Speech architecture.

Lifecycle per turn:
  UserTurnCompleteFrame → consumed (not forwarded); launches background inference task
  LLMTextFrame          → emitted for each token from Speech LLM
  LLMResponseEndFrame   → emitted when playout is exhausted (after possible Stage 2)

Speech-first architecture:
  Speech LLM streams the user-facing response first. FastJudge runs only after
  Speech text exists, then commits fields and may trigger a Stage 2 Speech call
  in the same playout or a fresh playout when the previous context has ended.

Provisional/commit/rollback (user-side):
  - Preemptive turn  → history.append_user(text, provisional=True)
  - Non-preemptive   → history.append_user(text, provisional=False)
  - Same turn_id formal confirmation (text matches) → commit, no restart
    + fire deferred FastJudge if speech is already complete
  - Same turn_id formal confirmation (text differs) → cancel, rollback, restart
  - LLMProcessor does NOT commit assistant text (OutputTransport's job)

Cancellation:
  An InterruptionFrame or CancelFrame triggers cancel_inflight() → _cancel_inference().
  on_flush() additionally rolls back provisional history.

Flow integration:
  Uses FlowRuntime for context building, FastJudge message construction, field
  commit, completion checks, and transition management.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import Awaitable
from dataclasses import dataclass, is_dataclass
from dataclasses import fields as dataclass_fields
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Protocol

from src.backends.llm_dashscope import StreamEvent, StreamEventType
from src.core.config import settings
from src.core.logging import get_logger
from src.flows.ablation_switches import (
    is_bridge_phrase_enabled,
    is_judge_disabled,
)
from src.flows.fast_judge_runner import FastJudgeFailure, FastJudgeRunner
from src.flows.models import FieldStatus, JudgeResult
from src.flows.question_predicate import speech_asked_question
from src.flows.stage1_followup_gate import should_fire_stage1_followup
from src.flows.state_committer import CommitDecision, NodeActionResult, StateCommitter
from src.handoff.models import TransferPriority, TransferReason
from src.pipeline.events import JudgeDecisionEvent, TransferTriggeredEvent
from src.pipeline.frames import (
    BotStoppedSpeakingFrame,
    Frame,
    FrameDirection,
    InterruptionFrame,
    LLMResponseEndFrame,
    LLMTextFrame,
    SessionBootstrapFrame,
    UserStartedSpeakingFrame,
    UserTurnCompleteFrame,
)
from src.pipeline.processor import FrameProcessor
from src.processors.silent_rescue_state import SilentRescueState

if TYPE_CHECKING:
    from src.core.call_tracer import CallTracer
    from src.flows.memory_extractor import MemoryExtractorScheduler
    from src.flows.models import TurnPlan
    from src.flows.runtime import FlowRuntime
    from src.flows.short_term_memory import ShortTermMemoryScheduler
    from src.flows.sop_progress import SopProgressScheduler
    from src.session.conversation_history import ConversationHistory
    from src.session.lifecycle import SessionLifecycleHelper
    from src.session.playout_gate import PlayoutGate

SIPTransferFn = Callable[[str], Awaitable[None]]
DebugPublisher = Callable[[dict], None]

logger = get_logger(__name__)

# Large/nested fields that bloat server stdout — kept in the data-channel
# payload (frontend trace needs them) but stripped before logging.
_LOG_OMIT_KEYS = frozenset(
    {
        "prompt_messages",
        "response_text",
        "rescue_text",
        "text",
        "commit_decision",
        "judge_result",
        "turn_assessment",
        "extracted_fields",
        "collected_fields",
        "field_candidates",
        "pending_fields",
        "future_observations",
        "speech_risks",
        "knowledge_query_detail",
        "handoff",
        "payload",
        "contradictions",
    }
)


def _shrink_for_log(kwargs: dict) -> dict:
    """Drop large nested payloads and truncate long strings so server log stays readable."""
    out: dict = {}
    for k, v in kwargs.items():
        if k in _LOG_OMIT_KEYS:
            continue
        if isinstance(v, str) and len(v) > 200:
            out[k] = v[:200] + "…"
        else:
            out[k] = v
    return out


# ── Protocol ──────────────────────────────────────────────────────────────────


class LLMBackend(Protocol):
    def stream_with_tools(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        session_id: str = "",
        turn_id: str = "",
    ) -> AsyncIterator[StreamEvent]: ...


def _map_judge_transfer_reason(
    transfer_reason: str,
    risk_level: str,
) -> tuple[TransferReason, TransferPriority]:
    """Map Judge LLM output to handoff model enums."""
    match transfer_reason:
        case "user_request":
            return TransferReason.USER_REQUEST, TransferPriority.NORMAL
        case "emotional_escalation":
            return TransferReason.EMOTIONAL_DISTRESS, TransferPriority.HIGH
        case "medical_risk":
            if risk_level == "urgent":
                return TransferReason.SAFETY_SYMPTOM, TransferPriority.URGENT
            return TransferReason.MEDICAL_COMPLEX, TransferPriority.HIGH
        case _:
            return TransferReason.OTHER, TransferPriority.NORMAL


def _is_fast_judge_decision(decision: CommitDecision) -> bool:
    return decision.decision_id.startswith("fast_judge:")


def _fast_judge_should_run_stage2(
    decision: CommitDecision,
    node_action: NodeActionResult,
) -> tuple[bool, str]:
    """Decide whether to run a Stage 2 inference after the Fast Judge returns.

    Stage 2 always runs in a separate playout (its own ctx_id), so there is no
    longer a PlayoutGate `can_append` race — this function only answers
    "is Stage 2 semantically requested by Fast Judge / node_action?".
    """
    if node_action.followup_requested:
        return False, "fast_judge_followup_wait_for_user"
    if node_action.terminal_requested:
        return False, "terminal_requested"
    if decision.legacy_judge_result.transfer_to_human:
        return False, "handoff"
    if decision.legacy_judge_result.risk_level == "urgent":
        return False, "urgent_handoff_risk"
    if not node_action.advanced:
        return False, "no_advance"
    inferred_stage2 = (
        node_action.advanced
        and decision.node_action == "advance"
        and (not decision.stage2_target_node or decision.stage2_target_node == node_action.node_after)
    )
    if not decision.stage2_should_start and not inferred_stage2:
        return False, "fast_judge_stage2_not_requested"
    if decision.stage2_target_node and decision.stage2_target_node != node_action.node_after:
        return False, "fast_judge_stage2_target_mismatch"
    return True, "allowed"


def _fast_judge_stage1_already_asked_next_node(decision: CommitDecision) -> bool:
    directive_code = getattr(decision.speech_directive, "code", "") if decision.speech_directive else ""
    return decision.stage2_reason == "already_asked_next_node" or directive_code == "already_asked_next_node"


# Single source of truth for the stage1-followup gate, shared with the offline
# eval replay harness so their Stage 2 gating cannot drift. Re-exported under the
# historical private name to keep existing imports/tests working.
_should_fire_stage1_followup = should_fire_stage1_followup


def _force_advance_if_capped(decision, collectable_missing: list[str], node_id: str, cap: int, state) -> tuple:
    """If all collectable-missing fields are at cap, rewrite followup→advance.

    Route override only — does NOT synthesize FieldStatus values; skipped fields
    stay unresolved. Returns (decision, forced).
    """
    from dataclasses import replace

    if decision.node_action != "followup" or not collectable_missing:
        return decision, False
    if any(state.under_cap(node_id, f, cap) for f in collectable_missing):
        return decision, False
    forced = replace(
        decision,
        node_action="advance",
        legacy_judge_result=replace(decision.legacy_judge_result, stage_complete=True),
    )
    return forced, True


def _chunk_text(text: str, chunk_size: int = 20) -> list[str]:
    """Split text into chunks for LLMTextFrame streaming."""
    if len(text) <= chunk_size:
        return [text]
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _dump_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _dump_model(getattr(value, field.name)) for field in dataclass_fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _dump_model(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump_model(item) for item in value]
    return value


def _serialize_commit_decision(decision: CommitDecision) -> dict[str, Any]:
    return {
        "decision_id": decision.decision_id,
        "accepted": decision.accepted,
        "node_before": decision.node_before,
        "node_after": decision.node_after,
        "node_action": decision.node_action,
        "idempotent": decision.idempotent,
        "reject_reason": decision.reject_reason,
        "append_allowed_candidate": decision.append_allowed_candidate,
        "continuation_prefix": decision.continuation_prefix,
        "field_commits": [
            {
                "field": commit.field,
                "flow_value": _dump_model(commit.flow_value),
                "memory_value": _dump_model(commit.memory_value),
            }
            for commit in decision.field_commits
        ],
        "next_speech_strategy": _dump_model(decision.next_speech_strategy)
        if decision.next_speech_strategy is not None
        else None,
        "followup_missing_fields": list(decision.followup_missing_fields),
        "stage2_should_start": decision.stage2_should_start,
        "stage2_target_node": decision.stage2_target_node,
        "stage2_reason": decision.stage2_reason,
        "speech_risks": [_dump_model(risk) for risk in decision.speech_risks],
        "stage1_posture": decision.stage1_posture,
    }


@dataclass(frozen=True)
class SpeechInferenceResult:
    clean_text: str
    raw_text: str


# ── Processor ─────────────────────────────────────────────────────────────────


def resolve_speech_model_override(registry: Any, node_id: str) -> str | None:
    """Per-node speech model override for the speech call site.

    Returns the override model only if the registry yields a real non-empty
    string for this node's speech lane; otherwise None (use backend default).
    Tolerates a missing registry or a mock whose ``model_override`` returns a
    non-string, so the real-time loop never forwards a bogus ``model`` kwarg."""
    fn = getattr(registry, "model_override", None)
    if fn is None:
        return None
    value = fn(node_id, "speech")
    return value if isinstance(value, str) and value else None


class LLMProcessor(FrameProcessor):
    """Consumes UserTurnCompleteFrame and streams LLMTextFrame tokens.

    Speech LLM generates conversation tokens first; FastJudge extracts fields
    and checks stage completion after Speech text is available.
    """

    def __init__(
        self,
        name: str,
        *,
        llm_backend: LLMBackend,
        fast_judge_backend: LLMBackend | None = None,
        history: ConversationHistory,
        flow_runtime: FlowRuntime,
        turn_metrics=None,
        debug_publish: DebugPublisher | None = None,
        sip_transfer: SIPTransferFn | None = None,
        lifecycle: SessionLifecycleHelper | None = None,
        call_tracer: CallTracer | None = None,
        playout_gate: PlayoutGate | None = None,
        memory_scheduler: MemoryExtractorScheduler | None = None,
        short_term_memory_scheduler: ShortTermMemoryScheduler | None = None,
        sop_progress_scheduler: SopProgressScheduler | None = None,
    ) -> None:
        super().__init__(name)
        self._backend = llm_backend
        self._fast_judge_backend = fast_judge_backend
        self._history = history
        self._flow = flow_runtime
        self._sip_transfer = sip_transfer
        self._lifecycle = lifecycle
        self._call_tracer = call_tracer
        self._playout_gate = playout_gate
        self._memory_scheduler = memory_scheduler
        self._short_term_memory_scheduler = short_term_memory_scheduler
        self._sop_progress_scheduler = sop_progress_scheduler
        self._turn_metrics = turn_metrics
        self._debug_publish = debug_publish

        # Background inference state
        self._inference_task: asyncio.Task | None = None
        self._deferred_judge_task: asyncio.Task | None = None
        self._speculative_text: str = ""
        self._current_turn_id: str | None = None
        self._completed_speech_results: dict[str, SpeechInferenceResult] = {}
        self._inference_start_mono: float = 0.0
        self._ttft_recorded: bool = False  # only record TTFT for Stage 1, not Stage 2

        # Deferred judge: set when speculative→formal upgrade happens mid-speech
        self._judge_requested = asyncio.Event()

        # Transfer playout barrier
        self._transfer_playout_barrier: asyncio.Event | None = None
        self._transfer_playout_context_id: str = ""

        # Goodbye / terminal hook state
        self._speech_detected_goodbye = False
        self._pending_terminal_hook = False
        self._bootstrapped = False

        # ADR 0006 silent-rescue state
        self._silent_rescue_state = SilentRescueState()
        self._silent_rescue_task: asyncio.Task | None = None
        self._silent_rescue_current_playout_id: str | None = None

        # stage1-followup cap state (Tasks 9 + 12)
        from src.processors.stage1_followup_state import Stage1FollowupState

        self._stage1_followup_state = Stage1FollowupState()
        self._pending_stage1_followup: tuple | None = None  # (node_id, field, inference_id)
        # Task 12.2: user-speaking guard — set by _on_user_speech_signal, cleared at new-turn start
        self._user_speaking_since_last_playout: bool = False

        # Task 19: bind knowledge_qa orchestrator hooks to this processor's
        # TTS pipeline. Idempotent (last binding wins) — the orchestrator
        # module's hooks are process-level callables.
        self._bind_knowledge_orchestrator_hooks()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def setup(self) -> None:
        """Pre-warm LLM backend HTTP connection pools during pipeline setup."""
        backends = [self._backend, self._fast_judge_backend]
        warmup_coros = [b.warmup() for b in backends if b and hasattr(b, "warmup")]
        if warmup_coros:
            await asyncio.gather(*warmup_coros, return_exceptions=True)

    # ── Data frame handling ───────────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, UserTurnCompleteFrame):
            # Consume: do NOT forward downstream
            await self._handle_user_turn_flow(frame)
            return

        if isinstance(frame, SessionBootstrapFrame):
            await self._handle_session_bootstrap()
            return

        await self.push_frame(frame, direction)

    # ── Turn handling ────────────────────────────────────────────────────────

    async def _handle_user_turn_flow(self, frame: UserTurnCompleteFrame) -> None:
        """Speech-first flow path: Speech LLM, then FastJudge after Speech text exists."""
        turn_id = frame.turn_id
        user_text = frame.text
        is_speculative = frame.is_preemptive

        if turn_id == self._current_turn_id and self._inference_task is not None:
            # Same turn — formal confirmation
            if self._is_text_match(user_text):
                self._history.commit_provisional()
                self._flow.upgrade_to_formal()
                # Signal deferred judge if speech is still running
                if not self._inference_task.done():
                    self._judge_requested.set()
                else:
                    # Speech already done — run deferred judge as separate playout
                    self._deferred_judge_task = asyncio.create_task(self._run_deferred_judge(user_text, turn_id))
                return  # Don't restart inference
            else:
                await self._cancel_inference()
                self._history.rollback_provisional()
                self._history.append_user(user_text, provisional=False)
        else:
            await self._cancel_inference()
            self._current_turn_id = turn_id
            self._flow.begin_user_turn(turn_id)
            if is_speculative:
                self._history.append_user(user_text, provisional=True)
                self._speculative_text = user_text
            else:
                self._history.append_user(user_text, provisional=False)
                self._speculative_text = ""

        self._inference_task = asyncio.create_task(self._run_inference_flow(user_text, is_speculative))

    def _is_text_match(self, text: str) -> bool:
        """Compare incoming text with the provisional speculative text."""
        return text == self._speculative_text

    # ── ADR 0006 Silent-Rescue ───────────────────────────────────────────────

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        """Override to hook into our own LLMResponseEndFrame emits.

        Any LLMResponseEndFrame pushed by this processor (Stage 1 / Stage 2 /
        silent-rescue itself) means "AI is done speaking the current playout";
        schedule a silent-rescue timer unless a terminal/transfer is pending.

        Task 12.1: playout-aware cap increment for stage1-followup.
        When the LLMResponseEndFrame for the pending followup playout arrives,
        check the PlayoutGate snapshot for that context_id:
          - Not interrupted (or no gate): record_asked → consume retry.
          - Interrupted: clear pending WITHOUT recording → retry preserved.
        Fallback path: there is no per-playout playback-start callback wired
        from the transport back to this processor (mark_started fires in
        LivekitOutputTransport asynchronously). Using the LLMResponseEndFrame
        hook is the earliest in-processor signal that inference completed and
        the playout was submitted. Barge-in after inference but before first
        audio frame does NOT set mark_interrupted (transport checks _is_speaking),
        so those retries are also preserved via the _cancel_inference clear below.
        """
        await super().push_frame(frame, direction)
        if isinstance(frame, LLMResponseEndFrame):
            # Task 12.1: check pending followup cap lifecycle before silent-rescue scheduling
            ctx_id = frame.context_id or ""
            pending = self._pending_stage1_followup
            if pending is not None and ctx_id == pending[2]:
                node_id, field, _ = pending
                _snapshot_fn = getattr(self._playout_gate, "snapshot", None) if self._playout_gate else None
                interrupted = _snapshot_fn is not None and getattr(_snapshot_fn(ctx_id), "interrupted", False)
                if interrupted:
                    # Barge-in during streaming: playout was interrupted before/during
                    # playback → do NOT consume the retry.
                    self._emit_debug(
                        "stage1_followup_cap_skip",
                        node=node_id,
                        field=field,
                        reason="interrupted_before_playback",
                        playout_id=ctx_id,
                    )
                else:
                    self._stage1_followup_state.record_asked(node_id, field)
                    self._emit_debug(
                        "stage1_followup_cap_record",
                        node=node_id,
                        field=field,
                        count=self._stage1_followup_state._asked.get((node_id, field), 0),
                        playout_id=ctx_id,
                    )
                self._pending_stage1_followup = None

            if self._pending_terminal_hook or getattr(self, "_speech_detected_transfer", False):
                return
            self._schedule_silent_rescue(ctx_id)

    def _schedule_silent_rescue(self, playout_id: str) -> None:
        """ADR 0006 §4.1: schedule a silent-rescue task for the just-closed playout.

        Cancels any prior task. No-op if disabled or cap reached.
        """
        if not settings.silent_rescue_enabled:
            return
        if not self._silent_rescue_state.allowed(settings.silent_rescue_max_per_call):
            self._emit_debug(
                "silent_rescue_cap_reached",
                total=self._silent_rescue_state.total_count,
                cap=settings.silent_rescue_max_per_call,
            )
            return
        # Cancel any prior task before starting a new one
        self._cancel_silent_rescue()
        self._silent_rescue_current_playout_id = playout_id
        self._silent_rescue_task = asyncio.create_task(self._silent_rescue_pipeline(playout_id))
        self._emit_debug(
            "silent_rescue_timer_start",
            playout_id=playout_id,
            timeout_ms=settings.silent_rescue_timeout_ms,
        )

    def _cancel_silent_rescue(self) -> None:
        """Cancel any pending silent-rescue task. Safe to call anytime."""
        if self._silent_rescue_task is not None and not self._silent_rescue_task.done():
            self._silent_rescue_task.cancel()
        self._silent_rescue_task = None

    def _on_user_speech_signal(self) -> None:
        """ADR 0006 §4.3: VAD user-started-speaking → cancel timer + reset chain.
        Task 12.2: also set user-speaking guard so stage1-followup is suppressed."""
        had_task = self._silent_rescue_task is not None and not self._silent_rescue_task.done()
        self._cancel_silent_rescue()
        self._silent_rescue_state.on_user_speech()
        self._user_speaking_since_last_playout = True
        if had_task:
            self._emit_debug(
                "silent_rescue_timer_cancel",
                reason="user_started_speaking",
                playout_id=self._silent_rescue_current_playout_id,
            )

    async def _silent_rescue_pipeline(self, playout_id: str) -> None:
        """Wait for the audio source to ACTUALLY drain, then sleep timeout_ms, then fire.

        CosyVoice duplex streams audio into LiveKit's audio source faster than
        realtime, so ``TTSPlayoutDoneFrame`` / ``mark_playout_done`` fire when
        *generation* completes, not when the patient has finished *hearing* the
        audio. A server-side estimate (``playback_start + audio_duration``) runs
        short on TTS underruns / jitter-buffer lag, so the silence window opened
        while the AI was still talking and rescue spoke over its own sentence
        (production 2026-05-25). We now key off the PlayoutGate ``drained`` flag,
        which the transport sets from ``rtc.AudioSource.wait_for_playout()`` —
        ground truth for "every captured frame has been emitted". A small fixed
        margin then covers the network + earpiece jitter buffer the server can't
        observe, before the 5s silence window begins.
        """
        try:
            snapshot_fn = getattr(self._playout_gate, "snapshot", None) if self._playout_gate else None
            if snapshot_fn is not None:
                # Safety bound: even a long utterance + drain should comfortably
                # fit in 30s; if we exceed this the snapshot is stuck (e.g. a
                # missing drain callback), fall through rather than block forever.
                safety_deadline = time.monotonic() + 30.0
                while time.monotonic() < safety_deadline:
                    snap = snapshot_fn(playout_id)
                    if getattr(snap, "interrupted", False):
                        self._emit_debug(
                            "silent_rescue_timer_cancel",
                            reason="interrupted",
                            playout_id=playout_id,
                        )
                        return
                    if getattr(snap, "drained", False):
                        break
                    await asyncio.sleep(0.05)
                # Jitter-buffer margin beyond the server-side source drain.
                margin_ms = settings.silent_rescue_playout_margin_ms
                if margin_ms > 0:
                    await asyncio.sleep(margin_ms / 1000.0)
            # Stage B: silence window, anchored to actual playback drain.
            await asyncio.sleep(settings.silent_rescue_timeout_ms / 1000.0)
            # Stage C: fire
            await self._fire_silent_rescue(playout_id)
        except asyncio.CancelledError:
            return

    async def _fire_silent_rescue(self, playout_id: str) -> None:
        """ADR 0006 §4.2: re-run Stage 1 prompt with full conversation history.

        Speech model self-decides what to say (more empathy / pivot question /
        suggest patience). No directive, no post-check. Output flows through
        the standard playout path; its own LLMResponseEndFrame will re-arm timer.
        """
        # Last gate before speaking — defense-in-depth against a missed/late
        # cancel race (the timer task elapsed before _on_user_speech_signal or a
        # barge-in cancelled it). Never talk over a patient who is being handled.
        if self._playout_gate is not None and getattr(self._playout_gate.snapshot(playout_id), "interrupted", False):
            self._emit_debug("silent_rescue_skipped", reason="interrupted_at_fire", playout_id=playout_id)
            return
        if self._inference_task is not None and not self._inference_task.done():
            self._emit_debug("silent_rescue_skipped", reason="inference_in_flight", playout_id=playout_id)
            return
        current_node = self._flow._state.current_node
        plan = await self._flow.prepare_silent_rescue_plan(node_id=current_node)
        if plan is None:
            self._emit_debug("silent_rescue_skipped", reason="no_plan", node=current_node)
            return
        self._silent_rescue_state.record_trigger()
        silent_rescue_pid = f"{playout_id}:silent-rescue:{self._silent_rescue_state.total_count - 1}"
        self._emit_debug(
            "silent_rescue_fire",
            playout_id=playout_id,
            silent_rescue_playout_id=silent_rescue_pid,
            chain_length=self._silent_rescue_state.chain_length,
            total=self._silent_rescue_state.total_count,
            node=current_node,
        )
        result = await self._run_speech_inference(
            plan,
            silent_rescue_pid,
            latency_role="silent_rescue",
        )
        if result is None or not getattr(result, "raw_text", ""):
            self._emit_debug("silent_rescue_empty_output", silent_rescue_playout_id=silent_rescue_pid)
            return

    # ── stage1-followup (Task 9) ─────────────────────────────────────────────

    def _apply_stage1_followup_cap(self, decision, stage1_text: str = ""):
        """Force-advance rewrite — call right after the Judge returns and BEFORE
        apply_node_action, on BOTH foreground and deferred paths (spec §4.5)."""
        if not settings.stage1_followup_enabled or not decision.accepted:
            return decision
        # Part C2 (fail-closed): never force-advance over an unanswered Stage 1
        # question — the open-question invariant dominates force-advance.
        if settings.judge_advance_question_guard_enabled and speech_asked_question(stage1_text, interrogatives=True):
            self._emit_debug(
                "force_advance_suppressed_open_question",
                node=self._flow._state.current_node,
            )
            return decision
        collectable = self._flow.collectable_missing_fields()
        decision, forced = _force_advance_if_capped(
            decision,
            collectable,
            self._flow._state.current_node,
            settings.stage1_followup_max_per_field,
            self._stage1_followup_state,
        )
        if forced:
            self._emit_debug(
                "stage1_followup_force_advance",
                node=self._flow._state.current_node,
                fields=collectable,
                reason="followup_cap_exhausted",
            )
        return decision

    def _user_started_speaking_since_stage1(self) -> bool:
        # Task 12.2: backed by _user_speaking_since_last_playout, initialized in __init__
        return self._user_speaking_since_last_playout

    def _notify_field_commits_resolved(self, decision, node_id: str) -> None:
        """Task 12.4: for each field in an accepted decision that is now resolved
        (CONFIRMED / REFUSED / NOT_APPLICABLE), call on_field_resolved so the
        stage1-followup cap counter for that field is cleared.

        We use the field_commits from the CommitDecision (set by StateCommitter
        from the Judge envelope's field_delta). This is called right after
        _apply_stage1_followup_cap so the counters reflect the most recent state.

        Choice rationale: CommitDecision.field_commits lists every field that
        was actually committed by apply_field_commits — no need to diff
        collected_fields ourselves. INFERRED and UNKNOWN are not cleared
        because the spec defines "resolved" as CONFIRMED / accepted REFUSED /
        NOT_APPLICABLE (see CompletionPolicy.unsatisfied_fields).
        """
        _RESOLVED = {FieldStatus.CONFIRMED, FieldStatus.REFUSED, FieldStatus.NOT_APPLICABLE}
        if not decision.accepted:
            return
        for fc in decision.field_commits:
            if fc.flow_value.status in _RESOLVED:
                self._stage1_followup_state.on_field_resolved(node_id, fc.field)

    async def _maybe_prepare_stage1_followup(
        self,
        decision,
        node_id: str,
        continuation_prefix: str,
    ):
        if not settings.stage1_followup_enabled or not _is_fast_judge_decision(decision):
            return None

        collectable = self._flow.collectable_missing_fields()
        cap = settings.stage1_followup_max_per_field
        any_under_cap = any(self._stage1_followup_state.under_cap(node_id, f, cap) for f in collectable)

        posture = decision.stage1_posture  # m1: use field directly (not getattr)
        det_asked = speech_asked_question(continuation_prefix)
        fire, reason = _should_fire_stage1_followup(decision, collectable, any_under_cap, det_asked)
        self._emit_debug(
            "stage1_followup_decision",
            node=node_id,
            fire=fire,
            reason=reason,
            posture=posture,
            det_asked=det_asked,
            posture_vs_det_disagree=((posture == "closed_unasked") != det_asked),
            collectable=collectable,
        )

        if not fire:
            return None
        if self._user_started_speaking_since_stage1():
            self._emit_debug("stage1_followup_suppressed", node=node_id, reason="user_speaking")
            return None
        # I1: cap-race / StopIteration guard
        target = next(
            (f for f in collectable if self._stage1_followup_state.under_cap(node_id, f, cap)),
            None,
        )
        if target is None:
            self._emit_debug("stage1_followup_suppressed", node=node_id, reason="cap_race")
            return None
        # C1: use real CommitDecision field, not getattr fallback
        sd = decision.speech_directive
        directive = (sd.text if sd is not None else "") or ""
        self._cancel_silent_rescue()
        # I3: clear _pending_stage1_followup when no plan
        plan = await self._flow.prepare_stage1_followup(
            missing_fields=[target],
            directive=directive,
            continuation_prefix=continuation_prefix,
        )
        if plan is None:
            self._pending_stage1_followup = None
            return None
        self._pending_stage1_followup = (node_id, target, plan.inference_id)
        return plan

    # ── Dual-lane inference flow ─────────────────────────────────────────────

    def _emit_debug(self, event: str, *, turn_id: str | None = None, **kwargs) -> None:
        """Publish a debug event to the frontend via data channel AND server log.

        Frontend gets the full payload (needs prompt_messages / structured dicts to
        render). Server log gets a shrunk copy so stdout stays readable when
        debugging from terminal / tailing logs.
        """
        resolved_turn_id = turn_id or self._current_turn_id or ""
        if self._debug_publish is not None:
            payload = {
                "topic": "dual_llm_debug",
                "event": event,
                "turn_id": resolved_turn_id,
                "ts": time.time(),
                **kwargs,
            }
            self._debug_publish(payload)
        logger.info(f"agent.{event}", turn_id=resolved_turn_id, **_shrink_for_log(kwargs))
        if self._debug_publish is None:
            return
        if self._bus and event == "judge_done" and "error" not in kwargs:
            extracted = kwargs.get("extracted_fields", {})
            self._bus.emit(
                JudgeDecisionEvent(
                    turn_id=turn_id or self._current_turn_id or "",
                    node=kwargs.get("node", ""),
                    decision="complete" if kwargs.get("stage_complete") else "incomplete",
                    fields=extracted if isinstance(extracted, dict) else {},
                    confidence=0.0,
                    latency_ms=float(kwargs.get("latency_ms", -1)),
                )
            )

    def _emit_advance_decision(self, current_node: str, judge_result: JudgeResult, judge_ran: bool) -> None:
        """Emit full diagnostics for the advance/hold decision."""
        node_cfg = self._flow._current_node_config()
        state = self._flow._state

        collected = {k: {"value": v.value, "status": v.status.value} for k, v in state.collected_fields.items()}

        policy_satisfied = False
        if node_cfg.completion_policy is not None:
            policy_satisfied = node_cfg.completion_policy.is_satisfied(state)
        policy_missing = self._flow.get_missing_fields()

        judge_satisfied = policy_satisfied or judge_result.stage_complete
        will_advance = judge_satisfied

        self._emit_debug(
            "advance_decision",
            node=current_node,
            judge_stage_complete=judge_result.stage_complete,
            policy_satisfied=policy_satisfied,
            policy_missing=policy_missing,
            judge_ran=judge_ran,
            speech_advance=None,
            will_advance=will_advance,
            collected_fields=collected,
        )

    async def _run_inference_flow(self, user_text: str, is_speculative: bool) -> None:
        """Top-level: Speech runs first, FastJudge evaluates the completed
        Speech text, then optional Stage 2 runs.
        """
        try:
            # Task 12.2: reset user-speaking guard at the start of every new playout so
            # stage1-followup is not permanently suppressed across turns.
            self._user_speaking_since_last_playout = False
            plan = await self._flow.prepare_user_turn(user_text, is_speculative)
            playout_id = plan.inference_id
            self._ttft_recorded = False
            self._judge_requested.clear()

            current_node = self._flow._state.current_node

            # Emit: Speech LLM starts
            self._emit_debug(
                "speech_start",
                asr_text=user_text,
                node=current_node,
                is_speculative=is_speculative,
                prompt_messages=_prepare_debug_messages(plan.messages),
            )

            self._inference_start_mono = time.monotonic()
            if self._turn_metrics and self._current_turn_id:
                self._turn_metrics.record_llm_request_start(
                    turn_id=self._current_turn_id,
                    actual_text=user_text,
                )

            # Followup buffer: push a short phrase to TTS immediately before
            # Stage 1 starts, covering thinking-model latency (e.g. Flash+think512
            # takes 3-5s before emitting content tokens).  The buffer gives the
            # patient instant audio feedback ("好的，") while the LLM thinks.
            # ABLATION: see docs/superpowers/specs/2026-04-15-thesis-ch4-eval-roadmap.md §4.6
            # When BRIDGE_PHRASE_ENABLED=0, the TTFA B-experiment disables the pre-played
            # bridge audio; TTS then only starts when Speech emits its first token.
            node_cfg = self._flow._current_node_config()
            if node_cfg.followup_buffer and is_bridge_phrase_enabled():
                await self.push_frame(
                    LLMTextFrame(
                        text=node_cfg.followup_buffer,
                        turn_id=self._current_turn_id or "",
                        metadata={"context_id": playout_id, "latency_role": "bridge"},
                    ),
                    FrameDirection.DOWNSTREAM,
                )

            # Stage 1: Speech LLM streams tokens to TTS. FastJudge runs after this text exists.
            stage1_result = await self._run_speech_inference(plan, playout_id)
            if self._current_turn_id:
                self._completed_speech_results[self._current_turn_id] = stage1_result
            stage1_text = stage1_result.clean_text
            # Safety: strip any residual markers that leaked past detection
            # (e.g. leading whitespace caused _could_be_marker_prefix to fail)
            for _m in self._ALL_MARKERS:
                if _m in stage1_text:
                    stage1_text = stage1_text.replace(_m, "", 1).lstrip()

            # Emit: Speech LLM done
            self._emit_debug(
                "speech_done",
                node=current_node,
                response_text=stage1_result.raw_text,
                latency_ms=round((time.monotonic() - self._inference_start_mono) * 1000),
            )

            # Push Stage 1 end-of-playout immediately so the TTS session can
            # flush audio without waiting for Fast Judge to return. For very
            # short Stage 1 responses ("好的，感谢您的配合"), CosyVoice duplex
            # otherwise buffers until it receives `send_finish()`; pushing
            # LLMResponseEndFrame here unblocks the audio path. Stage 2 (if
            # any) runs in a separate playout with its own ctx_id, so we are
            # safe to close this one out now.
            await self.push_frame(
                LLMResponseEndFrame(
                    context_id=playout_id,
                    turn_id=self._current_turn_id or "",
                ),
                FrameDirection.DOWNSTREAM,
            )

            # Speech-detected transfer: [TRANSFER] marker found in response
            if self._speech_detected_transfer:
                cmd = self._flow.request_human_transfer(
                    reason=TransferReason.USER_REQUEST,
                    priority=TransferPriority.NORMAL,
                    trigger_text=user_text,
                )
                # Speech already said the transition message — skip TTS in execute
                await self._execute_human_transfer(cmd, skip_tts=True)
                return

            # Speech-detected goodbye: [GOODBYE] marker — direct terminal hook.
            # Stage 1 end frame already pushed above; just set the terminal hook.
            if self._speech_detected_goodbye:
                self._emit_debug("goodbye_detected", node=current_node, text=stage1_text)
                self._pending_terminal_hook = True
                return

            # Run Judge only after Speech text exists.
            judge_result = JudgeResult.EMPTY
            decision: CommitDecision | None = None
            node_cfg = self._flow._current_node_config()
            should_run_judge = not is_speculative or self._judge_requested.is_set()
            if should_run_judge and self._fast_judge_backend is not None and not is_judge_disabled():
                if node_cfg.field_definitions:
                    speech_result_id = f"{self._current_turn_id}:stage1"
                    decision = await self._run_fast_judge_with_timeout(
                        user_text=user_text,
                        assistant_text=stage1_result.raw_text,
                        speech_result_id=speech_result_id,
                    )
                    judge_result = decision.legacy_judge_result
                    decision = self._apply_stage1_followup_cap(decision, stage1_text=stage1_result.raw_text)
                    judge_result = decision.legacy_judge_result
                    # Task 12.4: clear cap counters for fields that are now resolved
                    self._notify_field_commits_resolved(decision, current_node)

                    if decision.accepted and judge_result.transfer_to_human:
                        reason, priority = _map_judge_transfer_reason(
                            judge_result.transfer_reason,
                            judge_result.risk_level,
                        )
                        cmd = self._flow.request_human_transfer(
                            reason=reason,
                            priority=priority,
                            trigger_text=user_text,
                        )
                        await self._execute_human_transfer(cmd)
                        return

            # Emit collected_fields snapshot + advance decision diagnostics
            self._emit_advance_decision(current_node, judge_result, decision is not None and decision.accepted)

            if decision is not None and not decision.accepted:
                self._emit_debug("judge_rejected", node=current_node, reason=decision.reject_reason)

            node_action = await self._flow.apply_node_action(decision) if decision and decision.accepted else None
            if node_action and node_action.terminal_requested:
                self._emit_debug("route_terminal", node=node_action.node_before)
                # Stage 1 end frame already pushed above; just set the terminal hook.
                self._pending_terminal_hook = True
                return

            # ADR 0006: ADR 0005 Judge-driven rescue dispatch was removed here.
            # Followup paths now go through unified Stage 1 prompt (共情 + 掰回
            # 问题) — see config/prompts/packs/sop-v1/speech/*.md. If Stage 1
            # output is empty or misses the followup question, silent-rescue
            # (ADR 0006 §4) catches it after 3s of patient silence.

            # Stage 1 LLMResponseEndFrame was already pushed right after the
            # Stage 1 inference returned. Stage 2 (if any) always runs in a
            # SEPARATE playout with its own ctx_id — there is no PlayoutGate
            # `can_append` race because the two playouts are independent.
            if node_action and (node_action.advanced or node_action.followup_requested):
                from src.flows.runtime import KNOWLEDGE_QA_NODE

                active_decision = decision
                assert active_decision is not None
                new_node = self._flow._state.current_node
                if node_action.advanced:
                    self._emit_debug(
                        "stage_advance",
                        old_node=current_node,
                        new_node=new_node,
                        reason="knowledge_divert" if new_node == KNOWLEDGE_QA_NODE else active_decision.node_action,
                    )
                    # Task 12.3: node changed → reset followup counters for the new node
                    self._stage1_followup_state.on_node_change(new_node)

                if _is_fast_judge_decision(active_decision):
                    should_run_stage2, stage2_skip_reason = _fast_judge_should_run_stage2(
                        active_decision,
                        node_action,
                    )
                else:
                    has_severe_context_risk = any(
                        risk.severity in {"high", "critical"} for risk in active_decision.speech_risks
                    )
                    should_run_stage2 = not has_severe_context_risk and (
                        node_action.advanced or node_action.followup_requested
                    )
                    stage2_skip_reason = "severe_speech_risk" if has_severe_context_risk else "allowed"

                if not should_run_stage2:
                    # Preserve the historical `same_playout_append_skipped`
                    # event name so the frontend trace UI ("Stage 2 未发声"
                    # panel) and existing tests keep working. The name is
                    # legacy — there is no longer a same-playout append; the
                    # event now means "Stage 2 inference was not run for the
                    # reason given".
                    self._emit_debug(
                        "same_playout_append_skipped",
                        node=new_node,
                        reason=stage2_skip_reason,
                    )

                stage1_already_asked_next_node = _fast_judge_stage1_already_asked_next_node(active_decision)

                if should_run_stage2:
                    post_commit_context = self._flow.build_weighted_context_view(
                        "", lane="speech", exclude_current_turn_from_recent=True
                    )
                    if node_action.followup_requested:
                        next_plan = await self._flow.prepare_followup(node_action.followup_missing_fields)
                    elif new_node == KNOWLEDGE_QA_NODE:
                        from src.core.pilot_guardrails import knowledge_qa_review_enabled

                        if knowledge_qa_review_enabled():
                            from src.flows.knowledge_turn import handle_knowledge_query_turn

                            history_msgs = (
                                self._history.build_messages() if hasattr(self._history, "build_messages") else []
                            )
                            kb_hash = getattr(self._flow, "kb_hash", "") or ""  # TODO Task 21: compute kb_hash
                            call_id = getattr(self._lifecycle, "_session_id", "") or "" if self._lifecycle else ""
                            await handle_knowledge_query_turn(
                                user_text=user_text,
                                history=history_msgs,
                                flow_state=self._flow._state,
                                kb_hash=kb_hash,
                                call_id=call_id,
                            )
                            # State machine emitted its own TTS; no Stage 2 follow-up needed.
                            next_plan = None
                        else:
                            next_plan = await self._flow.prepare_divert_entry(user_text=user_text)
                    else:
                        if (
                            settings.stage2_entry_prompt_enabled
                            and node_action.advanced
                            and not stage1_already_asked_next_node
                        ):
                            next_plan = await self._flow.prepare_stage2_entry(
                                previous_speech_text=stage1_result.raw_text,
                                user_text=user_text,
                            )
                        else:
                            next_plan = await self._flow.prepare_appended_segment(
                                active_decision,
                                post_commit_context,
                                previous_speech_text=stage1_result.raw_text,
                            )
                else:
                    next_plan = await self._maybe_prepare_stage1_followup(
                        decision=active_decision,
                        node_id=new_node,
                        continuation_prefix=stage1_result.raw_text,
                    )

                if next_plan:
                    stage2_context_id = next_plan.inference_id
                    self._emit_debug(
                        "stage2_start",
                        node=new_node,
                        separate_playout=True,
                        prompt_messages=_prepare_debug_messages(next_plan.messages),
                    )
                    stage2_start = time.monotonic()
                    stage2_result = await self._run_speech_inference(next_plan, stage2_context_id)
                    self._emit_debug(
                        "stage2_done",
                        node=new_node,
                        response_text=stage2_result.raw_text,
                        latency_ms=round((time.monotonic() - stage2_start) * 1000),
                    )
                    # Mark the new node's question as already asked by Stage 2
                    # so the next turn's Stage 1 (running in the same node)
                    # doesn't ask it again. Only set when we actually advanced
                    # into a fresh node — followup keeps the same node;
                    # divert hands off to knowledge_qa.
                    # Only lock the node as "already asked" if Stage 2 actually
                    # asked a question; a bridge-only Stage 2 would otherwise
                    # permanently suppress the next Stage 1 from ever asking it.
                    stage2_asked_question = "？" in stage2_result.raw_text or "?" in stage2_result.raw_text
                    if (
                        node_action.advanced
                        and not node_action.followup_requested
                        and new_node != KNOWLEDGE_QA_NODE
                        and (stage2_asked_question or stage1_already_asked_next_node)
                    ):
                        self._flow._state.node_question_already_asked[new_node] = True
                    await self.push_frame(
                        LLMResponseEndFrame(
                            context_id=stage2_context_id,
                            turn_id=self._current_turn_id or "",
                        ),
                        FrameDirection.DOWNSTREAM,
                    )

                if self._flow.is_terminal_node():
                    self._pending_terminal_hook = True

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("llm.inference_flow_error", error=str(e))
            cmd = await self._flow.force_transfer_to_human(str(e))
            await self._execute_human_transfer(cmd)

    _TRANSFER_MARKER = "[TRANSFER]"
    _GOODBYE_MARKER = "[GOODBYE]"
    # Markers that set a flag via setattr when detected
    _MARKERS = {
        _TRANSFER_MARKER: "_speech_detected_transfer",
        _GOODBYE_MARKER: "_speech_detected_goodbye",
    }
    _ALL_MARKERS = list(_MARKERS.keys())

    async def _run_speech_inference(
        self,
        plan: TurnPlan,
        playout_id: str,
        *,
        latency_role: str = "speech",
    ) -> SpeechInferenceResult:
        """Stream Speech LLM tokens to TTS.

        If the response starts with [TRANSFER] or [GOODBYE], sets the corresponding
        flag and strips the marker before sending to TTS. Only buffers the first
        few tokens for marker detection; all subsequent tokens stream normally.

        ``latency_role`` is stamped on every emitted ``LLMTextFrame.metadata``;
        defaults to ``"speech"`` for Stage 1 / Stage 2 callers. The silent-rescue
        path (ADR 0006) passes ``"silent_rescue"``.
        """
        accumulated: list[str] = []
        raw_accumulated: list[str] = []
        first_token = True
        self._speech_detected_transfer = False
        self._speech_detected_goodbye = False
        _FLAG_MARKERS = self._MARKERS
        _ALL = self._ALL_MARKERS

        _prefix_buf: list[str] = []
        _marker_checked = False

        async def _push_text(text: str) -> None:
            await self.push_frame(
                LLMTextFrame(
                    text=text,
                    turn_id=self._current_turn_id or "",
                    metadata={"context_id": playout_id, "latency_role": latency_role},
                ),
                FrameDirection.DOWNSTREAM,
            )

        async def _flush_prefix_buf() -> None:
            """Marker not found — flush buffered tokens individually."""
            nonlocal _marker_checked
            _marker_checked = True
            for t in _prefix_buf:
                accumulated.append(t)
                await _push_text(t)
            _prefix_buf.clear()

        def _check_markers(joined: str) -> str | None:
            """Check if any marker is present in joined text. Returns marker or None."""
            for marker in _ALL:
                if marker in joined:
                    return marker
            return None

        def _could_be_marker_prefix(joined: str) -> bool:
            """Check if joined text could still be a prefix of any marker.

            Leading whitespace is ignored: LLMs sometimes emit \\n or spaces
            before the marker token, which must not abort marker detection.
            """
            stripped = joined.lstrip()
            if not stripped:
                return True  # Only whitespace so far — may still precede a marker
            return any(marker.startswith(stripped) for marker in _ALL)

        # Per-node speech model override (e.g. knowledge_qa → a stronger model).
        # Resolved from the prompt pack's node_overrides; only a real str override is
        # forwarded — None / mock registries fall back to the backend's default model,
        # and we DON'T pass model= at all then, so backends lacking the kwarg are unaffected.
        _speech_model = resolve_speech_model_override(getattr(self._flow, "_prompt_registry", None), plan.node_id)
        _stream_kwargs: dict = {"tools": None, "session_id": "", "turn_id": self._current_turn_id or ""}
        if _speech_model is not None:
            _stream_kwargs["model"] = _speech_model

        async for event in self._backend.stream_with_tools(plan.messages, **_stream_kwargs):
            if event.type == StreamEventType.TEXT and event.text:
                raw_accumulated.append(event.text)
                if first_token:
                    first_token = False
                    if not self._ttft_recorded and self._turn_metrics and self._current_turn_id:
                        ttft_ms = (time.monotonic() - self._inference_start_mono) * 1000
                        self._turn_metrics.record_llm_ttft(turn_id=self._current_turn_id, ttft_ms=ttft_ms)
                        self._ttft_recorded = True

                if not _marker_checked:
                    _prefix_buf.append(event.text)
                    joined = "".join(_prefix_buf)
                    found_marker = _check_markers(joined)
                    if found_marker is not None:
                        attr = _FLAG_MARKERS[found_marker]
                        setattr(self, attr, True)
                        _marker_checked = True
                        clean = joined.replace(found_marker, "", 1).lstrip()
                        _prefix_buf.clear()
                        if clean:
                            accumulated.append(clean)
                            await _push_text(clean)
                    elif not _could_be_marker_prefix(joined):
                        await _flush_prefix_buf()
                else:
                    accumulated.append(event.text)
                    await _push_text(event.text)

            elif event.type == StreamEventType.DONE:
                break

        # Flush any remaining buffered prefix (very short response without marker)
        if not _marker_checked and _prefix_buf:
            joined = "".join(_prefix_buf)
            found_marker = _check_markers(joined)
            if found_marker is not None:
                setattr(self, _FLAG_MARKERS[found_marker], True)
                joined = joined.replace(found_marker, "", 1).lstrip()
            if joined:
                accumulated.append(joined)
                await _push_text(joined)

        return SpeechInferenceResult(
            clean_text="".join(accumulated),
            raw_text="".join(raw_accumulated),
        )

    async def _run_fast_judge_with_timeout(
        self,
        *,
        user_text: str,
        assistant_text: str,
        speech_result_id: str,
    ) -> CommitDecision:
        """Run Fast Judge after Speech and commit through StateCommitter."""
        fast_judge_start_mono = time.monotonic()
        origin_turn_id = self._current_turn_id or self._flow._state.active_turn_id or ""
        node_name = self._flow._state.current_node

        try:
            messages, identity = self._flow.build_fast_judge_messages(
                user_text=user_text,
                assistant_text=assistant_text,
                speech_result_id=speech_result_id,
            )
        except Exception as exc:
            logger.warning("llm.fast_judge_prompt_failed", turn_id=origin_turn_id, error=str(exc))
            return CommitDecision(
                decision_id=f"fast_judge:{origin_turn_id}:{speech_result_id}:{node_name}",
                accepted=False,
                reject_reason="prompt_failed",
                node_before=node_name,
                node_after=self._flow._state.current_node,
                node_action="stay",
            )

        self._emit_debug(
            "fast_judge_start",
            node=identity.node_id,
            asr_text=user_text,
            assistant_text=assistant_text,
            speech_result_id=speech_result_id,
            prompt_messages=_prepare_debug_messages(messages),
        )

        try:
            runner = FastJudgeRunner(
                backend=self._fast_judge_backend,
                model=settings.fast_judge_model,
                max_tokens=settings.fast_judge_max_tokens,
                temperature=settings.fast_judge_temperature,
            )
            envelope = await asyncio.wait_for(
                runner.run(
                    messages=messages,
                    identity=identity,
                    session_id="",
                    turn_id=identity.turn_id,
                    stage1_text=assistant_text,
                ),
                timeout=settings.fast_judge_timeout_s,
            )
            decision = StateCommitter(self._flow).commit_fast_judge_result(envelope, expected=identity)
        except asyncio.TimeoutError:
            logger.warning("llm.fast_judge_timeout", turn_id=origin_turn_id)
            self._emit_debug("fast_judge_done", node=identity.node_id, error="timeout")
            return CommitDecision(
                decision_id=f"fast_judge:{identity.turn_id}:{identity.speech_result_id}:{identity.node_id}",
                accepted=False,
                reject_reason="timeout",
                node_before=identity.node_id,
                node_after=self._flow._state.current_node,
                node_action="stay",
            )
        except FastJudgeFailure as exc:
            logger.warning("llm.fast_judge_failed", turn_id=origin_turn_id, reason=str(exc))
            self._emit_debug(
                "fast_judge_done", node=identity.node_id, error=str(exc), response_text=runner.last_raw_text
            )
            return CommitDecision(
                decision_id=f"fast_judge:{identity.turn_id}:{identity.speech_result_id}:{identity.node_id}",
                accepted=False,
                reject_reason=str(exc),
                node_before=identity.node_id,
                node_after=self._flow._state.current_node,
                node_action="stay",
            )

        if decision.accepted:
            result = decision.legacy_judge_result
            # last_raw_text is the raw LLM output (pre-mapping); v2 prompt has
            # the LLM write top-level `reason`, while reason_source is set by
            # _prepare_fast_judge_payload after parsing. Emit reason_source
            # and the post-mapping judge_result so frontend trace and
            # scripts/fast_judge_v2_shadow_eval.py can read what actually
            # shipped, not just what the model wrote.
            self._emit_debug(
                "fast_judge_done",
                node=identity.node_id,
                response_text=getattr(runner, "last_raw_text", "") or envelope.result.model_dump_json(),
                judge_result=envelope.result.model_dump(mode="json"),
                reason_source=envelope.result.answer_assessment.reason_source,
                stage_complete=result.stage_complete,
                transfer_to_human=result.transfer_to_human,
                knowledge_query=result.knowledge_query,
                node_action=decision.node_action,
                reason=result.reason,
                stage2_should_start=decision.stage2_should_start,
                stage2_target_node=decision.stage2_target_node,
                stage2_reason=decision.stage2_reason,
                cache_usage=getattr(runner, "last_usage", None),
                commit_decision=_serialize_commit_decision(decision),
                turn_assessment={
                    "recommended_action": decision.node_action,
                    "stage_complete": result.stage_complete,
                    "reason": result.reason,
                },
                latency_ms=round((time.monotonic() - fast_judge_start_mono) * 1000),
            )
        return decision

    async def _run_deferred_judge(self, user_text: str, origin_turn_id: str) -> None:
        """Late-formal path: speech already finished, TTS session already closed.
        Stage 2 (if needed) plays as a SEPARATE utterance — degraded but acceptable.
        """
        # ABLATION: see docs/superpowers/specs/2026-04-15-thesis-ch4-eval-roadmap.md §4.6
        # When ABLATION_DISABLE_JUDGE=1 the deferred fast-judge is a no-op; Speech LLM
        # is the sole source of field extraction / advance decisions.
        if is_judge_disabled():
            return
        if self._fast_judge_backend is None:
            return
        node = self._flow._current_node_config()
        if not node.field_definitions:
            return

        speech_result = self._completed_speech_results.get(origin_turn_id)
        if speech_result is None:
            logger.warning("llm.deferred_judge_missing_speech", turn_id=origin_turn_id)
            return

        speech_result_id = f"{origin_turn_id}:stage1"
        decision = await self._run_fast_judge_with_timeout(
            user_text=user_text,
            assistant_text=speech_result.raw_text,
            speech_result_id=speech_result_id,
        )
        judge_result = decision.legacy_judge_result

        # Stale-turn guard: if a new turn started while judge was running, discard
        if self._current_turn_id != origin_turn_id:
            logger.info("llm.deferred_judge_stale", origin=origin_turn_id, current=self._current_turn_id)
            return

        # Apply force-advance cap BEFORE emitting the advance_decision event so that
        # the debug event truthfully reflects the rewritten decision (I1 fix).
        # _apply_stage1_followup_cap no-ops when decision.accepted is False, so calling
        # it here — before the accepted guard below — is safe.
        decision = self._apply_stage1_followup_cap(decision, stage1_text=speech_result.raw_text)
        judge_result = decision.legacy_judge_result
        # Task 12.4 (deferred path): clear cap counters for fields that are now resolved
        self._notify_field_commits_resolved(decision, node.name)

        self._emit_advance_decision(node.name, judge_result, decision.accepted)

        if not decision.accepted:
            self._emit_debug("judge_rejected", turn_id=origin_turn_id, node=node.name, reason=decision.reject_reason)
            return

        if judge_result.transfer_to_human:
            reason, priority = _map_judge_transfer_reason(
                judge_result.transfer_reason,
                judge_result.risk_level,
            )
            cmd = self._flow.request_human_transfer(
                reason=reason,
                priority=priority,
                trigger_text=user_text,
            )
            await self._execute_human_transfer(cmd)
            return

        node_action = await self._flow.apply_node_action(decision)
        if node_action.terminal_requested:
            self._emit_debug("route_terminal_deferred", node=node_action.node_before)
            self._pending_terminal_hook = True
            await self._execute_terminal_hook()
            return
        if self._flow.is_terminal_node() and not (settings.stage2_entry_prompt_enabled and node_action.advanced):
            self._emit_debug("route_terminal_deferred", node=self._flow._state.current_node)
            self._pending_terminal_hook = True
            await self._execute_terminal_hook()
            return
        if node_action.advanced:
            from src.flows.runtime import KNOWLEDGE_QA_NODE

            playout_id = str(uuid.uuid4())
            new_node = self._flow._state.current_node
            # Task 12.3 (deferred path): node changed → reset followup counters
            self._stage1_followup_state.on_node_change(new_node)
            stage1_already_asked_next_node = _fast_judge_stage1_already_asked_next_node(decision)

            if new_node == KNOWLEDGE_QA_NODE:
                from src.core.pilot_guardrails import knowledge_qa_review_enabled

                if knowledge_qa_review_enabled():
                    from src.flows.knowledge_turn import handle_knowledge_query_turn

                    history_msgs = self._history.build_messages() if hasattr(self._history, "build_messages") else []
                    kb_hash = getattr(self._flow, "kb_hash", "") or ""  # TODO Task 21: compute kb_hash
                    call_id = getattr(self._lifecycle, "_session_id", "") or "" if self._lifecycle else ""
                    await handle_knowledge_query_turn(
                        user_text=user_text,
                        history=history_msgs,
                        flow_state=self._flow._state,
                        kb_hash=kb_hash,
                        call_id=call_id,
                    )
                    # State machine emitted its own TTS; skip the legacy Stage 2 path.
                    next_plan = None
                else:
                    next_plan = await self._flow.prepare_divert_entry(user_text=user_text)
            else:
                if settings.stage2_entry_prompt_enabled and not stage1_already_asked_next_node:
                    next_plan = await self._flow.prepare_stage2_entry(
                        previous_speech_text=speech_result.raw_text,
                    )
                else:
                    next_plan = await self._flow.prepare_node_entry(continuation_prefix=speech_result.raw_text)

            if next_plan:
                self._flow.remap_context_node(playout_id)
                self._emit_debug(
                    "stage2_start",
                    node=new_node,
                    separate_playout=True,
                    prompt_messages=_prepare_debug_messages(next_plan.messages),
                )
                stage2_start = time.monotonic()
                stage2_result = await self._run_speech_inference(next_plan, playout_id)
                self._emit_debug(
                    "stage2_done",
                    node=new_node,
                    response_text=stage2_result.raw_text,
                    latency_ms=round((time.monotonic() - stage2_start) * 1000),
                )
                # Only lock the node as "already asked" if Stage 2 actually asked a
                # question; a bridge-only Stage 2 would otherwise permanently suppress
                # the next Stage 1 from ever asking it. (Mirrors the fast-judge path.)
                stage2_asked_question = "？" in stage2_result.raw_text or "?" in stage2_result.raw_text
                if new_node != KNOWLEDGE_QA_NODE and (stage2_asked_question or stage1_already_asked_next_node):
                    self._flow._state.node_question_already_asked[new_node] = True
                if self._flow.is_terminal_node():
                    self._pending_terminal_hook = True
                await self.push_frame(
                    LLMResponseEndFrame(
                        context_id=playout_id,
                        turn_id=self._current_turn_id or "",
                    ),
                    FrameDirection.DOWNSTREAM,
                )
        elif node_action.followup_requested:
            new_node = self._flow._state.current_node
            next_plan = await self._maybe_prepare_stage1_followup(
                decision=decision,
                node_id=new_node,
                continuation_prefix=speech_result.raw_text,
            )
            if next_plan:
                ctx_id = next_plan.inference_id
                self._flow.remap_context_node(ctx_id)
                await self._run_speech_inference(next_plan, ctx_id)
                await self.push_frame(
                    LLMResponseEndFrame(context_id=ctx_id, turn_id=self._current_turn_id or ""),
                    FrameDirection.DOWNSTREAM,
                )

    # ── Task 19: knowledge_qa orchestrator hooks ──────────────────────────────

    def _bind_knowledge_orchestrator_hooks(self) -> None:
        """Replace src.flows.knowledge_turn module-level hooks with bound impls
        that route through this LLMProcessor's TTS pipeline.

        Called once at __init__ — the hooks are module-level callables shared
        across the process; rebinding is idempotent (last binding wins).
        """
        from src.flows import knowledge_turn

        knowledge_turn._emit_bridge_phrase = self._knowledge_emit_bridge
        knowledge_turn._emit_refuse_phrase = self._knowledge_emit_refuse
        knowledge_turn._emit_escalation_phrase = self._knowledge_emit_escalation
        knowledge_turn._run_speech_followup = self._knowledge_speech_followup
        knowledge_turn._run_speech_answer = self._knowledge_speech_answer
        knowledge_turn._run_speech_exit_check = self._knowledge_speech_exit_check
        knowledge_turn._resume = self._knowledge_resume
        knowledge_turn._get_review_runner = self._knowledge_get_review_runner

    async def _knowledge_emit_bridge(self, flow_state, text: str) -> None:
        """Knowledge-QA Bridge phrase: e.g. '我帮您查看一下，请稍等'."""
        logger.info("knowledge_emit_bridge text=%s", text)
        playout_id = f"{self._current_turn_id or 'unknown'}.knowledge_bridge"
        await self._emit_knowledge_phrase(text, playout_id)

    async def _knowledge_emit_refuse(self, flow_state, phrase_key: str) -> None:
        """Knowledge-QA [B1] refuse phrase from phrase_bank by key."""
        from src.prompts.phrase_bank import load_knowledge_phrases

        phrase = load_knowledge_phrases()[phrase_key]
        logger.info("knowledge_emit_refuse phrase_key=%s", phrase_key)
        playout_id = f"{self._current_turn_id or 'unknown'}.knowledge_refuse.{phrase_key}"
        await self._emit_knowledge_phrase(phrase, playout_id)

    async def _knowledge_emit_escalation(self, flow_state, phrase_key: str) -> None:
        """Knowledge-QA escalation phrase (Type E1/E2 from phrase_bank)."""
        from src.prompts.phrase_bank import load_knowledge_phrases

        phrase = load_knowledge_phrases()[phrase_key]
        logger.info("knowledge_emit_escalation phrase_key=%s", phrase_key)
        playout_id = f"{self._current_turn_id or 'unknown'}.knowledge_escalate.{phrase_key}"
        await self._emit_knowledge_phrase(phrase, playout_id)

    async def _knowledge_speech_followup(self, flow_state, missing_prereqs: list, on_audio_start) -> None:
        """[B2] followup question. For Task 19, we use a deterministic fallback
        question built from the slots — the [B2] LLM call lives in a follow-up
        revision (the validator + fallback are wired now to keep the OFF path
        free of LLM regressions while we ship the safety-critical refuse path
        first).
        """
        from src.flows.knowledge_validator import _fallback_question

        text = _fallback_question(missing_prereqs)
        logger.info("knowledge_speech_followup missing=%s text=%s", missing_prereqs, text)
        playout_id = f"{self._current_turn_id or 'unknown'}.knowledge_followup"
        await self._emit_knowledge_phrase(text, playout_id)
        if on_audio_start is not None:
            on_audio_start()

    async def _knowledge_speech_answer(self, flow_state, user_text: str, facts) -> None:
        """[D] answer Speech: delegate to the existing knowledge_qa Speech path.

        Reuses ``_flow.prepare_divert_entry(user_text)`` so the existing
        Qwen3.6-Flash + full-KB pipeline runs unchanged. This is the [D] in
        the spec — only its prompt changed (Task 10), not the mechanism.
        """
        logger.info("knowledge_speech_answer user_text=%s", user_text[:80])
        next_plan = await self._flow.prepare_divert_entry(user_text=user_text)
        if next_plan is None:
            logger.warning("knowledge_speech_answer: prepare_divert_entry returned None")
            return
        next_context_id = next_plan.inference_id
        stage2_result = await self._run_speech_inference(next_plan, next_context_id)
        self._emit_debug(
            "knowledge_speech_answer.done",
            response_text=getattr(stage2_result, "raw_text", ""),
        )

    async def _knowledge_speech_exit_check(self, flow_state) -> None:
        """[E] Exit-check phrase: fixed '您这个问题还有不清楚的吗？'."""
        text = "您这个问题还有不清楚的吗？"
        logger.info("knowledge_speech_exit_check")
        playout_id = f"{self._current_turn_id or 'unknown'}.knowledge_exit_check"
        await self._emit_knowledge_phrase(text, playout_id)

    async def _knowledge_resume(self, flow_state) -> None:
        """RESUME: close the current KnowledgeSession and let the existing flow
        runtime mechanism return control to the suspended SOP node on the next
        user turn. Do NOT decrement sessions_in_current_suspension (cap counts
        are per-suspension lifetime, not per-active-session).
        """
        rs = flow_state.knowledge_review_state
        if rs is not None and rs.current_session is not None:
            rs.current_session.closed = True
            rs.current_session = None
        logger.info("knowledge_resume: session closed, awaiting next user turn for SOP resume")

    def _knowledge_get_review_runner(self):
        """Singleton KnowledgeReviewRunner.from_manifest(backend=self._backend).

        The Speech LLMBackend at self._backend is qwen3.6-flash; the [A] runner
        config-loads the review model id from defaults.review BUT needs an
        HTTPX-style backend. For the pilot, reuse self._backend (DashScope
        OpenAI-compatible), and let the runner override the model string when
        calling .complete_structured.
        """
        if not hasattr(self, "_knowledge_review_runner_singleton"):
            from src.flows.knowledge_review_runner import KnowledgeReviewRunner

            self._knowledge_review_runner_singleton = KnowledgeReviewRunner.from_manifest(
                backend=self._backend,
            )
        return self._knowledge_review_runner_singleton

    async def _emit_knowledge_phrase(
        self,
        directive_text: str,
        playout_id: str,
    ) -> str:
        """TTS a deterministic knowledge-QA phrase verbatim, bypassing Speech LLM."""
        cleaned = directive_text or ""
        for marker in self._ALL_MARKERS:
            cleaned = cleaned.replace(marker, "")
        cleaned = cleaned.strip()
        if not cleaned:
            self._emit_debug("knowledge_phrase_empty", playout_id=playout_id)
            return "handoff_empty_directive"

        for chunk in _chunk_text(cleaned, chunk_size=20):
            await self.push_frame(
                LLMTextFrame(
                    text=chunk,
                    turn_id=self._current_turn_id or "",
                    metadata={"context_id": playout_id, "latency_role": "knowledge_phrase"},
                ),
                FrameDirection.DOWNSTREAM,
            )
        return "ok"

    async def _cancel_inference(self) -> None:
        """Cancel background inference + deferred judge tasks if running.

        Task 12.1: also clears _pending_stage1_followup so a barge-in that
        cancels the inference task before the LLMResponseEndFrame is pushed
        (i.e. before the playout-aware cap hook in push_frame fires) leaves
        the field under cap — retry preserved.
        """
        if self._inference_task is not None and not self._inference_task.done():
            self._inference_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._inference_task
        self._inference_task = None

        if self._deferred_judge_task is not None and not self._deferred_judge_task.done():
            self._deferred_judge_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._deferred_judge_task
        self._deferred_judge_task = None

        # Clear any pending followup playout whose LLMResponseEndFrame was never pushed
        # (inference cancelled mid-stream before reaching push_frame for that ctx_id).
        self._pending_stage1_followup = None

    # ── System frame handling (C-stage hook) ──────────────────────────────────

    async def handle_system_frame(self, frame: Frame, direction: FrameDirection) -> None:
        # ADR 0006: VAD says the patient began speaking → cancel silent-rescue
        # timer and reset chain length (also sets the stage1-followup
        # user-speaking guard, Task 12.2). UserStartedSpeakingFrame is a
        # SystemFrame, so the base dispatcher routes it here (handle_system_frame),
        # NOT to process_frame — wiring the cancel on process_frame was dead code
        # and let rescue fire after the patient had already spoken.
        if isinstance(frame, UserStartedSpeakingFrame):
            self._on_user_speech_signal()
            # fall through — still forward the frame downstream via super()
        if isinstance(frame, BotStoppedSpeakingFrame):
            if self._flow and self._flow.has_deferred_post_actions():
                await self._flow.on_node_reply_finished()
            if self._memory_scheduler is not None:
                self._memory_scheduler.on_playout_done(frame)
            if self._short_term_memory_scheduler is not None:
                self._short_term_memory_scheduler.on_playout_done(frame)
            if self._sop_progress_scheduler is not None:
                self._sop_progress_scheduler.on_playout_done(frame)
            # Transfer playout barrier — match context_id
            if self._transfer_playout_barrier is not None and frame.context_id == self._transfer_playout_context_id:
                self._transfer_playout_barrier.set()
            # Terminal hook — ONLY fires via explicit _pending_terminal_hook flag
            if self._pending_terminal_hook:
                await self._execute_terminal_hook()
        await super().handle_system_frame(frame, direction)

    # ── Terminal hook ──────────────────────────────────────────────────────────

    async def _execute_terminal_hook(self) -> None:
        """Unified terminal hook — form_submit + call_complete."""
        from src.pipeline.frames import EndFrame

        if not self._pending_terminal_hook:
            return
        self._pending_terminal_hook = False

        payload = self._flow.build_terminal_payload()
        self._emit_debug("terminal_hook", **payload)

        if self._call_tracer:
            trace_form_submitted = getattr(self._call_tracer, "trace_form_submitted", None)
            if callable(trace_form_submitted):
                trace_form_submitted(payload)

        logger.info("llm.terminal_hook_executed", fields=len(payload.get("collected_fields", {})))
        await self.push_frame(EndFrame(), FrameDirection.DOWNSTREAM)

    async def _end_playout_and_set_terminal(self, playout_id: str) -> None:
        """End current playout and schedule terminal hook on next BotStoppedSpeaking."""
        await self.push_frame(
            LLMResponseEndFrame(context_id=playout_id, turn_id=self._current_turn_id or ""),
            FrameDirection.DOWNSTREAM,
        )
        self._pending_terminal_hook = True

    # ── Session bootstrap ─────────────────────────────────────────────────────

    async def _handle_session_bootstrap(self) -> None:
        """Handle initial greeting generation — once-only."""
        if self._bootstrapped:
            logger.warning("llm.bootstrap_duplicate_ignored")
            return
        self._bootstrapped = True
        self._inference_start_mono = time.monotonic()
        self._ttft_recorded = False
        plan = await self._flow.prepare_node_entry(continuation_prefix="")
        if plan:
            playout_id = plan.inference_id
            self._emit_debug("bootstrap_start", node=self._flow._state.current_node)
            stage1_result = await self._run_speech_inference(plan, playout_id)
            self._emit_debug("bootstrap_done", node=self._flow._state.current_node, text=stage1_result.raw_text)
            await self.push_frame(
                LLMResponseEndFrame(context_id=playout_id, turn_id=""),
                FrameDirection.DOWNSTREAM,
            )

    # ── Overrides ─────────────────────────────────────────────────────────────

    async def cancel_inflight(self) -> None:
        await super().cancel_inflight()
        await self._cancel_inference()

    async def on_flush(self, frame: Frame) -> None:
        await self._cancel_inference()
        self._history.rollback_provisional()
        self._current_turn_id = None
        await super().on_flush(frame)

    # ── Human transfer orchestration ───────────────────────────────────────────

    async def _execute_human_transfer(self, cmd, *, skip_tts: bool = False) -> None:
        """Orchestrate full transfer flow. Called from _run_inference_flow.

        Args:
            skip_tts: If True, Speech LLM already said the transition message —
                skip the interrupt + TTS playout steps.
        """
        from src.pipeline.frames import EndFrame

        if not skip_tts:
            # 1. Stop current Speech TTS
            await self.push_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

            # 2. Play transition message + wait for playout via barrier
            transfer_playout_id = str(uuid.uuid4())
            self._transfer_playout_context_id = transfer_playout_id
            self._transfer_playout_barrier = asyncio.Event()

            for token in _chunk_text(cmd.transition_message):
                await self.push_frame(
                    LLMTextFrame(
                        text=token,
                        turn_id=self._current_turn_id or "",
                        metadata={"context_id": transfer_playout_id},
                    ),
                    FrameDirection.DOWNSTREAM,
                )
            await self.push_frame(
                LLMResponseEndFrame(
                    context_id=transfer_playout_id,
                    turn_id=self._current_turn_id or "",
                ),
                FrameDirection.DOWNSTREAM,
            )

            try:
                await asyncio.wait_for(self._transfer_playout_barrier.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("transfer.playout_barrier_timeout", playout_id=transfer_playout_id)
            finally:
                self._transfer_playout_barrier = None
                self._transfer_playout_context_id = ""

        # 3. Fire-and-forget summary push
        asyncio.create_task(self._push_transfer_summary(cmd))

        # 4. Debug event
        self._emit_debug(
            "transfer_to_human",
            reason=cmd.reason,
            priority=cmd.priority,
            source_node=cmd.source_node,
        )
        if self._bus:
            self._bus.emit(
                TransferTriggeredEvent(
                    turn_id=self._current_turn_id or "",
                    reason=cmd.reason if hasattr(cmd, "reason") else "",
                    node=cmd.source_node if hasattr(cmd, "source_node") else "",
                    destination=settings.human_agent_sip_destination or "",
                )
            )

        # 5. SIP REFER
        destination = settings.human_agent_sip_destination
        if not destination or self._sip_transfer is None:
            logger.error("transfer.no_destination_or_sip_fn", destination=destination)
            await self._recover_from_failed_transfer(cmd)
            return

        try:
            await self._sip_transfer(destination)
        except Exception as e:
            logger.error("transfer.sip_refer_failed", error=str(e))
            await self._recover_from_failed_transfer(cmd)
            return

        # 6. Commit terminal (FlowState mutates HERE, only on success)
        self._flow.commit_transfer_terminal(reason=cmd.reason, priority=cmd.priority)

        # 7. Post-call pipeline (once-only)
        if self._call_tracer:
            self._call_tracer.trace_handoff_requested(cmd.reason, cmd.priority)
        if self._lifecycle:
            outcome = "transferred_safety" if cmd.priority == "urgent" else "transferred"
            self._lifecycle.schedule_post_call_once(outcome=outcome, transfer_reason=cmd.reason)

        # 8. Exit
        await self.push_frame(EndFrame(), FrameDirection.DOWNSTREAM)

    async def _push_transfer_summary(self, cmd) -> None:
        """Fire-and-forget: generate LLM summary + push to webhook. Best-effort."""
        import httpx

        webhook_url = settings.transfer_summary_webhook_url
        if not webhook_url:
            logger.info("transfer.summary_push_skipped", reason="no_webhook_url")
            return

        # 1. Generate structured handoff context
        from src.handoff.transfer_service import get_transfer_service

        transfer_service = get_transfer_service()
        flow_state = self._flow._state
        reason_enum = (
            TransferReason(cmd.reason) if cmd.reason in TransferReason._value2member_map_ else TransferReason.OTHER
        )
        handoff_text = transfer_service.prepare_handoff_from_flow_state(
            flow_state=flow_state,
            reason=reason_enum,
            trigger_text=cmd.trigger_text,
        )

        # 2. LLM summary (3s timeout, fallback to None)
        llm_summary = None
        if self._fast_judge_backend is not None:
            summary_prompt = (
                "你是一个对话摘要器。用3句话总结以下随访对话要点，"
                "供人工客服快速了解情况。包含：患者身份、主要诉求、转接原因。\n\n"
                f"{handoff_text}"
            )
            try:
                full_text = ""
                async for event in self._fast_judge_backend.stream_with_tools(
                    [{"role": "system", "content": summary_prompt}],
                    tools=None,
                    session_id="",
                    turn_id="",
                ):
                    if event.type == StreamEventType.TEXT and event.text:
                        full_text += event.text
                llm_summary = full_text or None
            except Exception:
                logger.warning("transfer.summary_llm_failed")

        # 3. Build payload
        from datetime import datetime, timedelta, timezone

        tz = timezone(timedelta(hours=8))
        payload = {
            "call_id": self._flow._state.active_turn_id or "",
            "timestamp": datetime.now(tz=tz).isoformat(),
            "transfer_reason": cmd.reason,
            "priority": cmd.priority,
            "patient": cmd.patient_info,
            "collected_fields": cmd.collected_fields,
            "summary": llm_summary,
            "conversation_turns": cmd.conversation_turns,
            "current_node": cmd.source_node,
            "nodes_completed": cmd.nodes_completed,
        }

        # 4. POST webhook
        try:
            async with httpx.AsyncClient(timeout=settings.transfer_webhook_timeout_s) as client:
                await client.post(webhook_url, json=payload)
            logger.info("transfer.summary_pushed", url=webhook_url)
        except Exception:
            logger.warning("transfer.summary_push_failed", url=webhook_url)

    async def _recover_from_failed_transfer(self, cmd) -> None:
        """SIP REFER failed. FlowState unchanged — just apologize and continue."""
        apology = "抱歉，暂时无法转接人工客服，我继续为您服务。"
        apology_playout_id = str(uuid.uuid4())
        for token in _chunk_text(apology):
            await self.push_frame(
                LLMTextFrame(
                    text=token,
                    turn_id=self._current_turn_id or "",
                    metadata={"context_id": apology_playout_id},
                ),
                FrameDirection.DOWNSTREAM,
            )
        await self.push_frame(
            LLMResponseEndFrame(
                context_id=apology_playout_id,
                turn_id=self._current_turn_id or "",
            ),
            FrameDirection.DOWNSTREAM,
        )
        logger.warning("transfer.failed_recovery", source_node=cmd.source_node)


# ── Module-level helpers ──────────────────────────────────────────────────────


def _prepare_debug_messages(messages: list[dict]) -> list[dict]:
    """Normalize messages for debug display while preserving order and text."""
    out: list[dict] = []
    for msg in messages:
        clean_msg = {k: v for k, v in msg.items() if not k.startswith("_")}
        content = clean_msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(part.get("text", ""))
                else:
                    parts.append(str(part))
            clean_msg["content"] = "".join(parts)
        out.append(clean_msg)
    return out
