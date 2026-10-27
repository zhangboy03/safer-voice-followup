"""Top-level orchestrator for a knowledge_qa turn.

Per spec §Runtime Changes sketch. The state machine lives here:
  Bridge → [A] Review → envelope guard → router → {[B1], [B2], [D]} → [E].

Entry: handle_knowledge_query_turn(user_text, history, flow_state).
"""

from __future__ import annotations

import asyncio  # noqa: F401 — used by Task 18 orchestrator body
import hashlib  # noqa: F401 — used by Task 18 orchestrator body
import time
from typing import Any

from src.core.logging import get_logger
from src.flows.knowledge_review_models import (
    CallbackPriority,  # noqa: F401 — used by Task 18 orchestrator body
    EscalationKind,  # noqa: F401 — used by Task 18 orchestrator body
    KnowledgeReviewEnvelope,
    RefuseKind,  # noqa: F401 — used by Task 18 orchestrator body
)
from src.flows.knowledge_router import (
    RouteAction,  # noqa: F401 — used by Task 18 orchestrator body
    is_backchannel,
    is_corrective_recovery_hit,
    route,  # noqa: F401 — used by Task 18 orchestrator body
)
from src.flows.knowledge_session import (
    KnowledgeSession,  # noqa: F401 — used by Task 18 orchestrator body
)
from src.flows.knowledge_validator import (
    FactsValidationError,  # noqa: F401 — used by Task 18 orchestrator body
    validate_facts,  # noqa: F401 — used by Task 18 orchestrator body
)
from src.handoff.callback_queue import (
    get_callback_queue_writer,  # noqa: F401 — used by Task 18 orchestrator body
)
from src.prompts.phrase_bank import (
    PHRASE_TYPE_1,  # noqa: F401 — used by Task 18 orchestrator body
    PHRASE_TYPE_2,  # noqa: F401 — used by Task 18 orchestrator body
    PHRASE_TYPE_3,  # noqa: F401 — used by Task 18 orchestrator body
    PHRASE_TYPE_E1,  # noqa: F401 — used by Task 18 orchestrator body
    PHRASE_TYPE_E2,  # noqa: F401 — used by Task 18 orchestrator body
    load_knowledge_phrases,  # noqa: F401 — used by Task 18 orchestrator body
)

logger = get_logger(__name__)

MIN_CONFIDENCE = 0.5  # spec §4 — placeholder, calibrated post-eval
REVIEW_WALL_TIMEOUT_S = 3.0

# ── Event bus wiring (Task 21) ────────────────────────────────────────────────

_event_bus = None


def bind_event_bus(bus) -> None:
    """Wire a process-wide event bus for KnowledgeMetricEvent emission.
    LLMProcessor.__init__ should call this once at startup."""
    global _event_bus
    _event_bus = bus


def _emit_knowledge_metric(**fields) -> None:
    from src.pipeline.events import KnowledgeMetricEvent

    if _event_bus is None:
        logger.info("knowledge_metric (no bus) %s", fields)
        return
    _event_bus.emit(KnowledgeMetricEvent(**fields))


def _cache_hit_rate(usage: dict | None) -> float:
    if not isinstance(usage, dict):
        return -1.0
    pt = usage.get("prompt_tokens") or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    if not isinstance(pt, (int, float)) or pt <= 0:
        return -1.0
    return min(1.0, max(0.0, cached / pt))


def envelope_matches_full(
    envelope: KnowledgeReviewEnvelope,
    flow_state: Any,
    *,
    current_input_hash: str,
    current_kb_hash: str,
    current_prompt_hash: str,
) -> bool:
    """All 9 conditions (0–8) per spec §Data Model envelope match contract.

    Reads identity from `flow_state` directly using REAL FlowState field
    names (active_turn_id / flow_epoch / divert_return_node). The
    suspension anchor lives on the session (set at DIVERT time); we
    require it to still match the live flow_state.divert_return_node —
    a mismatch means a RESUME interleaved and the envelope is stale.

    Resolves current_session from flow_state at check time — does NOT take
    a local session var (which could be stale).
    """
    rs = flow_state.knowledge_review_state
    if rs is None:
        return False
    current_session = rs.current_session
    if current_session is None:
        return False
    if current_session.closed:
        return False
    # Suspension anchor: session captured at DIVERT vs. live FlowState's divert_return_node.
    if current_session.suspended_sop_node != flow_state.divert_return_node:
        return False
    if envelope.turn_id != (flow_state.active_turn_id or ""):
        return False
    if envelope.flow_epoch != flow_state.flow_epoch:
        return False
    if envelope.session_id != current_session.session_id:
        return False
    if envelope.followup_round_at_request != current_session.followup_round:
        return False
    if envelope.input_hash != current_input_hash:
        return False
    if envelope.kb_hash != current_kb_hash:
        return False
    if envelope.prompt_hash != current_prompt_hash:
        return False
    return True


async def handle_knowledge_query_turn(
    user_text: str,
    history: list,
    flow_state: Any,
    *,
    kb_hash: str,
    call_id: str,
) -> None:
    """Drive the [A]→router→{B1|B2|D} machine for one knowledge-query turn.

    Per spec §Runtime Changes. [E] runs on the NEXT user turn, not in this
    function — this returns after [D]/[B1]/[B2] emission + RESUME (B1) or
    waiting (B2/D). kb_hash and call_id are caller-provided (they live on
    SessionLifecycleHelper / runtime, not on FlowState).
    """
    divert_t0 = time.monotonic()
    rs = flow_state.knowledge_review_state
    if rs.current_session is None:
        rs.current_session = KnowledgeSession(
            session_id=_new_session_id(),
            divert_entry_turn_id=flow_state.active_turn_id or "",
            suspended_sop_node=flow_state.divert_return_node or "",
        )
        rs.sessions_in_current_suspension += 1

    # Depth cap (per Codex r3 round-3 High #4)
    if rs.sessions_in_current_suspension > 3:
        await _emit_escalation_phrase(flow_state, PHRASE_TYPE_E1)
        _enqueue_callback(
            flow_state=flow_state,
            call_id=call_id,
            kind=EscalationKind.DEPTH_CAP,
            out_of_scope_class=None,
            priority=CallbackPriority.NORMAL,
            user_text=user_text,
            reason_text="depth_cap_hit",
        )
        await _resume(flow_state)
        return

    session = rs.current_session

    # Bridge phrase (covers [A] latency)
    bridge_t0 = time.monotonic()
    await _emit_bridge_phrase(flow_state, "我帮您查看一下，请稍等")
    bridge_ttfa_ms = (time.monotonic() - bridge_t0) * 1000

    # [A] Review with hard timeout
    runner = _get_review_runner()
    review_t0 = time.monotonic()
    try:
        facts = await asyncio.wait_for(
            runner.run(
                user_text=user_text,
                recent_dialogue=history[-3:],
                session_id=session.session_id,
                followup_round=session.followup_round,
                turn_id=flow_state.active_turn_id or "",
                flow_epoch=flow_state.flow_epoch,
                kb_hash=kb_hash,
                prompt_hash=None,
            ),
            timeout=REVIEW_WALL_TIMEOUT_S,
        )
    except asyncio.TimeoutError as exc:
        review_wall_ms = (time.monotonic() - review_t0) * 1000
        logger.warning("knowledge_review.infra_fallback %r", exc)
        _emit_knowledge_metric(
            turn_id=flow_state.active_turn_id or "",
            route="b1",
            bridge_ttfa_ms=bridge_ttfa_ms,
            review_wall_ms=review_wall_ms,
            knowledge_first_audio_ms=(time.monotonic() - divert_t0) * 1000,
            failure_reason="review_timeout",
        )
        await _emit_refuse_phrase(flow_state, PHRASE_TYPE_2)
        await _resume(flow_state)
        return
    except RuntimeError as exc:
        review_wall_ms = (time.monotonic() - review_t0) * 1000
        logger.warning("knowledge_review.infra_fallback %r", exc)
        _emit_knowledge_metric(
            turn_id=flow_state.active_turn_id or "",
            route="b1",
            bridge_ttfa_ms=bridge_ttfa_ms,
            review_wall_ms=review_wall_ms,
            knowledge_first_audio_ms=(time.monotonic() - divert_t0) * 1000,
            failure_reason="review_timeout",
        )
        await _emit_refuse_phrase(flow_state, PHRASE_TYPE_2)
        await _resume(flow_state)
        return

    review_wall_ms = (time.monotonic() - review_t0) * 1000
    review_cache = _cache_hit_rate(getattr(runner, "last_usage", None))

    # Envelope identity guard (9 conditions, see Task 17)
    current_input_hash = _compute_input_hash(user_text, history)
    if not envelope_matches_full(
        facts.envelope,
        flow_state,
        current_input_hash=current_input_hash,
        current_kb_hash=kb_hash,
        current_prompt_hash=runner.prompt_hash,
    ):
        live = getattr(flow_state.knowledge_review_state.current_session, "session_id", None)
        logger.info("review_stale_drop session_id=%s envelope=%s", live, facts.envelope)
        return  # silent drop

    # Pre-routing facts validation
    try:
        validate_facts(facts, min_confidence=MIN_CONFIDENCE)
    except FactsValidationError as exc:
        logger.warning("knowledge_review.infra_fallback code=%s", exc.code)
        _emit_knowledge_metric(
            turn_id=flow_state.active_turn_id or "",
            route="b1",
            bridge_ttfa_ms=bridge_ttfa_ms,
            review_wall_ms=review_wall_ms,
            review_cache_hit_rate=review_cache,
            knowledge_first_audio_ms=(time.monotonic() - divert_t0) * 1000,
            review_response_format_kind="json_schema",
            failure_reason="review_schema",
        )
        await _emit_refuse_phrase(flow_state, PHRASE_TYPE_2)
        await _resume(flow_state)
        return

    session.last_facts = facts

    # Route
    decision = route(facts, session)

    if decision.kind == RouteAction.REFUSE_LEGITIMATE:
        phrase_key, priority = _select_legitimate_phrase(decision.out_of_scope_class)
        _emit_knowledge_metric(
            turn_id=flow_state.active_turn_id or "",
            route="b1",
            bridge_ttfa_ms=bridge_ttfa_ms,
            review_wall_ms=review_wall_ms,
            review_cache_hit_rate=review_cache,
            knowledge_first_audio_ms=(time.monotonic() - divert_t0) * 1000,
            review_response_format_kind="json_schema",
        )
        await _emit_refuse_phrase(flow_state, phrase_key)
        if decision.out_of_scope_class == "medical_judgment":
            _enqueue_callback(
                flow_state=flow_state,
                call_id=call_id,
                kind=RefuseKind.LEGITIMATE,
                out_of_scope_class="medical_judgment",
                priority=CallbackPriority.NORMAL,
                user_text=user_text,
                reason_text=facts.reason,
            )
        elif decision.out_of_scope_class == "safety_risk":
            _enqueue_callback(
                flow_state=flow_state,
                call_id=call_id,
                kind=RefuseKind.LEGITIMATE,
                out_of_scope_class="safety_risk",
                priority=CallbackPriority.HIGH,
                user_text=user_text,
                reason_text=facts.reason,
            )
        await _resume(flow_state)

    elif decision.kind == RouteAction.REFUSE_COERCED:
        _emit_knowledge_metric(
            turn_id=flow_state.active_turn_id or "",
            route="b1",
            bridge_ttfa_ms=bridge_ttfa_ms,
            review_wall_ms=review_wall_ms,
            review_cache_hit_rate=review_cache,
            knowledge_first_audio_ms=(time.monotonic() - divert_t0) * 1000,
            review_response_format_kind="json_schema",
        )
        await _emit_refuse_phrase(flow_state, PHRASE_TYPE_2)
        await _resume(flow_state)

    elif decision.kind == RouteAction.FOLLOWUP:
        _emit_knowledge_metric(
            turn_id=flow_state.active_turn_id or "",
            route="b2",
            bridge_ttfa_ms=bridge_ttfa_ms,
            review_wall_ms=review_wall_ms,
            review_cache_hit_rate=review_cache,
            knowledge_first_audio_ms=(time.monotonic() - divert_t0) * 1000,
            review_response_format_kind="json_schema",
        )
        # Cap consumed at TTS audio start (incremented per round; cap = MAX_FOLLOWUP_ROUNDS)
        session.followup_heard_status = "pending"

        def _on_audio_start():
            session.followup_round += 1

        await _run_speech_followup(
            flow_state,
            facts.missing_prerequisites,
            _on_audio_start,
        )

    elif decision.kind == RouteAction.ANSWER:
        _emit_knowledge_metric(
            turn_id=flow_state.active_turn_id or "",
            route="d",
            bridge_ttfa_ms=bridge_ttfa_ms,
            review_wall_ms=review_wall_ms,
            review_cache_hit_rate=review_cache,
            knowledge_first_audio_ms=(time.monotonic() - divert_t0) * 1000,
            review_response_format_kind="json_schema",
        )
        await _run_speech_answer(flow_state, user_text, facts)
        await _run_speech_exit_check(flow_state)


def _select_legitimate_phrase(out_of_scope_class: str | None):
    if out_of_scope_class == "medical_judgment":
        return PHRASE_TYPE_1, CallbackPriority.NORMAL
    if out_of_scope_class == "safety_risk":
        return PHRASE_TYPE_3, CallbackPriority.HIGH
    return PHRASE_TYPE_2, None


def _hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _new_session_id() -> str:
    import uuid

    return uuid.uuid4().hex


def _compute_input_hash(user_text: str, history: list) -> str:
    """Canonical input_hash recipe. MUST match KnowledgeReviewRunner._input_hash
    (src/flows/knowledge_review_runner.py) so the envelope guard succeeds."""
    from src.flows.knowledge_review_runner import _input_hash

    return _input_hash(user_text, history[-3:])


# --- Module-level hooks the integration layer wires up.
# Tests monkeypatch these directly; Task 19 (llm_processor wiring) overrides
# them at process startup via bind_orchestrator_hooks().
# Each hook raises NotImplementedError until Task 19 binds the real impl —
# this gives a catastrophic-fail loud failure if anything reaches these
# stubs in production without being wired.


async def _emit_bridge_phrase(flow_state, text: str) -> None:
    raise NotImplementedError("emit_bridge_phrase not wired by bind_orchestrator_hooks()")


async def _emit_refuse_phrase(flow_state, phrase_key: str) -> None:
    raise NotImplementedError("emit_refuse_phrase not wired")


async def _emit_escalation_phrase(flow_state, phrase_key: str) -> None:
    raise NotImplementedError("emit_escalation_phrase not wired")


async def _run_speech_followup(flow_state, missing_prereqs, on_audio_start) -> None:
    raise NotImplementedError("run_speech_followup not wired")


async def _run_speech_answer(flow_state, user_text, facts) -> None:
    raise NotImplementedError("run_speech_answer not wired")


async def _run_speech_exit_check(flow_state) -> None:
    raise NotImplementedError("run_speech_exit_check not wired")


async def _resume(flow_state) -> None:
    raise NotImplementedError("resume not wired")


def _get_review_runner():
    """Return a singleton KnowledgeReviewRunner. Tests monkeypatch this hook
    to inject a fake runner; production wiring sets it once at startup."""
    raise NotImplementedError("_get_review_runner not wired by bind_orchestrator_hooks()")


def _enqueue_callback(**kwargs):
    return get_callback_queue_writer().enqueue(**kwargs)


# --- [E] Exit judge driver + recovery window (Task 20) ---


async def handle_exit_check_turn(
    user_text: str,
    history: list,
    flow_state: Any,
    *,
    kb_hash: str,
    call_id: str,
) -> None:
    """Driver for the user reply AFTER [D] TTS heard-complete + [E] speech
    emitted. Runs [E] judge to classify done / new_question / continue / escalate.

    Caller (LLMProcessor) is responsible for routing the next user turn here
    when knowledge_review_state.current_session is open + last assistant speech
    was the [E] exit-check phrase.
    """
    runner = _get_exit_runner()
    rs = flow_state.knowledge_review_state
    session = rs.current_session
    if session is None:
        logger.warning("handle_exit_check_turn: no current_session — likely caller bug")
        return

    exit_t0 = time.monotonic()
    try:
        decision = await asyncio.wait_for(
            runner.run(
                last_assistant_text="您这个问题还有不清楚的吗？",
                user_text=user_text,
                turn_id=flow_state.active_turn_id or "",
                flow_epoch=flow_state.flow_epoch,
                session_id=session.session_id,
                kb_hash=kb_hash,
            ),
            timeout=REVIEW_WALL_TIMEOUT_S,
        )
    except (asyncio.TimeoutError, RuntimeError) as exc:
        logger.warning("knowledge_exit.infra_fallback %r", exc)
        await _resume(flow_state)
        return

    exit_wall_ms = (time.monotonic() - exit_t0) * 1000
    exit_cache = _cache_hit_rate(getattr(runner, "last_usage", None))

    intent = decision.exit_intent

    if intent == "done":
        # Arm recovery only on ambiguous backchannel
        if is_backchannel(user_text):
            rs.exit_recovery_armed = True
        _emit_knowledge_metric(
            turn_id=flow_state.active_turn_id or "",
            route="e1",
            exit_wall_ms=exit_wall_ms,
            exit_cache_hit_rate=exit_cache,
            exit_response_format_kind="json_schema",
        )
        await _resume(flow_state)

    elif intent in ("new_question", "continue"):
        _emit_knowledge_metric(
            turn_id=flow_state.active_turn_id or "",
            route="b1",
            exit_wall_ms=exit_wall_ms,
            exit_cache_hit_rate=exit_cache,
            exit_response_format_kind="json_schema",
        )
        # Close current session, then re-enter [A] with the new text
        session.closed = True
        rs.current_session = None
        await handle_knowledge_query_turn(
            user_text=user_text,
            history=history,
            flow_state=flow_state,
            kb_hash=kb_hash,
            call_id=call_id,
        )

    elif intent == "escalate":
        _emit_knowledge_metric(
            turn_id=flow_state.active_turn_id or "",
            route="e2",
            exit_wall_ms=exit_wall_ms,
            exit_cache_hit_rate=exit_cache,
            exit_response_format_kind="json_schema",
        )
        await _emit_escalation_phrase(flow_state, PHRASE_TYPE_E2)
        _enqueue_callback(
            flow_state=flow_state,
            call_id=call_id,
            kind=EscalationKind.EXIT_ESCALATE,
            out_of_scope_class=None,
            priority=CallbackPriority.NORMAL,
            user_text=user_text,
            reason_text="exit_escalate",
        )
        await _resume(flow_state)


async def handle_post_exit_recovery_turn(
    user_text: str,
    history: list,
    flow_state: Any,
    *,
    kb_hash: str,
    call_id: str,
) -> bool:
    """If armed, check recovery regex. Return True if recovery handled the
    turn (re-DIVERT'd); False if non-match (caller proceeds to normal SOP
    routing). Always disarms after one turn (regardless of hit/miss).

    Caller (LLMProcessor) wires this BEFORE normal SOP routing while
    flow_state.knowledge_review_state.exit_recovery_armed is True. See
    Task 23 integration test for the full wiring.
    """
    rs = flow_state.knowledge_review_state
    if not rs.exit_recovery_armed:
        return False
    rs.exit_recovery_armed = False  # disarm regardless of outcome

    if not is_corrective_recovery_hit(user_text):
        return False

    # Recovery hit → re-DIVERT with raw text
    if rs.current_session is not None:
        rs.current_session.closed = True
        rs.current_session = None
    await handle_knowledge_query_turn(
        user_text=user_text,
        history=history,
        flow_state=flow_state,
        kb_hash=kb_hash,
        call_id=call_id,
    )
    return True


def _get_exit_runner():
    """Singleton accessor — tests monkeypatch this. Production wiring
    (Task 23 LLMProcessor) should swap in a process-wide singleton.
    # TODO Task 23 integration wiring
    """
    from src.flows.knowledge_exit_runner import KnowledgeExitRunner

    return KnowledgeExitRunner.from_manifest()
