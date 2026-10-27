# src/eval/study3/pinchi_adapter.py
"""Pins two-stage replay: real route() over injected Review facts.

NOT the handle_knowledge_query_turn orchestrator (it needs startup-injected
hooks). For ANSWER we call the Speech fn with retrieved KB; for non-ANSWER we
return the phrase-bank text the production orchestrator would emit, and capture
the callback priority side-effect as white-box diagnostics."""

from __future__ import annotations

from typing import Awaitable, Callable

from src.eval.study3.adapters import SystemOutput
from src.flows.knowledge_review_models import KnowledgeReviewFacts
from src.flows.knowledge_router import RouteAction, route
from src.flows.knowledge_session import KnowledgeSession

ReviewFn = Callable[[str], Awaitable[KnowledgeReviewFacts]]
SpeechFn = Callable[[str, list], Awaitable[str]]

# phrase-key + callback priority by route/out_of_scope_class, mirroring
# knowledge_turn._select_legitimate_phrase + the enqueue side-effects.
_PHRASE_BY_OOS = {
    "medical_judgment": ("type_1", "NORMAL"),
    "safety_risk": ("type_3", "HIGH"),
    # kb_out_of_coverage / adversarial_or_pii / None → generic refuse, no callback
}


async def run_pinchi(
    question: str,
    *,
    review_fn: ReviewFn,
    speech_fn: SpeechFn,
    phrases: dict[str, str],
) -> SystemOutput:
    try:
        facts = await review_fn(question)
    except Exception:
        return SystemOutput(system="pinchi_full", patient_text="", status="adapter_error")

    session = KnowledgeSession(session_id="eval", divert_entry_turn_id="t", suspended_sop_node="")
    decision = route(facts, session)
    wb = {
        "scope": facts.scope,
        "out_of_scope_class": facts.out_of_scope_class,
        "answer_readiness": facts.answer_readiness,
        "route_action": decision.kind.name,
        "callback_priority": None,
    }

    if decision.kind == RouteAction.ANSWER:
        # Faithful to production: NO retrieval. The speech node injects the FULL KB
        # via its prompt template (speech/knowledge_qa.md embeds all 245 entries),
        # exactly like the live render_speech_prompt path — prod never calls the
        # retriever. We pass [] so no redundant top-k is appended on top.
        try:
            text = await speech_fn(question, [])
        except Exception:
            return SystemOutput(system="pinchi_full", patient_text="", status="adapter_error", whitebox=wb)
        return SystemOutput(
            system="pinchi_full",
            patient_text=text,
            kb_refs=[],  # whole KB is in-prompt; no retrieved subset to cite
            whitebox=wb,
        )

    if decision.kind == RouteAction.FOLLOWUP:
        # production asks a clarification via speech; for replay emit a fixed clarify stub.
        return SystemOutput(
            system="pinchi_full",
            patient_text="您能再说明一下具体情况吗？需要更多信息才能帮到您。",
            whitebox=wb,
        )

    if decision.kind == RouteAction.REFUSE_LEGITIMATE:
        phrase_key, priority = _PHRASE_BY_OOS.get(decision.out_of_scope_class or "", ("type_2", None))
        wb["callback_priority"] = priority
        return SystemOutput(system="pinchi_full", patient_text=phrases[phrase_key], whitebox=wb)

    # REFUSE_COERCED / REFUSE_INFRA → generic refuse phrase, no callback
    return SystemOutput(system="pinchi_full", patient_text=phrases["type_2"], whitebox=wb)
