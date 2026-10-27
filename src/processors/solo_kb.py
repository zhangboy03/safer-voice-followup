"""Real production three-stage KB, decoupled from FlowRuntime.
review -> route -> KbDecision. The DRIVER speaks the result (grounded answer
streamed; frozen refuse verbatim; deterministic clarify) and enqueues callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from src.flows import knowledge_router
from src.flows.knowledge_review_models import CallbackPriority, RefuseKind
from src.flows.knowledge_router import RouteAction
from src.flows.knowledge_session import KnowledgeSession
from src.flows.knowledge_turn import _select_legitimate_phrase
from src.flows.knowledge_validator import _fallback_question
from src.prompts.phrase_bank import load_knowledge_phrases


class KbDecisionKind(Enum):
    ANSWER = auto()
    REFUSE = auto()
    CLARIFY = auto()


@dataclass
class KbDecision:
    kind: KbDecisionKind
    # REFUSE:
    verbatim_text: str | None = None
    phrase_key: str | None = None
    priority: CallbackPriority | None = None
    refuse_kind: RefuseKind | None = None
    out_of_scope_class: str | None = None
    # CLARIFY:
    clarify_text: str | None = None
    # always: the user query that triggered this (for callback user_text)
    query: str = ""


class KbThreeStage:
    def __init__(self, *, review_runner, kb_hash: str, flow_epoch: int = 0, language: str = "zh"):
        self._language = language
        self._review = review_runner
        self._kb_hash = kb_hash
        self._flow_epoch = flow_epoch

    async def decide(
        self,
        *,
        query: str,
        recent_dialogue: list[dict],
        session: KnowledgeSession,
        turn_id: str,
    ) -> KbDecision:
        facts = await self._review.run(
            user_text=query,
            recent_dialogue=recent_dialogue[-3:],  # prod uses last-3 (matches _input_hash recipe)
            session_id=session.session_id,
            followup_round=session.followup_round,
            turn_id=turn_id,
            flow_epoch=self._flow_epoch,
            kb_hash=self._kb_hash,
        )
        session.last_facts = facts
        decision = knowledge_router.route(facts, session)
        phrases = load_knowledge_phrases(self._language)
        if decision.kind is RouteAction.ANSWER:
            return KbDecision(kind=KbDecisionKind.ANSWER, query=query)
        if decision.kind is RouteAction.FOLLOWUP:
            return KbDecision(
                kind=KbDecisionKind.CLARIFY,
                clarify_text=(
                    "Could you give me a little more detail about your question?"
                    if self._language == "en" else _fallback_question(facts.missing_prerequisites)
                ),
                query=query,
            )
        # REFUSE_LEGITIMATE / REFUSE_COERCED / REFUSE_INFRA
        if decision.kind is RouteAction.REFUSE_LEGITIMATE:
            phrase_key, priority = _select_legitimate_phrase(decision.out_of_scope_class)
            refuse_kind = RefuseKind.LEGITIMATE
        elif decision.kind is RouteAction.REFUSE_COERCED:
            phrase_key, priority, refuse_kind = (
                "type_2",
                CallbackPriority.NORMAL,
                RefuseKind.COERCED,
            )
        else:  # REFUSE_INFRA
            phrase_key, priority, refuse_kind = (
                "type_2",
                CallbackPriority.NORMAL,
                RefuseKind.INFRA_FALLBACK,
            )
        return KbDecision(
            kind=KbDecisionKind.REFUSE,
            verbatim_text=phrases[phrase_key],
            phrase_key=phrase_key,
            priority=priority or CallbackPriority.NORMAL,
            refuse_kind=refuse_kind,
            out_of_scope_class=decision.out_of_scope_class,
            query=query,
        )
