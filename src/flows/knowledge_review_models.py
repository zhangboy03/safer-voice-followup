"""Data models for knowledge_qa review architecture.

Per spec docs/superpowers/specs/2026-05-16-knowledge-qa-review-design.md §Data Model.

Frozen dataclasses: facts and decisions are immutable once emitted by an LLM
call so the deterministic router cannot mutate them mid-routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional


@dataclass(frozen=True)
class KnowledgeDecisionEnvelope:
    """Base envelope — shared between [A] and [E] LLM call results."""

    turn_id: str
    flow_epoch: int
    session_id: str
    input_hash: str
    kb_hash: str
    prompt_hash: str


@dataclass(frozen=True)
class KnowledgeReviewEnvelope(KnowledgeDecisionEnvelope):
    """[A] envelope — adds followup_round_at_request for cap-consumption identity."""

    followup_round_at_request: int = 0


@dataclass(frozen=True)
class KnowledgeReviewFacts:
    """Output of [A] reviewer LLM — route facts only, no decision."""

    scope: Literal["in_scope", "out_of_scope"]
    out_of_scope_class: Optional[
        Literal[
            "medical_judgment",
            "kb_out_of_coverage",
            "adversarial_or_pii",
            "safety_risk",
        ]
    ]
    missing_prerequisites: list[str]
    answer_readiness: Optional[Literal["ready", "needs_clarification"]]
    reason: str
    confidence: float
    envelope: KnowledgeReviewEnvelope


@dataclass(frozen=True)
class KnowledgeExitDecision:
    """Output of [E] exit judge LLM."""

    exit_intent: Literal["done", "new_question", "continue", "escalate"]
    envelope: KnowledgeDecisionEnvelope
    raw_user_reply: str


class RefuseKind(Enum):
    """Why we refused; distinct from EscalationKind."""

    LEGITIMATE = "legitimate"
    COERCED = "coerced"
    INFRA_FALLBACK = "infra_fallback"


class EscalationKind(Enum):
    """Why we escalated (RESUME + callback); distinct from RefuseKind."""

    DEPTH_CAP = "depth_cap"
    EXIT_ESCALATE = "exit_escalate"


class CallbackPriority(Enum):
    NORMAL = "normal"
    HIGH = "high"
