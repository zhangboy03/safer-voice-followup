"""Deterministic router + exit-recovery utilities for knowledge_qa.

Per spec §Architecture + §Data Model + §7.3. The router is a pure
function: facts + session → RouteAction. No I/O, no LLM, no mutation.
"""

from __future__ import annotations

import hashlib as _hashlib
import logging as _logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.flows.knowledge_review_models import KnowledgeReviewFacts
from src.flows.knowledge_session import KnowledgeSession


class RouteAction(Enum):
    ANSWER = "answer"  # [D]
    FOLLOWUP = "followup"  # [B2]
    REFUSE_LEGITIMATE = "refuse_legitimate"  # [B1] Type 1/2/3
    REFUSE_COERCED = "refuse_coerced"  # [B1] Type 2
    REFUSE_INFRA = "refuse_infra"  # [B1] Type 2


# 最多两轮澄清追问：followup_round 0/1 仍 needs_clarification → 再追问 (FOLLOWUP)；
# followup_round 2 仍 needs_clarification → 收口 (REFUSE_COERCED)。
MAX_FOLLOWUP_ROUNDS = 2


@dataclass(frozen=True)
class RouteDecision:
    kind: RouteAction
    out_of_scope_class: Optional[str] = None  # used by REFUSE_LEGITIMATE


def route(facts: KnowledgeReviewFacts, session: KnowledgeSession) -> RouteDecision:
    """Map (facts, session) → RouteDecision.

    Rule 1: out_of_scope → REFUSE_LEGITIMATE (phrase per out_of_scope_class)
    Rule 2: in_scope + ready → ANSWER
    Rule 3: in_scope + needs_clarification + followup_round < 2 → FOLLOWUP (最多两轮)
    Rule 4: in_scope + needs_clarification + followup_round == 2 → REFUSE_COERCED
    """
    if facts.scope == "out_of_scope":
        return RouteDecision(
            kind=RouteAction.REFUSE_LEGITIMATE,
            out_of_scope_class=facts.out_of_scope_class,
        )
    if facts.answer_readiness == "ready":
        return RouteDecision(kind=RouteAction.ANSWER)
    if facts.answer_readiness == "needs_clarification":
        if session.followup_round < MAX_FOLLOWUP_ROUNDS:
            return RouteDecision(kind=RouteAction.FOLLOWUP)
        return RouteDecision(kind=RouteAction.REFUSE_COERCED)
    raise RuntimeError(f"unroutable facts: {facts}")


# --- Exit recovery window (per §7.3) ---
#
# Strong corrective markers anywhere in the utterance.
# Leading-position markers (only count when they appear at chars [0:2]).
# NOT hot-reloadable: process restart required; CI hashes into startup log.

EXIT_RECOVERY_BACKCHANNEL_RE = re.compile(r"^(嗯嗯|哦哦|啊嗯|好的好的|嗯|啊|哦|是|好的|行|对|可以)[，。、！～\s]*$")
EXIT_RECOVERY_CORRECTIVE_STRONG_RE = re.compile(r"(不是|不对|等等|等一下|刚才那个|我还想|我刚才|刚说的)")
EXIT_RECOVERY_CORRECTIVE_LEADING_RE = re.compile(r"^(诶|哎|那个我)")

# Operator-visible startup hash (logged once at module import).
_LOG = _logging.getLogger(__name__)
_RE_BLOB = "\n".join(
    [
        EXIT_RECOVERY_BACKCHANNEL_RE.pattern,
        EXIT_RECOVERY_CORRECTIVE_STRONG_RE.pattern,
        EXIT_RECOVERY_CORRECTIVE_LEADING_RE.pattern,
    ]
)
_LOG.info(
    "knowledge_router.regex.loaded sha256:%s",
    _hashlib.sha256(_RE_BLOB.encode("utf-8")).hexdigest(),
)


def is_backchannel(user_text: str) -> bool:
    """True iff user_text is a pure ambiguous backchannel that should arm the recovery window."""
    return bool(EXIT_RECOVERY_BACKCHANNEL_RE.match(user_text.strip()))


def is_corrective_recovery_hit(user_text: str) -> bool:
    """True iff user_text contains a corrective marker (strong anywhere
    OR leading-position) — should re-DIVERT into [A] with raw user_text.

    Leading-marker check uses re.match (anchored at position 0). The earlier
    draft used search(user_text[:2]) which silently dropped the 3-char
    "那个我" marker — Codex Gate 3 No-Ship #6.
    """
    if EXIT_RECOVERY_CORRECTIVE_STRONG_RE.search(user_text):
        return True
    if EXIT_RECOVERY_CORRECTIVE_LEADING_RE.match(user_text):
        return True
    return False
