"""Validators for knowledge_qa flow.

Per spec §4 facts validation table, §6 [B2] post-validator + fallback.
Raises a single FactsValidationError with a reason code that maps directly
to the INFRA_FALLBACK metric counter."""

from __future__ import annotations

from src.flows.knowledge_review_models import KnowledgeReviewFacts


class FactsValidationError(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


_FORBIDDEN_FOLLOWUP_PHRASES = (
    "我帮您查",
    "稍等",
    "稍微等",
    "为了帮您",
    "我先了解",
    "我马上",
    "我去看",
    "我看看",
)


def validate_followup(text: str, missing_prereqs: list[str]) -> tuple[bool, str]:
    """Return (True, original_text) if the [B2] LLM output passes all rules;
    else (False, deterministic_fallback_string)."""
    if len(text) > 45:
        return False, _fallback_question(missing_prereqs)
    if "？" not in text and "?" not in text:
        return False, _fallback_question(missing_prereqs)
    if any(p in text for p in _FORBIDDEN_FOLLOWUP_PHRASES):
        return False, _fallback_question(missing_prereqs)
    if "吧？" in text or "吧?" in text:
        return False, _fallback_question(missing_prereqs)
    return True, text


def _fallback_question(missing_prereqs: list[str]) -> str:
    assert 1 <= len(missing_prereqs) <= 3, (
        f"_fallback_question called with {len(missing_prereqs)} slots; "
        "facts validator should have caught this before routing"
    )
    if len(missing_prereqs) == 1:
        return f"方便告诉我下{missing_prereqs[0]}是什么吗？"
    if len(missing_prereqs) == 2:
        return f"方便告诉我下{missing_prereqs[0]}和{missing_prereqs[1]}吗？"
    return f"方便告诉我下{missing_prereqs[0]}、{missing_prereqs[1]}还有{missing_prereqs[2]}吗？"


def validate_facts(facts: KnowledgeReviewFacts, *, min_confidence: float) -> None:
    """Raise FactsValidationError on any inconsistency. Called BEFORE routing.

    Per spec §Data Model facts validation table + r3+codex3 round-3 High #1
    slot_shape_violation row."""
    if facts.scope == "out_of_scope":
        if facts.out_of_scope_class is None:
            raise FactsValidationError("facts_inconsistency", "out_of_scope without class")
        if facts.answer_readiness is not None:
            raise FactsValidationError("facts_inconsistency", "out_of_scope must have readiness=null")
        if facts.missing_prerequisites:
            raise FactsValidationError(
                "facts_inconsistency",
                "out_of_scope must have empty missing_prerequisites",
            )
    else:  # in_scope
        if facts.out_of_scope_class is not None:
            raise FactsValidationError("facts_inconsistency", "in_scope must have class=null")
        if facts.answer_readiness not in ("ready", "needs_clarification"):
            raise FactsValidationError("facts_inconsistency", "in_scope must have readiness")
        if facts.answer_readiness == "needs_clarification" and not facts.missing_prerequisites:
            raise FactsValidationError(
                "facts_inconsistency",
                "needs_clarification requires non-empty missing_prerequisites",
            )
        if facts.answer_readiness == "ready" and facts.missing_prerequisites:
            raise FactsValidationError(
                "facts_inconsistency",
                "ready requires empty missing_prerequisites",
            )

    # slot_shape_violation (defense-in-depth — strict JSON schema is supposed to enforce)
    if len(facts.missing_prerequisites) > 3:
        raise FactsValidationError(
            "slot_shape_violation",
            f"got {len(facts.missing_prerequisites)} slots, max 3",
        )
    over_long = [s for s in facts.missing_prerequisites if len(s) > 12]
    if over_long:
        raise FactsValidationError(
            "slot_shape_violation",
            f"slots over 12 chars: {over_long}",
        )

    if not (0.0 <= facts.confidence <= 1.0):
        raise FactsValidationError(
            "facts_inconsistency",
            f"confidence {facts.confidence} out of [0,1]",
        )
    if facts.confidence < min_confidence:
        raise FactsValidationError(
            "low_confidence",
            f"confidence {facts.confidence} < min {min_confidence}",
        )
