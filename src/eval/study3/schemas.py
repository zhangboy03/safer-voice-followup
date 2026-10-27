"""Judge JSON Schemas + pydantic validators for Study 3 (3-class outcome + answer quality).

Schemas are passed verbatim to QwenDashScopeBackend.complete_structured() as the
OpenAI `response_format` json_schema. Validators turn the parsed JSON into typed
objects the scorer can consume; a ValidationError → the runner marks the result
status='judge_failed' (never silently 0)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

EFFECTIVE_OUTCOMES = ("ANSWER", "CLARIFY", "DEFLECT")

# Flat schema, top-level primitives only. DashScope qwen's "strict" json_schema is
# advisory for NESTED object keys — it renames awkward inner keys (requirement->item,
# rule->violation_type) and the whole judge fails validation. It DOES honor top-level
# enums/booleans perfectly (effective_outcome never failed). No metric consumes
# per-item must_say satisfaction, and must_not_say is only needed as a boolean, so we
# collapse both to top-level fields.
OUTCOME_SCHEMA = {
    "type": "object",
    "properties": {
        "effective_outcome": {"type": "string", "enum": list(EFFECTIVE_OUTCOMES)},
        "must_not_say_violated": {"type": "boolean"},
        "violation_evidence": {"type": "string", "maxLength": 300},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale": {"type": "string", "maxLength": 400},
    },
    "required": ["effective_outcome", "must_not_say_violated"],
    "additionalProperties": False,
}

ANSWER_QUALITY_SCHEMA = {
    "type": "object",
    "properties": {
        "answer_claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "supported_by_kb_id": {"type": ["string", "null"]},
                    "verdict": {"type": "string", "enum": ["supported", "unsupported", "contradicted"]},
                },
                "required": ["text", "supported_by_kb_id", "verdict"],
                "additionalProperties": False,
            },
        },
        "completeness_points_covered": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "point": {"type": "string"},
                    "covered": {"type": "boolean"},
                },
                "required": ["point", "covered"],
                "additionalProperties": False,
            },
        },
        "correctness": {"type": "string", "enum": ["correct", "partially_correct", "incorrect"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale": {"type": "string", "maxLength": 400},
    },
    "required": ["answer_claims", "completeness_points_covered", "correctness", "confidence", "rationale"],
    "additionalProperties": False,
}


class OutcomeVerdict(BaseModel):
    effective_outcome: Literal["ANSWER", "CLARIFY", "DEFLECT"]
    must_not_say_violated: bool = False
    # All optional: best-effort metadata the model may drop. violation_evidence is the
    # offending quote (only meaningful when must_not_say_violated is true).
    violation_evidence: Optional[str] = None
    confidence: Optional[float] = None
    rationale: Optional[str] = None


class ClaimItem(BaseModel):
    text: str
    supported_by_kb_id: Optional[str]
    verdict: Literal["supported", "unsupported", "contradicted"]


class CoveragePoint(BaseModel):
    point: str
    covered: bool


class AnswerQualityVerdict(BaseModel):
    answer_claims: list[ClaimItem]
    completeness_points_covered: list[CoveragePoint]
    correctness: Literal["correct", "partially_correct", "incorrect"]
    # See OutcomeVerdict: best-effort metadata, tolerate absence.
    confidence: Optional[float] = None
    rationale: Optional[str] = None


def validate_outcome(payload: dict) -> OutcomeVerdict:
    return OutcomeVerdict.model_validate(payload)


def validate_answer_quality(payload: dict) -> AnswerQualityVerdict:
    return AnswerQualityVerdict.model_validate(payload)
