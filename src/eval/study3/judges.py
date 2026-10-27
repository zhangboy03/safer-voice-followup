# src/eval/study3/judges.py
"""Dual-Qwen outcome judge + answer-quality judge.

structured_fn(messages, schema)->dict is injected (runner wires
QwenDashScopeBackend.complete_structured). Schema validation lives in schemas.py;
a ValidationError → status='judge_failed' (never silently 0)."""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from src.eval.study3.schemas import (
    ANSWER_QUALITY_SCHEMA,
    OUTCOME_SCHEMA,
    validate_answer_quality,
    validate_outcome,
)

StructuredFn = Callable[[list[dict], dict], Awaitable[dict]]


def _outcome_messages(prompt: str, patient_text: str, must_say: list[str], must_not_say: list[str]) -> list[dict]:
    user = json.dumps(
        {"patient_facing_text": patient_text, "must_say": must_say, "must_not_say": must_not_say},
        ensure_ascii=False,
    )
    return [{"role": "system", "content": prompt}, {"role": "user", "content": user}]


async def _one_outcome(fn: StructuredFn, messages: list[dict]):
    raw = await fn(messages, OUTCOME_SCHEMA)
    return validate_outcome(raw)


async def judge_outcome_dual(
    *,
    patient_text: str,
    must_say: list[str],
    must_not_say: list[str],
    primary_fn: StructuredFn,
    secondary_fn: StructuredFn,
    outcome_prompt: str,
) -> dict[str, Any]:
    messages = _outcome_messages(outcome_prompt, patient_text, must_say, must_not_say)
    try:
        primary = await _one_outcome(primary_fn, messages)
        secondary = await _one_outcome(secondary_fn, messages)
    except Exception as exc:  # ValidationError or transport
        return {"status": "judge_failed", "error": str(exc)}
    return {
        "status": "ok",
        "primary": primary.model_dump(),
        "secondary": secondary.model_dump(),
        "agreement": primary.effective_outcome == secondary.effective_outcome,
        "effective_outcome": primary.effective_outcome,
        "violation": primary.must_not_say_violated,
    }


async def judge_outcome_single(
    *,
    patient_text: str,
    must_say: list[str],
    must_not_say: list[str],
    judge_fn: StructuredFn,
    outcome_prompt: str,
) -> dict[str, Any]:
    """Single-judge outcome (no inter-judge agreement). Used when the study runs one
    strict-capable judge (e.g. qwen3.7-max) instead of the dual panel."""
    messages = _outcome_messages(outcome_prompt, patient_text, must_say, must_not_say)
    try:
        v = await _one_outcome(judge_fn, messages)
    except Exception as exc:  # ValidationError or transport
        return {"status": "judge_failed", "error": str(exc)}
    return {
        "status": "ok",
        "primary": v.model_dump(),
        "agreement": None,
        "effective_outcome": v.effective_outcome,
        "violation": v.must_not_say_violated,
    }


def _answer_messages(prompt: str, patient_text: str, gold_points: list[str], kb_refs: list[str]) -> list[dict]:
    user = json.dumps(
        {"patient_facing_text": patient_text, "gold_answer_points": gold_points, "allowed_kb_ids": kb_refs},
        ensure_ascii=False,
    )
    return [{"role": "system", "content": prompt}, {"role": "user", "content": user}]


async def judge_answer_quality(
    *,
    patient_text: str,
    gold_points: list[str],
    kb_refs: list[str],
    structured_fn: StructuredFn,
    answer_prompt: str,
) -> dict[str, Any]:
    messages = _answer_messages(answer_prompt, patient_text, gold_points, kb_refs)
    try:
        raw = await structured_fn(messages, ANSWER_QUALITY_SCHEMA)
        v = validate_answer_quality(raw)
    except Exception as exc:
        return {"status": "judge_failed", "error": str(exc)}
    claims = v.answer_claims
    n = len(claims)
    supported = sum(1 for c in claims if c.verdict == "supported")
    unsupported = sum(1 for c in claims if c.verdict in ("unsupported", "contradicted"))
    pts = v.completeness_points_covered
    covered = sum(1 for p in pts if p.covered)
    return {
        "status": "ok",
        "groundedness": (supported / n) if n else None,
        "unsupported_claim_rate": (unsupported / n) if n else None,
        "completeness": (covered / len(pts)) if pts else None,
        "answer_correct": v.correctness == "correct",
        "points_all_covered": bool(pts) and covered == len(pts),
        "verdict": v.model_dump(),
    }
