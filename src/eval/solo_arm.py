"""Solo-arch evaluation arm: drives SoloDriverProcessor turn-by-turn against the
study2_v3 SimulatedPatient and scores coverage with the SAME scorers as the brain
arm — a clean A/B without touching the protected faithfulness run paths
(driver.py / spa_agent.py / our_arm.py are NEVER imported here).

Public surface
--------------
run_solo_arm(card, *, drive_assistant_turn, patient_respond_fn,
             coverage_judge_fn, spa_system_prompt, max_turns) -> SoloArmResult

All LLM calls are injectable so this module is unit-testable with stubs and
production-wireable without changing any logic here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from src.eval.study2_v3.cards import Card
from src.eval.study2_v3.coverage_judge import (
    aggregate_coverage,
    build_coverage_judge_messages,
    fields_for_judge,
    normalize_coverage_judge_result,
)
from src.eval.study2_v3.simulated_patient import SimulatedPatient
from src.eval.study2_v3.spa_prompt import build_spa_system_prompt


@dataclass
class SoloArmResult:
    """Scored outcome of one solo-arm run."""

    case_id: str
    transcript: str
    coverage: dict[str, Any]
    turns: int


async def run_solo_arm(
    card: Card,
    *,
    drive_assistant_turn: Callable[[str], Awaitable[str]],
    patient_respond_fn: Callable[[list[dict]], Awaitable[dict]],
    coverage_judge_fn: Callable[[list[dict]], Awaitable[Any]],
    spa_system_prompt: str | None = None,
    max_turns: int = 24,
) -> SoloArmResult:
    """Closed-loop A/B arm: patient speaks → solo assistant replies → repeat.

    Parameters
    ----------
    card:
        Study-2 patient card with fields, persona, and node facts.
    drive_assistant_turn:
        Coroutine that feeds one patient utterance into the solo pipeline and
        returns the assistant's spoken text.  Production wires this to a
        ``SoloDriverProcessor`` harness; unit tests stub it.  Pass ``""`` to
        request the opening (assistant-first) turn.
    patient_respond_fn:
        Coroutine matching ``Callable[[list[dict]], Awaitable[dict]]`` — the
        LLM backend used by ``SimulatedPatient``.  Injectable for stubs.
    coverage_judge_fn:
        Coroutine matching ``Callable[[list[dict]], Awaitable[Any]]`` — accepts
        a ``messages`` list (system + user) and returns either a parsed JSON
        ``dict`` or a raw text string.  The raw-text fallback path is handled
        via ``normalize_coverage_judge_result``.
    spa_system_prompt:
        System prompt for the simulated patient.  Defaults to the study2_v3
        canonical prompt built by ``build_spa_system_prompt()``.
    max_turns:
        Hard cap on patient turns to prevent infinite loops.

    Returns
    -------
    SoloArmResult
        Transcript, coverage dict (completeness / accuracy / refusal_compliance
        + field-level verdicts), and turn count.
    """
    if spa_system_prompt is None:
        spa_system_prompt = build_spa_system_prompt()

    patient = SimulatedPatient(patient_respond_fn, spa_system_prompt)
    history_lines: list[str] = []
    disclosed: list[str] = []

    # --- opening turn: assistant speaks first (patient utterance = "") ---
    assistant_text = await drive_assistant_turn("")
    if assistant_text:
        history_lines.append(f"客服：{assistant_text}")

    turns = 0
    while turns < max_turns:
        turns += 1

        pt = await patient.respond(
            card,
            history="\n".join(history_lines),
            current_assistant_text=assistant_text,
            disclosed_fields=disclosed,
        )

        if pt.patient_utterance:
            history_lines.append(f"患者：{pt.patient_utterance}")

        disclosed = sorted(set(disclosed) | set(pt.revealed_fields))

        if pt.call_should_end:
            break

        assistant_text = await drive_assistant_turn(pt.patient_utterance)
        if assistant_text:
            history_lines.append(f"客服：{assistant_text}")

    transcript = "\n".join(history_lines)

    # --- score coverage with the same frozen judge as the brain arm ---
    fields = fields_for_judge(card)
    messages = build_coverage_judge_messages(transcript=transcript, fields=fields)
    raw_result = await coverage_judge_fn(messages)

    # coverage_judge_fn may return a parsed dict or a raw string depending on
    # the backend wrapper the caller provides.
    if isinstance(raw_result, dict):
        parsed_json = raw_result
        raw_text = ""
    else:
        # Treat as raw text; normalization will mark judge_valid=False.
        parsed_json = None
        raw_text = str(raw_result)

    norm = normalize_coverage_judge_result(parsed_json, raw_text=raw_text)
    coverage = aggregate_coverage(card, norm)
    coverage["verdicts"] = norm

    return SoloArmResult(
        case_id=card.case_id,
        transcript=transcript,
        coverage=coverage,
        turns=turns,
    )
