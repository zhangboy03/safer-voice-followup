"""Closed-loop driver for Study 2 v3.

Reuses the exact v2 被测链路 (Speech LLM -> Fast Judge -> StateCommitter ->
FlowRuntime -> conditional Stage2). The ONLY change from v2 is the source of
the patient turn: instead of a pre-baked fixed script, each turn is produced
live by a SimulatedPatient conditioned on the real running transcript.

The driver receives all heavy collaborators (flow, judge runner, committer,
speech runner, patient) by injection so the orchestration can be unit-tested
with fakes; production wiring lives in the runner script.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from src.eval.study2_v3.cards import Card
from src.eval.study2_v3.simulated_patient import SimulatedPatient
from src.flows.question_predicate import speech_asked_question
from src.flows.runtime import KNOWLEDGE_QA_NODE

SpeechRunner = Callable[..., Awaitable[dict]]


def render_history(turns: list[dict[str, Any]], opening_text: str) -> str:
    """Build the transcript string handed to the simulated patient. This IS the
    closed-loop guarantee surface: the patient only ever sees real prior turns."""
    lines: list[str] = []
    if opening_text:
        lines.append(f"Assistant: {opening_text}")
    for t in turns:
        if t.get("user_text"):
            lines.append(f"Patient: {t['user_text']}")
        combined = t.get("combined_assistant_text", "")
        if combined:
            lines.append(f"Assistant: {combined}")
    return "\n".join(lines)


def _accepted_fields(decision) -> list[dict[str, Any]]:
    out = []
    for commit in decision.field_commits:
        out.append(
            {
                "field": commit.field,
                "value": commit.memory_value.value,
                "status": commit.memory_value.status,
                "evidence": commit.memory_value.evidence,
            }
        )
    return out


def _strip_markers(text: str) -> str:
    out = text or ""
    for marker in ("[TRANSFER]", "[GOODBYE]"):
        out = out.replace(marker, "")
    return out.strip()


async def run_journey(
    card: Card,
    *,
    flow,
    fast_judge_runner,
    committer,
    speech_runner: SpeechRunner,
    patient: SimulatedPatient,
    run_id: str,
    max_turns: int = 18,
    on_opening=None,
    on_turn=None,
) -> dict[str, Any]:
    actual_items: list[dict[str, Any]] = []
    disclosed_fields: list[str] = []

    # assistant opening
    opening_text = ""
    opening_plan = await flow.prepare_node_entry()
    if opening_plan is not None:
        res = await speech_runner(opening_plan.messages, session_id=run_id, turn_id="opening")
        opening_text = _strip_markers(res["clean_text"])
        flow._history.append_assistant(opening_text, node_id="greeting")
        if on_opening is not None and opening_text:
            await on_opening(opening_text)

    incomplete = True
    for idx in range(1, max_turns + 1):
        turn_id = f"turn-{idx}"
        node_before = flow._state.current_node
        previous_assistant_text = flow._history.get_last_assistant_text() or ""

        # --- live simulated patient turn (closed loop) ---
        history = render_history(actual_items, opening_text)
        patient_turn = await patient.respond(
            card,
            history=history,
            current_assistant_text=previous_assistant_text,
            disclosed_fields=list(disclosed_fields),
        )
        user_text = patient_turn.patient_utterance
        if not user_text:
            user_text = "嗯。"

        # --- Stage 1 speech (assistant reply) ---
        flow.begin_user_turn(turn_id)
        stage1_plan = await flow.prepare_user_turn(user_text, is_speculative=False)
        stage1_res = await speech_runner(stage1_plan.messages, session_id=run_id, turn_id=f"{turn_id}:stage1")
        stage1_text = _strip_markers(stage1_res["clean_text"])

        # --- Fast Judge (被测系统) ---
        messages, identity = flow.build_fast_judge_messages(
            user_text=user_text,
            assistant_text=stage1_res["raw_text"],
            speech_result_id=f"{turn_id}:stage1",
        )
        envelope = await fast_judge_runner.run(
            messages=messages,
            identity=identity,
            session_id=run_id,
            turn_id=turn_id,
            stage1_text=stage1_res["raw_text"],
        )
        decision = committer.commit_fast_judge_result(envelope, expected=identity)
        action_result = await flow.apply_node_action(decision)

        # --- conditional Stage 2 ---
        stage2_text = ""
        stage2_plan = None
        if action_result.followup_requested and not speech_asked_question(stage1_text, interrogatives=True):
            directive = decision.speech_directive.text if decision.speech_directive else ""
            stage2_plan = await flow.prepare_stage1_followup(
                action_result.followup_missing_fields, directive, stage1_text
            )
        elif action_result.advanced and action_result.node_after != KNOWLEDGE_QA_NODE:
            stage2_plan = await flow.prepare_stage2_entry(previous_speech_text=stage1_text)

        if stage2_plan is not None:
            stage2_res = await speech_runner(stage2_plan.messages, session_id=run_id, turn_id=f"{turn_id}:stage2")
            stage2_text = _strip_markers(stage2_res["clean_text"])

        combined = "\n".join(t for t in [stage1_text, stage2_text] if t)
        flow._history.append_user(user_text)
        flow._history.append_assistant(combined, node_id=node_before)
        disclosed_fields.extend(f for f in patient_turn.revealed_fields if f not in disclosed_fields)

        accepted = _accepted_fields(decision)
        item = {
            "run_id": run_id,
            "case_id": card.case_id,
            "turn_id": turn_id,
            "node_before": node_before,
            "previous_assistant_text": previous_assistant_text,
            "user_text": user_text,
            "stage1_text": stage1_text,
            "stage2_text": stage2_text,
            "combined_assistant_text": combined,
            "actual_node_action": decision.node_action,
            "node_after": action_result.node_after,
            "accepted_fields": accepted,
            "asked_question": speech_asked_question(combined, interrogatives=True),
            "revealed_fields": list(patient_turn.revealed_fields),
            "fast_judge_action": decision.node_action,
            "ttft_ms": stage1_res.get("ttft_ms"),
        }
        actual_items.append(item)

        if on_turn is not None:
            await on_turn(item)

        if decision.node_action == "terminal" or patient_turn.call_should_end:
            incomplete = False
            break

    return {
        "case_id": card.case_id,
        "run_id": run_id,
        "opening_text": opening_text,
        "actual_items": actual_items,
        "incomplete_journey": incomplete,
        "turn_count": len(actual_items),
    }
