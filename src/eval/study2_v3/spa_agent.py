"""Static-Prompt-Agent free-talk arm (JourneyBench SPA baseline).

No Fast Judge, no FlowRuntime, no stage2 — just a single static SOP prompt
free-running the conversation against the SimulatedPatient. Produces a flat
transcript scored later by the SAME transcript judge as the our-system arm.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from src.eval.study2_v3.cards import Card
from src.eval.study2_v3.driver import render_history
from src.eval.study2_v3.simulated_patient import SimulatedPatient
from src.eval.study2_v3.spa_prompt import build_spa_system_prompt

SpeechRunner = Callable[..., Awaitable[dict]]


def _build_messages(system_prompt: str, turns: list[dict[str, Any]], user_text: str) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for t in turns:
        messages.append({"role": "user", "content": t["user_text"]})
        messages.append({"role": "assistant", "content": t["assistant_text"]})
    messages.append({"role": "user", "content": user_text})
    return messages


async def run_spa_journey(
    card: Card,
    *,
    speech_runner: SpeechRunner,
    patient: SimulatedPatient,
    run_id: str,
    max_turns: int = 18,
    on_turn=None,
) -> dict[str, Any]:
    system_prompt = build_spa_system_prompt()
    actual_items: list[dict[str, Any]] = []
    disclosed_fields: list[str] = []

    incomplete = True
    for idx in range(1, max_turns + 1):
        turn_id = f"spa-turn-{idx}"
        history = render_history(
            [{"user_text": it["user_text"], "combined_assistant_text": it["assistant_text"]} for it in actual_items],
            opening_text="",
        )
        last_assistant = actual_items[-1]["assistant_text"] if actual_items else ""
        patient_turn = await patient.respond(
            card,
            history=history,
            current_assistant_text=last_assistant,
            disclosed_fields=list(disclosed_fields),
        )
        user_text = patient_turn.patient_utterance or "嗯。"

        messages = _build_messages(system_prompt, actual_items, user_text)
        res = await speech_runner(messages, session_id=run_id, turn_id=turn_id)

        actual_items.append(
            {
                "run_id": run_id,
                "case_id": card.case_id,
                "turn_id": turn_id,
                "user_text": user_text,
                "assistant_text": res["clean_text"],
                "ttft_ms": res.get("ttft_ms"),
                "revealed_fields": list(patient_turn.revealed_fields),
            }
        )
        if on_turn is not None:
            await on_turn(actual_items[-1])
        disclosed_fields.extend(f for f in patient_turn.revealed_fields if f not in disclosed_fields)

        if patient_turn.call_should_end:
            incomplete = False
            break

    return {
        "case_id": card.case_id,
        "run_id": run_id,
        "arm": "spa",
        "actual_items": actual_items,
        "incomplete_journey": incomplete,
        "turn_count": len(actual_items),
    }
