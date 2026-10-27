"""Live simulated patient: verbalizes one natural utterance per turn, strictly
obeying the card's per-field disclosure policy, conditioned on the assistant's
actual words (closed loop).

The component takes an injectable ``respond_fn`` so the orchestration logic can
be unit-tested with a stub backend while production wires it to the LLM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from src.eval.study2_v3.cards import Card


@dataclass
class PatientTurn:
    patient_utterance: str
    revealed_fields: list[str]
    call_should_end: bool


def build_patient_payload(
    card: Card,
    *,
    history: str,
    current_assistant_text: str,
    disclosed_fields: list[str],
) -> dict[str, Any]:
    return {
        "case_id": card.case_id,
        "persona": card.persona,
        "node_facts": card.node_facts,
        "field_disclosure": [
            {
                "field": f.name,
                "policy": f.policy,
                "node": f.node,
                "value": f.gold_value,
                "status": f.status,
            }
            for f in card.fields
        ],
        "conversation_history": history,
        "current_assistant_text": current_assistant_text,
        "disclosed_fields": disclosed_fields,
        "instruction": "只输出 JSON：patient_utterance / revealed_fields / call_should_end。",
    }


class SimulatedPatient:
    def __init__(
        self,
        respond_fn: Callable[[list[dict]], Awaitable[dict]],
        system_prompt: str,
    ) -> None:
        self._respond = respond_fn
        self._sys = system_prompt

    async def respond(
        self,
        card: Card,
        *,
        history: str,
        current_assistant_text: str,
        disclosed_fields: list[str],
    ) -> PatientTurn:
        payload = build_patient_payload(
            card,
            history=history,
            current_assistant_text=current_assistant_text,
            disclosed_fields=disclosed_fields,
        )
        messages = [
            {"role": "system", "content": self._sys},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        raw = await self._respond(messages)
        if not isinstance(raw, dict):
            raw = {}
        revealed = raw.get("revealed_fields", []) or []
        if not isinstance(revealed, list):
            revealed = []
        return PatientTurn(
            patient_utterance=str(raw.get("patient_utterance", "") or "").strip(),
            revealed_fields=[str(x) for x in revealed],
            call_should_end=bool(raw.get("call_should_end", False)),
        )
