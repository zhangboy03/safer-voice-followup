"""Queue-backed patient: yields the next human-typed utterance to the brain loop.

Duck-types SimulatedPatient.respond so driver.run_journey / run_spa_journey can use it
unchanged. The human ends the call via the session (task cancel), not via this object,
so call_should_end is always False here.
"""

from __future__ import annotations

import asyncio

from src.eval.study2_v3.simulated_patient import PatientTurn


class HumanPatient:
    def __init__(self) -> None:
        self._inbox: asyncio.Queue[str] = asyncio.Queue()

    async def submit(self, text: str) -> None:
        await self._inbox.put(text)

    async def respond(self, card, *, history, current_assistant_text, disclosed_fields) -> PatientTurn:
        text = await self._inbox.get()
        return PatientTurn(patient_utterance=text, revealed_fields=[], call_should_end=False)
