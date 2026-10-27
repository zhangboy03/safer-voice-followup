"""Minimal stand-in for FlowRuntime state, scoped to the KB modules' contracts.
A KB *episode* = one patient question + its <=2 clarify rounds + the answer.
Stable session_id across barge-in retry of the SAME episode; new per question."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from src.flows.knowledge_session import KnowledgeReviewState, KnowledgeSession


@dataclass
class FlowStateShim:
    """Exactly the surface callback_queue.enqueue + transfer_service read."""

    knowledge_review_state: KnowledgeReviewState
    patient_info: dict
    collected_fields: dict = field(default_factory=dict)


@dataclass
class KbEpisode:
    session: KnowledgeSession
    flow_state: FlowStateShim


class KbEpisodeManager:
    def __init__(self, *, patient_info: dict | None = None):
        self._patient_info = patient_info or {}
        self._episode: KbEpisode | None = None

    def open_or_continue(self, *, turn_id: str) -> KbEpisode:
        if self._episode is None:
            session = KnowledgeSession(
                session_id=str(uuid.uuid4()),
                divert_entry_turn_id=turn_id,
                suspended_sop_node="",  # no SOP node graph in solo
            )
            krs = KnowledgeReviewState(current_session=session)
            self._episode = KbEpisode(
                session=session,
                flow_state=FlowStateShim(knowledge_review_state=krs, patient_info=self._patient_info),
            )
        return self._episode

    def mark_followup(self) -> None:
        # increment at TTS-audio-start (caller decides the moment); see driver.
        if self._episode is not None:
            self._episode.session.followup_round += 1

    def close(self) -> None:
        if self._episode is not None:
            self._episode.session.closed = True
        self._episode = None

    @property
    def is_open(self) -> bool:
        return self._episode is not None
