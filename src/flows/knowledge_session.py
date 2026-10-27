"""Mutable session state for an in-flight knowledge_qa interaction.

Per spec §Data Model. Lifecycle: created on first DIVERT, closed on RESUME.
At most ONE session is `current` at a time; up to 3 per suspended SOP node
(depth cap).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from src.flows.knowledge_review_models import KnowledgeReviewFacts


@dataclass
class KnowledgeSession:
    session_id: str
    divert_entry_turn_id: str
    suspended_sop_node: str
    followup_round: int = 0  # 0/1/2; incremented at TTS audio start (cap = 2 followups)
    followup_heard_status: Literal["n/a", "pending", "completed", "barge_in"] = "n/a"
    closed: bool = False  # True after RESUME or escalate
    last_facts: Optional[KnowledgeReviewFacts] = None


@dataclass
class KnowledgeReviewState:
    """Lives on FlowState. One per call."""

    current_session: Optional[KnowledgeSession] = None
    sessions_in_current_suspension: int = 0
    exit_recovery_armed: bool = False
