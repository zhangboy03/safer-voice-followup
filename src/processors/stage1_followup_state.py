"""Per-call stage1-followup cap accounting (2026-05-25 spec §4.5).

Tracks how many times each (node, field) has been auto-asked via an in-turn
stage1-followup, so we re-ask at most `cap` times before force-advancing.
Counting is driven by the LLMProcessor and must be incremented only when the
followup audio actually started playing (playout-aware), never on inference
dispatch — see llm_processor wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Stage1FollowupState:
    # (node_id, field) -> times auto-asked this node-visit
    _asked: dict[tuple[str, str], int] = field(default_factory=dict)
    total_count: int = 0

    def under_cap(self, node_id: str, field_name: str, cap: int) -> bool:
        return self._asked.get((node_id, field_name), 0) < cap

    def record_asked(self, node_id: str, field_name: str) -> None:
        self._asked[(node_id, field_name)] = self._asked.get((node_id, field_name), 0) + 1
        self.total_count += 1

    def on_field_resolved(self, node_id: str, field_name: str) -> None:
        self._asked.pop((node_id, field_name), None)

    def on_node_change(self, new_node_id: str) -> None:
        # Drop all counters; a new node-visit starts fresh.
        self._asked.clear()
