"""Shared gate for the in-turn stage1-followup decision (spec §2, §5).

This is the single source of truth for "should we voice an in-turn
stage1-followup second segment?". Both the live pipeline
(``src.processors.llm_processor``) and the offline eval replay harness
(``EMNLP/.../run_v2_batch_replay.py``) import this so their Stage 2 gating
cannot drift apart. The two paths drifted once (the replay fired a followup
on ``ack_only`` posture while the live path suppressed it), which made the
eval over-count redundant second segments — see study_2 v2_001_t08.

The function is intentionally dependency-free: it only reads attributes off
the committed decision, so it stays importable from anywhere without pulling
in the processor / pipecat stack.
"""

from __future__ import annotations


def should_fire_stage1_followup(
    decision, collectable_missing: list[str], any_under_cap: bool, det_asked: bool
) -> tuple[bool, str]:
    """Decide whether to voice an in-turn stage1-followup (spec §2, §5).

    Caller handles the flag check, user-speaking guard, and fast-judge-path check.

    AND-gated on the deterministic no-question signal (spec §5): even when the
    Judge calls ``closed_unasked``, if Stage 1's text actually contains a question
    (``det_asked``) we do NOT fire — this defends against a spurious posture call
    re-asking a question the patient is already answering.
    """
    if decision.node_action != "followup":
        return False, "not_followup"
    posture = getattr(decision, "stage1_posture", "asked")
    if posture == "asked":
        return False, "posture_asked"
    if posture == "ack_only":
        return False, "posture_ack_only"
    if det_asked:
        return False, "det_asked_disagree"
    if not collectable_missing:
        return False, "no_collectable_missing"
    if not any_under_cap:
        return False, "capped"
    return True, "allowed"
