"""Coverage-based scorer for Study 2 v3.

Replaces v2's strict per-turn symbol equality. The journey is scored on:
milestone coverage, SOP order validity, journey-global field recall/precision,
evidence-supported commit rate, premature-advance / stale-followup structural
violations, and SJCS_coverage. Every coverage gap is attributed to either the
agent (agent_miss) or the simulator (simulator_gap), or flagged for LLM audit.

See spec module 4.
"""

from __future__ import annotations

from typing import Any

from src.eval.study2_v3.cards import SOP_ORDER, Card, Milestone


def _accepted_names(item: dict[str, Any]) -> set[str]:
    return {f["field"] for f in item.get("accepted_fields", [])}


def attribute_gap(
    field_name: str,
    node: str,
    policy: str,
    actual_items: list[dict[str, Any]],
) -> str:
    """Decide whether an uncollected required field is the agent's fault or the
    simulator's. Decision tree from spec module 4."""
    node_turns = [it for it in actual_items if it.get("node_before") == node]
    asked_turns = [it for it in node_turns if it.get("asked_question")]
    revealed = any(field_name in it.get("revealed_fields", []) for it in actual_items)

    # 1. Never asked at the node at all -> agent failed to engage.
    if not asked_turns:
        return "agent_miss"
    # 2. Patient did reveal it, but it never got committed -> committer/judge failed.
    if revealed:
        return "agent_miss"
    # 3. on_followup but the agent only asked once (no follow-up) -> agent_miss.
    if policy == "on_followup" and len(asked_turns) < 2:
        return "agent_miss"
    # 4. Asked (and re-asked where required) but the simulator never gave it.
    if asked_turns:
        return "simulator_gap"
    return "needs_audit"


def _order_validity(actual_items: list[dict[str, Any]]) -> int:
    """1 if every node transition moves forward in SOP_ORDER or stays put."""
    for it in actual_items:
        before = it.get("node_before")
        after = it.get("node_after")
        if before not in SOP_ORDER or after not in SOP_ORDER:
            # unknown node (e.g. knowledge_qa side path) — do not penalize ordering
            continue
        if SOP_ORDER.index(after) < SOP_ORDER.index(before):
            return 0
    return 1


def score_journey(
    card: Card,
    milestones: list[Milestone],
    actual_items: list[dict[str, Any]],
    field_results: list[dict[str, Any]],
) -> dict[str, Any]:
    # --- field gold sets (journey-global) ---
    required_gold = {f.name for f in card.fields if f.scoring == "required" and f.policy != "refuse"}
    optional_gold = {f.name for f in card.fields if f.scoring == "optional"}
    policy_of = {f.name: f.policy for f in card.fields}
    node_of = {f.name: f.node for f in card.fields}

    correct = {r["field"] for r in field_results if r.get("field_correct")}
    supported = {r["field"] for r in field_results if r.get("support_label") == "supported"}

    required_correct = required_gold & correct
    optional_correct = optional_gold & correct
    committed_required = {r["field"] for r in field_results if r["field"] in required_gold}

    field_recall = len(required_correct) / len(required_gold) if required_gold else 1.0
    field_precision = len(required_correct) / len(committed_required) if committed_required else 1.0
    field_f1 = (
        2 * field_precision * field_recall / (field_precision + field_recall)
        if (field_precision + field_recall)
        else 0.0
    )
    evidence_rate = len(supported) / len(field_results) if field_results else 1.0

    # --- premature advance / stale followup (cumulative, evidence-aware) ---
    cumulative: set[str] = set()
    premature_advance = 0
    stale_followup = 0
    ms_by_node = {m.node: m for m in milestones}
    for it in actual_items:
        node = it.get("node_before")
        action = it.get("actual_node_action")
        ms = ms_by_node.get(node)
        if ms and action == "followup":
            # all required already collected (before this turn) -> asking again is stale
            if ms.required_fields and set(ms.required_fields) <= (cumulative & supported):
                stale_followup += 1
        cumulative |= _accepted_names(it)
        if ms and action in ("advance", "terminal"):
            need = set(ms.required_fields)
            if need and not need <= (cumulative & supported):
                premature_advance += 1

    # --- milestone coverage ---
    terminal_seen = any(it.get("actual_node_action") == "terminal" for it in actual_items)
    achieved = 0
    for m in milestones:
        if m.terminal:
            if terminal_seen:
                achieved += 1
            continue
        need = set(m.required_fields)
        if not need or need <= correct:
            achieved += 1
    milestone_coverage = achieved / len(milestones) if milestones else 1.0

    sop_order_validity = _order_validity(actual_items)

    # --- coverage gaps with attribution ---
    gaps: list[dict[str, Any]] = []
    for fname in sorted(required_gold - correct):
        gaps.append(
            {
                "field": fname,
                "node": node_of.get(fname, ""),
                "policy": policy_of.get(fname, ""),
                "attribution": attribute_gap(fname, node_of.get(fname, ""), policy_of.get(fname, ""), actual_items),
            }
        )

    sjcs = milestone_coverage * field_f1 * evidence_rate * sop_order_validity

    return {
        "milestone_coverage": milestone_coverage,
        "milestones_total": len(milestones),
        "milestones_achieved": achieved,
        "sop_order_validity": sop_order_validity,
        "field_recall": field_recall,
        "field_precision": field_precision,
        "field_f1": field_f1,
        "evidence_supported_commit_rate": evidence_rate,
        "optional_field_recall": (len(optional_correct) / len(optional_gold) if optional_gold else 1.0),
        "premature_advance_count": premature_advance,
        "stale_followup_count": stale_followup,
        "sjcs_coverage": sjcs,
        "gaps": gaps,
        "required_gold_count": len(required_gold),
        "required_correct_count": len(required_correct),
    }
