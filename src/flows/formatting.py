"""SOP progress formatting utilities for system prompt injection."""

from __future__ import annotations

from collections import defaultdict

from src.flows.models import FieldStatus, FieldValue, FlowState

_DISPLAYABLE = frozenset({FieldStatus.CONFIRMED, FieldStatus.INFERRED})


def format_sop_progress(
    flow_state: FlowState,
    current_node: str,
    node_to_group: dict[str, str],
) -> str | None:
    """Format collected fields as SOP progress text for the system prompt.

    Rules:
    - Only CONFIRMED / INFERRED fields are shown.
    - Fields belonging to the *current* node are listed in detail.
    - Fields from other nodes are compressed into one line per group.
    - Returns ``None`` when there is nothing displayable.
    """
    # Filter to displayable fields only.
    displayable: dict[str, FieldValue] = {
        k: v for k, v in flow_state.collected_fields.items() if v.status in _DISPLAYABLE
    }
    if not displayable:
        return None

    # Group fields by source_node.
    by_node: dict[str, dict[str, FieldValue]] = defaultdict(dict)
    for field_name, fv in displayable.items():
        by_node[fv.source_node][field_name] = fv

    lines: list[str] = []

    # --- Completed groups (nodes != current_node) ---
    for node, fields in by_node.items():
        if node == current_node:
            continue
        group_label = node_to_group.get(node, node)
        pairs = ", ".join(f"{k}={v.value}" for k, v in fields.items())
        lines.append(f"[{group_label} 已完成] {pairs}")

    # --- Current group ---
    current_fields = by_node.get(current_node)
    if current_fields:
        group_label = node_to_group.get(current_node, current_node)
        lines.append(f"[{group_label} 当前]")
        for field_name, fv in current_fields.items():
            lines.append(f"- {field_name}: {fv.value}")

    if not lines:
        return None

    return "\n".join(lines)
