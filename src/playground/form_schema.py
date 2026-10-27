"""Canonical 随访 questionnaire schema, derived from the PRODUCTION flow config
(src/flows/pinchi_flow_config) — NOT eval cards (no gold values). Single source
of truth for the solo prompt's ★ list, the extractor schema, and the UI panel."""

from __future__ import annotations

from dataclasses import dataclass

from src.flows.pinchi_flow_config import NODE_TO_GROUP, build_agent_config


@dataclass(frozen=True)
class FieldSpec:
    name: str
    group: str
    definition: str
    required: bool


def field_catalog() -> list[FieldSpec]:
    config = build_agent_config()
    out: list[FieldSpec] = []
    for node in NODE_TO_GROUP:  # production SOP order (greeting … insurance … closing)
        node_config = config.nodes.get(node)
        if node_config is None or not node_config.field_definitions:
            continue  # skip closing / knowledge_qa / human_transfer (no fields)
        required = set(node_config.completion_policy.required_fields)
        for name, definition in node_config.field_definitions.items():  # stable insertion order
            out.append(
                FieldSpec(name=name, group=NODE_TO_GROUP[node], definition=definition, required=name in required)
            )
    return out
