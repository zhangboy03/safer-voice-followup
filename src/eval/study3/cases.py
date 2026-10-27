"""Load + validate the Study 3 case manifest into typed Case objects.

The manifest is local-only YAML (gitignored EMNLP path); the loader is tracked
so unit tests can assert distribution + KB anchoring against it."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass(frozen=True)
class AnswerPoint:
    point: str
    kb_id: str


@dataclass(frozen=True)
class Case:
    case_id: str
    case_type: str
    patient_question: str
    turn_context_kind: str
    gold_routing: str  # ANSWER | CLARIFY | DEFLECT
    risk_level: str  # low | medium | high | severe
    escalation_required: bool
    source: str
    knowledge_refs: list[str] = field(default_factory=list)
    gold_answer_points: list[AnswerPoint] = field(default_factory=list)
    must_say: list[str] = field(default_factory=list)
    must_not_say: list[str] = field(default_factory=list)
    expected_callback_priority: Optional[str] = None
    serialized_context: Optional[str] = None


def load_cases(manifest_path: Path) -> list[Case]:
    data = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8"))
    out: list[Case] = []
    for raw in data["cases"]:
        gold = raw["gold_routing"]
        if gold not in ("ANSWER", "CLARIFY", "DEFLECT"):
            raise ValueError(f"{raw['case_id']}: bad gold_routing {gold!r}")
        points = [AnswerPoint(point=p["point"], kb_id=p["kb_id"]) for p in (raw.get("gold_answer_points") or [])]
        out.append(
            Case(
                case_id=raw["case_id"],
                case_type=raw["case_type"],
                patient_question=raw["patient_question"],
                turn_context_kind=raw["turn_context_kind"],
                gold_routing=gold,
                risk_level=raw["risk_level"],
                escalation_required=bool(raw["escalation_required"]),
                source=raw["source"],
                knowledge_refs=list(raw.get("knowledge_refs") or []),
                gold_answer_points=points,
                must_say=list(raw.get("must_say") or []),
                must_not_say=list(raw.get("must_not_say") or []),
                expected_callback_priority=raw.get("expected_callback_priority"),
                serialized_context=raw.get("serialized_context"),
            )
        )
    return out


def kb_ids_in_manifest(cases: list[Case]) -> set[str]:
    ids: set[str] = set()
    for c in cases:
        ids.update(c.knowledge_refs)
        ids.update(p.kb_id for p in c.gold_answer_points)
    return ids
