"""PatientPersona — data model for virtual patient personas."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class PersonaBasic:
    age_group: str = ""
    gender: str = ""
    dialect_hint: str | None = None


@dataclass
class PersonaMedical:
    device_summary: str = ""
    history: list[str] = field(default_factory=list)
    primary_concern_paraphrase: str = ""


@dataclass
class PersonaPersonality:
    patience: str = "medium"  # high/medium/low
    verbosity: str = "medium"  # terse/medium/verbose
    memory_quality: str = "clear"  # clear/fuzzy/very_fuzzy
    emotion: str = "calm"  # calm/anxious/frustrated/warm


@dataclass
class PersonaInteraction:
    tendency_to_digress: float = 0.0
    needs_clarification: float = 0.0
    decision_style: str = "decisive"  # decisive/hesitant/defer_to_family


@dataclass
class PersonaSource:
    cluster_id: str = ""
    extraction_run_tag: str = ""
    example_count: int = 0


@dataclass
class PatientPersona:
    id: str = ""
    tier: str = "standard"  # standard/challenge/edge
    basic: PersonaBasic = field(default_factory=PersonaBasic)
    medical: PersonaMedical = field(default_factory=PersonaMedical)
    personality: PersonaPersonality = field(default_factory=PersonaPersonality)
    interaction: PersonaInteraction = field(default_factory=PersonaInteraction)
    forbidden: list[str] = field(default_factory=list)
    style_examples: list[str] = field(default_factory=list)
    source: PersonaSource = field(default_factory=PersonaSource)

    @classmethod
    def from_yaml(cls, path: Path) -> "PatientPersona":
        """Load persona from YAML file."""
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            id=data.get("id", ""),
            tier=data.get("tier", "standard"),
            basic=PersonaBasic(**data.get("basic", {})),
            medical=PersonaMedical(**data.get("medical", {})),
            personality=PersonaPersonality(**data.get("personality", {})),
            interaction=PersonaInteraction(**data.get("interaction", {})),
            forbidden=data.get("forbidden", []),
            style_examples=data.get("style_examples", []),
            source=PersonaSource(**data.get("source", {})),
        )
