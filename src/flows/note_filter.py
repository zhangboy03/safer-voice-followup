"""Deterministic filter pass for Pins incremental notes.

Used by ShortTermMemoryCommitter before merging analyzer patches. No LLM calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.flows.sop_decision_models import ShortTermMemoryItem

_MEDICAL_SHORT_WORDS = frozenset(
    {
        # symptoms
        "胸痛",
        "发热",
        "头痛",
        "呕吐",
        "心悸",
        "晕厥",
        "出血",
        "昏迷",
        "抽搐",
        # attributes
        "孕妇",
        "高龄",
        "婴幼儿",
        "失能",
        "失智",
        "卧床",
    }
)

_REMOVE_PROTECTION_THRESHOLD = 0.9  # spec §6.4 — policy value, do not inline

_BLOOD_TYPE_RE = re.compile(r"^[ABO型阳阴性\s]+$")
_VALUE_UNIT_RE = re.compile(r"^\d+(\.\d+)?[a-zA-Z℃%岁次/]+$")
_DURATION_RE = re.compile(r"^\d+(天|周|月|年|小时|分钟)$")


def hits_blacklist(text: str, blacklist: list[str]) -> tuple[bool, str]:
    """Return (hit, matched_phrase).

    Rules: exact match OR blacklist phrases covering ≥80% of trimmed text chars.
    Pure substring presence does NOT count (avoids '我可以的，但是…').
    """
    trimmed = text.strip()
    if not trimmed:
        return False, ""
    if trimmed in blacklist:
        return True, trimmed

    covered = 0
    matched = ""
    for phrase in blacklist:
        if phrase and phrase in trimmed:
            # Count non-overlapping occurrences — conservative.
            count = trimmed.count(phrase)
            covered += len(phrase) * count
            matched = matched or phrase
    if covered >= 0.8 * len(trimmed):
        return True, matched
    return False, ""


def hits_medical_allowlist(text: str) -> bool:
    """Return True if `text` matches the medical short-word allowlist.

    Used to rescue short salient_fact items (e.g. 胸痛 / 3 天 / 38.5℃)
    from the generic len<6 cutoff in Filter A's relation 3.
    """
    trimmed = text.strip()
    if not trimmed:
        return False
    if trimmed in _MEDICAL_SHORT_WORDS:
        return True
    if _BLOOD_TYPE_RE.match(trimmed):
        return True
    if _VALUE_UNIT_RE.match(trimmed):
        return True
    if _DURATION_RE.match(trimmed):
        return True
    return False


@dataclass
class FilterConfig:
    confidence_threshold: float = 0.5
    blacklist: list[str] = field(default_factory=list)


def filter_upserts(
    items: list[ShortTermMemoryItem],
    cfg: FilterConfig,
) -> tuple[list[ShortTermMemoryItem], list[tuple[ShortTermMemoryItem, str]]]:
    kept: list[ShortTermMemoryItem] = []
    dropped: list[tuple[ShortTermMemoryItem, str]] = []
    for item in items:
        # Gate 1: confidence
        if item.confidence < cfg.confidence_threshold:
            dropped.append((item, f"confidence<{cfg.confidence_threshold}"))
            continue
        # Gate 2: blacklist
        hit, matched = hits_blacklist(item.text, cfg.blacklist)
        if hit:
            dropped.append((item, f"blacklist:{matched}"))
            continue
        # Gate 3: salient_fact short-text fallback (medical allowlist rescues)
        text = item.text.strip()
        if item.category == "salient_fact" and len(text) < 6 and not hits_medical_allowlist(text):
            dropped.append((item, "salient_fact_too_short"))
            continue
        kept.append(item)
    return kept, dropped


def filter_removes(
    remove_keys: list[str],
    *,
    kept_upserts: list[ShortTermMemoryItem],
    existing_items: list[ShortTermMemoryItem],
    recent_turn_ids: set[str],
) -> tuple[list[str], list[tuple[str, str]]]:
    """Filter `remove_keys` against safety rules with explicit escape clauses.

    `recent_turn_ids` is the set of last 2 turn_ids (caller decides exact window).
    """
    kept: list[str] = []
    dropped: list[tuple[str, str]] = []
    existing_by_key = {it.key: it for it in existing_items}
    upsert_keys = {it.key for it in kept_upserts}
    has_correction_upsert = any(it.category == "correction_hint" for it in kept_upserts)

    for key in remove_keys:
        if key not in existing_by_key:
            dropped.append((key, "remove_target_not_found"))
            continue
        existing = existing_by_key[key]
        is_high_conf_recent = existing.confidence >= _REMOVE_PROTECTION_THRESHOLD and any(
            tid in recent_turn_ids for tid in existing.source_turn_ids
        )
        if not is_high_conf_recent:
            kept.append(key)
            continue
        # Escape clauses (spec §6.4)
        if key in upsert_keys:
            kept.append(key)
            continue
        if existing.category in ("correction_hint", "future_node_hint"):
            kept.append(key)
            continue
        if has_correction_upsert:
            kept.append(key)
            continue
        dropped.append((key, "remove_high_confidence_recent_item"))
    return kept, dropped
