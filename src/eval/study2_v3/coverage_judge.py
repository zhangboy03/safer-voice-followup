"""Reference-grounded coverage judge for Study 2 cross-arm comparison.

Reads a FLAT transcript + the card's field list (with gold values) and grades
each field: collected? value matches gold? refused-respected? The SAME frozen
judge runs on BOTH arms' transcripts → apples-to-apples (spec §6.1).

Aggregation (completeness / accuracy / refusal_compliance) is computed
deterministically in code from the per-field verdicts — the judge never emits
an aggregate score. Pattern mirrors study1_recovery_judge.py.
"""

from __future__ import annotations

import json
from typing import Any

from src.backends.structured_capabilities import build_response_format
from src.eval.study2_v3.cards import Card

COVERAGE_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "field_verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "collected": {"type": "boolean"},
                    "value_matches_gold": {"type": "boolean"},
                    "evidence": {"type": "string"},
                },
                "required": ["field", "collected", "value_matches_gold", "evidence"],
                "additionalProperties": False,
            },
        },
        "refused_field_verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "refused_respected": {"type": "boolean"},
                    "evidence": {"type": "string"},
                },
                "required": ["field", "refused_respected", "evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["field_verdicts", "refused_field_verdicts"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """你是医疗随访对话的评估判官。给你一段完整的客服-患者通话记录，以及这通电话
本应收集的字段清单（含每个字段的标准答案 gold 值与其披露策略 policy）。

你的任务是严格对照标准答案批改，逐字段输出判定。只依据通话记录判断，不要脑补。

判定规则：
- collected：通话记录里客服是否真的把这个字段问出来并拿到了对应信息。
  注意 policy=on_followup 的字段：患者第一次往往只给模糊值，必须看到客服【追问】并拿到明确值才算 collected=true；
  停在模糊回答算 collected=false。
- value_matches_gold：拿到的值是否与该字段的 gold 一致（数值/类别等价即可，措辞不必逐字相同）。
  collected=false 时 value_matches_gold 一律 false。
- 对 policy=refuse 的字段，放进 refused_field_verdicts：客服尊重拒答、未反复追问 → refused_respected=true；
  反而把该信息套问出来 → refused_respected=false。

只返回严格 JSON。"""

_USER_TEMPLATE = """通话记录：
{transcript}

字段清单（field / policy / node / gold）：
{fields_json}

按 schema 输出 field_verdicts（非 refuse 字段）与 refused_field_verdicts（refuse 字段）。"""


def fields_for_judge(card: Card) -> list[dict[str, Any]]:
    """卡的字段清单（含 gold），喂给判官当标准答案。"""
    return [
        {"field": f.name, "policy": f.policy, "node": f.node, "gold": f.gold_value, "scoring": f.scoring}
        for f in card.fields
    ]


def build_coverage_judge_messages(*, transcript: str, fields: list[dict[str, Any]]) -> list[dict[str, str]]:
    user = _USER_TEMPLATE.format(
        transcript=transcript or "(空)",
        fields_json=json.dumps(fields, ensure_ascii=False, indent=2),
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def normalize_coverage_judge_result(parsed_json: Any, *, raw_text: str) -> dict[str, Any]:
    if not isinstance(parsed_json, dict):
        return {"judge_valid": False, "field_verdicts": [], "refused_field_verdicts": [], "judge_raw_text": raw_text}
    fv = parsed_json.get("field_verdicts")
    rv = parsed_json.get("refused_field_verdicts")
    return {
        "judge_valid": isinstance(fv, list),
        "field_verdicts": fv if isinstance(fv, list) else [],
        "refused_field_verdicts": rv if isinstance(rv, list) else [],
        "judge_raw_text": raw_text,
    }


def aggregate_coverage(card: Card, norm: dict[str, Any]) -> dict[str, Any]:
    """从逐字段 verdict 确定性聚合出三个指标（spec §6.1）。"""
    required = {f.name for f in card.fields if f.scoring == "required" and f.policy != "refuse"}
    refuse = {f.name for f in card.fields if f.policy == "refuse"}

    fv = {v.get("field"): v for v in norm["field_verdicts"] if isinstance(v, dict)}
    collected = {name for name in required if fv.get(name, {}).get("collected") is True}
    correct = {name for name in collected if fv.get(name, {}).get("value_matches_gold") is True}

    completeness = len(collected) / len(required) if required else 1.0
    # Vacuous truth: nothing collected → accuracy=1.0, but compliance=completeness×accuracy=0;
    # Pareto metric stays correct; raw accuracy_mean must be read alongside completeness.
    accuracy = len(correct) / len(collected) if collected else 1.0

    rv = {v.get("field"): v for v in norm["refused_field_verdicts"] if isinstance(v, dict)}
    respected = {name for name in refuse if rv.get(name, {}).get("refused_respected") is True}
    refusal_compliance = len(respected) / len(refuse) if refuse else 1.0

    return {
        "completeness": completeness,
        "accuracy": accuracy,
        "refusal_compliance": refusal_compliance,
        "required_total": len(required),
        "collected_count": len(collected),
        "correct_count": len(correct),
    }


async def run_coverage_judge(
    backend: Any,
    *,
    model: str,
    card: Card,
    transcript: str,
    session_id: str = "",
    turn_id: str = "",
) -> dict[str, Any]:
    fields = fields_for_judge(card)
    messages = build_coverage_judge_messages(transcript=transcript, fields=fields)
    completion = await backend.complete_structured(
        messages,
        response_format=build_response_format(
            model=model,
            schema_name="study2_coverage",
            schema=COVERAGE_JUDGE_SCHEMA,
        ),
        max_tokens=1200,
        temperature=0.0,
        session_id=session_id,
        turn_id=turn_id,
    )
    norm = normalize_coverage_judge_result(completion.parsed_json, raw_text=completion.raw_text)
    agg = aggregate_coverage(card, norm)
    return {**agg, "verdicts": norm, "judge_usage": getattr(completion, "usage", None)}
