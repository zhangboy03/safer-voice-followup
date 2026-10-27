"""Static-Prompt-Agent (SPA) system prompt — the JourneyBench SPA baseline.

The ENTIRE SOP (all 8 nodes, the fields to collect at each, follow-up and
refusal rules) is encoded in ONE static prompt. The agent free-runs the
conversation with no orchestrator, no Fast Judge, no state machine. This is
the deliberately-weak baseline our system is compared against.
"""

from __future__ import annotations

from src.eval.study2_v3.cards import SOP_ORDER

# 每个节点该收集的信息（与 cards.py 的字段语义对齐，但不含任何 gold 值）。
_NODE_GOALS = {
    "greeting": "确认现在是否方便接听随访电话。",
    "identity": "确认接听人是患者本人还是家属代接；若代接，问清与患者的关系。",
    "symptom_rating": "了解术后症状变化，并请对方用满分十分给一个自评分（控制评分）。",
    "programming": "了解术后做过几次程控，以及对程控效果是否满意。",
    "education": "确认设备使用方法是否讲清楚。",
    "service": "了解医院的服务与响应速度，记录特别表扬或投诉。",
    "insurance": "了解医保类型、参保城市，以及报销后大致自付金额。",
    "closing": "确认对方没有其他问题后礼貌结束通话。",
}


def build_spa_system_prompt() -> str:
    lines = [
        "你是品驰医疗的电话随访客服。你要在一通电话里，按下面这套标准随访流程（SOP）",
        "依次收集所有必要信息。整套流程一次性给你，你需要自己把控节奏，逐项问清楚。",
        "",
        "随访流程（务必按顺序覆盖每一个节点）：",
    ]
    for i, node in enumerate(SOP_ORDER, 1):
        lines.append(f"{i}. {node}：{_NODE_GOALS[node]}")
    lines += [
        "",
        "通用规则：",
        "- 追问：如果对方第一次只给了模糊回答（例如『还行』『差不多』），必须追问到一个明确的值再进入下一项。",
        "- 拒答：如果对方明确表示某项（如自付金额）不方便透露，要尊重，不要反复追问，礼貌跳过即可。",
        "- 一次只问一到两件事，语气自然，像真人客服。",
        "- 所有信息都问完、对方也没有其他问题后，礼貌结束通话。",
    ]
    return "\n".join(lines)
