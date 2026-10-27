"""Study 1 latency turn selection (pure, event-driven).

长句里有停顿时，EnergyVAD/SmartTurn 会把一句话切成多个 turn。harness 若直接取
``last(TurnMetricsEvent)``，会落在一个时间锚错位的碎片 turn 上：该 turn 的
``first_audio`` 早于它自己的 ``turn_stop``（agent 攒了前几段才开口，正压在用户最后一段
还没说完时），导致 ``e2e_ms`` / ``user_stop_to_emit_ms`` 为负、被 turn_metrics 序列化成
-1，而 ``speech_end_to_audio_ms`` 只靠加 VAD 尾巴才勉强为正——于是报出一个假的低延迟。

本模块把"选哪个 turn 的指标"抽成纯函数：按 turn_id 取每个 turn 最完整的一条 metrics，
优先选时间锚自洽（e2e>0 且 user_stop_to_emit>=0 且 emit_to_audio>0，即
turn_stop < smart_turn_emit < first_audio）的那个；并报告这一 case 是否被切成多段
（segmented）。判定全部基于可复现事件，不手工估计、不把缺失当 0。
"""

from __future__ import annotations

from typing import Any


def _positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and value > 0


def _non_negative(value: Any) -> bool:
    return isinstance(value, (int, float)) and value >= 0


def anchors_consistent(metrics: dict[str, Any]) -> bool:
    """时间锚自洽：turn_stop < smart_turn_emit < first_audio。

    体现为序列化后的三个差值都非负/正（任一为负会被 turn_metrics 写成 -1）。
    """

    return (
        _positive(metrics.get("e2e_ms"))
        and _non_negative(metrics.get("user_stop_to_emit_ms"))
        and _positive(metrics.get("emit_to_audio_ms"))
    )


def select_latency_metrics(
    metrics_events: list[dict[str, Any]],
    *,
    user_turn_count: int = 0,
) -> dict[str, Any]:
    """从一个 latency case 的所有 TurnMetricsEvent 里选出可用于报数的那条。

    ``metrics_events`` 是该 case 窗口内全部 TurnMetricsEvent（一个 turn 会随里程碑多次
    发射，后到的更完整）。``user_turn_count`` 是该 case 的 UserStartedSpeakingEvent 数，
    用于判断长句是否被切成多段。返回选中的 metrics 和诊断字段，不修改入参。
    """

    by_turn: dict[str, dict[str, Any]] = {}
    for metrics in metrics_events:
        turn_id = metrics.get("turn_id")
        if turn_id:
            by_turn[turn_id] = metrics
    turns = list(by_turn.values())

    consistent = [m for m in turns if anchors_consistent(m)]
    chosen = consistent[0] if consistent else (turns[-1] if turns else None)

    metrics_turn_count = len(turns)
    segmented = metrics_turn_count > 1 or user_turn_count > 1

    if chosen is None:
        reason = "no_turn_metrics"
    elif not consistent:
        reason = "no_consistent_anchor_turn"
    elif segmented:
        reason = "segmented_but_recovered"
    else:
        reason = ""

    return {
        "chosen": chosen,
        "metrics_turn_count": metrics_turn_count,
        "user_turn_count": user_turn_count,
        "segmented": segmented,
        "consistent": bool(consistent),
        "reason": reason,
    }
