"""置信度计算与决策模块

职责:
1. 计算综合置信度分数
2. 根据阈值确定决策
3. 聚合 Top-K 结果的风险等级

延迟预算: < 10ms

置信度公式 (v3):
    置信度 = 0.70 × top1_score + 0.30 × discrimination

改进点:
- 移除 coverage 因子（边际贡献最小）
- 风险等级与决策阈值解耦（风险仅控制 prompt 措辞）
- discrimination 在 Min-Max 归一化让 top-1 虚高时提供关键区分信号

阈值参数化（Phase 3 GraphRAG）:
- 环境变量 GRAPH_CONFIDENCE_HIGH / GRAPH_CONFIDENCE_LOW 可覆盖默认阈值
- 融合后分数分布变化时，用 shadow 数据重标定
"""

import logging
from typing import List, Optional, Tuple

from src.knowledge.models import (
    RAG_CONFIG,
    RAGDecision,
    SearchResult,
)

logger = logging.getLogger(__name__)


# 风险等级优先级
RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


def aggregate_risk_level(results: List[SearchResult]) -> Tuple[str, Optional[str]]:
    """
    聚合 Top-5 结果中的最高风险等级 (安全优先)

    Args:
        results: 检索结果列表

    Returns:
        (risk_level, default_decision):
        - risk_level: 聚合后的风险等级
        - default_decision: 如果结果为空，默认决策 ("transfer")，否则为 None
    """
    if not results:
        # 空结果直接转人工
        return ("low", "transfer")

    max_risk = "low"
    for result in results[:5]:  # Top-5
        item_risk = result.item.risk_level
        if RISK_ORDER.get(item_risk, 0) > RISK_ORDER.get(max_risk, 0):
            max_risk = item_risk

    return (max_risk, None)


def compute_discrimination_score(results: List[SearchResult]) -> float:
    """
    计算区分度分数

    Top-1 与 Top-2 的分数差距越大，表示结果越可信（更明确的最佳匹配）。
    当 Min-Max 归一化让 top-1 虚高时，如果 top-1 和 top-2 都接近 1.0，
    说明匹配不精准，disc 会很低，拉低总分。

    Args:
        results: 检索结果列表

    Returns:
        区分度分数 [0, 1]
    """
    if not results:
        return 0.0

    if len(results) < 2:
        # 只有一个结果，认为区分度较高
        return 1.0 if results[0].score > 0.5 else 0.5

    # 计算 Top-1 与 Top-2 的分数差距
    gap = max(0.0, results[0].score - results[1].score)
    # gap 在 [0, 0.5] 之间映射到 [0, 1]
    return min(1.0, gap * 2)


def compute_confidence(
    search_results: List[SearchResult],
) -> Tuple[float, str]:
    """
    计算综合置信度 (v3)

    新公式:
    置信度 = 0.70 × top1_score + 0.30 × discrimination

    Args:
        search_results: 检索结果列表

    Returns:
        (confidence, reasoning): 置信度分数和计算推理说明
    """
    if not search_results:
        return 0.0, "no results"

    top1 = search_results[0].score
    disc = compute_discrimination_score(search_results)
    confidence = 0.70 * top1 + 0.30 * disc

    reasoning = f"top1={top1:.2f}*0.70 + disc={disc:.2f}*0.30 => {confidence:.2f}"
    return confidence, reasoning


def make_decision(
    confidence: float,
) -> RAGDecision:
    """
    根据置信度做出决策 (v3: 风险解耦)

    阈值表:
    | 置信度范围         | 决策      |
    |-------------------|----------|
    | >= 0.75           | ANSWER   |
    | 0.55 - 0.75       | CAUTIOUS |
    | < 0.55            | TRANSFER |

    Args:
        confidence: 置信度分数

    Returns:
        RAGDecision (answer/cautious/transfer)
    """
    high = RAG_CONFIG["confidence_high_threshold"]
    low = RAG_CONFIG["confidence_low_threshold"]

    if confidence >= high:
        return RAGDecision.ANSWER
    elif confidence >= low:
        return RAGDecision.CAUTIOUS
    else:
        return RAGDecision.TRANSFER


class ConfidenceCalculator:
    """置信度计算器 (v3)"""

    def calculate(
        self,
        search_results: List[SearchResult],
    ) -> Tuple[float, str, str, RAGDecision]:
        """
        计算置信度并做出决策 (v3)

        Args:
            search_results: 检索结果列表

        Returns:
            (confidence, risk_level, reasoning, decision)
        """
        # 聚合风险等级
        risk_level, default_decision = aggregate_risk_level(search_results)

        # 如果有默认决策 (空结果)，直接返回
        if default_decision:
            return (
                0.0,
                risk_level,
                "No search results, transferring to human",
                RAGDecision.TRANSFER,
            )

        # 计算置信度 (v3)
        confidence, reasoning = compute_confidence(search_results)

        # 做出决策 (v3: 不再传 risk_level)
        decision = make_decision(confidence)

        # 扩展推理说明（风险等级仍记录用于可观测性）
        full_reasoning = f"risk={risk_level}, {reasoning} => {decision.value}"

        return confidence, risk_level, full_reasoning, decision
