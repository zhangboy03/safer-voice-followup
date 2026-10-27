"""知识强化模块

职责:
1. 前置条件检查 - 检查回答前需要的上下文信息
2. 禁忌项检查 - 检查是否有安全警告需要添加

延迟预算: < 10ms
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.knowledge.models import (
    Contraindication,
    KnowledgeItem,
    SearchResult,
)

logger = logging.getLogger(__name__)


# 会话上下文类型 (从 handoff 层导入，这里用 Dict 表示)
SessionContext = Dict[str, Any]


# 白名单条件函数 (安全的条件检查，避免 eval)
CONDITION_FUNCTIONS: Dict[str, Callable[[Dict[str, Any], Any], bool]] = {
    # 设备型号精确匹配
    "device_model_is": lambda ctx, val: ctx.get("device_model") == val,
    # 设备型号在列表中
    "device_model_in": lambda ctx, val: ctx.get("device_model") in (val if isinstance(val, list) else [val]),
    # 产品线精确匹配
    "product_line_is": lambda ctx, val: ctx.get("product_line") == val,
    # 检查上下文中是否存在某个字段
    "has_context": lambda ctx, val: val in ctx and ctx[val] is not None,
    # 检查上下文中某个字段不存在或为空
    "missing_context": lambda ctx, val: val not in ctx or ctx[val] is None,
    # 设备型号前缀匹配 (如 G106 系列)
    "device_model_starts_with": lambda ctx, val: (
        ctx.get("device_model", "").startswith(val) if ctx.get("device_model") else False
    ),
}


def check_contraindication(
    contraindication: Contraindication,
    context: Dict[str, Any],
) -> bool:
    """
    安全地检查禁忌项条件

    Args:
        contraindication: 禁忌项配置
        context: 会话上下文 (字典形式)

    Returns:
        True 如果条件匹配 (需要显示警告)
    """
    cond_type = contraindication.condition_type
    cond_value = contraindication.condition_value

    if cond_type not in CONDITION_FUNCTIONS:
        logger.warning(f"Unknown condition_type: {cond_type}, skipping")
        return False

    try:
        return CONDITION_FUNCTIONS[cond_type](context, cond_value)
    except Exception as e:
        logger.error(f"Error checking contraindication: {e}")
        return False


def check_prerequisites(
    item: KnowledgeItem,
    context: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """
    检查前置条件是否满足

    Args:
        item: 知识项
        context: 会话上下文

    Returns:
        (is_satisfied, clarifying_question):
        - is_satisfied: 前置条件是否都满足
        - clarifying_question: 如果不满足，返回追问话术
    """
    if not item.prerequisites:
        return (True, None)

    required = item.prerequisites.required_context
    if not required:
        return (True, None)

    # 检查每个必需的上下文字段
    for field in required:
        if field not in context or context[field] is None:
            # 缺少必需信息
            clarifying = item.prerequisites.clarifying_question
            if not clarifying:
                # 生成默认追问
                clarifying = f"请问能提供您的{field}信息吗？"
            return (False, clarifying)

    return (True, None)


class KnowledgeEnhancer:
    """知识强化器"""

    def __init__(self):
        """初始化知识强化器"""
        pass

    def enhance(
        self,
        search_results: List[SearchResult],
        context: SessionContext,
    ) -> Tuple[bool, Optional[str], List[str]]:
        """
        对检索结果进行知识强化

        Args:
            search_results: 检索结果列表
            context: 会话上下文

        Returns:
            (needs_clarification, clarifying_question, warnings):
            - needs_clarification: 是否需要追问
            - clarifying_question: 追问话术
            - warnings: 警告信息列表
        """
        if not search_results:
            return (False, None, [])

        # 转换上下文为字典形式
        ctx_dict = self._context_to_dict(context)

        needs_clarification = False
        clarifying_question = None
        warnings: List[str] = []

        # 处理 Top-3 结果
        for result in search_results[:3]:
            item = result.item

            # 检查前置条件
            if not needs_clarification:  # 只返回第一个追问
                satisfied, question = check_prerequisites(item, ctx_dict)
                if not satisfied:
                    needs_clarification = True
                    clarifying_question = question

            # 检查禁忌项
            for contra in item.contraindications:
                if check_contraindication(contra, ctx_dict):
                    if contra.warning and contra.warning not in warnings:
                        warnings.append(contra.warning)

        return (needs_clarification, clarifying_question, warnings)

    def _context_to_dict(self, context: SessionContext) -> Dict[str, Any]:
        """将 SessionContext 转换为字典"""
        # 如果已经是字典，直接返回
        if isinstance(context, dict):
            return context

        # 否则尝试提取常见字段
        ctx = {
            "session_id": getattr(context, "session_id", None),
            "patient_id": getattr(context, "patient_id", None),
            "product_line": getattr(context, "product_line", None),
            "device_model": getattr(context, "device_model", None),
            "intent_type": getattr(context, "intent_type", None),
            "intent_confidence": getattr(context, "intent_confidence", 0.0),
            "current_sop_step": getattr(context, "current_sop_step", None),
        }
        # 合并扩展上下文 (用于禁忌项检查)
        extra = getattr(context, "extra_context", None)
        if extra:
            ctx.update(extra)
        return ctx
