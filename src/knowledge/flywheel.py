"""数据飞轮模块

职责:
1. 收集低置信度问题
2. 存储到问题池 (pending_questions.json)
3. 提供问题池管理接口
4. 自动分类和聚合相似问题

注意: MVP 阶段使用 JSON 文件存储，单进程运行。
未来多进程/多实例部署时需迁移至 SQLite 或数据库。
"""

import json
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.knowledge.models import QUESTION_CATEGORIES, Evidence, PendingQuestion

logger = logging.getLogger(__name__)


# 低置信度阈值
LOW_CONFIDENCE_THRESHOLD = 0.6

# 最大问题池大小
MAX_POOL_SIZE = 1000


def auto_categorize(query: str) -> str:
    """
    基于关键词自动分类问题

    Args:
        query: 问题文本

    Returns:
        分类名称 (charging/mri/device/symptom/maintenance/other)
    """
    query_lower = query.lower()

    for category, keywords in QUESTION_CATEGORIES.items():
        for keyword in keywords:
            if keyword in query_lower:
                return category

    return "other"


def compute_review_priority(confidence: float, risk_level: str, frequency: int = 1) -> int:
    """
    计算审核优先级

    优先级基于：
    - 风险等级（高风险优先）
    - 置信度（越低越优先，说明越需要补充知识）
    - 出现频率（越高越优先）

    Args:
        confidence: 置信度分数
        risk_level: 风险等级
        frequency: 出现频率（相似问题数量）

    Returns:
        优先级分数（0-100，越高越优先）
    """
    # 风险等级分数
    risk_scores = {"high": 40, "medium": 20, "low": 0}
    risk_score = risk_scores.get(risk_level, 0)

    # 置信度分数（置信度越低，分数越高）
    confidence_score = int((1 - confidence) * 40)

    # 频率分数
    frequency_score = min(20, frequency * 5)

    return risk_score + confidence_score + frequency_score


class DataFlywheel:
    """数据飞轮"""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    ):
        """
        初始化数据飞轮

        Args:
            data_dir: 数据目录路径
            low_confidence_threshold: 低置信度阈值
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data"

        self._data_dir = data_dir
        self._pending_file = data_dir / "pending_questions.json"
        self._low_confidence_threshold = low_confidence_threshold

        # 线程锁 (保护文件操作)
        self._lock = threading.Lock()

        # 确保数据目录存在
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # 初始化问题池文件
        if not self._pending_file.exists():
            self._save_pool([])

    def collect(
        self,
        evidence: Evidence,
        session_id: str,
        suggested_answer: str = "",
    ) -> Optional[str]:
        """
        收集低置信度问题

        Args:
            evidence: 知识层输出
            session_id: 会话 ID
            suggested_answer: AI 建议的答案（可选）

        Returns:
            问题 ID (如果被收集)，否则 None
        """
        # 检查是否需要收集
        if evidence.confidence >= self._low_confidence_threshold:
            return None

        # 创建待处理问题记录
        question_id = str(uuid.uuid4())[:8]

        # 获取风险等级字符串
        risk_level = evidence.risk_level
        if hasattr(risk_level, "value"):
            risk_level = risk_level.value

        # v2: 自动分类和计算优先级
        category = auto_categorize(evidence.original_query)
        review_priority = compute_review_priority(evidence.confidence, risk_level)

        pending = PendingQuestion(
            id=question_id,
            original_query=evidence.original_query,
            rewritten_query=evidence.rewritten_query,
            confidence=evidence.confidence,
            risk_level=risk_level,
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            top_results=[r.item.id for r in evidence.search_results[:3]],
            status="pending",
            # v2 新增字段
            category=category,
            suggested_answer=suggested_answer,
            review_priority=review_priority,
        )

        # 添加到问题池
        self._add_to_pool(pending)

        logger.info(
            f"Collected low confidence question: id={question_id}, "
            f"confidence={evidence.confidence:.2f}, "
            f"category={category}, priority={review_priority}, "
            f"query='{evidence.original_query[:50]}...'"
        )

        return question_id

    def _add_to_pool(self, question: PendingQuestion) -> None:
        """添加问题到问题池"""
        with self._lock:
            pool = self._load_pool()

            # 检查重复 (相同的 rewritten_query)
            for existing in pool:
                if existing.get("rewritten_query") == question.rewritten_query:
                    logger.debug(f"Duplicate question skipped: {question.rewritten_query[:30]}...")
                    return

            # 添加新问题
            pool.append(self._question_to_dict(question))

            # 限制问题池大小
            if len(pool) > MAX_POOL_SIZE:
                # 优先移除已处理的 (resolved/rejected)，然后移除最旧的 pending
                # Sort key: (is_pending, timestamp)
                # is_pending=True sorts after is_pending=False (True > False)
                # Within same status, older timestamps sort first
                # Then take the LAST MAX_POOL_SIZE items (keep newest pending)
                pool = sorted(
                    pool,
                    key=lambda x: (
                        x.get("status") == "pending",  # False (0) < True (1), so resolved/rejected first
                        x.get("timestamp", ""),
                    ),
                )
                # Remove from the front (resolved/rejected and oldest pending)
                pool = pool[-(MAX_POOL_SIZE):]

            self._save_pool(pool)

    def _load_pool(self) -> List[Dict]:
        """加载问题池"""
        try:
            with open(self._pending_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_pool(self, pool: List[Dict]) -> None:
        """保存问题池"""
        with open(self._pending_file, "w", encoding="utf-8") as f:
            json.dump(pool, f, ensure_ascii=False, indent=2)

    def _question_to_dict(self, question: PendingQuestion) -> Dict:
        """将 PendingQuestion 转换为字典"""
        return {
            "id": question.id,
            "original_query": question.original_query,
            "rewritten_query": question.rewritten_query,
            "confidence": question.confidence,
            "risk_level": question.risk_level,
            "timestamp": question.timestamp,
            "session_id": question.session_id,
            "top_results": question.top_results,
            "status": question.status,
            "resolution": question.resolution,
            "new_knowledge_id": question.new_knowledge_id,
            # v2 新增字段
            "category": question.category,
            "suggested_answer": question.suggested_answer,
            "similar_questions": question.similar_questions,
            "review_priority": question.review_priority,
        }

    def get_pending_questions(
        self,
        limit: int = 50,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
    ) -> List[PendingQuestion]:
        """
        获取待处理问题列表

        Args:
            limit: 最大返回数量
            min_confidence: 最小置信度过滤
            max_confidence: 最大置信度过滤

        Returns:
            PendingQuestion 列表
        """
        with self._lock:
            pool = self._load_pool()

        results = []
        for item in pool:
            if item.get("status") != "pending":
                continue

            confidence = item.get("confidence", 0)
            if min_confidence is not None and confidence < min_confidence:
                continue
            if max_confidence is not None and confidence > max_confidence:
                continue

            results.append(
                PendingQuestion(
                    id=item.get("id", ""),
                    original_query=item.get("original_query", ""),
                    rewritten_query=item.get("rewritten_query", ""),
                    confidence=confidence,
                    risk_level=item.get("risk_level", "low"),
                    timestamp=item.get("timestamp", ""),
                    session_id=item.get("session_id", ""),
                    top_results=item.get("top_results", []),
                    status=item.get("status", "pending"),
                    resolution=item.get("resolution"),
                    new_knowledge_id=item.get("new_knowledge_id"),
                    # v2 新增字段
                    category=item.get("category", ""),
                    suggested_answer=item.get("suggested_answer", ""),
                    similar_questions=item.get("similar_questions", []),
                    review_priority=item.get("review_priority", 0),
                )
            )

            if len(results) >= limit:
                break

        return results

    def resolve_question(
        self,
        question_id: str,
        resolution: str,
        new_knowledge_id: Optional[str] = None,
    ) -> bool:
        """
        标记问题为已解决

        Args:
            question_id: 问题 ID
            resolution: 解决方案描述
            new_knowledge_id: 新增知识项 ID (如果有)

        Returns:
            是否成功
        """
        with self._lock:
            pool = self._load_pool()

            for item in pool:
                if item.get("id") == question_id:
                    item["status"] = "resolved"
                    item["resolution"] = resolution
                    item["new_knowledge_id"] = new_knowledge_id
                    self._save_pool(pool)
                    logger.info(f"Question {question_id} resolved: {resolution[:50]}...")
                    return True

            logger.warning(f"Question {question_id} not found")
            return False

    def reject_question(
        self,
        question_id: str,
        reason: str = "Not actionable",
    ) -> bool:
        """
        拒绝/忽略问题

        Args:
            question_id: 问题 ID
            reason: 拒绝原因

        Returns:
            是否成功
        """
        with self._lock:
            pool = self._load_pool()

            for item in pool:
                if item.get("id") == question_id:
                    item["status"] = "rejected"
                    item["resolution"] = reason
                    self._save_pool(pool)
                    logger.info(f"Question {question_id} rejected: {reason}")
                    return True

            logger.warning(f"Question {question_id} not found")
            return False

    def get_statistics(self) -> Dict:
        """
        获取问题池统计信息

        Returns:
            统计信息字典
        """
        with self._lock:
            pool = self._load_pool()

        stats = {
            "total": len(pool),
            "pending": 0,
            "resolved": 0,
            "rejected": 0,
            "by_risk_level": {"low": 0, "medium": 0, "high": 0},
            "by_category": {},  # v2 新增
            "avg_confidence": 0.0,
            "avg_priority": 0.0,  # v2 新增
        }

        confidence_sum = 0.0
        priority_sum = 0
        for item in pool:
            status = item.get("status", "pending")
            if status == "pending":
                stats["pending"] += 1
            elif status == "resolved":
                stats["resolved"] += 1
            elif status == "rejected":
                stats["rejected"] += 1

            risk = item.get("risk_level", "low")
            if risk in stats["by_risk_level"]:
                stats["by_risk_level"][risk] += 1

            # v2: 按分类统计
            category = item.get("category", "other") or "other"
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1

            confidence_sum += item.get("confidence", 0)
            priority_sum += item.get("review_priority", 0)

        if stats["total"] > 0:
            stats["avg_confidence"] = confidence_sum / stats["total"]
            stats["avg_priority"] = priority_sum / stats["total"]

        return stats

    def get_questions_by_priority(
        self,
        limit: int = 20,
        category: Optional[str] = None,
    ) -> List[PendingQuestion]:
        """
        按优先级排序获取待处理问题

        Args:
            limit: 最大返回数量
            category: 可选的分类过滤

        Returns:
            按优先级降序排列的 PendingQuestion 列表
        """
        with self._lock:
            pool = self._load_pool()

        results = []
        for item in pool:
            if item.get("status") != "pending":
                continue

            if category and item.get("category") != category:
                continue

            results.append(
                PendingQuestion(
                    id=item.get("id", ""),
                    original_query=item.get("original_query", ""),
                    rewritten_query=item.get("rewritten_query", ""),
                    confidence=item.get("confidence", 0),
                    risk_level=item.get("risk_level", "low"),
                    timestamp=item.get("timestamp", ""),
                    session_id=item.get("session_id", ""),
                    top_results=item.get("top_results", []),
                    status=item.get("status", "pending"),
                    resolution=item.get("resolution"),
                    new_knowledge_id=item.get("new_knowledge_id"),
                    category=item.get("category", ""),
                    suggested_answer=item.get("suggested_answer", ""),
                    similar_questions=item.get("similar_questions", []),
                    review_priority=item.get("review_priority", 0),
                )
            )

        # 按优先级降序排序
        results.sort(key=lambda x: x.review_priority, reverse=True)
        return results[:limit]

    def get_question_by_id(self, question_id: str) -> Optional[PendingQuestion]:
        """
        根据 ID 获取问题详情

        Args:
            question_id: 问题 ID

        Returns:
            PendingQuestion 或 None
        """
        with self._lock:
            pool = self._load_pool()

        for item in pool:
            if item.get("id") == question_id:
                return PendingQuestion(
                    id=item.get("id", ""),
                    original_query=item.get("original_query", ""),
                    rewritten_query=item.get("rewritten_query", ""),
                    confidence=item.get("confidence", 0),
                    risk_level=item.get("risk_level", "low"),
                    timestamp=item.get("timestamp", ""),
                    session_id=item.get("session_id", ""),
                    top_results=item.get("top_results", []),
                    status=item.get("status", "pending"),
                    resolution=item.get("resolution"),
                    new_knowledge_id=item.get("new_knowledge_id"),
                    category=item.get("category", ""),
                    suggested_answer=item.get("suggested_answer", ""),
                    similar_questions=item.get("similar_questions", []),
                    review_priority=item.get("review_priority", 0),
                )

        return None


# 全局单例
_flywheel: Optional[DataFlywheel] = None


def get_flywheel(data_dir: Optional[Path] = None) -> DataFlywheel:
    """获取数据飞轮单例"""
    global _flywheel

    if _flywheel is None:
        _flywheel = DataFlywheel(data_dir)

    return _flywheel
