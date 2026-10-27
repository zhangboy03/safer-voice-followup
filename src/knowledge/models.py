"""知识层数据模型定义

定义知识层的输出契约：Evidence
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional


class RiskLevel(str, Enum):
    """风险等级"""

    LOW = "low"  # 设备相关，一般性咨询
    MEDIUM = "medium"  # 医疗相关，症状咨询
    HIGH = "high"  # 核磁检查等高风险操作


class RAGDecision(str, Enum):
    """RAG 决策结果"""

    ANSWER = "answer"  # 直接回答
    CAUTIOUS = "cautious"  # 谨慎回答（加提示语）
    TRANSFER = "transfer"  # 转人工


@dataclass
class Prerequisites:
    """前置条件"""

    required_context: List[str] = field(default_factory=list)
    clarifying_question: str = ""


@dataclass
class Contraindication:
    """禁忌项"""

    condition_type: str  # 条件类型 (device_model_is, device_model_in, etc.)
    condition_value: Any  # 条件值
    warning: str  # 警告信息


@dataclass
class RetrievalHints:
    """检索提示"""

    keywords: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)


@dataclass
class KnowledgeItemMetadata:
    """知识项元数据"""

    tags: List[str] = field(default_factory=list)
    source: str = ""
    version: str = ""
    last_verified: str = ""


@dataclass
class KnowledgeItem:
    """知识库条目"""

    id: str
    question: str
    answer: str

    # 分类信息
    module_id: str = ""  # 知识模块 ID，如 device.charger, medical.mri
    product_line: str = "ALL"  # 产品线: DBS/VNS/SCS/SNM/ALL
    device_models: List[str] = field(default_factory=list)  # 适用设备型号
    risk_level: str = "low"  # 风险等级

    # 检索增强
    question_variants: List[str] = field(default_factory=list)  # 问题变体
    hypothetical_questions: List[str] = field(default_factory=list)  # 假设问题（LLM 生成）
    keywords: List[str] = field(default_factory=list)  # 检索关键词
    synonyms: List[str] = field(default_factory=list)  # 同义词

    # 增强字段
    prerequisites: Optional[Prerequisites] = None  # 前置条件
    contraindications: List[Contraindication] = field(default_factory=list)  # 禁忌项
    retrieval_hints: Optional[RetrievalHints] = None  # 检索提示

    # 旧字段（兼容）
    topic: str = ""
    metadata: Optional[KnowledgeItemMetadata] = None

    def get_searchable_text(self) -> str:
        """获取用于 BM25 检索的文本 (question + variants + hypothetical + hints)"""
        parts = [self.question]
        parts.extend(self.question_variants)
        parts.extend(self.hypothetical_questions)  # 假设问题也用于 BM25
        if self.retrieval_hints:
            parts.extend(self.retrieval_hints.keywords)
            parts.extend(self.retrieval_hints.synonyms)
        return " ".join(parts)


@dataclass
class SearchResult:
    """检索结果项"""

    item: KnowledgeItem
    score: float  # 归一化分数 [0, 1]
    source: str  # "keyword", "vector", "both"


@dataclass
class Evidence:
    """知识层输出

    这是知识层向上层传递的检索结果和置信度评估。
    """

    # 检索结果
    items: List[KnowledgeItem] = field(default_factory=list)
    search_results: List[SearchResult] = field(default_factory=list)

    # 置信度评估
    confidence: float = 0.0  # 综合置信度 [0, 1]
    risk_level: RiskLevel = RiskLevel.LOW

    # 决策
    decision: RAGDecision = RAGDecision.ANSWER

    # 追问（KnowledgeEnhancer 产出，保留兼容）
    needs_clarification: bool = False
    clarifying_question: Optional[str] = None

    # 警告（禁忌项等）
    warnings: List[str] = field(default_factory=list)

    # 可观测性
    original_query: str = ""
    rewritten_query: str = ""
    reasoning: str = ""
    latency_ms: float = 0.0

    def get_best_answer(self) -> Optional[str]:
        """获取最佳答案"""
        if self.search_results:
            return self.search_results[0].item.answer
        if self.items:
            return self.items[0].answer
        return None

    def get_top_k_answers(self, k: int = 3) -> List[str]:
        """获取 Top-K 答案"""
        if self.search_results:
            return [r.item.answer for r in self.search_results[:k]]
        return [item.answer for item in self.items[:k]]

    def should_transfer(self) -> bool:
        """是否应该转人工"""
        return self.decision == RAGDecision.TRANSFER

    def should_be_cautious(self) -> bool:
        """是否需要谨慎回答"""
        return self.decision == RAGDecision.CAUTIOUS


@dataclass
class PendingQuestion:
    """待处理问题（数据飞轮）"""

    id: str
    original_query: str
    rewritten_query: str
    confidence: float
    risk_level: str
    timestamp: str
    session_id: str

    # 检索上下文
    top_results: List[str] = field(default_factory=list)  # Top-3 结果 ID

    # 处理状态
    status: str = "pending"  # pending/resolved/rejected
    resolution: Optional[str] = None  # 解决方案描述
    new_knowledge_id: Optional[str] = None  # 新增知识项 ID（如果有）

    # v2 新增字段
    category: str = ""  # 问题分类：charging/mri/device/symptom/maintenance/other
    suggested_answer: str = ""  # AI 建议的答案（可选）
    similar_questions: List[str] = field(default_factory=list)  # 类似问题 ID 列表（聚合用）
    review_priority: int = 0  # 审核优先级（基于频率和风险，越高越优先）


# 配置常量
# 注意: embedding_dim 现在由 provider 动态提供，此处保留默认值用于兼容
RAG_CONFIG = {
    # Embedding 配置
    # 注意: embedding_dim 已弃用，维度由 EmbeddingProvider.dim 动态获取
    # 本地模型 (bge-base-zh): 768
    # 远程 API (DashScope text-embedding-v4): 1024
    "embedding_timeout_ms": 80,
    # 外层混合检索超时（rag_service.query_async）。默认 250ms 保持原生产行为；
    # 离线评测/批处理可调大，让远程 embedding（~500ms）真正参与向量检索。
    "search_timeout_ms": 250,
    "embedding_model": "text-embedding-v4",  # 仅用于 dashscope provider
    "embedding_dim": 1024,  # 默认值，实际由 provider 决定
    # 检索配置
    "keyword_top_k": 5,
    "vector_top_k": 5,
    "final_top_k": 5,
    # 置信度阈值 (v3: 风险解耦，统一阈值)
    # 公式: 0.70 × top1_score + 0.30 × discrimination
    "confidence_high_threshold": 0.75,  # >= 此值 → ANSWER
    "confidence_low_threshold": 0.55,  # >= 此值 → CAUTIOUS, < 此值 → TRANSFER
}


# 问题分类关键词映射
QUESTION_CATEGORIES = {
    "charging": ["充电", "电量", "信号格", "发热", "温度", "电池", "续航"],
    "mri": ["核磁", "MRI", "磁共振", "CT", "检查", "拍片", "造影"],
    "device": ["程控器", "充电器", "刺激器", "脉冲", "型号", "设备", "遥控器"],
    "symptom": ["不舒服", "疼", "痛", "抖", "副作用", "反应", "头晕", "麻", "紧"],
    "maintenance": ["维修", "更换", "坏了", "换新", "保修", "故障", "失灵"],
}
