"""知识层 (Knowledge Layer)

职责：检索相关知识，评估可靠性（仅 RAG 路由调用）

组件：
- RAGService: RAG 协调器
- QueryRewriter: 查询改写
- HybridRetriever: 混合检索（向量+关键词）
- ConfidenceCalculator: 置信度计算
- KnowledgeEnhancer: 知识强化

输出契约：Evidence
"""

from .confidence import ConfidenceCalculator, compute_confidence, compute_discrimination_score, make_decision
from .embedding import (
    compute_cosine_similarity,
    get_embedding_provider,
    reset_provider,
)
from .flywheel import DataFlywheel, get_flywheel
from .knowledge_enhancer import KnowledgeEnhancer
from .knowledge_migrator import load_knowledge_base, migrate_knowledge_base
from .models import (
    RAG_CONFIG,
    Contraindication,
    Evidence,
    KnowledgeItem,
    KnowledgeItemMetadata,
    PendingQuestion,
    Prerequisites,
    RAGDecision,
    RetrievalHints,
    RiskLevel,
    SearchResult,
)
from .query_rewriter import QueryRewriter
from .rag_service import RAGService, get_rag_service
from .retriever import (
    HybridRetriever,
    KeywordRetriever,
    VectorRetriever,
)

__all__ = [
    # 核心服务
    "RAGService",
    "get_rag_service",
    # 核心输出
    "Evidence",
    # 知识条目
    "KnowledgeItem",
    "SearchResult",
    # 枚举
    "RiskLevel",
    "RAGDecision",
    # 辅助类
    "Prerequisites",
    "Contraindication",
    "RetrievalHints",
    "KnowledgeItemMetadata",
    "PendingQuestion",
    # 配置
    "RAG_CONFIG",
    # 检索器
    "KeywordRetriever",
    "VectorRetriever",
    "HybridRetriever",
    # 其他组件
    "QueryRewriter",
    "ConfidenceCalculator",
    "compute_confidence",
    "compute_discrimination_score",
    "make_decision",
    "KnowledgeEnhancer",
    "DataFlywheel",
    "get_flywheel",
    "load_knowledge_base",
    "migrate_knowledge_base",
    # Embedding (from utils)
    "get_embedding_provider",
    "reset_provider",
    "compute_cosine_similarity",
]
