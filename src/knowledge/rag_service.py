"""RAG 服务主入口

职责:
1. 协调查询改写、检索、置信度计算、知识强化
2. 提供异步查询接口
3. 管理数据飞轮

总延迟预算: < 150ms
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.knowledge.confidence import ConfidenceCalculator
from src.knowledge.flywheel import LOW_CONFIDENCE_THRESHOLD, DataFlywheel
from src.knowledge.knowledge_enhancer import KnowledgeEnhancer
from src.knowledge.knowledge_migrator import load_knowledge_base
from src.knowledge.models import (
    RAG_CONFIG,
    Evidence,
    KnowledgeItem,
    RiskLevel,
)
from src.knowledge.query_rewriter import QueryRewriter
from src.knowledge.retriever import HybridRetriever, KeywordRetriever, VectorRetriever

logger = logging.getLogger(__name__)


class RAGService:
    """RAG 服务"""

    def __init__(
        self,
        config_dir: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        enable_flywheel: bool = True,
        enable_vector: bool = True,
    ):
        """
        初始化 RAG 服务

        Args:
            config_dir: 配置目录路径 (包含 knowledge_base.json, term_mappings.json)
            data_dir: 数据目录路径 (用于飞轮存储)
            enable_flywheel: 是否启用数据飞轮
            enable_vector: 是否启用向量检索
        """
        # 默认目录
        if config_dir is None:
            config_dir = Path(__file__).parent.parent.parent / "config"
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data"

        self._config_dir = config_dir
        self._data_dir = data_dir
        self._enable_flywheel = enable_flywheel
        self._enable_vector = enable_vector

        # 组件
        self._query_rewriter: Optional[QueryRewriter] = None
        self._keyword_retriever: Optional[KeywordRetriever] = None
        self._vector_retriever: Optional[VectorRetriever] = None
        self._hybrid_retriever: Optional[HybridRetriever] = None
        self._confidence_calculator: Optional[ConfidenceCalculator] = None
        self._knowledge_enhancer: Optional[KnowledgeEnhancer] = None
        self._flywheel: Optional[DataFlywheel] = None

        # 知识库
        self._knowledge_items: List[KnowledgeItem] = []

        # 初始化状态
        self._initialized = False

    def initialize(self) -> None:
        """
        初始化所有组件

        应在服务启动时调用一次
        """
        if self._initialized:
            logger.warning("RAG service already initialized")
            return

        start_time = time.perf_counter()
        logger.info("Initializing RAG service...")

        # 1. 加载知识库
        knowledge_base_path = self._config_dir / "knowledge_base.json"
        self._knowledge_items = load_knowledge_base(knowledge_base_path)
        logger.info(f"Loaded {len(self._knowledge_items)} knowledge items")

        # 2. 初始化查询改写器
        term_mappings_path = self._config_dir / "term_mappings.json"
        self._query_rewriter = QueryRewriter(term_mappings_path)

        # 3. 初始化关键词检索器
        self._keyword_retriever = KeywordRetriever()
        self._keyword_retriever.build_index(self._knowledge_items, self._config_dir)

        # 4. 初始化向量检索器 (可选，支持缓存)
        if self._enable_vector:
            try:
                self._vector_retriever = VectorRetriever()
                cache_dir = self._data_dir / "vector_index"
                # 尝试从缓存加载（跳过 embedding 计算）
                if self._vector_retriever.load_index(cache_dir, self._knowledge_items):
                    logger.info("Vector index loaded from cache — skipping embedding computation")
                    # 仍需 warmup embedding 模型（首次推理 JIT）
                    warmup_start = time.perf_counter()
                    self._vector_retriever._provider.embed_sync("warmup")
                    warmup_ms = (time.perf_counter() - warmup_start) * 1000
                    logger.info(f"Embedding warmup completed in {warmup_ms:.1f}ms")
                else:
                    logger.info("No valid cache found, building vector index from scratch")
                    self._vector_retriever.build_index(self._knowledge_items)
                    # 保存到缓存供下次启动使用
                    self._vector_retriever.save_index(cache_dir)
            except Exception as e:
                logger.warning(f"Failed to initialize vector retriever: {e}")
                self._vector_retriever = VectorRetriever()  # 空的检索器
        else:
            self._vector_retriever = VectorRetriever()

        # 5. 初始化混合检索器
        self._hybrid_retriever = HybridRetriever(
            self._keyword_retriever,
            self._vector_retriever,
        )

        # 6. 初始化置信度计算器
        self._confidence_calculator = ConfidenceCalculator()

        # 7. 初始化知识强化器
        self._knowledge_enhancer = KnowledgeEnhancer()

        # 8. 初始化数据飞轮 (可选)
        if self._enable_flywheel:
            self._flywheel = DataFlywheel(self._data_dir)

        self._initialized = True
        latency = (time.perf_counter() - start_time) * 1000
        logger.info(f"RAG service initialized in {latency:.2f}ms")

    async def query_async(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Evidence:
        """
        异步查询接口 (向量检索带超时降级)

        Args:
            query: 用户查询
            context: 会话上下文

        Returns:
            Evidence
        """
        import asyncio

        if not self._initialized:
            self.initialize()

        start_time = time.perf_counter()

        # 默认上下文
        if context is None:
            context = {"session_id": "default"}

        # 1. 查询改写 (同步，很快)
        rewritten_query, replacements, rewrite_latency = self._query_rewriter.rewrite(query)

        # 2. 异步混合检索（带超时保护）
        product_line_filter = context.get("product_line")
        device_model_filter = context.get("device_model")
        try:
            search_results = await asyncio.wait_for(
                self._hybrid_retriever.search_async(
                    rewritten_query,
                    top_k=RAG_CONFIG.get("final_top_k", 5),
                    product_line_filter=product_line_filter,
                    device_model_filter=device_model_filter,
                ),
                timeout=RAG_CONFIG.get("search_timeout_ms", 250) / 1000,
            )
        except asyncio.TimeoutError:
            logger.warning("RAG search timeout, using keyword fallback")
            # 降级为纯关键词检索（走 HybridRetriever 保证 RRF 归一化）
            search_results = self._hybrid_retriever.search(
                rewritten_query,
                top_k=RAG_CONFIG.get("final_top_k", 5),
                product_line_filter=product_line_filter,
                device_model_filter=device_model_filter,
                enable_vector=False,
            )

        # 3. 置信度计算
        confidence, risk_level, reasoning, decision = self._confidence_calculator.calculate(
            search_results,
        )

        # 4. 知识强化
        needs_clarification, clarifying_question, warnings = self._knowledge_enhancer.enhance(
            search_results,
            context,
        )

        # 构建结果
        latency = (time.perf_counter() - start_time) * 1000
        result = Evidence(
            original_query=query,
            rewritten_query=rewritten_query,
            items=[r.item for r in search_results],
            search_results=search_results,
            confidence=confidence,
            risk_level=RiskLevel(risk_level),
            decision=decision,
            needs_clarification=needs_clarification,
            clarifying_question=clarifying_question,
            warnings=warnings,
            reasoning=reasoning,
            latency_ms=latency,
        )

        # 5. 数据飞轮收集 (基于置信度阈值)
        if self._flywheel and result.confidence < LOW_CONFIDENCE_THRESHOLD:
            session_id = context.get("session_id", "unknown")
            self._flywheel.collect(result, session_id)

        logger.info(
            f"RAG async query completed: confidence={result.confidence:.2f}, "
            f"risk={risk_level}, decision={result.decision.value}, "
            f"latency={latency:.2f}ms"
        )

        return result

    @property
    def knowledge_items(self) -> List[KnowledgeItem]:
        """获取知识库条目"""
        return self._knowledge_items

    @property
    def flywheel(self) -> Optional[DataFlywheel]:
        """获取数据飞轮"""
        return self._flywheel

    @property
    def is_initialized(self) -> bool:
        """是否已初始化"""
        return self._initialized


# 全局单例
_rag_service: Optional[RAGService] = None


def get_rag_service(
    config_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
) -> RAGService:
    """获取 RAG 服务单例"""
    global _rag_service

    if _rag_service is None:
        _rag_service = RAGService(config_dir, data_dir)

    return _rag_service
