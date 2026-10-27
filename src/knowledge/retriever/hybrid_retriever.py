"""混合检索器

合并关键词检索和向量检索结果，使用 Reciprocal Rank Fusion (RRF)

RRF 公式: rrf_score = alpha / (k + rank_kw) + (1 - alpha) / (k + rank_vec)
- k = 60 (业界标准常量)
- alpha = 0.5 (关键词/向量等权)
- 缺失某路检索时该分量为 0

最终对 RRF 分数做 Min-Max 归一化到 [0,1]，再叠加设备型号加成。

延迟预算: < 15ms (合并排序 + 评分归一化)
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

from src.knowledge.models import RAG_CONFIG, KnowledgeItem, SearchResult
from src.knowledge.retriever.keyword_retriever import KeywordRetriever
from src.knowledge.retriever.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


def normalize_scores(scores: List[float]) -> List[float]:
    """
    将分数归一化到 [0, 1] 区间 (Min-Max 归一化)

    Args:
        scores: 原始分数列表

    Returns:
        归一化后的分数列表
    """
    if not scores:
        return []

    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [0.5] * len(scores)  # 避免除零，返回中间值

    return [(s - min_s) / (max_s - min_s) for s in scores]


class HybridRetriever:
    """混合检索器"""

    def __init__(
        self,
        keyword_retriever: KeywordRetriever,
        vector_retriever: VectorRetriever,
    ):
        """
        初始化混合检索器

        Args:
            keyword_retriever: 关键词检索器
            vector_retriever: 向量检索器
        """
        self._keyword_retriever = keyword_retriever
        self._vector_retriever = vector_retriever

    def search(
        self,
        query: str,
        top_k: int = 5,
        product_line_filter: Optional[str] = None,
        device_model_filter: Optional[str] = None,
        enable_vector: bool = True,
    ) -> List[SearchResult]:
        """
        混合检索

        Args:
            query: 查询文本
            top_k: 最终返回结果数量
            product_line_filter: 产品线过滤
            device_model_filter: 设备型号过滤 (优先匹配，然后 fallback)
            enable_vector: 是否启用向量检索 (超时或失败时可禁用)

        Returns:
            SearchResult 列表，按优先级和分数排序
        """
        start_time = time.perf_counter()

        keyword_top_k = RAG_CONFIG.get("keyword_top_k", 5)
        vector_top_k = RAG_CONFIG.get("vector_top_k", 5)

        # 关键词检索
        keyword_results = self._keyword_retriever.search(
            query,
            top_k=keyword_top_k,
            product_line_filter=product_line_filter,
            device_model_filter=device_model_filter,
        )

        # 向量检索 (可选)
        vector_results: List[Tuple[KnowledgeItem, float]] = []
        if enable_vector and self._vector_retriever.is_ready:
            vector_results = self._vector_retriever.search(
                query,
                top_k=vector_top_k,
                product_line_filter=product_line_filter,
                device_model_filter=device_model_filter,
            )

        # 合并结果
        merged = self._merge_results(
            keyword_results,
            vector_results,
            device_model_filter=device_model_filter,
        )

        # 取 Top-K
        results = merged[:top_k]

        latency = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"Hybrid search '{query}': "
            f"keyword={len(keyword_results)}, vector={len(vector_results)}, "
            f"merged={len(results)}, {latency:.2f}ms"
        )

        return results

    async def search_async(
        self,
        query: str,
        top_k: int = 5,
        product_line_filter: Optional[str] = None,
        device_model_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        异步混合检索 (向量检索带超时降级)

        Args:
            query: 查询文本
            top_k: 返回结果数量
            product_line_filter: 产品线过滤
            device_model_filter: 设备型号过滤

        Returns:
            SearchResult 列表
        """

        start_time = time.perf_counter()

        keyword_top_k = RAG_CONFIG.get("keyword_top_k", 5)
        vector_top_k = RAG_CONFIG.get("vector_top_k", 5)

        # 关键词检索 (同步，很快)
        keyword_results = self._keyword_retriever.search(
            query,
            top_k=keyword_top_k,
            product_line_filter=product_line_filter,
            device_model_filter=device_model_filter,
        )

        # 向量检索 (异步，带超时)
        vector_results: List[Tuple[KnowledgeItem, float]] = []
        if self._vector_retriever.is_ready:
            vector_results = await self._vector_retriever.search_async(
                query,
                top_k=vector_top_k,
                product_line_filter=product_line_filter,
                device_model_filter=device_model_filter,
            )

        # 合并结果
        merged = self._merge_results(
            keyword_results,
            vector_results,
            device_model_filter=device_model_filter,
        )
        results = merged[:top_k]

        latency = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"Async hybrid search '{query}': "
            f"keyword={len(keyword_results)}, vector={len(vector_results)}, "
            f"merged={len(results)}, {latency:.2f}ms"
        )

        return results

    def _merge_results(
        self,
        keyword_results: List[Tuple[KnowledgeItem, float]],
        vector_results: List[Tuple[KnowledgeItem, float]],
        device_model_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        合并两路检索结果 (Reciprocal Rank Fusion)

        RRF 公式: rrf_score = alpha / (k + rank_kw) + (1 - alpha) / (k + rank_vec)
        - k = 60, alpha = 0.5
        - 缺失某路检索时该分量为 0
        - 最终 min-max 归一化到 [0,1]，再叠加 model_boost

        Args:
            keyword_results: 关键词检索结果 [(item, score), ...]（已按分数降序）
            vector_results: 向量检索结果 [(item, score), ...]（已按分数降序）
            device_model_filter: 设备型号 (用于分数加成)

        Returns:
            合并后的 SearchResult 列表，按 RRF 分数降序
        """
        RRF_K = 60
        ALPHA = 0.5

        # 构建 rank 字典（1-indexed rank，基于输入顺序即分数降序）
        keyword_rank: Dict[str, int] = {}
        keyword_items: Dict[str, KnowledgeItem] = {}
        for rank, (item, _score) in enumerate(keyword_results, start=1):
            keyword_rank[item.id] = rank
            keyword_items[item.id] = item

        vector_rank: Dict[str, int] = {}
        vector_items: Dict[str, KnowledgeItem] = {}
        for rank, (item, _score) in enumerate(vector_results, start=1):
            vector_rank[item.id] = rank
            vector_items[item.id] = item

        # 收集所有 item
        all_ids = set(keyword_rank.keys()) | set(vector_rank.keys())
        all_items: Dict[str, KnowledgeItem] = {**vector_items, **keyword_items}

        # 计算 RRF 分数
        rrf_scores: Dict[str, float] = {}
        sources: Dict[str, str] = {}
        for item_id in all_ids:
            kw_r = keyword_rank.get(item_id)
            vec_r = vector_rank.get(item_id)

            kw_component = ALPHA / (RRF_K + kw_r) if kw_r is not None else 0.0
            vec_component = (1 - ALPHA) / (RRF_K + vec_r) if vec_r is not None else 0.0
            rrf_scores[item_id] = kw_component + vec_component

            # source 标记（用于可观测性，不参与排序）
            if kw_r is not None and vec_r is not None:
                sources[item_id] = "both"
            elif kw_r is not None:
                sources[item_id] = "keyword"
            else:
                sources[item_id] = "vector"

        # Min-Max 归一化 RRF 分数到 [0, 1]
        raw_rrf_list = list(rrf_scores.values())
        if raw_rrf_list:
            normalized = normalize_scores(raw_rrf_list)
            rrf_ids = list(rrf_scores.keys())
            for i, item_id in enumerate(rrf_ids):
                rrf_scores[item_id] = normalized[i]

        # 按型号匹配分层排序 + 层内按 RRF 分数排序
        results: List[SearchResult] = []
        for item_id in all_ids:
            item = all_items[item_id]
            results.append(
                SearchResult(
                    item=item,
                    score=rrf_scores[item_id],
                    source=sources[item_id],
                )
            )

        if device_model_filter:
            # 医疗安全：型号匹配项 > 通用项 > 型号不匹配项，层内按分数降序
            results.sort(
                key=lambda x: (
                    self._model_match_tier(x.item, device_model_filter),
                    -x.score,
                )
            )
        else:
            results.sort(key=lambda x: -x.score)
        return results

    # 型号匹配层级（值越小优先级越高）
    _TIER_EXACT = 0  # 精确匹配
    _TIER_SERIES = 1  # 系列匹配
    _TIER_GENERIC = 2  # 通用（无型号限定）
    _TIER_MISMATCH = 3  # 型号不匹配

    def _model_match_tier(
        self,
        item: KnowledgeItem,
        device_model_filter: str,
    ) -> int:
        """
        确定知识项相对于查询型号的匹配层级。

        医疗安全场景下，排序必须满足：
        - 型号匹配的知识排在通用知识前面
        - 错误型号的专属知识排在通用知识后面
        """
        device_models = item.device_models
        if not device_models:
            return self._TIER_GENERIC

        if device_model_filter in device_models:
            return self._TIER_EXACT

        for model in device_models:
            if device_model_filter.startswith(model) or model.startswith(device_model_filter):
                return self._TIER_SERIES

        return self._TIER_MISMATCH

    @property
    def keyword_retriever(self) -> KeywordRetriever:
        """获取关键词检索器"""
        return self._keyword_retriever

    @property
    def vector_retriever(self) -> VectorRetriever:
        """获取向量检索器"""
        return self._vector_retriever
