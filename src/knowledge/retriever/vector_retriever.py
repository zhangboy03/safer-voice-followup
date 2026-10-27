"""向量检索器

使用 FAISS IndexFlatIP 进行向量检索

延迟预算: < 10ms (FAISS 检索) + 20-80ms (Embedding)
- 本地模型 (bge-base-zh): ~20-30ms
- 远程 API (DashScope): ~80-200ms
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.knowledge.embedding import EmbeddingProvider, get_embedding_provider
from src.knowledge.models import RAG_CONFIG, KnowledgeItem

logger = logging.getLogger(__name__)


class VectorRetriever:
    """向量检索器

    支持多向量索引：每个知识项可以有多个向量（主问题 + 问题变体 + 假设问题），
    检索时按 item_id 聚合，取 max score。
    """

    def __init__(
        self,
        knowledge_items: Optional[List[KnowledgeItem]] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ):
        """
        初始化向量检索器

        Args:
            knowledge_items: 知识库条目列表
            embedding_provider: Embedding Provider (可选，默认使用配置的 provider)
        """
        self._provider = embedding_provider or get_embedding_provider()
        self._index = None  # FAISS 索引
        self._id_map: List[str] = []  # 索引位置到 item_id 的映射（可重复）
        self._items: Dict[str, KnowledgeItem] = {}  # item_id 到 KnowledgeItem 的映射
        # 从 provider 获取维度，而非硬编码
        self._embedding_dim = self._provider.dim
        self._model_name = self._provider.model_name
        self._total_vectors = 0  # 索引中的向量总数

        logger.info(f"VectorRetriever initialized: model={self._model_name}, dim={self._embedding_dim}")

        if knowledge_items:
            self.build_index(knowledge_items)

    def build_index(
        self,
        knowledge_items: List[KnowledgeItem],
        use_cache: bool = False,  # 多向量模式下缓存不适用
    ) -> None:
        """
        构建多向量 FAISS 索引

        为每个知识项创建多个向量：
        1. 主问题
        2. 问题变体 (question_variants)
        3. 假设问题 (hypothetical_questions) - LLM 生成的患者可能问法

        检索时按 item_id 聚合，取 max score。

        Args:
            knowledge_items: 知识库条目列表
            use_cache: 已弃用，多向量模式下不使用缓存
        """
        _ = use_cache
        try:
            import faiss
        except ImportError:
            logger.error("faiss package not installed. Run: pip install faiss-cpu")
            raise ImportError("Please install faiss-cpu: pip install faiss-cpu")

        start_time = time.perf_counter()

        # 清空现有索引
        self._id_map.clear()
        self._items.clear()

        # 收集所有需要计算 embedding 的文本
        texts_to_embed: List[str] = []

        for item in knowledge_items:
            self._items[item.id] = item

            # 1. 主问题
            texts_to_embed.append(item.question)
            self._id_map.append(item.id)

            # 2. 问题变体
            for variant in item.question_variants:
                if variant and variant.strip():
                    texts_to_embed.append(variant.strip())
                    self._id_map.append(item.id)

            # 3. 假设问题（核心新增）
            for hq in item.hypothetical_questions:
                if hq and hq.strip():
                    texts_to_embed.append(hq.strip())
                    self._id_map.append(item.id)

        logger.info(f"Building multi-vector index: {len(texts_to_embed)} vectors for {len(knowledge_items)} items")

        # 批量计算 embedding
        embeddings = self._provider.embed_batch_sync(texts_to_embed)

        # 过滤无效 embedding
        valid_embeddings: List[np.ndarray] = []
        valid_ids: List[str] = []
        for emb, item_id in zip(embeddings, self._id_map):
            if emb is not None:
                valid_embeddings.append(emb)
                valid_ids.append(item_id)
            else:
                logger.warning(f"Failed to compute embedding for item {item_id}")

        self._id_map = valid_ids

        if not valid_embeddings:
            logger.warning("No valid embeddings, vector retriever disabled")
            return

        # 构建 FAISS 索引 (使用内积，需要先归一化)
        embeddings_matrix = np.vstack(valid_embeddings).astype(np.float32)

        # L2 归一化 (使内积等价于余弦相似度)
        faiss.normalize_L2(embeddings_matrix)

        # 创建 IndexFlatIP (内积索引)
        self._index = faiss.IndexFlatIP(self._embedding_dim)
        self._index.add(embeddings_matrix)

        self._total_vectors = self._index.ntotal

        latency = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Built multi-vector index: {self._total_vectors} vectors for {len(knowledge_items)} items, "
            f"dim={self._embedding_dim}, "
            f"{latency:.2f}ms"
        )

        # Warmup: 触发 PyTorch JIT 编译路径，消除首次查询的延迟尖刺
        warmup_start = time.perf_counter()
        self._provider.embed_sync("warmup")
        warmup_ms = (time.perf_counter() - warmup_start) * 1000
        logger.info(f"Embedding warmup completed in {warmup_ms:.1f}ms")

    def search(
        self,
        query: str,
        top_k: int = 5,
        product_line_filter: Optional[str] = None,
        device_model_filter: Optional[str] = None,
    ) -> List[Tuple[KnowledgeItem, float]]:
        """
        向量检索（多向量聚合）

        搜索时会匹配所有向量（主问题、问题变体、假设问题），
        然后按 item_id 聚合，取每个 item 的 max score。

        Args:
            query: 查询文本
            top_k: 返回结果数量
            product_line_filter: 产品线过滤 (可选)
            device_model_filter: 设备型号过滤 (可选，用于优先排序)

        Returns:
            [(KnowledgeItem, score), ...] 按相似度降序
        """
        if self._index is None or self._index.ntotal == 0:
            logger.warning("Vector index not built or empty")
            return []

        start_time = time.perf_counter()

        # 计算查询 embedding
        query_embedding = self._provider.embed_sync(query)
        if query_embedding is None:
            logger.warning("Failed to compute query embedding, falling back to empty result")
            return []

        # 归一化
        try:
            import faiss

            query_vec = query_embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_vec)
        except ImportError:
            # 手动归一化
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_vec = (query_embedding / norm).reshape(1, -1).astype(np.float32)
            else:
                return []

        # 搜索更多结果以便聚合后仍有足够结果
        # 多向量模式下，同一个 item 可能有多个匹配，需要多取一些
        search_k = min(top_k * 10, self._index.ntotal)
        distances, indices = self._index.search(query_vec, search_k)

        # 按 item_id 聚合，取 max score
        item_scores: Dict[str, float] = {}
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue

            item_id = self._id_map[idx]
            item = self._items.get(item_id)
            if not item:
                continue

            # 产品线过滤 (硬过滤)
            if product_line_filter:
                if item.product_line not in (product_line_filter, "ALL"):
                    continue

            # 内积分数 (归一化后等于余弦相似度，范围 [-1, 1])
            # 转换到 [0, 1] 范围
            score = (float(dist) + 1) / 2

            # 取同一 item 的 max score
            if item_id not in item_scores or score > item_scores[item_id]:
                item_scores[item_id] = score

        # 按分数降序排序，取 top_k
        sorted_items = sorted(item_scores.items(), key=lambda x: -x[1])[:top_k]
        results = [(self._items[item_id], score) for item_id, score in sorted_items]

        latency = (time.perf_counter() - start_time) * 1000
        logger.debug(f"Vector search '{query}' returned {len(results)} results in {latency:.2f}ms")

        return results

    async def search_async(
        self,
        query: str,
        top_k: int = 5,
        product_line_filter: Optional[str] = None,
        device_model_filter: Optional[str] = None,
    ) -> List[Tuple[KnowledgeItem, float]]:
        """
        异步向量检索 (带超时降级)

        Args:
            query: 查询文本
            top_k: 返回结果数量
            product_line_filter: 产品线过滤
            device_model_filter: 设备型号过滤

        Returns:
            [(KnowledgeItem, score), ...] 超时返回空列表
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self.search(query, top_k, product_line_filter, device_model_filter)),
                timeout=RAG_CONFIG["embedding_timeout_ms"] / 1000,
            )
            return result
        except asyncio.TimeoutError:
            logger.warning("Vector search timeout, returning empty result")
            return []
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    def get_item(self, item_id: str) -> Optional[KnowledgeItem]:
        """根据 ID 获取知识项"""
        return self._items.get(item_id)

    @property
    def total_docs(self) -> int:
        """索引中的知识项数量（去重后）"""
        return len(self._items)

    @property
    def total_vectors(self) -> int:
        """索引中的向量数量（包含假设问题等多向量）"""
        return self._total_vectors if self._index else 0

    @property
    def is_ready(self) -> bool:
        """索引是否就绪"""
        return self._index is not None and self._index.ntotal > 0

    def save_index(self, cache_dir: Path) -> None:
        """保存 FAISS 索引 + id_map 到目录"""
        if self._index is None:
            logger.warning("No index to save")
            return

        try:
            import json

            import faiss

            cache_dir.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(cache_dir / "index.faiss"))

            meta = {
                "id_map": self._id_map,
                "embedding_dim": self._embedding_dim,
                "model_name": self._model_name,
                "total_vectors": self._total_vectors,
                "item_count": len(self._items),
            }
            (cache_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
            logger.info(f"Saved vector index cache to {cache_dir} ({self._total_vectors} vectors)")
        except Exception as e:
            logger.error(f"Failed to save index cache: {e}")

    def load_index(self, cache_dir: Path, knowledge_items: List[KnowledgeItem]) -> bool:
        """从缓存加载 FAISS 索引 + id_map

        Returns:
            True 加载成功，False 需要重建
        """
        try:
            import json

            import faiss

            index_path = cache_dir / "index.faiss"
            meta_path = cache_dir / "meta.json"
            if not index_path.exists() or not meta_path.exists():
                return False

            meta = json.loads(meta_path.read_text())

            # 验证：维度和模型必须匹配
            if meta.get("embedding_dim") != self._embedding_dim:
                logger.warning("Cache dim mismatch, rebuilding")
                return False
            if meta.get("model_name") != self._model_name:
                logger.warning("Cache model mismatch, rebuilding")
                return False
            # 验证：知识库条目数必须匹配
            if meta.get("item_count") != len(knowledge_items):
                logger.warning(
                    f"Cache item count mismatch: {meta.get('item_count')} vs {len(knowledge_items)}, rebuilding"
                )
                return False

            self._index = faiss.read_index(str(index_path))
            self._id_map = meta["id_map"]
            self._total_vectors = self._index.ntotal

            # 重建 _items 映射（从传入的 knowledge_items）
            self._items = {item.id: item for item in knowledge_items}

            logger.info(f"Loaded vector index from cache: {self._total_vectors} vectors, {len(self._items)} items")
            return True

        except Exception as e:
            logger.warning(f"Failed to load index cache: {e}, will rebuild")
            return False
