"""本地 Embedding Provider

使用 sentence-transformers 加载本地模型，推荐 BAAI/bge-base-zh-v1.5
"""

import logging
import time
from typing import List, Optional

import numpy as np

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)


def _detect_device(requested: str) -> str:
    """检测可用设备

    Args:
        requested: 请求的设备 (auto/cpu/cuda/mps)

    Returns:
        实际使用的设备名称
    """
    if requested != "auto":
        return requested

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"


class LocalEmbeddingProvider(EmbeddingProvider):
    """本地 Embedding Provider

    使用 sentence-transformers 加载本地模型。
    推荐模型: BAAI/bge-base-zh-v1.5 (768维，~400MB)

    特点:
    - 延迟低 (~20-30ms vs API ~200ms)
    - 无网络依赖
    - 支持 CPU/CUDA/MPS 加速
    - LRU 缓存：重复查询 <0.01ms
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-zh-v1.5",
        device: str = "auto",
        cache_maxsize: int = 512,
    ):
        """
        初始化本地 Embedding Provider

        Args:
            model_name: HuggingFace 模型名称或本地路径
            device: 设备 (auto/cpu/cuda/mps)
            cache_maxsize: embedding 缓存大小 (0 表示禁用)
        """
        self._model_name = model_name
        self._requested_device = device
        self._device = _detect_device(device)
        self._model = None
        self._dim: Optional[int] = None

        # LRU 缓存: text -> np.ndarray
        self._cache: dict[str, np.ndarray] = {}
        self._cache_order: list[str] = []  # 访问顺序，最近使用的在末尾
        self._cache_maxsize = cache_maxsize

        logger.info(f"LocalEmbeddingProvider: model={model_name}, device={self._device}, cache={cache_maxsize}")

    def _load_model(self):
        """懒加载模型"""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers package not installed. Run: pip install sentence-transformers")

        start_time = time.perf_counter()
        logger.info(f"Loading embedding model: {self._model_name} on {self._device}...")

        self._model = SentenceTransformer(
            self._model_name,
            device=self._device,
        )

        # 获取维度
        self._dim = self._model.get_sentence_embedding_dimension()

        latency = (time.perf_counter() - start_time) * 1000
        logger.info(f"Loaded embedding model: dim={self._dim}, device={self._device}, latency={latency:.0f}ms")

    @property
    def dim(self) -> int:
        """返回 embedding 维度"""
        if self._dim is None:
            self._load_model()
        return self._dim  # type: ignore

    @property
    def model_name(self) -> str:
        """返回模型名称"""
        return self._model_name

    def _cache_get(self, text: str) -> Optional[np.ndarray]:
        """从缓存中获取 embedding，命中时更新访问顺序"""
        if text in self._cache:
            # 移到末尾（最近使用）
            self._cache_order.remove(text)
            self._cache_order.append(text)
            return self._cache[text]
        return None

    def _cache_put(self, text: str, embedding: np.ndarray) -> None:
        """将 embedding 放入缓存，超出容量时淘汰最久未使用的"""
        if self._cache_maxsize <= 0:
            return
        if text in self._cache:
            self._cache_order.remove(text)
        elif len(self._cache) >= self._cache_maxsize:
            evicted = self._cache_order.pop(0)
            del self._cache[evicted]
        self._cache[text] = embedding
        self._cache_order.append(text)

    def embed_sync(self, text: str) -> Optional[np.ndarray]:
        """
        同步获取单个文本的 embedding

        Args:
            text: 输入文本

        Returns:
            embedding 向量 (已归一化)，失败返回 None
        """
        # 缓存查找
        cached = self._cache_get(text)
        if cached is not None:
            logger.debug("[Local] Cache hit for embedding")
            return cached

        try:
            self._load_model()

            start_time = time.perf_counter()

            # sentence-transformers 默认返回 numpy 数组
            embedding = self._model.encode(
                text,
                normalize_embeddings=True,  # L2 归一化
                convert_to_numpy=True,
            )

            latency = (time.perf_counter() - start_time) * 1000
            logger.debug(f"[Local] Embedded 1 text in {latency:.2f}ms")

            result = embedding.astype(np.float32)
            self._cache_put(text, result)
            return result

        except Exception as e:
            logger.error(f"Local embedding error: {e}")
            return None

    def embed_batch_sync(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        同步批量获取 embedding

        Args:
            texts: 输入文本列表

        Returns:
            embedding 向量列表 (已归一化)，失败的位置为 None
        """
        if not texts:
            return []

        # 分离缓存命中和未命中
        results: list[Optional[np.ndarray]] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self._cache_get(text)
            if cached is not None:
                results[i] = cached
            else:
                miss_indices.append(i)
                miss_texts.append(text)

        if miss_indices:
            logger.debug(
                f"[Local] Batch embed: {len(texts) - len(miss_indices)} cache hits, {len(miss_indices)} misses"
            )

        if not miss_texts:
            logger.debug(f"[Local] All {len(texts)} texts served from cache")
            return results

        try:
            self._load_model()

            start_time = time.perf_counter()

            # 批量编码未命中的文本
            embeddings = self._model.encode(
                miss_texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            latency = (time.perf_counter() - start_time) * 1000
            logger.debug(f"[Local] Embedded {len(miss_texts)} texts in {latency:.2f}ms")

            # 填充结果并写入缓存
            for idx, emb in zip(miss_indices, embeddings):
                result = emb.astype(np.float32)
                results[idx] = result
                self._cache_put(texts[idx], result)

            return results

        except Exception as e:
            logger.error(f"Local batch embedding error: {e}")
            return [None] * len(texts)
