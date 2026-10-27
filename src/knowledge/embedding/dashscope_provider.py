"""DashScope Embedding Provider

使用阿里云 DashScope text-embedding-v4 模型
"""

import logging
import time
from typing import List, Optional

import numpy as np

from src.core.config import settings

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)

# 配置
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIM = 1024
DEFAULT_TIMEOUT_MS = 80
MAX_BATCH_SIZE = 10  # DashScope 单次最大批量（API 限制）


class DashScopeEmbeddingProvider(EmbeddingProvider):
    """DashScope Embedding Provider"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ):
        """
        初始化 DashScope Embedding Provider

        Args:
            api_key: DashScope API Key (默认从配置读取)
            timeout_ms: 超时时间 (毫秒)
        """
        self.api_key = api_key or settings.dashscope_api_key
        self.timeout_ms = timeout_ms
        self._client = None

        if not self.api_key:
            logger.warning("DASHSCOPE_API_KEY not set, embedding service disabled")

    @property
    def dim(self) -> int:
        """返回 embedding 维度"""
        return EMBEDDING_DIM

    @property
    def model_name(self) -> str:
        """返回模型名称"""
        return EMBEDDING_MODEL

    def _get_client(self):
        """懒加载 DashScope 客户端"""
        if self._client is None:
            try:
                import dashscope

                dashscope.api_key = self.api_key
                dashscope.base_http_api_url = settings.active_dashscope_http_api_url
                self._client = dashscope.TextEmbedding
            except ImportError:
                logger.error("dashscope package not installed")
                raise ImportError("Please install dashscope: pip install dashscope")
        return self._client

    def embed_sync(self, text: str) -> Optional[np.ndarray]:
        """
        同步获取单个文本的 embedding

        Args:
            text: 输入文本

        Returns:
            embedding 向量，失败返回 None
        """
        result = self.embed_batch_sync([text])
        return result[0] if result else None

    def embed_batch_sync(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        同步批量获取 embedding

        Args:
            texts: 输入文本列表

        Returns:
            embedding 向量列表，失败的位置为 None
        """
        if not self.api_key:
            logger.warning("Embedding service disabled, returning None")
            return [None] * len(texts)

        if not texts:
            return []

        start_time = time.perf_counter()
        results: List[Optional[np.ndarray]] = []

        try:
            client = self._get_client()

            # 分批处理
            for i in range(0, len(texts), MAX_BATCH_SIZE):
                batch = texts[i : i + MAX_BATCH_SIZE]

                response = client.call(
                    model=EMBEDDING_MODEL,
                    input=batch,
                )

                if response.status_code == 200:
                    embeddings = response.output.get("embeddings", [])
                    for emb in embeddings:
                        vector = emb.get("embedding")
                        if vector:
                            results.append(np.array(vector, dtype=np.float32))
                        else:
                            results.append(None)
                else:
                    logger.error(f"Embedding API error: {response.code} - {response.message}")
                    results.extend([None] * len(batch))

            latency = (time.perf_counter() - start_time) * 1000
            logger.debug(f"[DashScope] Embedded {len(texts)} texts in {latency:.2f}ms")

            # 检查是否超时 (仅警告，不影响结果)
            if latency > self.timeout_ms:
                logger.warning(f"Embedding latency {latency:.2f}ms exceeded timeout {self.timeout_ms}ms")

            return results

        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return [None] * len(texts)

    async def embed(self, text: str) -> Optional[np.ndarray]:
        """
        异步获取单个文本的 embedding

        Args:
            text: 输入文本

        Returns:
            embedding 向量，失败或超时返回 None
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self.embed_sync, text), timeout=self.timeout_ms / 1000
            )
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Embedding timeout after {self.timeout_ms}ms")
            return None
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None

    async def embed_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        异步批量获取 embedding

        Args:
            texts: 输入文本列表

        Returns:
            embedding 向量列表，失败的位置为 None
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self.embed_batch_sync, texts),
                timeout=self.timeout_ms / 1000 * (len(texts) // MAX_BATCH_SIZE + 1),
            )
            return result
        except asyncio.TimeoutError:
            logger.warning("Batch embedding timeout")
            return [None] * len(texts)
        except Exception as e:
            logger.error(f"Batch embedding error: {e}")
            return [None] * len(texts)
