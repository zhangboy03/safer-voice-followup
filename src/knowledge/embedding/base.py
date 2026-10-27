"""Embedding Provider 协议定义"""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class EmbeddingProvider(ABC):
    """Embedding Provider 协议

    所有 embedding 服务实现需遵循此协议。
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """返回 embedding 维度"""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """返回模型名称"""
        ...

    @abstractmethod
    def embed_sync(self, text: str) -> Optional[np.ndarray]:
        """
        同步获取单个文本的 embedding

        Args:
            text: 输入文本

        Returns:
            embedding 向量，失败返回 None
        """
        ...

    @abstractmethod
    def embed_batch_sync(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        同步批量获取 embedding

        Args:
            texts: 输入文本列表

        Returns:
            embedding 向量列表，失败的位置为 None
        """
        ...

    async def embed(self, text: str) -> Optional[np.ndarray]:
        """
        异步获取单个文本的 embedding (默认实现使用线程池)

        Args:
            text: 输入文本

        Returns:
            embedding 向量，失败返回 None
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_sync, text)

    async def embed_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        异步批量获取 embedding (默认实现使用线程池)

        Args:
            texts: 输入文本列表

        Returns:
            embedding 向量列表，失败的位置为 None
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_batch_sync, texts)
