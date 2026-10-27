"""Embedding Provider 工厂

根据配置创建对应的 embedding provider。
"""

import logging
from typing import Optional

from src.core.config import settings

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)

# 全局单例
_provider: Optional[EmbeddingProvider] = None


def get_embedding_provider() -> EmbeddingProvider:
    """获取 Embedding Provider 单例

    根据 settings.embedding_provider 配置返回对应的 provider:
    - "dashscope": DashScope 远程 API (text-embedding-v4)
    - "local": 本地模型 (默认 bge-base-zh-v1.5)

    Returns:
        EmbeddingProvider 实例
    """
    global _provider

    if _provider is not None:
        return _provider

    provider_type = settings.embedding_provider

    if provider_type == "local":
        try:
            from .local_provider import LocalEmbeddingProvider
        except ImportError as exc:
            raise ImportError(
                "Local embedding dependencies are missing. "
                "Install knowledge-local extras (uv sync --extra knowledge-local) to use EMBEDDING_PROVIDER=local."
            ) from exc

        _provider = LocalEmbeddingProvider(
            model_name=settings.embedding_local_model,
            device=settings.embedding_local_device,
        )
        logger.info(
            f"Using local embedding provider: "
            f"model={settings.embedding_local_model}, "
            f"device={settings.embedding_local_device}"
        )

    else:  # dashscope (default)
        from .dashscope_provider import DashScopeEmbeddingProvider

        _provider = DashScopeEmbeddingProvider()
        logger.info("Using DashScope embedding provider")

    return _provider


def reset_provider() -> None:
    """重置 provider 单例 (用于测试或切换配置)"""
    global _provider
    _provider = None
