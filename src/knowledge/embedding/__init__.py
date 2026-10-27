"""Embedding 服务包

提供可配置的 embedding 服务，支持：
- DashScope 远程 API (text-embedding-v4, 1024维)
- 本地模型 (bge-base-zh-v1.5, 768维)

使用方式:
    from src.knowledge.embedding import get_embedding_provider

    provider = get_embedding_provider()
    embedding = provider.embed_sync("你好")

配置:
    # .env
    EMBEDDING_PROVIDER=local  # 或 dashscope (默认)
    EMBEDDING_LOCAL_MODEL=BAAI/bge-base-zh-v1.5
    EMBEDDING_LOCAL_DEVICE=auto  # cpu/cuda/mps/auto
"""

from .base import EmbeddingProvider
from .factory import get_embedding_provider, reset_provider


def compute_cosine_similarity(a, b) -> float:
    """计算余弦相似度"""
    import numpy as np

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


__all__ = [
    "EmbeddingProvider",
    "get_embedding_provider",
    "reset_provider",
    "compute_cosine_similarity",
]
