"""查询改写模块

职责: 术语映射改写 - 将口语化表达转为标准医学术语

延迟预算: < 5ms (纯内存哈希表查找)
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QueryRewriter:
    """查询改写器"""

    def __init__(self, term_mappings_path: Optional[Path] = None):
        """
        初始化查询改写器

        Args:
            term_mappings_path: 术语映射文件路径
        """
        self._reverse_lookup: Dict[str, Dict[str, str]] = {}
        self._sorted_keys: List[str] = []  # 按长度降序排列的键

        if term_mappings_path:
            self._load_term_mappings(term_mappings_path)

    def _load_term_mappings(self, path: Path) -> None:
        """加载术语映射表"""
        logger.info(f"Loading term mappings from {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 获取 reverse_lookup
        reverse_lookup = data.get("reverse_lookup", {})

        # 过滤掉以 _ 开头的元数据字段
        self._reverse_lookup = {k: v for k, v in reverse_lookup.items() if not k.startswith("_")}

        # 按长度降序排列，确保长词优先匹配
        self._sorted_keys = sorted(self._reverse_lookup.keys(), key=len, reverse=True)

        logger.info(f"Loaded {len(self._reverse_lookup)} term mappings")

    def rewrite(self, query: str) -> Tuple[str, List[Dict[str, str]], float]:
        """
        改写查询

        Args:
            query: 原始查询

        Returns:
            (改写后的查询, 替换记录列表, 耗时毫秒)
            替换记录: [{"original": "...", "standard": "...", "category": "..."}]
        """
        start_time = time.perf_counter()

        if not self._reverse_lookup:
            latency = (time.perf_counter() - start_time) * 1000
            return query, [], latency

        rewritten = query
        replacements = []

        # 按长度降序匹配，避免短词误匹配
        for colloquial in self._sorted_keys:
            if colloquial in rewritten:
                mapping = self._reverse_lookup[colloquial]
                standard = mapping.get("standard", colloquial)
                category = mapping.get("category", "unknown")

                # 替换
                rewritten = rewritten.replace(colloquial, standard)

                replacements.append(
                    {
                        "original": colloquial,
                        "standard": standard,
                        "category": category,
                    }
                )

        latency = (time.perf_counter() - start_time) * 1000
        logger.debug(f"Query rewrite: '{query}' -> '{rewritten}' ({latency:.2f}ms)")

        return rewritten, replacements, latency
