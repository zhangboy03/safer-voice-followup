"""关键词检索器

使用 jieba 分词 + 倒排索引 + BM25 评分

延迟预算: < 10ms
"""

import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import jieba

from src.knowledge.models import KnowledgeItem

logger = logging.getLogger(__name__)


# BM25 参数
BM25_K1 = 1.5
BM25_B = 0.75


class KeywordRetriever:
    """关键词检索器"""

    def __init__(self, knowledge_items: Optional[List[KnowledgeItem]] = None):
        """
        初始化关键词检索器

        Args:
            knowledge_items: 知识库条目列表
        """
        # 倒排索引: {term: {item_id: term_frequency}}
        self._inverted_index: Dict[str, Dict[str, int]] = defaultdict(dict)

        # 文档长度
        self._doc_lengths: Dict[str, int] = {}

        # 平均文档长度
        self._avg_doc_length: float = 0.0

        # 文档总数
        self._total_docs: int = 0

        # ID 到知识项映射
        self._items: Dict[str, KnowledgeItem] = {}

        # 自定义词典是否已加载
        self._custom_dict_loaded = False

        if knowledge_items:
            self.build_index(knowledge_items)

    def _load_custom_dict(self, config_dir: Optional[Path] = None) -> None:
        """加载自定义词典"""
        if self._custom_dict_loaded:
            return

        # 添加设备型号
        device_models = [
            "G101A",
            "G102",
            "G102R",
            "G102RZ",
            "G106",
            "G106R",
            "G106RS",
            "G107",
            "G107R",
            "G111",
            "G112",
            "G113",
            "G114R",
            "C701",
            "C702",
            "R801",
            "R802",
        ]
        for model in device_models:
            jieba.add_word(model, freq=1000)

        # 添加产品线缩写
        product_lines = ["DBS", "VNS", "SCS", "SNM", "IPG", "MRI", "CT"]
        for pl in product_lines:
            jieba.add_word(pl, freq=1000)

        # 添加医学术语
        medical_terms = [
            "刺激器",
            "程控器",
            "充电器",
            "电极",
            "延伸导线",
            "脉冲发生器",
            "核磁检查",
            "磁共振",
            "帕金森",
            "肌张力障碍",
            "迷走神经",
            "脊髓电刺激",
            "骶神经",
            "震颤",
            "僵硬",
            "异动",
            "运动障碍",
            "程控",
            "远程程控",
            "起搏器自诊",
        ]
        for term in medical_terms:
            jieba.add_word(term, freq=500)

        # 尝试从 term_mappings.json 加载更多术语
        if config_dir:
            term_mappings_path = config_dir / "term_mappings.json"
            if term_mappings_path.exists():
                try:
                    with open(term_mappings_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # 从 mappings 中提取标准术语
                    mappings = data.get("mappings", {})
                    for category, terms in mappings.items():
                        if category.startswith("_"):
                            continue
                        for standard_term in terms.keys():
                            if not standard_term.startswith("_"):
                                jieba.add_word(standard_term, freq=500)

                    logger.info("Loaded custom terms from term_mappings.json")
                except Exception as e:
                    logger.warning(f"Failed to load term_mappings.json: {e}")

        self._custom_dict_loaded = True
        logger.info("Custom dictionary loaded for jieba")

    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        # 确保自定义词典已加载
        if not self._custom_dict_loaded:
            self._load_custom_dict()

        # jieba 分词
        tokens = list(jieba.cut(text))

        # 过滤停用词和单字符
        stopwords = {"的", "了", "是", "在", "有", "和", "与", "或", "吗", "呢", "啊", "呀", "吧"}
        tokens = [t.strip() for t in tokens if t.strip() and len(t.strip()) > 1 and t not in stopwords]

        return tokens

    def build_index(
        self,
        knowledge_items: List[KnowledgeItem],
        config_dir: Optional[Path] = None,
    ) -> None:
        """
        构建倒排索引

        Args:
            knowledge_items: 知识库条目列表
            config_dir: 配置目录 (用于加载自定义词典)
        """
        start_time = time.perf_counter()

        # 加载自定义词典
        self._load_custom_dict(config_dir)

        # 清空现有索引
        self._inverted_index.clear()
        self._doc_lengths.clear()
        self._items.clear()

        total_length = 0

        for item in knowledge_items:
            self._items[item.id] = item

            # 获取可检索文本
            searchable_text = item.get_searchable_text()
            tokens = self._tokenize(searchable_text)

            # 记录文档长度
            self._doc_lengths[item.id] = len(tokens)
            total_length += len(tokens)

            # 构建倒排索引
            term_freq: Dict[str, int] = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1

            for term, freq in term_freq.items():
                self._inverted_index[term][item.id] = freq

        self._total_docs = len(knowledge_items)
        self._avg_doc_length = total_length / max(self._total_docs, 1)

        latency = (time.perf_counter() - start_time) * 1000
        logger.info(f"Built keyword index: {self._total_docs} docs, {len(self._inverted_index)} terms, {latency:.2f}ms")

    def _compute_bm25(
        self,
        query_tokens: List[str],
        item_id: str,
    ) -> float:
        """
        计算 BM25 分数

        Args:
            query_tokens: 查询分词结果
            item_id: 文档 ID

        Returns:
            BM25 分数
        """
        score = 0.0
        doc_length = self._doc_lengths.get(item_id, 0)

        for token in query_tokens:
            if token not in self._inverted_index:
                continue

            posting = self._inverted_index[token]
            if item_id not in posting:
                continue

            # 词频
            tf = posting[item_id]

            # 文档频率
            df = len(posting)

            # IDF
            idf = math.log((self._total_docs - df + 0.5) / (df + 0.5) + 1)

            # BM25 公式
            numerator = tf * (BM25_K1 + 1)
            denominator = tf + BM25_K1 * (1 - BM25_B + BM25_B * doc_length / self._avg_doc_length)
            score += idf * numerator / denominator

        return score

    def search(
        self,
        query: str,
        top_k: int = 5,
        product_line_filter: Optional[str] = None,
        device_model_filter: Optional[str] = None,
    ) -> List[Tuple[KnowledgeItem, float]]:
        """
        关键词检索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            product_line_filter: 产品线过滤 (可选)
            device_model_filter: 设备型号过滤 (可选，用于优先排序而非硬过滤)

        Returns:
            [(KnowledgeItem, score), ...] 按分数降序
        """
        start_time = time.perf_counter()

        query_tokens = self._tokenize(query)
        if not query_tokens:
            logger.debug(f"Empty query tokens for: {query}")
            return []

        # 获取候选文档 (包含任一查询词的文档)
        candidate_ids: Set[str] = set()
        for token in query_tokens:
            if token in self._inverted_index:
                candidate_ids.update(self._inverted_index[token].keys())

        # 计算 BM25 分数
        scores: List[Tuple[str, float]] = []
        for item_id in candidate_ids:
            item = self._items.get(item_id)
            if not item:
                continue

            # 产品线过滤 (硬过滤)
            if product_line_filter:
                if item.product_line not in (product_line_filter, "ALL"):
                    continue

            score = self._compute_bm25(query_tokens, item_id)
            if score > 0:
                scores.append((item_id, score))

        # 按分数降序排序
        scores.sort(key=lambda x: -x[1])

        # 取 Top-K (多取一些以便 hybrid ranker 可以应用设备型号加成后重新排序)
        # 返回 fetch_k 个结果，让 hybrid ranker 负责最终截断
        fetch_k = top_k * 2 if device_model_filter else top_k
        results = []
        for item_id, score in scores[:fetch_k]:
            item = self._items.get(item_id)
            if item:
                results.append((item, score))

        latency = (time.perf_counter() - start_time) * 1000
        logger.debug(f"Keyword search '{query}' returned {len(results)} results in {latency:.2f}ms")

        # 返回所有 fetch_k 个结果，hybrid ranker 会应用 model boost 后重新排序并截断
        return results

    def get_item(self, item_id: str) -> Optional[KnowledgeItem]:
        """根据 ID 获取知识项"""
        return self._items.get(item_id)

    @property
    def total_docs(self) -> int:
        """文档总数"""
        return self._total_docs

    @property
    def vocab_size(self) -> int:
        """词表大小"""
        return len(self._inverted_index)
