"""知识库结构迁移工具

将旧版知识库迁移到新结构，新增字段:
- module_id: 知识模块 ID
- product_line: 产品线 (DBS/VNS/SCS/SNM/ALL)
- device_models: 适用设备型号列表
- risk_level: 风险等级 (low/medium/high)
- question_variants: 问题变体
- prerequisites: 前置条件
- contraindications: 禁忌项
- retrieval_hints: 检索提示
"""

import json
import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.knowledge.models import (
    Contraindication,
    KnowledgeItem,
    KnowledgeItemMetadata,
    Prerequisites,
    RetrievalHints,
)

logger = logging.getLogger(__name__)


# 产品线关键词映射
PRODUCT_LINE_KEYWORDS = {
    "DBS": [
        "DBS",
        "脑深部",
        "脑起搏器",
        "帕金森",
        "肌张力障碍",
        "G101",
        "G102",
        "G106",
        "G107",
        "C701",
        "C702",
        "R801",
        "R802",
    ],
    "VNS": [
        "VNS",
        "迷走神经",
        "癫痫",
        "G111",
        "G112",
        "G113",
        "G114",
    ],
    "SCS": [
        "SCS",
        "脊髓电刺激",
        "疼痛",
    ],
    "SNM": [
        "SNM",
        "骶神经",
        "膀胱",
    ],
}

# 设备型号正则
DEVICE_MODEL_PATTERN = re.compile(
    r"\b(G101A?|G102R?Z?|G106R?S?|G107R?|G111|G112|G113|G114R?|C701|C702|R801|R802)\b", re.IGNORECASE
)

# 高风险关键词
HIGH_RISK_KEYWORDS = ["核磁", "MRI", "磁共振", "手术", "取出", "更换"]
MEDIUM_RISK_KEYWORDS = ["症状", "不适", "疼痛", "异常", "故障", "副作用"]

# module_id 推断规则
MODULE_ID_RULES = [
    # (关键词列表, module_id)
    (["核磁", "MRI", "磁共振"], "medical.mri"),
    (["CT", "ct"], "medical.ct"),
    (["手术", "植入", "更换", "取出"], "medical.surgery"),
    (["症状", "不适", "震颤", "僵硬"], "medical.symptom"),
    (["充电器", "R801", "R802"], "device.charger"),
    (["程控器", "C701", "C702"], "device.controller"),
    (["刺激器", "IPG", "脉冲发生器", "电池"], "device.ipg"),
    (["电极", "导线"], "device.electrode"),
    (["充电", "充不上", "充不进"], "operation.charging"),
    (["程控", "调参数", "调机器"], "operation.control"),
    (["开机", "关机", "开关"], "operation.switch"),
    (["出行", "飞机", "高铁", "安检"], "daily.travel"),
    (["洗澡", "游泳", "运动", "日常"], "daily.activity"),
    (["电磁", "干扰", "手机", "微波炉"], "daily.environment"),
    (["联系", "电话", "客服"], "support.contact"),
    (["保修", "维修", "更换"], "support.warranty"),
]


def infer_product_line(item: Dict[str, Any]) -> str:
    """从知识项推断产品线"""
    text = f"{item.get('topic', '')} {item.get('question', '')} {item.get('answer', '')}"

    # 检查是否明确包含产品线标识
    for product_line, keywords in PRODUCT_LINE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                return product_line

    return "ALL"


def extract_device_models(item: Dict[str, Any]) -> List[str]:
    """从知识项提取设备型号"""
    text = f"{item.get('question', '')} {item.get('answer', '')}"
    matches = DEVICE_MODEL_PATTERN.findall(text)
    # 去重并保持顺序
    seen = set()
    result = []
    for m in matches:
        upper = m.upper()
        if upper not in seen:
            seen.add(upper)
            result.append(upper)
    return result


def infer_risk_level(item: Dict[str, Any]) -> str:
    """推断风险等级"""
    text = f"{item.get('topic', '')} {item.get('question', '')} {item.get('answer', '')}"

    # 高风险检查
    for keyword in HIGH_RISK_KEYWORDS:
        if keyword in text:
            return "high"

    # 中风险检查
    for keyword in MEDIUM_RISK_KEYWORDS:
        if keyword in text:
            return "medium"

    return "low"


def infer_module_id(item: Dict[str, Any]) -> str:
    """推断模块 ID"""
    text = f"{item.get('topic', '')} {item.get('question', '')} {item.get('answer', '')}"

    for keywords, module_id in MODULE_ID_RULES:
        for keyword in keywords:
            if keyword in text:
                return module_id

    # 默认分类
    topic = item.get("topic", "")
    if topic:
        return f"general.{topic[:20]}"
    return "general.unknown"


def extract_question_variants(item: Dict[str, Any]) -> List[str]:
    """从 question 字段提取问题变体

    原始数据中有些 question 字段包含多个问题变体，用换行符分隔
    """
    question = item.get("question", "")
    # 按换行符分割
    lines = [line.strip() for line in question.split("\n") if line.strip()]

    if len(lines) <= 1:
        return []

    # 第一行是主问题，其余是变体
    return lines[1:]


def build_prerequisites(item: Dict[str, Any]) -> Optional[Prerequisites]:
    """构建前置条件"""
    text = f"{item.get('question', '')} {item.get('answer', '')}"

    # 核磁检查需要知道设备型号
    if any(kw in text for kw in ["核磁", "MRI", "磁共振"]):
        return Prerequisites(
            required_context=["device_model"],
            clarifying_question="请问您植入的是什么型号的刺激器？您可以在程控器或充电器上查看。",
        )

    # 设备功能相关需要知道具体设备型号
    if any(kw in text for kw in ["能不能", "可以", "支持", "功能"]) and any(kw in text for kw in ["远程", "调参"]):
        return Prerequisites(
            required_context=["device_model"],
            clarifying_question="请问您使用的是什么型号的程控器或充电器？设备背面有型号标识。",
        )

    return None


def build_contraindications(item: Dict[str, Any]) -> List[Contraindication]:
    """构建禁忌项列表"""
    contraindications = []
    answer = item.get("answer", "")

    # G101A 不支持核磁
    if "G101A" in answer and any(kw in answer for kw in ["不支持", "不能", "不建议"]):
        if any(kw in answer for kw in ["核磁", "MRI"]):
            contraindications.append(
                Contraindication(
                    condition_type="device_model_is",
                    condition_value="G101A",
                    warning="G101A 型号不支持核磁检查，请勿进行 MRI 检查。",
                )
            )

    # 非核磁兼容电极
    if "非核磁兼容" in answer:
        contraindications.append(
            Contraindication(
                condition_type="has_context",
                condition_value="non_mri_compatible_electrode",
                warning="您的电极为非核磁兼容型号，不建议进行磁共振检查。",
            )
        )

    return contraindications


def build_retrieval_hints(item: Dict[str, Any]) -> RetrievalHints:
    """构建检索提示"""
    keywords = []
    synonyms = []

    topic = item.get("topic", "")
    tags = item.get("metadata", {}).get("tags", [])

    # 从标签提取关键词
    keywords.extend(tags)

    # 从 topic 提取
    if topic and topic not in keywords:
        keywords.append(topic)

    # 设备型号作为关键词
    models = extract_device_models(item)
    keywords.extend(models)

    # 去重
    keywords = list(dict.fromkeys(keywords))

    return RetrievalHints(keywords=keywords, synonyms=synonyms)


def migrate_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """迁移单个知识项到新结构"""
    # 提取问题变体
    question_variants = extract_question_variants(item)

    # 清理主问题 (只保留第一行)
    question = item.get("question", "")
    main_question = question.split("\n")[0].strip()

    # 构建新字段
    migrated = {
        # 原有字段
        "id": item.get("id", ""),
        "topic": item.get("topic", ""),
        "question": main_question,
        "answer": item.get("answer", ""),
        "metadata": item.get("metadata", {}),
        # 新增字段
        "module_id": infer_module_id(item),
        "product_line": infer_product_line(item),
        "device_models": extract_device_models(item),
        "risk_level": infer_risk_level(item),
        "question_variants": question_variants,
        "retrieval_hints": asdict(build_retrieval_hints(item)),
    }

    # 前置条件 (可选)
    prerequisites = build_prerequisites(item)
    if prerequisites:
        migrated["prerequisites"] = asdict(prerequisites)

    # 禁忌项 (可选)
    contraindications = build_contraindications(item)
    if contraindications:
        migrated["contraindications"] = [asdict(c) for c in contraindications]

    return migrated


def migrate_knowledge_base(
    input_path: Path, output_path: Optional[Path] = None, dry_run: bool = False
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    迁移整个知识库

    Args:
        input_path: 输入知识库路径
        output_path: 输出路径 (None 则覆盖原文件)
        dry_run: 是否仅预览不写入

    Returns:
        (迁移后的数据, 统计信息)
    """
    logger.info(f"Loading knowledge base from {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)

    logger.info(f"Loaded {len(knowledge_base)} items")

    # 迁移统计
    stats = {
        "total": len(knowledge_base),
        "product_line": {"DBS": 0, "VNS": 0, "SCS": 0, "SNM": 0, "ALL": 0},
        "risk_level": {"low": 0, "medium": 0, "high": 0},
        "with_prerequisites": 0,
        "with_contraindications": 0,
        "with_variants": 0,
    }

    migrated_items = []
    for item in knowledge_base:
        migrated = migrate_item(item)
        migrated_items.append(migrated)

        # 更新统计
        pl = migrated.get("product_line", "ALL")
        if pl in stats["product_line"]:
            stats["product_line"][pl] += 1

        rl = migrated.get("risk_level", "low")
        if rl in stats["risk_level"]:
            stats["risk_level"][rl] += 1

        if migrated.get("prerequisites"):
            stats["with_prerequisites"] += 1
        if migrated.get("contraindications"):
            stats["with_contraindications"] += 1
        if migrated.get("question_variants"):
            stats["with_variants"] += 1

    logger.info(f"Migration stats: {stats}")

    if not dry_run:
        out = output_path or input_path
        with open(out, "w", encoding="utf-8") as f:
            json.dump(migrated_items, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved migrated knowledge base to {out}")

    return migrated_items, stats


def is_migration_complete(knowledge_base: List[Dict[str, Any]]) -> bool:
    """检查迁移是否完成 (所有项都有必要字段)"""
    required_fields = ["module_id", "risk_level", "product_line"]
    return all(all(f in item and item[f] for f in required_fields) for item in knowledge_base)


def load_knowledge_item(raw: Dict[str, Any]) -> KnowledgeItem:
    """
    兼容新旧 Schema 的加载函数

    新字段优先，旧字段兜底
    """
    # 处理 metadata
    metadata_raw = raw.get("metadata", {})
    metadata = KnowledgeItemMetadata(
        tags=metadata_raw.get("tags", []),
        source=metadata_raw.get("source", ""),
        version=metadata_raw.get("version", ""),
        last_verified=metadata_raw.get("last_verified", ""),
    )

    # 处理 prerequisites
    prereq_raw = raw.get("prerequisites")
    prerequisites = None
    if prereq_raw:
        prerequisites = Prerequisites(
            required_context=prereq_raw.get("required_context", []),
            clarifying_question=prereq_raw.get("clarifying_question", ""),
        )

    # 处理 contraindications
    contra_raw = raw.get("contraindications", [])
    contraindications = [
        Contraindication(
            condition_type=c.get("condition_type", ""),
            condition_value=c.get("condition_value"),
            warning=c.get("warning", ""),
        )
        for c in contra_raw
    ]

    # 处理 retrieval_hints
    hints_raw = raw.get("retrieval_hints", {})
    retrieval_hints = RetrievalHints(
        keywords=hints_raw.get("keywords", []),
        synonyms=hints_raw.get("synonyms", []),
    )

    # 提取问题变体 (兼容旧格式)
    question_variants = raw.get("question_variants", [])
    if not question_variants:
        # 尝试从 question 字段提取
        question_variants = extract_question_variants(raw)

    # 提取假设问题（LLM 生成的患者可能问法）
    hypothetical_questions = raw.get("hypothetical_questions", [])

    # 清理主问题
    question = raw.get("question", "")
    main_question = question.split("\n")[0].strip()

    return KnowledgeItem(
        id=raw.get("id", ""),
        question=main_question,
        answer=raw.get("answer", ""),
        # 新字段优先，旧字段兜底
        module_id=raw.get("module_id") or infer_module_id(raw),
        product_line=raw.get("product_line") or infer_product_line(raw),
        device_models=raw.get("device_models") or extract_device_models(raw),
        risk_level=raw.get("risk_level") or infer_risk_level(raw),
        question_variants=question_variants,
        hypothetical_questions=hypothetical_questions,
        prerequisites=prerequisites,
        contraindications=contraindications,
        retrieval_hints=retrieval_hints,
        # 旧字段
        topic=raw.get("topic", ""),
        metadata=metadata,
    )


def load_knowledge_base(path: Path) -> List[KnowledgeItem]:
    """加载知识库并转换为 KnowledgeItem 列表"""
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    items = [load_knowledge_item(raw) for raw in raw_data]
    logger.info(f"Loaded {len(items)} knowledge items from {path}")

    # 检查迁移状态
    if not is_migration_complete(raw_data):
        logger.warning("Knowledge base migration incomplete, using inferred values")

    return items


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python -m src.knowledge.knowledge_migrator <input_path> [output_path] [--dry-run]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    dry_run = "--dry-run" in sys.argv

    migrated, stats = migrate_knowledge_base(input_path, output_path, dry_run)

    print("\n=== Migration Summary ===")
    print(f"Total items: {stats['total']}")
    print("\nProduct line distribution:")
    for pl, count in stats["product_line"].items():
        print(f"  {pl}: {count}")
    print("\nRisk level distribution:")
    for rl, count in stats["risk_level"].items():
        print(f"  {rl}: {count}")
    print(f"\nWith prerequisites: {stats['with_prerequisites']}")
    print(f"With contraindications: {stats['with_contraindications']}")
    print(f"With question variants: {stats['with_variants']}")
