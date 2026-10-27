"""热词加载模块"""

import os


def load_hotwords(filepath: str = "config/hotwords.txt") -> str:
    """
    加载热词配置文件，返回逗号分隔的热词字符串

    Args:
        filepath: 热词文件路径

    Returns:
        逗号分隔的热词字符串，如 "品驰:5,DBS:5,..."
    """
    if not os.path.exists(filepath):
        return ""

    hotwords = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 跳过空行和注释
            if not line or line.startswith("#"):
                continue
            hotwords.append(line)

    return ",".join(hotwords)


def parse_hotword(hotword_str: str) -> tuple[str, int]:
    """
    解析单个热词字符串

    Args:
        hotword_str: 如 "品驰:5"

    Returns:
        (word, weight) 元组
    """
    if ":" in hotword_str:
        parts = hotword_str.rsplit(":", 1)
        return parts[0], int(parts[1])
    return hotword_str, 5  # 默认权重


def get_hotwords_list(filepath: str = "config/hotwords.txt") -> list[tuple[str, int]]:
    """
    获取热词列表

    Returns:
        [(word, weight), ...] 列表
    """
    hotwords_str = load_hotwords(filepath)
    if not hotwords_str:
        return []

    result = []
    for item in hotwords_str.split(","):
        if item.strip():
            result.append(parse_hotword(item.strip()))
    return result
