"""感知层数据模型定义

定义感知层的输出契约：SpeechEvent
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SpeechEventType(str, Enum):
    """语音事件类型"""

    PARTIAL = "partial"  # 中间识别结果
    FINAL = "final"  # 最终识别结果
    TURN_END = "turn_end"  # 话轮结束
    INTERRUPT = "interrupt"  # 用户打断


class TurnEndReason(str, Enum):
    """话轮结束原因"""

    SILENCE = "silence"  # 静音超时
    MAX_DURATION = "max_duration"  # 最大时长
    INTERRUPT = "interrupt"  # 被打断


@dataclass
class SpeechEvent:
    """感知层输出的结构化事件

    这是感知层向上层传递的统一数据契约。
    无论底层 ASR、VAD、话轮检测如何实现，
    上层只需处理 SpeechEvent。
    """

    # 事件类型
    event_type: SpeechEventType

    # ASR 识别文本
    text: str

    # ASR 是否确认（final）
    is_final: bool

    # 话轮结束原因（仅当 event_type=TURN_END 时有值）
    turn_end_reason: Optional[TurnEndReason] = None

    # ASR 置信度 [0, 1]，None = 未知/无数据
    confidence: Optional[float] = None

    # 时间戳（monotonic）
    timestamp: float = 0.0

    # 轮次 ID
    turn_id: str = ""

    def is_turn_ended(self) -> bool:
        """是否为话轮结束事件"""
        return self.event_type == SpeechEventType.TURN_END

    def is_interrupt(self) -> bool:
        """是否为打断事件"""
        return self.event_type == SpeechEventType.INTERRUPT
