"""跨通话记忆数据模型

MemoryItem: 低风险个性化信息（7个类别）
CallSummary: 通话摘要（SOP 完成状态、结局）
PatientMemoryContext: 会话开始时装载的完整记忆上下文
PostCallSnapshot: 通话结束时的不可变快照（异步提取使用）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryCategory(str, Enum):
    """MemoryItem 专用类别 — 仅低风险个性化信息"""

    COMMUNICATION_PREFERENCE = "communication_preference"  # 沟通偏好
    ADDRESS_PREFERENCE = "address_preference"  # 称呼偏好
    RECOVERY_MILESTONE = "recovery_milestone"  # 康复里程碑
    COMPLIANCE_PATTERN = "compliance_pattern"  # 配合度模式
    PATIENT_MOTIVATION = "patient_motivation"  # 动机/目标
    PATIENT_CONCERN = "patient_concern"  # 关切/担忧
    INTERACTION_QUALITY = "interaction_quality"  # 交互质量信号


@dataclass
class MemoryItem:
    """单条个性化记忆"""

    memory_id: str  # UUID
    patient_id: str
    category: MemoryCategory
    content: str  # 自然语言描述（第三人称）
    source_session_id: str
    source_call_date: str  # ISO date
    months_since_surgery: int
    confidence: float = 1.0
    is_active: bool = True  # 软删除/被新记忆替代
    created_at: str = ""
    updated_at: str = ""


@dataclass
class CallSummary:
    """通话摘要 — SOP 完成状态和结局"""

    call_id: str  # = session_id
    patient_id: str
    call_date: str
    duration_seconds: int = 0
    turn_count: int = 0
    sop_fields_collected: dict[str, Any] = field(default_factory=dict)
    sop_groups_completed: list[str] = field(default_factory=list)
    outcome: str = ""  # completed / partial / transferred / transferred_safety / disconnected
    transcript_summary: str = ""
    created_at: str = ""


@dataclass
class PatientMemoryContext:
    """会话开始时装载的完整记忆上下文"""

    patient_id: str
    previous_calls: list[CallSummary] = field(default_factory=list)
    active_memories: list[MemoryItem] = field(default_factory=list)
    last_call_date: str | None = None
    total_call_count: int = 0


@dataclass(frozen=True)
class PostCallSnapshot:
    """通话结束时的不可变快照 — 在清理前同步构造

    解决异步任务执行时状态已被清空的问题。
    """

    session_id: str
    patient_id: str
    patient_name: str
    product_line: str
    months_since_surgery: int
    transcript: tuple[dict[str, Any], ...]  # messages 的深拷贝（frozen 需要 tuple）
    plus_analysis: Any | None  # PlusAnalysis 的深拷贝
    turn_count: int
    outcome: str  # completed / transferred / transferred_safety / disconnected
    transfer_reason: str | None = None
    device_model: str = ""
    session_metrics: dict[str, Any] | None = None  # SessionMetrics.get_summary()
    collected_fields: dict[str, Any] | None = None  # SOP fields snapshot
