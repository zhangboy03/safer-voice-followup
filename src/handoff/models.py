"""编排层数据模型定义

保留患者档案、转人工相关模型。
SOP 状态管理已迁移到 LLM 中心架构 (src/conversation/)。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from src.flows.knowledge_review_models import CallbackPriority  # noqa: F401 — re-exported for callers


class CallType(str, Enum):
    """通话类型"""

    OUTBOUND_FOLLOWUP = "outbound_followup"  # AI 主动外呼随访
    INBOUND_CONSULT = "inbound_consult"  # 患者来电咨询


class TransferReason(str, Enum):
    """转人工原因"""

    USER_REQUEST = "user_request"  # 用户主动请求
    SAFETY_EMERGENCY = "safety_emergency"  # 紧急安全风险
    SAFETY_DEVICE = "safety_device"  # 设备安全风险
    SAFETY_SYMPTOM = "safety_symptom"  # 严重症状
    RAG_LOW_CONFIDENCE = "rag_low_confidence"  # RAG 置信度过低
    RAG_NO_ANSWER = "rag_no_answer"  # RAG 无法找到答案
    MEDICAL_COMPLEX = "medical_complex"  # 复杂医疗问题
    MEDICAL_ADJUSTMENT = "medical_adjustment"  # 参数调整需求
    EMOTIONAL_DISTRESS = "emotional_distress"  # 情绪问题
    BUSINESS_COMPLAINT = "business_complaint"  # 投诉
    BUSINESS_AFTER_SALES = "business_after_sales"  # 售后服务
    OTHER = "other"


class TransferPriority(str, Enum):
    """转接优先级"""

    URGENT = "urgent"  # 紧急（安全风险），需立即转接
    HIGH = "high"  # 高优先级（情绪问题），尽快转接
    NORMAL = "normal"  # 普通优先级


@dataclass
class Turn:
    """单轮对话"""

    turn_id: str
    user_input: str
    ai_response: str
    intent_type: str = ""
    timestamp: float = 0.0


@dataclass
class PatientProfile:
    """患者档案"""

    patient_id: str
    name: str
    product_line: str  # DBS/VNS/SCS/SNM
    device_model: Optional[str] = None
    surgery_date: Optional[str] = None
    hospital: Optional[str] = None
    main_symptom: Optional[str] = None


@dataclass
class TransferContext:
    """转接上下文"""

    reason: TransferReason
    priority: TransferPriority = TransferPriority.NORMAL
    trigger_text: str = ""
    session_id: str = ""
    patient_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # 会话信息
    conversation_summary: str = ""
    recent_turns: List[Dict[str, str]] = field(default_factory=list)
    collected_data: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass(frozen=True)
class CallbackRecord:
    call_id: str
    session_id: str
    patient_id: str
    reason: str  # serialized RefuseKind / EscalationKind value
    out_of_scope_class: Optional[str]
    priority: str  # CallbackPriority value (.value)
    context_text: str  # full handoff packet
    user_text: str
    timestamp: str
    idempotency_key: str  # f"{call_id}:{session_id}"

    def to_json(self) -> str:
        import json as _json
        from dataclasses import asdict

        return _json.dumps(asdict(self), ensure_ascii=False)


# 原因描述映射（用于交接单）
REASON_DESCRIPTIONS = {
    TransferReason.USER_REQUEST: "用户主动要求转人工",
    TransferReason.SAFETY_EMERGENCY: "紧急安全风险 - 需要立即人工介入",
    TransferReason.SAFETY_DEVICE: "设备安全问题",
    TransferReason.SAFETY_SYMPTOM: "严重症状报告",
    TransferReason.RAG_LOW_CONFIDENCE: "AI 无法确定答案，需要人工确认",
    TransferReason.RAG_NO_ANSWER: "知识库中未找到相关答案",
    TransferReason.MEDICAL_COMPLEX: "复杂医疗问题，需要专业人员解答",
    TransferReason.MEDICAL_ADJUSTMENT: "涉及设备参数调整",
    TransferReason.EMOTIONAL_DISTRESS: "用户情绪需要人工安抚",
    TransferReason.BUSINESS_COMPLAINT: "用户投诉",
    TransferReason.BUSINESS_AFTER_SALES: "售后服务需求",
    TransferReason.OTHER: "其他原因",
}


# 优先级对应的过渡话术
PRIORITY_TRANSITIONS = {
    TransferPriority.URGENT: "我立即为您转接人工客服，请稍等。",
    TransferPriority.HIGH: "我这就帮您转接人工客服，请稍等片刻。",
    TransferPriority.NORMAL: "好的，我帮您转接人工客服，请稍等。",
}
