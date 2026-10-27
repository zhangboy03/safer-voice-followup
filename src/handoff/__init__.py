"""转人工层 (Handoff Layer)

- PatientProfile: 患者档案
- load_patient_profile: 患者档案加载
- TransferService: 转人工服务
"""

from .models import (
    PRIORITY_TRANSITIONS,
    REASON_DESCRIPTIONS,
    CallType,
    # 患者档案
    PatientProfile,
    TransferContext,
    TransferPriority,
    # 转人工
    TransferReason,
    # 对话轮次
    Turn,
)
from .patient_loader import load_patient_profile
from .transfer_service import (
    TransferService,
    get_transfer_service,
)

__all__ = [
    # 患者档案
    "PatientProfile",
    "CallType",
    "load_patient_profile",
    # 对话轮次
    "Turn",
    # 转人工
    "TransferReason",
    "TransferPriority",
    "TransferContext",
    "TransferService",
    "get_transfer_service",
    "REASON_DESCRIPTIONS",
    "PRIORITY_TRANSITIONS",
]
