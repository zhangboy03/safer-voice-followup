"""转人工服务

提供转人工交接上下文的生成和管理。
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.pilot_guardrails import uses_callback_followup
from src.handoff.models import (
    PRIORITY_TRANSITIONS,
    REASON_DESCRIPTIONS,
    TransferPriority,
    TransferReason,
)

logger = logging.getLogger(__name__)


class TransferService:
    """转人工服务

    负责：
    1. 生成人工交接单（handoff context）
    2. 提供过渡话术
    3. 记录转接原因和优先级
    """

    def __init__(self):
        """初始化转人工服务"""
        pass

    def get_transition_message(
        self,
        reason: TransferReason,
        priority: TransferPriority = TransferPriority.NORMAL,
    ) -> str:
        """获取过渡话术

        当前内部试点不做 SIP live transfer。默认返回 callback-required
        话术，避免让模拟用户误以为系统正在实时转接人工。
        """
        if uses_callback_followup():
            return self.get_callback_required_message(reason, priority)

        # 优先使用优先级对应的话术
        base_message = PRIORITY_TRANSITIONS.get(priority, PRIORITY_TRANSITIONS[TransferPriority.NORMAL])

        # 安全风险添加安抚
        if reason == TransferReason.SAFETY_EMERGENCY:
            return "非常抱歉听到您这样说，请您不要担心，我立即为您转接专业人员，他们会帮助您。请稍等。"
        elif reason in (TransferReason.SAFETY_DEVICE, TransferReason.SAFETY_SYMPTOM):
            return f"这个问题非常重要，{base_message}"
        elif reason == TransferReason.EMOTIONAL_DISTRESS:
            return f"我理解您的感受，{base_message}"

        return base_message

    def get_callback_required_message(
        self,
        reason: TransferReason,
        priority: TransferPriority = TransferPriority.NORMAL,
    ) -> str:
        """Return pilot-safe wording for deferred human follow-up."""
        if priority == TransferPriority.URGENT or reason in (
            TransferReason.SAFETY_EMERGENCY,
            TransferReason.SAFETY_DEVICE,
            TransferReason.SAFETY_SYMPTOM,
            TransferReason.MEDICAL_COMPLEX,
            TransferReason.MEDICAL_ADJUSTMENT,
        ):
            return "这个情况需要人工同事进一步确认。我会先记录您的情况，本次通话先到这里，稍后会安排人工同事跟进。"
        if reason == TransferReason.USER_REQUEST:
            return "好的，我会记录您的人工沟通需求，本次通话先到这里，稍后会安排人工同事跟进。"
        if reason == TransferReason.EMOTIONAL_DISTRESS:
            return "我理解您的感受。我会记录下来，本次通话先到这里，稍后会安排人工同事跟进。"
        return "我会记录您的情况，本次通话先到这里，稍后会安排人工同事跟进。"

    def prepare_handoff_context(
        self,
        session_manager: Optional[Any],
        context: Optional[Any],
        reason: TransferReason,
        trigger_text: str = "",
        notes: str = "",
    ) -> str:
        """准备人工交接单

        Args:
            session_manager: SessionManager 实例（可选）
            context: SessionContext 实例（可选）
            reason: 转接原因
            trigger_text: 触发转接的用户输入
            notes: 额外备注

        Returns:
            格式化的交接单文本
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 有 SessionManager 时用完整信息
        if session_manager and context:
            patient = session_manager.patient
            memory = session_manager.memory
            sop_state = session_manager.sop_state

            # 获取最近对话
            recent_turns_text = self._format_recent_turns(memory.recent_turns)

            # 获取已采集数据
            collected_data_text = self._format_collected_data(sop_state.collected_data)

            # 获取会话摘要
            summary = memory.older_summary or "（无历史摘要）"

            return f"""=== 转人工交接单 ===
时间：{now}

【患者信息】
- 姓名：{patient.name}
- 患者ID：{patient.patient_id}
- 产品线：{patient.product_line}
- 设备型号：{patient.device_model or "未知"}
- 手术日期：{patient.surgery_date or "未知"}
- 主要症状：{patient.main_symptom or "未知"}

【转接原因】
{REASON_DESCRIPTIONS.get(reason, str(reason))}

【触发内容】
{trigger_text or "（无）"}

【会话摘要】
{summary}

【最近对话】
{recent_turns_text}

【已采集数据】
{collected_data_text}

【备注】
{notes or "无"}
"""

        # Fallback: 最小信息
        return f"""=== 转人工交接单 ===
时间：{now}

【转接原因】
{REASON_DESCRIPTIONS.get(reason, str(reason))}

【触发内容】
{trigger_text or "（无）"}

【备注】
会话上下文不可用，请人工核实患者信息。
{notes or ""}
"""

    def prepare_handoff_from_flow_state(
        self,
        flow_state: Any,
        reason: "TransferReason",
        trigger_text: str = "",
    ) -> str:
        """Prepare handoff context from FlowState (dual-LLM architecture)."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        patient_info = flow_state.patient_info if flow_state else {}
        patient_name = patient_info.get("name", "未知")
        patient_id = patient_info.get("patient_id", "未知")

        collected_lines = []
        for field_name, fv in flow_state.collected_fields.items():
            value = fv.value if hasattr(fv, "value") else fv
            collected_lines.append(f"- {field_name}: {value}")
        collected_text = "\n".join(collected_lines) if collected_lines else "（无）"

        return f"""=== 转人工交接单 ===
时间：{now}

【患者信息】
- 姓名：{patient_name}
- 患者ID：{patient_id}

【转接原因】
{REASON_DESCRIPTIONS.get(reason, str(reason))}

【触发内容】
{trigger_text or "（无）"}

【已采集数据】
{collected_text}
"""

    def _format_recent_turns(self, turns: List[Any]) -> str:
        """格式化最近对话

        Args:
            turns: Turn 列表

        Returns:
            格式化文本
        """
        if not turns:
            return "（无对话记录）"

        lines = []
        for turn in turns[-5:]:  # 最多5轮
            lines.append(f"用户：{turn.user_input}")
            lines.append(f"AI：{turn.ai_response}")
            lines.append("")  # 空行分隔
        return "\n".join(lines).strip()

    def _format_collected_data(self, collected_data: Dict[str, Any]) -> str:
        """格式化已采集数据

        Args:
            collected_data: 采集的数据字典

        Returns:
            格式化文本
        """
        if not collected_data:
            return "（无）"

        lines = []
        for field_name, data in collected_data.items():
            value = data.value if hasattr(data, "value") else data
            lines.append(f"- {field_name}: {value}")
        return "\n".join(lines)

    def create_transfer_info(
        self,
        reason: TransferReason,
        priority: TransferPriority,
        handoff_context: str,
        trigger_text: str = "",
    ) -> Dict[str, Any]:
        """创建转接信息（用于 SessionEndMessage）

        Args:
            reason: 转接原因
            priority: 优先级
            handoff_context: 交接单内容
            trigger_text: 触发内容

        Returns:
            转接信息字典
        """
        return {
            "reason": reason.value,
            "priority": priority.value,
            "reason_description": REASON_DESCRIPTIONS.get(reason, str(reason)),
            "trigger_text": trigger_text,
            "handoff_context": handoff_context,
            "timestamp": datetime.now().isoformat(),
        }


# 全局单例
_transfer_service: Optional[TransferService] = None


def get_transfer_service() -> TransferService:
    """获取转人工服务单例"""
    global _transfer_service
    if _transfer_service is None:
        _transfer_service = TransferService()
    return _transfer_service
