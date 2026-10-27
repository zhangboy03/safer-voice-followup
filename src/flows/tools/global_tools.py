"""Global tools available in all nodes."""

from __future__ import annotations

from src.flows.models import ToolResult, TransitionKind


async def transfer_to_human(args: dict, ctx) -> ToolResult:
    """Patient requests human agent — callback model: notify internal CS queue
    and tell the user a human will call them back. Call ends after this
    announcement (TERMINAL transition)."""
    return ToolResult(
        output="好的，我已经把您的情况记录下来了，稍后会有专业的人工客服给您回电话。",
        transition_to="human_transfer",
        transition_kind=TransitionKind.TERMINAL,
        transition_reason="patient_requested",
    )
