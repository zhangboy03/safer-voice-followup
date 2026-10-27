"""Tools available in companion (emotional support) mode."""

from __future__ import annotations

from src.flows.models import ToolError, ToolResult, TransitionKind


async def return_to_sop(args: dict, ctx) -> ToolResult:
    """Return from companion mode to suspended SOP node."""
    target = ctx.resolve_return_target()
    return ToolResult(
        output=None,  # silent -- resumed node generates response
        transition_to=target,
        transition_kind=TransitionKind.RESUME,
        transition_reason="companion_complete",
    )


async def query_knowledge_base(args: dict, ctx) -> ToolResult:
    """Query knowledge base for patient questions. Placeholder for RAG integration."""
    question = args.get("question", "")
    if not question:
        raise ToolError("question parameter is required")
    # TODO: integrate with RAG service in production
    return ToolResult(output=f"[知识库查询] 问题: {question} — 暂无结果，建议咨询主治医生。")
