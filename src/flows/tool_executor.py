"""ToolExecutor — execute tool handlers with facade, validation, visibility."""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from src.core.logging import get_logger
from src.flows.models import (
    FieldStatus,
    FieldValue,
    FlowError,
    FlowState,
    InterruptPolicy,
    NodeConfig,
    NodeTransition,
    SideEffectLevel,
    ToolError,
    ToolOutcome,
    ToolResult,
    ToolSpec,
    TransitionKind,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

ToolHandler = Callable[[dict, "ToolContext"], Awaitable[ToolResult]]

# ---------------------------------------------------------------------------
# ToolContext — read-only facade over FlowState
# ---------------------------------------------------------------------------


class ToolContext:
    """Read-only facade that tool handlers receive instead of raw FlowState."""

    def __init__(
        self,
        flow_state: FlowState,
        current_node_config: NodeConfig,
        _field_writer: Callable[[str, Any, FieldStatus], Awaitable[None]],
        _return_resolver: Callable[[], str],
    ) -> None:
        self._state = flow_state
        self._node_config = current_node_config
        self._field_writer = _field_writer
        self._return_resolver = _return_resolver

    # -- read-only properties ------------------------------------------------

    @property
    def current_node(self) -> str:
        return self._state.current_node

    @property
    def patient_info(self) -> dict:
        return dict(self._state.patient_info)  # shallow copy

    @property
    def collected_fields(self) -> dict:
        return dict(self._state.collected_fields)  # shallow copy

    @property
    def node_turn_count(self) -> int:
        return self._state.node_turn_count

    # -- controlled writes ---------------------------------------------------

    async def update_field(self, field_name: str, value: Any, status: FieldStatus) -> None:
        """Write a field, validating against the node's allowed_fields."""
        if self._node_config.allowed_fields is not None:
            if field_name not in self._node_config.allowed_fields:
                raise ToolError(
                    f"field '{field_name}' not allowed in node "
                    f"'{self.current_node}'. "
                    f"Allowed: {self._node_config.allowed_fields}"
                )
        await self._field_writer(field_name, value, status)

    # -- navigation helpers --------------------------------------------------

    def resolve_return_target(self) -> str:
        return self._return_resolver()

    def get_successors(self) -> list[str]:
        return list(self._node_config.successors or [])


# ---------------------------------------------------------------------------
# PendingTool — async wrapper for background tool execution
# ---------------------------------------------------------------------------


@dataclass
class PendingTool:
    """Handle for a tool running as an asyncio.Task."""

    tool_call_id: str
    tool_name: str
    task: asyncio.Task[ToolOutcome]
    interrupt_policy: InterruptPolicy
    flow_epoch: int
    inference_id: str

    async def result(self) -> ToolOutcome:
        return await self.task

    def cancel(self) -> bool:
        if self.interrupt_policy == InterruptPolicy.CANCELLABLE:
            return self.task.cancel()
        return False


# ---------------------------------------------------------------------------
# BackgroundToolTracker
# ---------------------------------------------------------------------------


class BackgroundToolTracker:
    """Track pending background tools and drain them on demand."""

    def __init__(self) -> None:
        self._pending: list[PendingTool] = []

    def track(self, pending: PendingTool) -> None:
        self._pending.append(pending)

    async def drain(self, timeout_s: float = 5.0) -> list[ToolOutcome]:
        """Await all pending tools with a collective timeout. Returns outcomes."""
        if not self._pending:
            return []
        tasks = [p.task for p in self._pending]
        done, not_done = await asyncio.wait(tasks, timeout=timeout_s)
        # Cancel stragglers
        for t in not_done:
            t.cancel()
        results: list[ToolOutcome] = []
        for t in done:
            try:
                results.append(t.result())
            except Exception:
                pass
        self._pending.clear()
        return results


# ---------------------------------------------------------------------------
# ToolExecutor
# ---------------------------------------------------------------------------


class ToolExecutor:
    """Registry + executor for tool handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, ToolHandler] = {}

    # -- registration --------------------------------------------------------

    def register(self, handler_id: str, handler: ToolHandler) -> None:
        if handler_id in self._handlers:
            raise FlowError(f"handler '{handler_id}' already registered")
        self._handlers[handler_id] = handler

    def register_many(self, handlers: dict[str, ToolHandler]) -> None:
        for hid, handler in handlers.items():
            self.register(hid, handler)

    # -- execution -----------------------------------------------------------

    async def execute(
        self,
        tool_spec: ToolSpec,
        arguments: dict,
        flow_state: FlowState,
        node_config: NodeConfig,
    ) -> ToolOutcome:
        """Execute a tool synchronously (awaited). Returns ToolOutcome."""
        handler = self._handlers.get(tool_spec.handler_id)
        if handler is None:
            return ToolOutcome(
                tool_payload=None,
                result_text=f"No handler registered for '{tool_spec.handler_id}'",
                transition=None,
                side_effect_level=SideEffectLevel.NONE,
            )

        ctx = self._build_context(flow_state, node_config)

        try:
            if tool_spec.timeout_s is not None:
                result = await asyncio.wait_for(handler(arguments, ctx), timeout=tool_spec.timeout_s)
            else:
                result = await handler(arguments, ctx)
        except ToolError as exc:
            return ToolOutcome(
                tool_payload=None,
                result_text=exc.message,
                transition=None,
                side_effect_level=tool_spec.side_effect,
            )
        except asyncio.CancelledError:
            return ToolOutcome(
                tool_payload=None,
                result_text="Tool execution cancelled",
                transition=None,
                side_effect_level=tool_spec.side_effect,
            )
        except TimeoutError:
            return ToolOutcome(
                tool_payload=None,
                result_text=f"Timeout after {tool_spec.timeout_s}s",
                transition=None,
                side_effect_level=tool_spec.side_effect,
            )
        except Exception as exc:
            return ToolOutcome(
                tool_payload=None,
                result_text=f"Tool failed: {exc}",
                transition=None,
                side_effect_level=tool_spec.side_effect,
            )

        return self._wrap_outcome(result, tool_spec, node_config)

    def execute_async(
        self,
        tool_spec: ToolSpec,
        arguments: dict,
        flow_state: FlowState,
        node_config: NodeConfig,
        flow_epoch: int,
        inference_id: str,
    ) -> PendingTool:
        """Launch tool execution as a background task. Returns PendingTool."""
        task = asyncio.create_task(self.execute(tool_spec, arguments, flow_state, node_config))
        return PendingTool(
            tool_call_id=str(uuid.uuid4()),
            tool_name=tool_spec.name,
            task=task,
            interrupt_policy=tool_spec.interrupt_policy,
            flow_epoch=flow_epoch,
            inference_id=inference_id,
        )

    # -- visibility filtering ------------------------------------------------

    @staticmethod
    def filter_visible_tools(tools: dict[str, ToolSpec], current_node: str) -> dict[str, ToolSpec]:
        """Filter tools by visible_in_nodes / hidden_in_nodes."""
        result: dict[str, ToolSpec] = {}
        for tool_id, spec in tools.items():
            # If hidden_in_nodes is set and current node is in it, skip.
            if spec.hidden_in_nodes and current_node in spec.hidden_in_nodes:
                continue
            # If visible_in_nodes is set, current node must be in it.
            if spec.visible_in_nodes and current_node not in spec.visible_in_nodes:
                continue
            result[tool_id] = spec
        return result

    # -- internal helpers ----------------------------------------------------

    def _build_context(self, flow_state: FlowState, node_config: NodeConfig) -> ToolContext:
        async def field_writer(field_name: str, value: Any, status: FieldStatus) -> None:
            flow_state.collected_fields[field_name] = FieldValue(
                value=value,
                status=status,
                source_node=flow_state.current_node,
                source_turn=flow_state.turn_count,
            )

        def return_resolver() -> str:
            raise ToolError("resolve_return_target not supported in v1")

        return ToolContext(flow_state, node_config, field_writer, return_resolver)

    @staticmethod
    def _wrap_outcome(result: ToolResult, spec: ToolSpec, node_config: NodeConfig) -> ToolOutcome:
        # Serialize output
        if result.output is None:
            result_text = ""
        elif isinstance(result.output, dict):
            result_text = json.dumps(result.output, ensure_ascii=False)
        else:
            result_text = str(result.output)

        # Build transition if requested
        transition: NodeTransition | None = None
        if result.transition_to is not None:
            kind = result.transition_kind or TransitionKind.ADVANCE
            # Validate ADVANCE targets against successors
            if kind == TransitionKind.ADVANCE:
                successors = node_config.successors or []
                if result.transition_to not in successors:
                    raise FlowError(
                        f"ADVANCE to '{result.transition_to}' invalid: "
                        f"not in successors {successors} of node "
                        f"'{node_config.name}'"
                    )
            transition = NodeTransition(
                target_node=result.transition_to,
                kind=kind,
                reason=result.transition_reason or "",
            )

        return ToolOutcome(
            tool_payload=result.output,
            result_text=result_text,
            transition=transition,
            side_effect_level=spec.side_effect,
        )
