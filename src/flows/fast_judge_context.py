from __future__ import annotations

from src.flows.models import FieldStatus, FlowState, NodeConfig
from src.flows.sop_decision_models import FastJudgeSemanticContext, FieldSnapshot
from src.session.conversation_history import ConversationHistory


class FastJudgeContextBuilder:
    def build(
        self,
        *,
        flow_state: FlowState,
        node_config: NodeConfig,
        node_catalog: dict[str, NodeConfig] | None = None,
        user_text: str,
        speech_text: str,
        speech_result_id: str,
        conversation_history: ConversationHistory,
    ) -> FastJudgeSemanticContext:
        required = list(node_config.completion_policy.required_fields) if node_config.completion_policy else []
        allowed = node_config.allowed_fields or set()
        optional = sorted(allowed - set(required)) if allowed else []

        pending_required = (
            node_config.completion_policy.unsatisfied_fields(flow_state) if node_config.completion_policy else []
        )
        pending_optional = [
            field for field in optional if not self._is_optional_accepted(flow_state, node_config, field)
        ]
        pending_fields = pending_required + pending_optional

        known = {
            field: FieldSnapshot(
                value=value.value,
                status=value.status.value,
                source_node=value.source_node,
                source_turn=value.source_turn,
            )
            for field, value in sorted(flow_state.collected_fields.items())
        }
        candidates = sorted(
            flow_state.sop_progress_state.field_candidates,
            key=lambda item: (item.node_id, item.field, tuple(item.source_turn_ids)),
        )

        successor_path = self._successor_path(node_config, node_catalog or {})

        return FastJudgeSemanticContext(
            turn_id=flow_state.active_turn_id or "",
            turn_index=flow_state.turn_count,
            node_id=node_config.name,
            flow_epoch=flow_state.flow_epoch,
            speech_result_id=speech_result_id,
            speech_text=speech_text,
            user_text=user_text,
            speech_asked_this_turn=self._speech_asked(node_config, speech_text, pending_fields),
            intended_ask_fields=pending_fields,
            required_fields=required,
            eligible_optional_fields=optional,
            successor_path=successor_path,
            node_field_catalog=self._node_field_catalog(node_catalog or {}, successor_path),
            pending_fields_before_turn=pending_fields,
            known_fields_before_turn=known,
            candidate_fields_before_turn=candidates,
            recent_relevant_context=self._recent_context(conversation_history, node_config.name),
        )

    def _successor_path(self, node: NodeConfig, catalog: dict[str, NodeConfig]) -> list[str]:
        path = []
        current = node
        seen = {node.name}
        while current.successors:
            next_node = current.successors[0]
            if next_node in seen or next_node not in catalog:
                break
            path.append(next_node)
            seen.add(next_node)
            current = catalog[next_node]
        return path

    def _node_field_catalog(
        self,
        catalog: dict[str, NodeConfig],
        successor_path: list[str],
    ) -> dict[str, dict[str, str]]:
        return {
            node_id: dict(catalog[node_id].field_definitions)
            for node_id in successor_path
            if node_id in catalog and catalog[node_id].field_definitions
        }

    def _is_optional_accepted(self, state: FlowState, node: NodeConfig, field: str) -> bool:
        value = state.collected_fields.get(field)
        if value is None:
            return False
        if value.status in {FieldStatus.CONFIRMED, FieldStatus.INFERRED}:
            return True
        return bool(
            value.status == FieldStatus.REFUSED and node.completion_policy and node.completion_policy.accept_refused
        )

    def _speech_asked(self, node: NodeConfig, speech_text: str, fields: list[str]) -> str:
        parts = [
            node.speech_completion_hint,
            node.judge_task_description,
            f"intended_fields={','.join(fields)}" if fields else "",
            speech_text,
        ]
        return " | ".join(part for part in parts if part)[:1200]

    def _recent_context(self, history: ConversationHistory, node_id: str) -> str:
        journal = history.get_journal_entries()
        node_turns = {
            entry.turn_index
            for entry in journal
            if entry.node_id == node_id and isinstance(entry.message.get("content"), str)
        }
        node_messages = [
            entry.message
            for entry in journal
            if entry.turn_index in node_turns and isinstance(entry.message.get("content"), str)
        ]
        messages = node_messages[-4:] or history.get_summary_context(last_n=2)
        lines = [
            f"{message.get('role', '')}: {message.get('content', '')}"
            for message in messages
            if isinstance(message.get("content"), str)
        ]
        return "\n".join(lines)[-1200:]
