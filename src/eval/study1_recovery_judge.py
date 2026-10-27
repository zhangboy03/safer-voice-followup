from __future__ import annotations

import json
from typing import Any

from src.backends.structured_capabilities import build_response_format

VALID_RECOVERY_LABELS = {"success", "failure", "uncertain"}

RECOVERY_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "recovery_label": {"type": "string", "enum": ["success", "failure", "uncertain"]},
        "ignored_new_intent": {"type": "boolean"},
        "wrong_business_action": {"type": "boolean"},
        "evidence": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "recovery_label",
        "ignored_new_intent",
        "wrong_business_action",
        "evidence",
        "confidence",
    ],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """You are an evaluation judge for a Chinese medical follow-up voice agent.

Your task is to decide whether the agent successfully recovered after a user interrupted it while it was speaking.

Judge only the post-interruption recovery. Do not rate general politeness, empathy, medical correctness, or full SOP completion unless they are directly needed to decide whether the interrupted user intent was handled.

Important judging notes:
- interruption_asr_text comes from automatic speech recognition and may be truncated, partial, or misrecognized. Infer the user's likely intent from interruption_type together with the ASR text, and judge whether the agent's reply plausibly addresses that intent. Do not require the agent to echo the exact ASR words, and do not call a reply "unrelated" only because it mentions something the (possibly truncated) ASR text did not.
- observed_state_events and observed_audio_events are already scoped to AFTER the interruption. Set wrong_business_action only for an obviously wrong action the agent took in response to the interruption; do not infer it from routing or transitions that belong to the pre-interruption turn.
- Asking the user to repeat or to speak more clearly is acceptable (not a failure) ONLY when the interruption was genuinely unclear or garbled. If the user's intent is already clear, asking to repeat — or continuing the old SOP question instead of addressing the intent — is a failure.

Return strict JSON only."""

_USER_PROMPT_TEMPLATE = """Case metadata:
- case_id: {case_id}
- target_node: {target_node}
- interruption_type: {interruption_type}
- expected_dialogue_action: {expected_dialogue_action}
- expected_state_behavior: {expected_state_behavior}

Before interruption:
- agent_text_before_interrupt: {agent_text_before_interrupt}

User interruption:
- interruption_asr_text: {interruption_asr_text}

After interruption:
- agent_text_after_interrupt: {agent_text_after_interrupt}
- observed_state_events: {observed_state_events_json}
- observed_audio_events: {observed_audio_events_json}

Decision rules:
1. success:
   The agent stopped or abandoned the old response and the new user utterance clearly influenced the next reply or business decision.
   Examples:
   - stop_repeat: the agent repeats, slows down, apologizes for speaking too fast, or restarts the explanation.
   - correction_supplement: the agent acknowledges, reassures, or follows up on the new symptom/information, or preserves it in the decision context. Do not require the agent to repeat the exact medical detail or ask a new follow-up question. A brief natural acknowledgement such as "我记下来了" or a safe reassurance before returning to the prior scoring question can be success if it clearly responds to the correction/supplement.
   - further_inquiry: the agent addresses the new question, routes to knowledge handling, or explicitly says it will check/transfer.
   - topic_shift_defer: the agent acknowledges that the user wants to defer, is unavailable, or asks for callback/handoff instead of continuing the old question.

2. failure:
   The agent keeps answering the old topic, ignores the interruption, asks an unrelated question, asks the user to repeat even though the interruption intent was already clear, or takes a business action that contradicts the interruption.

3. uncertain:
   The evidence is insufficient, the transcript is missing, or the agent response is too generic to tell whether it handled the interruption.

Output JSON schema:
{{
  "recovery_label": "success | failure | uncertain",
  "ignored_new_intent": true,
  "wrong_business_action": false,
  "evidence": "One short Chinese sentence explaining the decisive evidence.",
  "confidence": 0.0
}}"""

_COMPACT_KEYS = (
    "event_type",
    "topic",
    "event",
    "turn_id",
    "text",
    "asr_text",
    "assistant_text",
    "response_text",
    "text_fed",
    "to_node",
    "node",
    "speech_directive",
    "tts_ttfa_ms",
    "emit_to_audio_ms",
)
_AUDIO_EVENT_TYPES = {
    "AsrFinalEvent",
    "TTSFirstAudioEvent",
    "TTSCompleteEvent",
    "TTSInterruptedEvent",
    "TurnMetricsEvent",
    "UserStartedSpeakingEvent",
    "UserStoppedSpeakingEvent",
}
_STATE_EVENT_TYPES = {
    "SopTransitionEvent",
    "TransferTriggeredEvent",
    "JudgeDecisionEvent",
    "RagDecisionEvent",
}


def build_recovery_judge_messages(
    *,
    case_id: str,
    target_node: str,
    interruption_type: str,
    expected_dialogue_action: str,
    expected_state_behavior: str,
    agent_text_before_interrupt: str,
    interruption_asr_text: str,
    agent_text_after_interrupt: str,
    observed_state_events: list[dict[str, Any]],
    observed_audio_events: list[dict[str, Any]],
) -> list[dict[str, str]]:
    user_prompt = _USER_PROMPT_TEMPLATE.format(
        case_id=case_id,
        target_node=target_node,
        interruption_type=interruption_type,
        expected_dialogue_action=expected_dialogue_action,
        expected_state_behavior=expected_state_behavior,
        agent_text_before_interrupt=agent_text_before_interrupt or "",
        interruption_asr_text=interruption_asr_text or "",
        agent_text_after_interrupt=agent_text_after_interrupt or "",
        observed_state_events_json=json.dumps(observed_state_events, ensure_ascii=False, separators=(",", ":")),
        observed_audio_events_json=json.dumps(observed_audio_events, ensure_ascii=False, separators=(",", ":")),
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def normalize_recovery_judge_result(parsed_json: Any, *, raw_text: str) -> dict[str, Any]:
    if not isinstance(parsed_json, dict):
        return {
            "judge_valid": False,
            "recovery_label": "uncertain",
            "ignored_new_intent": False,
            "wrong_business_action": False,
            "evidence": "judge did not return parseable JSON",
            "confidence": 0.0,
            "judge_raw_text": raw_text,
        }

    label = parsed_json.get("recovery_label")
    judge_valid = label in VALID_RECOVERY_LABELS
    confidence = _clamp_confidence(parsed_json.get("confidence"))
    evidence = parsed_json.get("evidence")
    return {
        "judge_valid": judge_valid,
        "recovery_label": label if judge_valid else "uncertain",
        "ignored_new_intent": bool(parsed_json.get("ignored_new_intent", False)),
        "wrong_business_action": bool(parsed_json.get("wrong_business_action", False)),
        "evidence": evidence if isinstance(evidence, str) else "",
        "confidence": confidence,
        "judge_raw_text": raw_text,
    }


def _interruption_index(events: list[dict[str, Any]], opener_turn_id: str | None) -> int:
    """打断在事件流里的起点：第一条 TTSInterruptedEvent 或第一条非 opener AsrFinal。

    用于把喂给 judge 的 observed events 限定在 *打断之后*，避免 opener turn 的
    RAG / SOP / advance 决策污染恢复判定（Pilot 中 correction_002 即因此被误判
    wrong_business_action）。
    """

    for i, event in enumerate(events):
        if event.get("event_type") == "TTSInterruptedEvent":
            return i
        if event.get("event_type") == "AsrFinalEvent" and event.get("turn_id") != opener_turn_id:
            text = event.get("text")
            if isinstance(text, str) and text.strip():
                return i
    return 0


def extract_recovery_context(events: list[dict[str, Any]], *, opener_turn_id: str | None) -> dict[str, Any]:
    agent_text_before = ""
    for event in events:
        if event.get("turn_id") == opener_turn_id:
            agent_text_before = _assistant_text(event) or agent_text_before

    interruption_idx = _interruption_index(events, opener_turn_id)

    # 打断常被 turn detection 切成多段；拼接所有非 opener 的 ASR final 还原完整意图，
    # 去掉相邻重复。(旧实现只取最后一条，丢失了完整诉求。)
    asr_parts: list[str] = []
    for event in events:
        if event.get("event_type") == "AsrFinalEvent" and event.get("turn_id") != opener_turn_id:
            text = event.get("text")
            if isinstance(text, str) and text.strip():
                stripped = text.strip()
                if not asr_parts or asr_parts[-1] != stripped:
                    asr_parts.append(stripped)
    interruption_asr_text = "".join(asr_parts)

    # 恢复回复：打断之后、非 opener turn 的最后一条 assistant 文本（最完整）。
    agent_text_after = ""
    for event in events[interruption_idx:]:
        if event.get("turn_id") == opener_turn_id:
            continue
        text = _assistant_text(event)
        if text:
            agent_text_after = text

    # observed events 只取打断之后，剔除 opener 的前置处理。
    compact_post = [_compact_event(e) for e in events[interruption_idx:]]
    return {
        "agent_text_before_interrupt": agent_text_before,
        "interruption_asr_text": interruption_asr_text,
        "agent_text_after_interrupt": agent_text_after,
        "observed_state_events": [e for e in compact_post if e.get("event_type") in _STATE_EVENT_TYPES],
        "observed_audio_events": [e for e in compact_post if e.get("event_type") in _AUDIO_EVENT_TYPES],
    }


async def run_recovery_judge(
    backend: Any,
    *,
    model: str,
    case_id: str,
    target_node: str,
    interruption_type: str,
    expected_dialogue_action: str,
    expected_state_behavior: str,
    agent_text_before_interrupt: str,
    interruption_asr_text: str,
    agent_text_after_interrupt: str,
    observed_state_events: list[dict[str, Any]],
    observed_audio_events: list[dict[str, Any]],
    session_id: str = "",
    turn_id: str = "",
) -> dict[str, Any]:
    messages = build_recovery_judge_messages(
        case_id=case_id,
        target_node=target_node,
        interruption_type=interruption_type,
        expected_dialogue_action=expected_dialogue_action,
        expected_state_behavior=expected_state_behavior,
        agent_text_before_interrupt=agent_text_before_interrupt,
        interruption_asr_text=interruption_asr_text,
        agent_text_after_interrupt=agent_text_after_interrupt,
        observed_state_events=observed_state_events,
        observed_audio_events=observed_audio_events,
    )
    completion = await backend.complete_structured(
        messages,
        response_format=build_response_format(
            model=model,
            schema_name="study1_barge_in_recovery",
            schema=RECOVERY_JUDGE_SCHEMA,
        ),
        max_tokens=220,
        temperature=0.0,
        session_id=session_id,
        turn_id=turn_id,
    )
    result = normalize_recovery_judge_result(completion.parsed_json, raw_text=completion.raw_text)
    result["judge_usage"] = getattr(completion, "usage", None)
    return result


def _assistant_text(event: dict[str, Any]) -> str:
    if event.get("topic") == "dual_llm_debug" and event.get("event") == "speech_done":
        text = event.get("response_text")
        return text.strip() if isinstance(text, str) else ""
    if event.get("event_type") == "TTSCompleteEvent":
        text = event.get("text_fed")
        return text.strip() if isinstance(text, str) else ""
    return ""


def _compact_event(event: dict[str, Any]) -> dict[str, Any]:
    return {key: event[key] for key in _COMPACT_KEYS if key in event}


def _clamp_confidence(value: Any) -> float:
    if not isinstance(value, (int, float)):
        return 0.0
    return max(0.0, min(1.0, float(value)))
