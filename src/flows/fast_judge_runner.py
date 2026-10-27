from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import ValidationError

from src.backends.structured_capabilities import build_response_format
from src.core.config import settings
from src.core.logging import get_logger
from src.flows.question_predicate import speech_asked_question
from src.flows.sop_decision_models import (
    FastJudgeAnswerAssessment,
    FastJudgeEnvelope,
    FastJudgeFieldDelta,
    FastJudgeInternalIdentity,
    FastJudgeKnowledgeQuestion,
    FastJudgeLookaheadFieldDelta,
    FastJudgeModelOutput,
    FastJudgeResult,
    FastJudgeSpeechDirective,
)

logger = get_logger(__name__)


class FastJudgeFailure(RuntimeError):
    pass


def hash_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


_FAST_JUDGE_STRING_DEFAULT_FIELDS = {
    "node_after_candidate",
    "next_node_goal",
    "stage2_target_node",
    "stage2_reason",
}
_FAST_JUDGE_LIST_DEFAULT_FIELDS = {
    "field_delta",
    "lookahead_field_delta",
    "missing_fields",
}
_ANSWER_ASSESSMENT_VALUES = {
    "answered",
    "partially_answered",
    "not_answered",
    "refused",
    "needs_reconfirmation",
    "unclear",
    "not_applicable",
}
_HANDOFF_REASONS = {"none", "user_request", "emotional_escalation", "medical_risk", "other"}
_HANDOFF_RISK_LEVELS = {"normal", "elevated", "urgent"}
_TRUE_TEXT_VALUES = {
    "true",
    "1",
    "yes",
    "y",
    "是",
    "是的",
    "本人",
    "患者本人",
    "confirmed",
    "同意",
    "方便",
    "愿意",
    "可以",
    "行",
}
_FALSE_TEXT_VALUES = {
    "false",
    "0",
    "no",
    "n",
    "否",
    "不是",
    "非本人",
    "refused",
    "拒绝",
    "不方便",
    "不愿意",
}
_CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
_CONTROL_SCORE_EXPLICIT_REFUSAL_MARKERS = (
    "不想打分",
    "不愿打分",
    "不打分",
    "拒绝打分",
    "不想评分",
    "不愿评分",
    "拒绝评分",
    "不想给分",
    "不愿给分",
    "不想说分",
    "分数不说",
    "评分不说",
    "不想说",
    "不愿说",
    "不说",
    "不回答",
    "拒绝回答",
    "不告诉",
    "别问分",
    "别问评分",
    "不要问分",
    "不要问评分",
)
_CONTROL_SCORE_COMPLAINT_MARKERS = (
    "还问",
    "问我这些",
    "问这些",
    "这些问题",
    "问题干嘛",
    "问题干什么",
    "都说了",
    "都告诉",
    "烦",
    "啰嗦",
    "罗嗦",
    "麻烦",
)


def _raw_model_text(raw_text: str, parsed_json: Any) -> str:
    raw = (raw_text or "").strip()
    if raw and raw != "{}":
        return raw_text
    if parsed_json is None:
        return raw_text or ""
    try:
        return json.dumps(parsed_json, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return raw_text or ""


def _strip_model_fields(value: dict[str, Any], *, allowed_fields: set[str]) -> dict[str, Any]:
    return {key: field_value for key, field_value in value.items() if key in allowed_fields and field_value is not None}


def _normalize_answer_assessment(value: Any) -> Any:
    if isinstance(value, dict):
        data = _strip_model_fields(value, allowed_fields=set(FastJudgeAnswerAssessment.model_fields))
        if data.get("current_question_answered") not in _ANSWER_ASSESSMENT_VALUES:
            data.pop("current_question_answered", None)
        return data
    if isinstance(value, str):
        if value in _ANSWER_ASSESSMENT_VALUES:
            return {"current_question_answered": value}
        return {"followup_reason": value}
    if value is None:
        return {}
    return value


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_TEXT_VALUES:
            return True
        if normalized in _FALSE_TEXT_VALUES:
            return False
    return None


def _normalize_handoff(value: Any, *, node_action: Any) -> Any:
    if isinstance(value, dict):
        transfer = value.get("transfer_to_human", value.get("triggered", value.get("required")))
        transfer_bool = _coerce_bool(transfer)
        if node_action == "handoff" and transfer_bool is not False:
            transfer_bool = True

        reason = value.get("reason")
        if reason not in _HANDOFF_REASONS:
            reason = "other" if transfer_bool or node_action == "handoff" else "none"

        risk_level = value.get("risk_level")
        if risk_level not in _HANDOFF_RISK_LEVELS:
            risk_level = "normal"

        data: dict[str, Any] = {"reason": reason, "risk_level": risk_level}
        if transfer_bool is not None:
            data["transfer_to_human"] = transfer_bool
        return data
    if value is True:
        return {"transfer_to_human": True, "reason": "other"}
    if value is None or value is False:
        if node_action == "handoff":
            return {"transfer_to_human": True, "reason": "other"}
        return {}
    return value


def _normalize_knowledge_question(value: Any, *, node_action: Any) -> Any:
    if isinstance(value, dict):
        data = _strip_model_fields(value, allowed_fields=set(FastJudgeKnowledgeQuestion.model_fields))
        detected = data.get("detected")
        detected_bool = _coerce_bool(detected)
        if detected_bool is not None:
            data["detected"] = detected_bool
        elif node_action == "divert" or data.get("query"):
            data["detected"] = True
        else:
            data.pop("detected", None)
        return data
    if value is True:
        return {"detected": True}
    if value is None or value is False:
        if node_action == "divert":
            return {"detected": True}
        return {}
    return value


def _normalize_speech_directive(value: Any) -> Any:
    if isinstance(value, dict):
        data = _strip_model_fields(value, allowed_fields=set(FastJudgeSpeechDirective.model_fields))
        for key in ("code", "text"):
            if data.get(key) is None:
                data[key] = ""
        return data
    if isinstance(value, str):
        return {"text": value}
    if value is None or value is False:
        return {}
    return value


def _parse_control_score(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return int(value) if 0 <= value <= 10 and float(value).is_integer() else value
    if not isinstance(value, str):
        return value
    normalized = value.strip()
    if normalized.isdigit():
        score = int(normalized)
        return score if 0 <= score <= 10 else value
    for char, digit in _CHINESE_DIGITS.items():
        if char in normalized:
            return digit
    return value


def _normalize_field_value(field: Any, value: Any) -> Any:
    if field == "call_accepted":
        accepted = _coerce_bool(value)
        if accepted is True:
            return "confirmed"
        if accepted is False:
            return "refused"
        return value
    if field == "is_patient":
        patient = _coerce_bool(value)
        return patient if patient is not None else value
    if field == "control_score":
        return _parse_control_score(value)
    return value


def _normalize_field_delta_items(value: Any, *, lookahead: bool = False) -> Any:
    if not isinstance(value, list):
        return value

    allowed_fields = set(FastJudgeLookaheadFieldDelta.model_fields if lookahead else FastJudgeFieldDelta.model_fields)
    normalized = []
    for item in value:
        if not isinstance(item, dict):
            normalized.append(item)
            continue
        data = _strip_model_fields(item, allowed_fields=allowed_fields)
        if "field" in data and "value" in data:
            data["value"] = _normalize_field_value(data["field"], data["value"])
        normalized.append(data)
    return normalized


def _is_nonexplicit_control_score_complaint(item: dict[str, Any]) -> bool:
    if item.get("field") != "control_score" or item.get("status") != "refused":
        return False
    evidence = f"{item.get('evidence_span', '')} {item.get('value', '')}".strip()
    if not evidence:
        return False
    if any(marker in evidence for marker in _CONTROL_SCORE_EXPLICIT_REFUSAL_MARKERS):
        return False
    return any(marker in evidence for marker in _CONTROL_SCORE_COMPLAINT_MARKERS)


def _downgrade_spurious_control_score_refusal(data: dict[str, Any], *, identity: FastJudgeInternalIdentity) -> str:
    if identity.node_id != "symptom_rating" or not isinstance(data.get("field_delta"), list):
        return ""

    kept = []
    downgraded = False
    for item in data["field_delta"]:
        if isinstance(item, dict) and _is_nonexplicit_control_score_complaint(item):
            downgraded = True
            continue
        kept.append(item)
    if not downgraded:
        return ""

    data["field_delta"] = kept
    missing = _normalize_field_name_list(data.get("missing_fields", []))
    if isinstance(missing, list) and "control_score" not in missing:
        missing.append("control_score")
    data["missing_fields"] = missing

    if data.get("node_action") == "advance":
        data["node_action"] = "followup"
        data["stage_complete"] = False
        data["stage2_should_start"] = False
        data["stage2_target_node"] = ""
        data["stage2_reason"] = ""

    code = _directive_code(data)
    text = _directive_text(data)
    if code.startswith("advance") or "推进" in text or "拒答" in text:
        data["speech_directive"] = {
            "code": "ask_control_score",
            "text": "先安抚用户对反复追问的烦躁，说明评分用于入档量化记录，再自然追问0到10分；禁止推进下一节点。",
        }
    return "用户是在抱怨追问，不是明确拒绝评分；继续安抚并追问 control_score"


def _normalize_field_name_list(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    normalized = []
    for item in value:
        if isinstance(item, str):
            normalized.append(item)
        elif isinstance(item, dict) and isinstance(item.get("field"), str):
            normalized.append(item["field"])
        else:
            normalized.append(item)
    return normalized


def _split_model_field_delta_items(
    value: Any,
    *,
    current_node_id: str,
) -> tuple[Any, Any]:
    if not isinstance(value, list):
        return value, []
    current = []
    lookahead = []
    for item in value:
        if not isinstance(item, dict):
            current.append(item)
            continue
        node_id = item.get("node_id")
        if isinstance(node_id, str) and node_id and node_id != current_node_id:
            lookahead.append(item)
            continue
        current.append({key: field_value for key, field_value in item.items() if key != "node_id"})
    return current, lookahead


def _directive_code(data: dict[str, Any]) -> str:
    directive = data.get("speech_directive")
    if isinstance(directive, dict) and isinstance(directive.get("code"), str):
        return directive["code"]
    return ""


def _directive_text(data: dict[str, Any]) -> str:
    directive = data.get("speech_directive")
    if isinstance(directive, dict) and isinstance(directive.get("text"), str):
        return directive["text"]
    if isinstance(directive, str):
        return directive
    return ""


def _derive_answer_assessment(data: dict[str, Any]) -> dict[str, Any]:
    action = data.get("node_action")
    answered_by_action = {
        "advance": "answered",
        "terminal": "refused",
        "followup": "not_answered",
        "repair": "unclear",
        "stay": "unclear",
        "divert": "not_applicable",
        "handoff": "not_applicable",
    }
    return {
        "current_question_answered": answered_by_action.get(action, "unclear"),
        "needs_followup": action in {"followup", "repair"},
        "followup_reason": _directive_text(data),
        "missing_fields": data.get("missing_fields", []),
    }


def _derive_handoff(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("node_action") != "handoff":
        return {}
    code = _directive_code(data).lower()
    reason = "other"
    risk_level = "normal"
    if "medical" in code or "risk" in code or "urgent" in code or "风险" in _directive_text(data):
        reason = "medical_risk"
        risk_level = "urgent"
    elif "emotional" in code or "情绪" in _directive_text(data):
        reason = "emotional_escalation"
        risk_level = "elevated"
    elif "user_request" in code or "confirm_handoff" in code or "人工" in _directive_text(data):
        reason = "user_request"
    return {"transfer_to_human": True, "reason": reason, "risk_level": risk_level}


def _derive_knowledge_question(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("node_action") != "divert":
        return {}
    return {"detected": True, "query": _directive_text(data)}


def _block_advance_over_open_question(
    data: dict[str, Any], *, stage1_text: str, identity: FastJudgeInternalIdentity
) -> str:
    """Fail-closed guard (spec Part C1): if the Judge advanced but Stage 1's last
    line is a question, downgrade to followup and strip current-node
    confirmed/inferred field commits so a contradicted field never reaches the
    SOP record. Emits the structured safety event `advance_blocked_open_question`.
    Returns a short reason string when it fires (for the v2 followup_reason), else "".

    No bypass: even a Judge-declared already_asked_next_node does not exempt it
    (the marker is self-attested and unreliable on the very turn the Judge erred).
    """
    if not settings.judge_advance_question_guard_enabled:
        return ""
    if data.get("node_action") != "advance":
        return ""
    if not speech_asked_question(stage1_text, interrogatives=True):
        return ""

    # Capture the (unreliable) jump claim BEFORE clearing, for telemetry only.
    claimed_next_node = (
        data.get("stage2_reason") == "already_asked_next_node" or _directive_code(data) == "already_asked_next_node"
    )

    data["node_action"] = "followup"
    data["stage_complete"] = False
    data["stage2_should_start"] = False
    data["stage2_target_node"] = ""
    data["stage2_reason"] = ""

    # Strip current-node confirmed/inferred field_delta (here field_delta is already
    # current-node only — lookahead was split out earlier). Keep refused (explicit
    # user statement). Never fabricate a FieldStatus.
    field_delta = data.get("field_delta")
    stripped: list[str] = []
    if isinstance(field_delta, list):
        kept = []
        for item in field_delta:
            if isinstance(item, dict) and item.get("status") in {"confirmed", "inferred"}:
                fname = item.get("field")
                if fname:
                    stripped.append(fname)
                continue
            kept.append(item)
        data["field_delta"] = kept
    missing = data.get("missing_fields")
    if not isinstance(missing, list):
        missing = []
    for fname in stripped:
        if fname not in missing:
            missing.append(fname)
    data["missing_fields"] = missing

    data["speech_directive"] = {
        "code": "await_stage1_answer",
        "text": "Stage 1 刚问的问题还没得到回答；继续等用户回答，不要推进下一节点、不要复述同一问题。",
    }
    logger.info(
        "fast_judge.advance_blocked_open_question",
        turn_id=identity.turn_id,
        node=identity.node_id,
        stripped_fields=stripped,
        claimed_already_asked_next_node=claimed_next_node,
    )
    return "advance_blocked_open_question: Stage 1 仍在问问题，降级 followup 等回答"


def _prepare_fast_judge_payload(payload: Any, *, identity: FastJudgeInternalIdentity, stage1_text: str = "") -> Any:
    if not isinstance(payload, dict):
        return payload

    data = {key: value for key, value in payload.items() if key in FastJudgeResult.model_fields}
    data.setdefault("schema_version", "fast_judge.v1")
    data["turn_id"] = identity.turn_id
    data["node_id"] = identity.node_id
    data["speech_result_id"] = identity.speech_result_id
    data["evaluated_until"] = "speech_text_done"

    current_delta, compact_lookahead_delta = _split_model_field_delta_items(
        data.get("field_delta", []),
        current_node_id=identity.node_id,
    )
    data["field_delta"] = current_delta
    if not data.get("lookahead_field_delta"):
        data["lookahead_field_delta"] = compact_lookahead_delta

    for field in _FAST_JUDGE_STRING_DEFAULT_FIELDS:
        if data.get(field) is None:
            data[field] = ""
    for field in _FAST_JUDGE_LIST_DEFAULT_FIELDS:
        if data.get(field) is None:
            data[field] = []
    data["field_delta"] = _normalize_field_delta_items(data.get("field_delta", []))
    data["lookahead_field_delta"] = _normalize_field_delta_items(data.get("lookahead_field_delta", []), lookahead=True)
    data["missing_fields"] = _normalize_field_name_list(data.get("missing_fields", []))

    node_action = data.get("node_action")
    data["speech_directive"] = _normalize_speech_directive(data.get("speech_directive"))
    reason_override = _downgrade_spurious_control_score_refusal(data, identity=identity)
    guard_reason = _block_advance_over_open_question(data, stage1_text=stage1_text, identity=identity)
    node_action = data.get("node_action")
    if settings.fast_judge_prompt_v2_enabled:
        # v2 path: map top-level reason into answer_assessment.followup_reason with reason_source
        # Note: payload["reason"] is extracted before FastJudgeResult field filtering (line 364)
        top_reason = reason_override or guard_reason or str(payload.get("reason", "") or "").strip()
        aa = data.get("answer_assessment")
        if not isinstance(aa, dict):
            aa = _derive_answer_assessment(data)
        if top_reason:
            aa["followup_reason"] = top_reason
            aa["reason_source"] = "model"
        else:
            # fallback: derive from speech_directive.text (up to 80 chars)
            directive_text = _directive_text(data) or ""
            aa["followup_reason"] = directive_text[:80]
            aa["reason_source"] = "fallback"
            logger.info(
                "fast_judge.reason_fallback_used",
                turn_id=identity.turn_id,
                node=identity.node_id,
            )
        data["answer_assessment"] = aa
    else:
        # v1 legacy path: preserve existing _derive_answer_assessment behavior exactly
        if not data.get("answer_assessment"):
            data["answer_assessment"] = _derive_answer_assessment(data)
    if not data.get("handoff"):
        data["handoff"] = _derive_handoff(data)
    if not data.get("knowledge_question"):
        data["knowledge_question"] = _derive_knowledge_question(data)
    if "stage_complete" not in data:
        data["stage_complete"] = node_action in {"advance", "terminal"}
    if "stage2_should_start" not in data:
        data["stage2_should_start"] = node_action == "advance"
    if "stage1_posture" not in data or data["stage1_posture"] not in {"asked", "ack_only", "closed_unasked"}:
        data["stage1_posture"] = "asked"
    if not data.get("stage2_reason") and data["stage2_should_start"]:
        data["stage2_reason"] = _directive_code(data) or "advance"
    data["answer_assessment"] = _normalize_answer_assessment(data.get("answer_assessment"))
    data["handoff"] = _normalize_handoff(data.get("handoff"), node_action=node_action)
    data["knowledge_question"] = _normalize_knowledge_question(data.get("knowledge_question"), node_action=node_action)
    return data


def _extract_cache_tokens(usage: Any) -> tuple[int, int]:
    """Return (cached_tokens, prompt_tokens) from a backend usage payload.

    OpenAILLMBackend.complete_structured() stores usage as a dict
    (usage.model_dump()), so attribute access via getattr returns the default
    every time — silently masking cache hit rate in shadow eval. DashScope's
    OpenAI-compat surface nests cached counts under prompt_tokens_details.
    Read both flat and nested shapes, tolerate None/missing keys.
    """
    if not isinstance(usage, dict):
        return 0, 0
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    cached_tokens = int(usage.get("cached_tokens") or 0)
    if cached_tokens == 0:
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            cached_tokens = int(details.get("cached_tokens") or 0)
    return cached_tokens, prompt_tokens


class FastJudgeRunner:
    def __init__(self, *, backend, model: str, max_tokens: int, temperature: float) -> None:
        self._backend = backend
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self.last_raw_text = ""
        self.last_usage = None

    async def run(
        self,
        *,
        messages: list[dict],
        identity: FastJudgeInternalIdentity,
        session_id: str,
        turn_id: str,
        stage1_text: str = "",
    ) -> FastJudgeEnvelope:
        response_format = build_response_format(
            model=self._model,
            schema_name="fast_judge",
            schema=FastJudgeModelOutput.model_json_schema(),
        )
        completion = await self._backend.complete_structured(
            messages,
            response_format=response_format,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            session_id=session_id,
            turn_id=turn_id,
        )
        self.last_raw_text = _raw_model_text(completion.raw_text, completion.parsed_json)
        self.last_usage = getattr(completion, "usage", None)
        if completion.parsed_json is None:
            raise FastJudgeFailure("parse_failed")
        # Spec §7 T1/T2: per-turn cache_usage metric. usage is a dict
        # (StructuredCompletionResult.usage: dict[str, Any] | None) populated
        # by OpenAILLMBackend.complete_structured(). DashScope's OpenAI-compat
        # surface reports cached tokens via prompt_tokens_details.cached_tokens;
        # other backends may flatten to top-level cached_tokens. Read both.
        cached_tokens, prompt_tokens = _extract_cache_tokens(self.last_usage)
        logger.info(
            "fast_judge.cache_usage",
            turn_id=identity.turn_id,
            node=identity.node_id,
            cached_tokens=cached_tokens,
            prompt_tokens=prompt_tokens,
            hit_rate=(cached_tokens / prompt_tokens) if prompt_tokens else 0.0,
            prompt_v2=settings.fast_judge_prompt_v2_enabled,
        )
        try:
            result = FastJudgeResult.model_validate(
                _prepare_fast_judge_payload(completion.parsed_json, identity=identity, stage1_text=stage1_text)
            )
        except ValidationError as exc:
            raise FastJudgeFailure("schema_validation_failed") from exc
        return FastJudgeEnvelope(result=result, identity=identity)
