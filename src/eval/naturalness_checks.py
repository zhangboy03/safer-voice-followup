from __future__ import annotations

import json
import re
from typing import Any

_ADVANCE_MARKERS = ("[ADVANCE]", "[CONTINUE]")
_CONFIRMED_STATUSES = {"confirmed", "committed", "collected"}
_REFUSED_STATUSES = {"refused"}
_DIRECT_PATIENT_PATTERNS = (
    "您术后",
    "你术后",
    "您现在",
    "你现在",
    "您感觉",
    "你感觉",
    "您最近",
    "你最近",
    "您有没有",
    "你有没有",
)


def run_deterministic_checks(trace: list[dict]) -> list[dict]:
    findings: list[dict] = []
    for entry in trace:
        findings.extend(_check_field_reasks(entry))
        finding = _check_proxy_speaking_frame(entry)
        if finding:
            findings.append(finding)
        finding = _check_stage2_repeat(entry)
        if finding:
            findings.append(finding)
        finding = _check_stage2_restart_opener(entry)
        if finding:
            findings.append(finding)
        finding = _check_future_completion(entry)
        if finding:
            findings.append(finding)
        findings.extend(_check_judge_commit_evidence(entry))
        finding = _check_stale_judge_acceptance(entry)
        if finding:
            findings.append(finding)
        finding = _check_interrupted_playback_context(entry)
        if finding:
            findings.append(finding)
        finding = _check_same_playout_append(entry)
        if finding:
            findings.append(finding)
        finding = _check_speech_prompt_markers(entry)
        if finding:
            findings.append(finding)
    return findings


def build_regression_candidate(
    *,
    call_id: str,
    finding: dict,
    prompt_hashes: dict,
    context_snapshot_id: str,
    judge_identity: dict,
) -> dict:
    return {
        "artifact_type": "sop_regression_candidate.v1",
        "call_id": call_id,
        "finding": finding,
        "prompt_hashes": prompt_hashes,
        "context_snapshot_id": context_snapshot_id,
        "judge_identity": judge_identity,
    }


def _check_field_reasks(entry: dict) -> list[dict]:
    asked = set(_as_list(entry.get("asked_fields")))
    asked.update(_as_list(_get_path(entry, ("speech", "asked_fields"))))
    asked.update(_as_list(_get_path(entry, ("playback", "asked_fields"))))
    asked.update(_as_list(entry.get("reasked_fields")))
    if not asked:
        return []

    findings: list[dict] = []
    fields = _get_path(entry, ("call_working_memory", "sop_fields"), default={})
    for field, status in _field_statuses(fields).items():
        if field not in asked:
            continue
        if status in _CONFIRMED_STATUSES:
            findings.append(
                _finding("committed_field_reasked", entry, "already committed field was asked again", field=field)
            )
        if status in _REFUSED_STATUSES:
            findings.append(_finding("refused_field_reasked", entry, "refused field was asked again", field=field))
    return findings


def _check_proxy_speaking_frame(entry: dict) -> dict | None:
    frame = _get_path(entry, ("call_working_memory", "speaking_frame"), default={})
    if _get_path(frame, ("caller_role",)) != "proxy_family":
        return None
    heard_text = _stringify(_get_path(entry, ("playback", "heard_text"), default=""))
    if any(pattern in heard_text for pattern in _DIRECT_PATIENT_PATTERNS):
        return _finding(
            "proxy_speaking_frame_violation",
            entry,
            "proxy caller was addressed as if they were the patient",
            relationship=_get_path(frame, ("relationship",)),
        )
    return None


def _check_stage2_repeat(entry: dict) -> dict | None:
    stage1 = _stringify(entry.get("stage1_text") or _get_path(entry, ("speech", "stage1_text"), default=""))
    stage2 = _stringify(entry.get("stage2_text") or _get_path(entry, ("speech", "stage2_text"), default=""))
    if not stage1 or not stage2:
        return None
    normalized_stage1 = _normalize_text(stage1)
    normalized_stage2 = _normalize_text(stage2)
    if len(normalized_stage1) >= 6 and (
        normalized_stage1 == normalized_stage2 or normalized_stage1 in normalized_stage2
    ):
        return _finding("stage2_repeated_stage1_question", entry, "stage2 repeated the stage1 question")
    return None


# 多字承接词（最长优先）；那-簇归一到「那」；单字需后接边界
_FILLER_MULTI = ("嗯好的", "好的", "那好")
_FILLER_NA_CLUSTER = ("那我先", "那我们", "那咱们", "那您")
_FILLER_SINGLE = ("好", "嗯", "对", "那")
_FILLER_BOUNDARY = "，,。.？?！!：:；; 、\t\n"


def _leading_filler(text: str) -> str:
    """Return the canonical opening discourse-marker of a line, or "".

    Mandarin has no spaces, so single-char markers (好/嗯/对/那) only count
    when followed by a boundary (so 好转/那边 don't false-trigger). 那-clusters
    (那我先…) normalize to 「那」 so a 那-vs-那 collision is still caught.
    """
    s = text.lstrip(" \t\n\"'“”‘’：:")
    for token in _FILLER_MULTI:  # longest first
        if s.startswith(token):
            return token
    for token in _FILLER_NA_CLUSTER:
        if s.startswith(token):
            return "那"
    for token in _FILLER_SINGLE:
        if s.startswith(token):
            rest = s[len(token) :]
            if rest == "" or rest[0] in _FILLER_BOUNDARY:
                return token
    return ""


def _check_stage2_restart_opener(entry: dict) -> dict | None:
    stage1 = _stringify(entry.get("stage1_text") or _get_path(entry, ("speech", "stage1_text"), default=""))
    stage2 = _stringify(entry.get("stage2_text") or _get_path(entry, ("speech", "stage2_text"), default=""))
    if not stage1 or not stage2:
        return None
    f1 = _leading_filler(stage1)
    f2 = _leading_filler(stage2)
    if f1 and f2 and f1 == f2:
        return _finding("stage2_restart_opener", entry, "stage2 reused stage1's opening filler")
    return None


def _check_future_completion(entry: dict) -> dict | None:
    node_id = entry.get("node_id")
    completed_node = entry.get("completed_node_id") or _get_path(entry, ("transition", "completed_node_id"))
    source_turn_index = entry.get("completion_source_turn_index") or _get_path(
        entry,
        ("transition", "completion_source_turn_index"),
    )
    turn_index = entry.get("turn_index")
    if completed_node == node_id and _is_future_index(source_turn_index, turn_index):
        return _finding(
            "future_observation_completed_active_node",
            entry,
            "active node was completed from a future observation",
            completion_source_turn_index=source_turn_index,
        )

    future = entry.get("future_observation") or {}
    if future.get("completed_node_id") == node_id and future.get("accepted") is True:
        return _finding(
            "future_observation_completed_active_node",
            entry,
            "future observation completed the active node",
        )
    return None


def _check_judge_commit_evidence(entry: dict) -> list[dict]:
    judge = entry.get("judge") or {}
    if judge.get("accepted") is not True:
        return []
    committed = _as_list(judge.get("committed_delta") or judge.get("field_commits"))
    findings: list[dict] = []
    for candidate in committed:
        if not isinstance(candidate, dict):
            continue
        if any(candidate.get(key) for key in ("evidence", "evidence_quote", "source_turn_id", "source_text")):
            continue
        findings.append(
            _finding(
                "judge_commit_without_evidence",
                entry,
                "judge committed a field without source evidence",
                field=candidate.get("field"),
            )
        )
    return findings


def _check_stale_judge_acceptance(entry: dict) -> dict | None:
    judge = entry.get("judge") or {}
    if judge.get("accepted") is not True:
        return None
    if judge.get("turn_id") not in (None, entry.get("turn_id")):
        return _finding("stale_turn_or_flow_epoch_accepted", entry, "accepted judge turn_id is stale")
    if judge.get("flow_epoch") not in (None, entry.get("flow_epoch")):
        return _finding("stale_turn_or_flow_epoch_accepted", entry, "accepted judge flow_epoch is stale")
    return None


def _check_interrupted_playback_context(entry: dict) -> dict | None:
    playback = entry.get("playback") or {}
    if playback.get("used_as_heard_context") is True:
        return _finding(
            "interrupted_playback_used_as_heard_context",
            entry,
            "interrupted playback was explicitly marked as heard context",
        )
    if playback.get("interrupted") is not True:
        return None
    heard_text = _stringify(playback.get("heard_text"))
    if not heard_text:
        return None
    context_text = _stringify(entry.get("weighted_context") or entry.get("heard_context"))
    if heard_text in context_text:
        return _finding(
            "interrupted_playback_used_as_heard_context",
            entry,
            "interrupted playback text appeared in heard context",
        )
    return None


def _check_same_playout_append(entry: dict) -> dict | None:
    append = entry.get("append") or {}
    if append.get("allowed") is not True or append.get("same_playout") is not True:
        return None
    if append.get("post_commit_context") or append.get("post_commit_context_snapshot_id"):
        return None
    return _finding(
        "same_playout_append_without_post_commit_context",
        entry,
        "same-playout append was allowed without post-commit context",
    )


def _check_speech_prompt_markers(entry: dict) -> dict | None:
    prompt_text = _stringify(
        entry.get("speech_prompt")
        or entry.get("prompt")
        or entry.get("messages")
        or _get_path(entry, ("speech", "messages"))
    )
    if any(marker in prompt_text for marker in _ADVANCE_MARKERS):
        return _finding(
            "speech_prompt_contains_advance_continue_marker",
            entry,
            "speech prompt contained legacy advance/continue marker instruction",
        )
    return None


def _field_statuses(fields: Any) -> dict[str, str]:
    if not isinstance(fields, dict):
        return {}
    statuses: dict[str, str] = {}
    for field, value in fields.items():
        status: Any
        if isinstance(value, dict):
            status = value.get("status")
        else:
            status = getattr(value, "status", value)
        statuses[str(field)] = str(status).lower()
    return statuses


def _finding(check: str, entry: dict, reason: str, **extra: Any) -> dict:
    result = {
        "check": check,
        "turn_id": entry.get("turn_id"),
        "node_id": entry.get("node_id"),
        "reason": reason,
    }
    result.update({key: value for key, value in extra.items() if value is not None})
    return result


def _get_path(value: Any, path: tuple[str, ...], default: Any = None) -> Any:
    current = value
    for key in path:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
        if current is default:
            return default
    return current


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple | set):
        return list(value)
    return [value]


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _normalize_text(text: str) -> str:
    return re.sub(r"[\s，。？！,.?!：:；;\"'“”‘’、]+", "", text)


def _is_future_index(source_turn_index: Any, turn_index: Any) -> bool:
    try:
        return int(source_turn_index) > int(turn_index)
    except (TypeError, ValueError):
        return False
