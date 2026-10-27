from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LATENCY_FIELDS = [
    "case_id",
    "text",
    "category",
    "voice",
    "tts_model",
    "latency_segmented",
    "latency_consistent",
    "latency_turn_count",
    "latency_select_reason",
    "asr_final_latency_ms",
    "user_stop_to_emit_ms",
    "llm_ttft_ms",
    "tts_ttfa_ms",
    "emit_to_audio_ms",
    "e2e_ms",
    "speech_end_to_audio_ms",
]
BARGE_IN_FIELDS = [
    "case_id",
    "mode",
    "interruption_type",
    "fixture_kind",
    "text",
    "voice",
    "tts_model",
    "delay_ms",
    "user_started",
    "tts_interrupted",
    "asr_interrupt_min_chars",
    "max_interrupt_asr_chars",
    "asr_interrupt_qualified",
    "interrupt_outcome",
    "false_interrupt_outcome",
    "active_opener_tts_at_injection",
    "opener_tts_injection_overlap_ms",
    "active_opener_tts_at_interrupt",
    "opener_tts_overlap_ms",
    "detection_ms",
    "old_tts_stop_ms",
    "new_business_turn_count",
    "followup_turn_metrics_count",
    "old_line_stopped",
    "false_trigger",
    "behavior_pass",
    "barge_in_recovered",
    "interruption_asr_text",
    "agent_text_after_interrupt",
    "invalid_injection",
    "injection_sim_opener",
    "judge_recovery_label",
    "judge_confidence",
]
# 合格打断尝试 = ASR 达阈值且旧音频仍在可听窗口内（drain 锚点）。这三种结局都属"真的发生了
# 一次合格打断"，进 SIR 分母；not_asr_qualified / no_active_tts / detection_after_tts_complete /
# no_user_speech 不是合格尝试，不进分母。
_SIR_QUALIFIED_OUTCOMES = ("interrupted", "soft_pivot", "stop_missing")
CASE_INDEX_FIELDS = [
    "case_id",
    "part",
    "mode",
    "status",
    "summary_path",
    "events_bytes",
    "raw_bytes",
]
JUDGE_FIELDS = [
    "case_id",
    "judge_model",
    "judge_valid",
    "judge_recovery_label",
    "judge_ignored_new_intent",
    "judge_wrong_business_action",
    "judge_confidence",
    "judge_evidence",
]


def aggregate_study1_run(
    run_dir: Path | str,
    *,
    case_manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    summaries = _load_case_summaries(run_path)
    latency_rows = [s for s in summaries if s.get("part") == "latency"]
    barge_rows = [s for s in summaries if s.get("part") == "barge_in"]
    judge_rows = [s for s in summaries if s.get("judge_valid") is not None]
    missing_rows = _missing_event_rows(summaries)

    # 口径（2026-06-03，判官式）：SIR 分子 = 停旧线（硬切或软转）AND judge 答对新意图。
    # 逐 case 落 barge_in_recovered，再在 run_manifest 汇总 SIR。就地写回 barge_rows。
    for row in barge_rows:
        if row.get("mode") == "interrupt":
            row["barge_in_recovered"] = bool(row.get("behavior_pass")) and row.get("judge_recovery_label") == "success"
    sir = _barge_in_sir(barge_rows)

    _write_csv(run_path / "case_index.csv", CASE_INDEX_FIELDS, [_case_index_row(run_path, s) for s in summaries])
    _write_csv(run_path / "latency_summary.csv", LATENCY_FIELDS, latency_rows)
    _write_csv(run_path / "barge_in_summary.csv", BARGE_IN_FIELDS, barge_rows)
    _write_jsonl(run_path / "judge_results.jsonl", [_project_fields(row, JUDGE_FIELDS) for row in judge_rows])
    _write_jsonl(run_path / "missing_events.jsonl", missing_rows)

    manifest = {
        "study": "study_1_voice_interaction",
        "run_id": run_path.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(summaries),
        "latency_count": len(latency_rows),
        "barge_in_count": len(barge_rows),
        "judge_count": len(judge_rows),
        "missing_event_count": len(missing_rows),
        "barge_in_sir": sir,
        "git_commit": _git_commit(),
    }
    if case_manifest_path is not None:
        path = Path(case_manifest_path)
        manifest["case_manifest_path"] = str(path)
        manifest["case_manifest_sha256"] = _sha256(path)
    (run_path / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _load_case_summaries(run_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(run_path.glob("*/r*/summary.json")):
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        data["_summary_path"] = str(summary_path)
        rows.append(data)
    return rows


def _case_index_row(run_path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    status = "pass"
    if any(row["case_id"] == summary.get("case_id") for row in _missing_event_rows([summary])):
        status = "diagnostic"
    return {
        "case_id": summary.get("case_id", ""),
        "part": summary.get("part", ""),
        "mode": summary.get("mode", ""),
        "status": status,
        "summary_path": _relative_to(summary.get("_summary_path", ""), run_path),
        "events_bytes": summary.get("events_bytes", ""),
        "raw_bytes": summary.get("raw_bytes", ""),
    }


def _barge_in_sir(barge_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """ASR-qualified Success-Interrupt Rate（判官式口径）。

    分母 = 真的发生了一次合格打断的 true-interrupt case（``interrupt_outcome`` ∈
    interrupted/soft_pivot/stop_missing）；分子 = 其中 ``barge_in_recovered``（停旧线 AND
    judge 答对新意图）。同时按 outcome 计数，便于诊断分母构成。MVP 样本小，rate 仅供参考。
    """

    interrupts = [r for r in barge_rows if r.get("mode") == "interrupt"]
    by_outcome: dict[str, int] = {}
    for r in interrupts:
        outcome = str(r.get("interrupt_outcome", "unknown"))
        by_outcome[outcome] = by_outcome.get(outcome, 0) + 1
    invalid_injections = sum(1 for r in interrupts if r.get("invalid_injection"))
    # 剔除 invalid_injection（ASR 抓到 opener 而非预期打断语）后再算 SIR/recovery，避免注入抖动污染分母。
    qualified = [
        r
        for r in interrupts
        if r.get("interrupt_outcome") in _SIR_QUALIFIED_OUTCOMES and not r.get("invalid_injection")
    ]
    recovered = [r for r in qualified if r.get("barge_in_recovered")]
    denom = len(qualified)
    return {
        "qualified_attempts": denom,
        "recovered": len(recovered),
        "rate": (len(recovered) / denom) if denom else None,
        "invalid_injections": invalid_injections,
        "by_outcome": by_outcome,
    }


def _missing_event_rows(summaries: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for summary in summaries:
        case_id = str(summary.get("case_id", ""))
        if summary.get("part") == "latency":
            # 新口径：长句被切成多段时，碎片 turn 的锚点不自洽（见 study1_latency）。
            # 这种情况只报一条 segmentation 诊断，不再逐字段报 -1（那些 -1 是 segmentation
            # 的症状，不是独立缺失）。latency_consistent 缺失（旧 summary）时退回逐字段判定。
            if summary.get("latency_consistent") is False:
                rows.append(
                    {
                        "case_id": case_id,
                        "field": "latency_consistent",
                        "reason": (
                            "latency anchors inconsistent / utterance segmented "
                            f"({summary.get('latency_select_reason', '')}); no usable TTFA"
                        ),
                    }
                )
            else:
                if _non_positive(summary.get("emit_to_audio_ms")):
                    rows.append(
                        {
                            "case_id": case_id,
                            "field": "emit_to_audio_ms",
                            "reason": "latency case did not produce first-agent-audio timing",
                        }
                    )
                if _non_positive(summary.get("user_stop_to_emit_ms")):
                    rows.append(
                        {
                            "case_id": case_id,
                            "field": "user_stop_to_emit_ms",
                            "reason": "latency case did not produce turn-boundary readiness timing",
                        }
                    )
                if _non_positive(summary.get("e2e_ms")):
                    rows.append(
                        {
                            "case_id": case_id,
                            "field": "e2e_ms",
                            "reason": "latency case did not produce complete end-to-end timing",
                        }
                    )
        if summary.get("part") != "barge_in":
            continue
        if summary.get("mode") == "interrupt":
            if not summary.get("user_started"):
                rows.append(
                    {
                        "case_id": case_id,
                        "field": "user_started",
                        "reason": "true barge-in did not emit UserStartedSpeakingEvent",
                    }
                )
            interrupt_outcome = summary.get("interrupt_outcome")
            if interrupt_outcome is not None:
                # 新口径：只有 "opener 仍在播、ASR 已达到线上打断阈值、却没停旧 TTS"
                # 才是真正缺失诊断；
                # no_active_tts（opener 太短，注入点落在回复结束后）属测试设计问题，
                # not_asr_qualified（ASR 未达到 3 个有效字符）属线上逻辑下的非打断，
                # 都在 barge_in_summary.csv 里单列，不进 missing_events。
                if interrupt_outcome == "stop_missing":
                    rows.append(
                        {
                            "case_id": case_id,
                            "field": "tts_interrupted",
                            "reason": "true barge-in: opener TTS still active but no TTSInterruptedEvent (agent did not stop within window)",
                        }
                    )
            elif not summary.get("tts_interrupted"):
                # legacy summary（无 interrupt_outcome 字段）退回旧判定
                rows.append(
                    {
                        "case_id": case_id,
                        "field": "tts_interrupted",
                        "reason": "true barge-in did not emit TTSInterruptedEvent",
                    }
                )
            if summary.get("judge_valid") is not True:
                rows.append(
                    {
                        "case_id": case_id,
                        "field": "judge_valid",
                        "reason": "true barge-in did not produce a valid recovery judge result",
                    }
                )
        elif summary.get("false_trigger"):
            # 新口径：false_trigger 只在 business_false_trigger（产生了新业务 turn）时为真；
            # acoustic_stop（仅声学暂停旧 TTS、无新业务 turn）不再算误打断。
            rows.append(
                {
                    "case_id": case_id,
                    "field": "false_trigger",
                    "reason": "false interrupt produced a new business turn (business false trigger)",
                }
            )
    return rows


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_project_fields(row, fields))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _project_fields(row: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in fields}


def _relative_to(path: str, root: Path) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return path


def _non_positive(value: Any) -> bool:
    return not isinstance(value, (int, float)) or value <= 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return ""
