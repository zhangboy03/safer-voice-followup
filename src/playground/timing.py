"""Per-step latency capture by wrapping the injected speech/judge runners.
Keeps driver.run_journey untouched (spec §3.1c)."""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable


def _prefix(turn_id: str) -> str:
    return turn_id.split(":", 1)[0]


def _label(turn_id: str) -> str:
    return turn_id.split(":", 1)[1] if ":" in turn_id else turn_id


class TimingCollector:
    def __init__(self) -> None:
        # key -> {label: {"ttft_ms": float|None, "total_ms": float}}
        self._records: dict[str, dict[str, dict]] = {}

    def record(self, turn_id: str, label: str, ttft_ms: float | None, total_ms: float) -> None:
        self._records.setdefault(_prefix(turn_id), {})[label] = {
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
        }

    def timeline_for(self, turn_id: str) -> dict[str, Any]:
        bucket = self._records.pop(_prefix(turn_id), None)
        if not bucket:
            return {}
        stage1 = bucket.get("stage1", {})
        judge = bucket.get("judge", {})
        stage2 = bucket.get("stage2", {})
        total = sum(s.get("total_ms", 0.0) for s in (stage1, judge, stage2))
        return {
            "stage1_ttft_ms": stage1.get("ttft_ms"),
            "stage1_total_ms": stage1.get("total_ms"),
            "judge_total_ms": judge.get("total_ms"),
            "stage2_ttft_ms": stage2.get("ttft_ms"),
            "stage2_total_ms": stage2.get("total_ms"),
            "brain_total_ms": total,
        }


def wrap_speech_runner(speech_runner: Callable[..., Awaitable[dict]], collector: TimingCollector):
    async def _wrapped(messages, *, session_id: str = "", turn_id: str = "") -> dict:
        started = time.monotonic()
        res = await speech_runner(messages, session_id=session_id, turn_id=turn_id)
        total = (time.monotonic() - started) * 1000.0
        collector.record(turn_id, _label(turn_id), res.get("ttft_ms"), total)
        return res

    return _wrapped


def wrap_judge_runner(fast_judge_runner: Any, collector: TimingCollector) -> Any:
    class _Wrapped:
        async def run(self, **kwargs):
            started = time.monotonic()
            res = await fast_judge_runner.run(**kwargs)
            total = (time.monotonic() - started) * 1000.0
            collector.record(kwargs.get("turn_id", ""), "judge", None, total)
            return res

    return _Wrapped()
