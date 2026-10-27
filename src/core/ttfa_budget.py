"""TTFA 预算管理器

对标 Polaris §3.3 + §5.2 — 实时推理延迟预算分配与执行。

总预算 1000ms 分配：
- ASR wait: 350ms (已由 asr_final_wait_timeout_ms 控制)
- Safety check: 10ms
- Prompt compile: 5ms
- LLM TTFT: 动态（= 剩余 - TTS 预留）
- LLM sentence accumulation: 动态
- TTS synthesis: 300ms (TTS 预留)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TTFABudget:
    """TTFA 预算分配器"""

    total_budget_ms: int = 1000
    tts_reserve_ms: int = 300

    # 内部追踪
    _turn_start_time: float = 0.0

    def start(self) -> None:
        """轮次开始时调用"""
        self._turn_start_time = time.monotonic()

    def elapsed_ms(self) -> float:
        """已耗时"""
        if self._turn_start_time == 0.0:
            return 0.0
        return (time.monotonic() - self._turn_start_time) * 1000

    def remaining_for_llm(self, elapsed_ms: float | None = None) -> float:
        """计算 LLM 可用时间 = 总预算 - 已耗时 - TTS 预留"""
        if elapsed_ms is None:
            elapsed_ms = self.elapsed_ms()
        return max(0, self.total_budget_ms - elapsed_ms - self.tts_reserve_ms)

    def is_over_budget(self, elapsed_ms: float | None = None) -> bool:
        """检查总预算是否已超出（不含 TTS 预留）"""
        if elapsed_ms is None:
            elapsed_ms = self.elapsed_ms()
        return elapsed_ms > (self.total_budget_ms - self.tts_reserve_ms)

    def should_degrade(self, elapsed_ms: float | None = None, threshold_ms: float = 200) -> bool:
        """LLM 可用时间不足 threshold_ms 时建议降级

        Args:
            elapsed_ms: 已耗时（不传则自动计算）
            threshold_ms: 触发降级的 LLM 剩余时间阈值

        Returns:
            True = 应降级 max_tokens
        """
        remaining = self.remaining_for_llm(elapsed_ms)
        return remaining < threshold_ms

    def to_dict(self, elapsed_ms: float | None = None) -> dict[str, Any]:
        """序列化当前预算状态"""
        if elapsed_ms is None:
            elapsed_ms = self.elapsed_ms()
        remaining = self.remaining_for_llm(elapsed_ms)
        degraded = self.should_degrade(elapsed_ms)
        return {
            "total_budget_ms": self.total_budget_ms,
            "elapsed_ms": round(elapsed_ms, 1),
            "remaining_for_llm_ms": round(remaining, 1),
            "tts_reserve_ms": self.tts_reserve_ms,
            "degraded": degraded,
        }
