"""Per-call silent-rescue counters used by LLMProcessor (ADR 0006 §4.5/§9.2).

LLMProcessor 持有一个 SilentRescueState 实例 / 通话，记录：
- total_count：通话总触发数（用于 metrics + cap 检查）
- chain_length：自上次患者发声以来的连续触发数（用于 metrics + soft constraint）
- max_chain_seen：本通话出现过的最大连续触发链（用于 session-end summary）
"""

from dataclasses import dataclass


@dataclass
class SilentRescueState:
    total_count: int = 0
    chain_length: int = 0
    max_chain_seen: int = 0

    def record_trigger(self) -> None:
        self.total_count += 1
        self.chain_length += 1
        if self.chain_length > self.max_chain_seen:
            self.max_chain_seen = self.chain_length

    def on_user_speech(self) -> None:
        """ASR partial 到达 → chain 归零；total / max_chain_seen 保留。"""
        self.chain_length = 0

    def allowed(self, max_per_call: int) -> bool:
        """检查是否还能触发一次。max_per_call=0 视作无限。"""
        if max_per_call <= 0:
            return True
        return self.total_count < max_per_call
