"""延迟指标追踪模块

提供轮次和会话级别的延迟指标收集与统计。

M10 增强:
- P50/P95/P99 百分位统计
- 延迟达标率计算
- 分阶段延迟统计
- 延迟来源标记
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

# 默认延迟阈值（毫秒）
# 各阶段定义：
#   - ASR: 用户停止说话 → ASR final（包含 VAD 等待 250-500ms + ASR 处理）
#   - RAG: RAG 调用开始 → 返回结果（纯 RAG 查询耗时）
#   - LLM: ASR final → 第一个分句返回（LLM 处理 + 分句）
#   - TTS: 第一个分句返回 → 前端播放（TTS 合成 + 网络传输）
#   - E2E: 用户停止说话 → 前端播放（端到端延迟）
DEFAULT_LATENCY_THRESHOLDS = {
    "asr": 450,  # VAD 等待 250-500ms + ASR 处理 ~100ms
    "rag": 50,  # 纯 RAG 查询，本地 Embedding 应该很快
    "llm": 400,  # ASR final → 第一个分句
    "tts": 200,  # 第一个分句 → 前端播放
    "e2e": 900,  # 内部达标阈值（比 <1s 目标更严格，留 100ms 余量）
    "ttfa": 900,  # Time-To-First-Audio = E2E
}


@dataclass
class TurnMetrics:
    """单轮对话延迟指标，使用 time.monotonic() 计时

    监测点定义：
    - ASR: speech_end_time → asr_final_time（包含 VAD 等待时间）
    - RAG: rag_start_time → rag_end_time（纯 RAG 查询耗时）
    - LLM: asr_final_time → llm_first_sentence_time（LLM 处理到第一个分句）
    - TTS: llm_first_sentence_time → tts_play_time（第一个分句到前端播放）
    - E2E: speech_end_time → tts_play_time（端到端延迟，包含 VAD 等待时间）
    """

    turn_id: str

    # 核心时间戳
    speech_end_time: Optional[float] = None  # 用户停止说话的时间（静音开始，VAD 等待前）
    asr_final_time: Optional[float] = None  # ASR 返回最终转录结果

    # RAG 时间戳（纯 RAG 查询耗时）
    rag_start_time: Optional[float] = None  # RAG 模块被调用
    rag_end_time: Optional[float] = None  # RAG 返回结果

    # === LLM 管道细分时间戳（拆解 asr_final → llm_first_sentence） ===
    safety_check_end_time: Optional[float] = None  # 安全+转人工检测完成
    analysis_start_time: Optional[float] = None  # 并行 SOP 分析 + RAG 开始
    analysis_end_time: Optional[float] = None  # 并行 SOP 分析 + RAG 全部完成
    llm_call_start_time: Optional[float] = None  # LLM 流式调用发起
    llm_first_token_time: Optional[float] = None  # LLM 返回第一个 token

    # LLM 时间戳
    llm_first_sentence_time: Optional[float] = None  # LLM 返回第一个完整分句

    # TTS 时间戳
    tts_first_audio_time: Optional[float] = None  # TTS 返回第一个音频块（服务端收到）
    tts_play_time: Optional[float] = None  # 前端开始播放（WebSocket 发送首字节）

    # ASR 观测字段（Part 0 基线）
    asr_text_char_count: int = 0  # final 文本字符数（去标点空格后）
    asr_confidence: Optional[float] = None  # ASR 置信度，None=未知
    speech_duration_ms: int = 0  # 发话时长 turn_start→speech_end
    vad_wait_ms: int = 0  # endpoint 等待 speech_end→当前

    # Part 4A: 上下文缓存观测
    flash_prompt_tokens: int = 0
    flash_cached_tokens: int = 0
    flash_completion_tokens: int = 0
    plus_prompt_tokens: int = 0
    plus_cached_tokens: int = 0

    # Part 1: ASR 置信度感知
    asr_low_confidence_hint_injected: bool = False

    # Part 4A: TTFA 预算观测
    ttfa_budget_total_ms: int = 0
    ttfa_budget_remaining_at_llm_ms: int = 0
    ttfa_budget_degraded: bool = False

    # Time-to-Main-Answer
    main_first_audio_ready_ms: int = 0  # speech_end → 主首音频 chunk 收到
    main_first_audio_play_ms: int = 0  # speech_end → 主首音频发到 WebSocket

    def get_asr_ms(self) -> int:
        """计算 ASR 延迟（毫秒）: 用户停止说话 → ASR final（包含 VAD 等待时间）"""
        if self.speech_end_time and self.asr_final_time:
            return int((self.asr_final_time - self.speech_end_time) * 1000)
        return 0

    def get_rag_ms(self) -> int:
        """计算 RAG 延迟（毫秒）: RAG 开始 → RAG 结束（纯查询耗时）"""
        if self.rag_start_time and self.rag_end_time:
            return int((self.rag_end_time - self.rag_start_time) * 1000)
        return 0

    def get_llm_ms(self) -> int:
        """计算 LLM 延迟（毫秒）: ASR final → 第一个分句返回"""
        if self.asr_final_time and self.llm_first_sentence_time:
            return int((self.llm_first_sentence_time - self.asr_final_time) * 1000)
        return 0

    def get_tts_ms(self) -> int:
        """计算 TTS 延迟（毫秒）: 第一个分句返回 → 前端播放"""
        if self.llm_first_sentence_time and self.tts_play_time:
            return int((self.tts_play_time - self.llm_first_sentence_time) * 1000)
        return 0

    def get_e2e_ms(self) -> int:
        """计算端到端延迟（毫秒）: 用户停止说话 → 前端播放（包含 VAD 等待时间）"""
        if self.speech_end_time and self.tts_play_time:
            return int((self.tts_play_time - self.speech_end_time) * 1000)
        return 0

    # === LLM 管道细分指标 ===

    def get_safety_ms(self) -> int:
        """安全+转人工检测耗时: ASR final → 安全检测完成"""
        if self.asr_final_time and self.safety_check_end_time:
            return int((self.safety_check_end_time - self.asr_final_time) * 1000)
        return 0

    def get_analysis_ms(self) -> int:
        """SOP分析+RAG并行耗时: 分析开始 → 全部完成"""
        if self.analysis_start_time and self.analysis_end_time:
            return int((self.analysis_end_time - self.analysis_start_time) * 1000)
        return 0

    def get_llm_ttft_ms(self) -> int:
        """LLM 首 token 延迟 (TTFT): LLM 调用发起 → 第一个 token"""
        if self.llm_call_start_time and self.llm_first_token_time:
            return int((self.llm_first_token_time - self.llm_call_start_time) * 1000)
        return 0

    def get_llm_sentence_ms(self) -> int:
        """分句积累耗时: 第一个 token → 第一个完整分句"""
        if self.llm_first_token_time and self.llm_first_sentence_time:
            return int((self.llm_first_sentence_time - self.llm_first_token_time) * 1000)
        return 0

    def get_tts_synthesis_ms(self) -> int:
        """TTS 合成耗时: 第一个分句 → TTS 返回第一个音频块"""
        if self.llm_first_sentence_time and self.tts_first_audio_time:
            return int((self.tts_first_audio_time - self.llm_first_sentence_time) * 1000)
        return 0

    def get_tts_network_ms(self) -> int:
        """TTS 网络+发送耗时: TTS 第一个音频块 → WebSocket 发送"""
        if self.tts_first_audio_time and self.tts_play_time:
            return int((self.tts_play_time - self.tts_first_audio_time) * 1000)
        return 0

    def get_ttfa_ms(self) -> int:
        """TTFA (Time-To-First-Audio) = E2E 别名"""
        return self.get_e2e_ms()

    def is_short_text(self, threshold: int = 3) -> bool:
        """短词判定（去标点后 <=threshold 字符）"""
        return self.asr_text_char_count <= threshold and self.asr_text_char_count > 0

    def has_valid_confidence(self) -> bool:
        """是否有有效置信度"""
        return self.asr_confidence is not None

    def is_latency_compliant(self, threshold_ms: int = 1000) -> bool:
        """检查端到端延迟是否达标"""
        e2e = self.get_e2e_ms()
        return e2e > 0 and e2e <= threshold_ms

    def to_message(self) -> dict:
        """转换为前端消息格式"""
        return {
            "type": "metrics",
            "turn_id": self.turn_id,
            "data": {
                "asr_ms": self.get_asr_ms(),
                "rag_ms": self.get_rag_ms(),
                "llm_ms": self.get_llm_ms(),
                "tts_ms": self.get_tts_ms(),
                "e2e_ms": self.get_e2e_ms(),
                # LLM 管道细分
                "safety_ms": self.get_safety_ms(),
                "analysis_ms": self.get_analysis_ms(),
                "llm_ttft_ms": self.get_llm_ttft_ms(),
                "llm_sentence_ms": self.get_llm_sentence_ms(),
                # TTS 细分
                "tts_synthesis_ms": self.get_tts_synthesis_ms(),
                "tts_network_ms": self.get_tts_network_ms(),
                # ASR 观测字段
                "asr_text_char_count": self.asr_text_char_count,
                "asr_confidence": self.asr_confidence,
                "speech_duration_ms": self.speech_duration_ms,
                "vad_wait_ms": self.vad_wait_ms,
                # Part 4A: 缓存观测
                "flash_prompt_tokens": self.flash_prompt_tokens,
                "flash_cached_tokens": self.flash_cached_tokens,
                "flash_cache_hit_rate": (
                    round(self.flash_cached_tokens / self.flash_prompt_tokens, 4)
                    if self.flash_prompt_tokens > 0
                    else 0.0
                ),
                # Part 1: ASR 置信度感知
                "asr_low_confidence_hint_injected": self.asr_low_confidence_hint_injected,
                # Part 4A: TTFA 预算
                "ttfa_budget_degraded": self.ttfa_budget_degraded,
                "main_first_audio_ready_ms": self.main_first_audio_ready_ms,
            },
        }

    def to_dict(self) -> dict:
        """转换为完整字典格式（用于日志和报告）"""
        return {
            "turn_id": self.turn_id,
            "asr_ms": self.get_asr_ms(),
            "rag_ms": self.get_rag_ms(),
            "llm_ms": self.get_llm_ms(),
            "tts_ms": self.get_tts_ms(),
            "e2e_ms": self.get_e2e_ms(),
            "is_compliant": self.is_latency_compliant(),
            # LLM 管道细分
            "safety_ms": self.get_safety_ms(),
            "analysis_ms": self.get_analysis_ms(),
            "llm_ttft_ms": self.get_llm_ttft_ms(),
            "llm_sentence_ms": self.get_llm_sentence_ms(),
            # TTS 细分
            "tts_synthesis_ms": self.get_tts_synthesis_ms(),
            "tts_network_ms": self.get_tts_network_ms(),
            # ASR 观测字段
            "asr_text_char_count": self.asr_text_char_count,
            "asr_confidence": self.asr_confidence,
            "speech_duration_ms": self.speech_duration_ms,
            "vad_wait_ms": self.vad_wait_ms,
            # Part 4A: 缓存观测
            "flash_prompt_tokens": self.flash_prompt_tokens,
            "flash_cached_tokens": self.flash_cached_tokens,
            "flash_cache_hit_rate": (
                round(self.flash_cached_tokens / self.flash_prompt_tokens, 4) if self.flash_prompt_tokens > 0 else 0.0
            ),
            "plus_prompt_tokens": self.plus_prompt_tokens,
            "plus_cached_tokens": self.plus_cached_tokens,
            # Part 1: ASR 置信度感知
            "asr_low_confidence_hint_injected": self.asr_low_confidence_hint_injected,
            # Part 4A: TTFA 预算
            "ttfa_budget_total_ms": self.ttfa_budget_total_ms,
            "ttfa_budget_remaining_at_llm_ms": self.ttfa_budget_remaining_at_llm_ms,
            "ttfa_budget_degraded": self.ttfa_budget_degraded,
            # Time-to-Main-Answer
            "main_first_audio_ready_ms": self.main_first_audio_ready_ms,
            "main_first_audio_play_ms": self.main_first_audio_play_ms,
        }


@dataclass
class SessionMetrics:
    """会话级别指标统计"""

    session_id: str
    start_time: float = field(default_factory=time.monotonic)
    total_turns: int = 0
    total_interrupts: int = 0
    total_clarifications: int = 0  # 预留（计数口径：触发判定次数）
    turn_metrics_history: list[TurnMetrics] = field(default_factory=list)

    # Part 3: 跨通话记忆指标
    memory_injected: bool = False  # 本次通话是否注入了记忆
    memory_items_count: int = 0  # 注入的记忆条数
    previous_calls_count: int = 0  # 历史通话数

    def add_turn(self, turn_metrics: TurnMetrics):
        """添加轮次指标"""
        self.total_turns += 1
        self.turn_metrics_history.append(turn_metrics)

    def record_interrupt(self):
        """记录打断次数"""
        self.total_interrupts += 1

    def record_clarification(self):
        """记录追问/澄清次数"""
        self.total_clarifications += 1

    # ============ 会话级统计率 ============

    def get_interrupt_rate(self) -> float:
        """打断率 = total_interrupts / total_turns"""
        if self.total_turns == 0:
            return 0.0
        return self.total_interrupts / self.total_turns

    def get_short_text_count(self) -> int:
        """短词轮次数"""
        return sum(1 for t in self.turn_metrics_history if t.is_short_text())

    def get_short_text_rate(self) -> float:
        """短词率 = short_text_count / total_turns"""
        if self.total_turns == 0:
            return 0.0
        return self.get_short_text_count() / self.total_turns

    def get_clarification_rate(self) -> float:
        """追问率 = total_clarifications / total_turns"""
        if self.total_turns == 0:
            return 0.0
        return self.total_clarifications / self.total_turns

    def get_avg_speech_duration_ms(self) -> float:
        """平均发话时长"""
        durations = [t.speech_duration_ms for t in self.turn_metrics_history if t.speech_duration_ms > 0]
        if not durations:
            return 0.0
        return sum(durations) / len(durations)

    def get_avg_asr_confidence(self) -> float:
        """平均 ASR 置信度（仅统计有效值）"""
        confs = [t.asr_confidence for t in self.turn_metrics_history if t.has_valid_confidence()]
        if not confs:
            return 0.0
        return sum(confs) / len(confs)

    # ============ Part 4A: 缓存与延迟漂移统计 ============

    def get_cache_hit_rate(self) -> float:
        """Flash 缓存命中率 = sum(cached_tokens) / sum(prompt_tokens)"""
        total_prompt = sum(t.flash_prompt_tokens for t in self.turn_metrics_history)
        total_cached = sum(t.flash_cached_tokens for t in self.turn_metrics_history)
        if total_prompt == 0:
            return 0.0
        return total_cached / total_prompt

    def get_plus_cache_hit_rate(self) -> float:
        """Plus 缓存命中率 = sum(plus_cached_tokens) / sum(plus_prompt_tokens)"""
        total_prompt = sum(t.plus_prompt_tokens for t in self.turn_metrics_history)
        total_cached = sum(t.plus_cached_tokens for t in self.turn_metrics_history)
        if total_prompt == 0:
            return 0.0
        return total_cached / total_prompt

    def get_ttft_drift(self) -> dict:
        """计算 TTFT 随轮次的漂移趋势

        Returns:
            {"early_median_ms": int, "recent_median_ms": int, "drift_ms": int}
        """
        ttft_values = [t.get_llm_ttft_ms() for t in self.turn_metrics_history if t.get_llm_ttft_ms() > 0]
        if len(ttft_values) < 4:
            return {"early_median_ms": 0, "recent_median_ms": 0, "drift_ms": 0}
        mid = len(ttft_values) // 2
        early = sorted(ttft_values[:mid])
        recent = sorted(ttft_values[mid:])
        early_median = early[len(early) // 2]
        recent_median = recent[len(recent) // 2]
        return {
            "early_median_ms": early_median,
            "recent_median_ms": recent_median,
            "drift_ms": recent_median - early_median,
        }

    def get_budget_degradation_count(self) -> int:
        """TTFA 预算降级触发次数"""
        return sum(1 for t in self.turn_metrics_history if t.ttfa_budget_degraded)

    # ============ M10 增强：统计函数 ============

    def get_e2e_latencies(self) -> list[int]:
        """获取所有轮次的端到端延迟列表"""
        latencies = []
        for turn in self.turn_metrics_history:
            e2e = turn.get_e2e_ms()
            if e2e > 0:
                latencies.append(e2e)
        return latencies

    def get_stage_latencies(self, stage: str) -> list[int]:
        """获取指定阶段的延迟列表

        Args:
            stage: 阶段名称（asr, rag, llm, tts, e2e, ttfa, vad_wait）

        Returns:
            延迟列表
        """
        latencies = []
        for turn in self.turn_metrics_history:
            if stage == "asr":
                val = turn.get_asr_ms()
            elif stage == "rag":
                val = turn.get_rag_ms()
            elif stage == "llm":
                val = turn.get_llm_ms()
            elif stage == "tts":
                val = turn.get_tts_ms()
            elif stage == "e2e":
                val = turn.get_e2e_ms()
            elif stage == "ttfa":
                val = turn.get_ttfa_ms()
            elif stage == "vad_wait":
                val = turn.vad_wait_ms
            else:
                val = 0

            if val > 0:
                latencies.append(val)

        return latencies

    def get_percentile(self, latencies: list[int], percentile: float) -> int:
        """计算百分位数

        Args:
            latencies: 延迟列表
            percentile: 百分位（0-1，如 0.95 表示 P95）

        Returns:
            百分位延迟值
        """
        if not latencies:
            return 0

        sorted_latencies = sorted(latencies)
        idx = int(len(sorted_latencies) * percentile)
        idx = min(idx, len(sorted_latencies) - 1)
        return sorted_latencies[idx]

    def get_average(self, latencies: list[int]) -> float:
        """计算平均值"""
        if not latencies:
            return 0.0
        return sum(latencies) / len(latencies)

    def get_compliance_rate(self, threshold_ms: int = 900) -> float:
        """计算延迟达标率

        Args:
            threshold_ms: 延迟阈值（毫秒）

        Returns:
            达标率（0-1）
        """
        latencies = self.get_e2e_latencies()
        if not latencies:
            return 0.0

        compliant = sum(1 for lat in latencies if lat <= threshold_ms)
        return compliant / len(latencies)

    def get_latency_stats(self) -> dict:
        """获取延迟统计数据

        Returns:
            包含 P50/P95/P99、平均值、达标率等统计数据
        """
        e2e_latencies = self.get_e2e_latencies()

        if not e2e_latencies:
            return {
                "count": 0,
                "avg_ms": 0,
                "p50_ms": 0,
                "p95_ms": 0,
                "p99_ms": 0,
                "min_ms": 0,
                "max_ms": 0,
                "compliance_rate": 0.0,
                "compliant_count": 0,
            }

        return {
            "count": len(e2e_latencies),
            "avg_ms": round(self.get_average(e2e_latencies), 1),
            "p50_ms": self.get_percentile(e2e_latencies, 0.50),
            "p95_ms": self.get_percentile(e2e_latencies, 0.95),
            "p99_ms": self.get_percentile(e2e_latencies, 0.99),
            "min_ms": min(e2e_latencies),
            "max_ms": max(e2e_latencies),
            "compliance_rate": round(self.get_compliance_rate(), 4),
            "compliant_count": sum(1 for lat in e2e_latencies if lat <= DEFAULT_LATENCY_THRESHOLDS["e2e"]),
        }

    def get_stage_stats(self) -> dict[str, dict]:
        """获取各阶段延迟统计

        Returns:
            各阶段的统计数据
        """
        stages = ["asr", "rag", "llm", "tts", "e2e", "ttfa"]
        result = {}

        for stage in stages:
            latencies = self.get_stage_latencies(stage)
            if latencies:
                threshold = DEFAULT_LATENCY_THRESHOLDS.get(stage, 800)
                compliant = sum(1 for lat in latencies if lat <= threshold)
                result[stage] = {
                    "count": len(latencies),
                    "avg_ms": round(self.get_average(latencies), 1),
                    "p95_ms": self.get_percentile(latencies, 0.95),
                    "threshold_ms": threshold,
                    "compliance_rate": round(compliant / len(latencies), 4),
                }

        return result

    def get_summary(self) -> dict:
        """获取会话摘要（增强版）"""
        duration = time.monotonic() - self.start_time
        latency_stats = self.get_latency_stats()

        return {
            "session_id": self.session_id,
            "duration_s": round(duration, 2),
            "total_turns": self.total_turns,
            "total_interrupts": self.total_interrupts,
            "total_clarifications": self.total_clarifications,
            # M10 增强：延迟统计
            "latency": latency_stats,
            # 会话级统计率
            "rates": {
                "interrupt_rate": round(self.get_interrupt_rate(), 4),
                "short_text_rate": round(self.get_short_text_rate(), 4),
                "clarification_rate": round(self.get_clarification_rate(), 4),
                "avg_speech_duration_ms": round(self.get_avg_speech_duration_ms(), 1),
                "avg_asr_confidence": round(self.get_avg_asr_confidence(), 4),
            },
            # Part 3: 跨通话记忆
            "memory": {
                "injected": self.memory_injected,
                "items_count": self.memory_items_count,
                "previous_calls_count": self.previous_calls_count,
            },
        }

    def get_detailed_summary(self) -> dict:
        """获取详细会话摘要（包含各阶段统计）"""
        summary = self.get_summary()
        summary["stages"] = self.get_stage_stats()
        # Part 4A: 缓存和延迟漂移
        summary["cache"] = {
            "flash_hit_rate": round(self.get_cache_hit_rate(), 4),
            "plus_hit_rate": round(self.get_plus_cache_hit_rate(), 4),
        }
        summary["ttft_drift"] = self.get_ttft_drift()
        summary["budget_degradation_count"] = self.get_budget_degradation_count()
        return summary


# ============ 工具函数 ============


def calculate_timely_success_rate(
    turn_metrics_list: list[TurnMetrics],
    latency_threshold_ms: int = 500,
    quality_scores: Optional[list[float]] = None,
    quality_threshold: float = 3.0,
) -> float:
    """计算及时成功率

    定义: (延迟 < 阈值 且 回答正确的轮次) / 总轮次

    Args:
        turn_metrics_list: 轮次指标列表
        latency_threshold_ms: 延迟阈值
        quality_scores: 质量分数列表（与轮次一一对应）
        quality_threshold: 质量分数阈值（默认 3/5）

    Returns:
        及时成功率（0-1）
    """
    if not turn_metrics_list:
        return 0.0

    # 如果没有质量分数，只考虑延迟
    if quality_scores is None:
        compliant = sum(1 for turn in turn_metrics_list if turn.is_latency_compliant(latency_threshold_ms))
        return compliant / len(turn_metrics_list)

    # 同时考虑延迟和质量
    if len(quality_scores) != len(turn_metrics_list):
        raise ValueError("quality_scores length must match turn_metrics_list")

    success_count = 0
    for turn, score in zip(turn_metrics_list, quality_scores):
        if turn.is_latency_compliant(latency_threshold_ms) and score >= quality_threshold:
            success_count += 1

    return success_count / len(turn_metrics_list)


@dataclass
class BaselineReport:
    """基线报告 — 单会话或多会话聚合"""

    session_ids: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    turn_details: list[dict] = field(default_factory=list)

    @classmethod
    def from_session(cls, session_metrics: SessionMetrics) -> "BaselineReport":
        """从单个会话构造基线报告"""
        turn_details = []
        for t in session_metrics.turn_metrics_history:
            turn_details.append(
                {
                    "turn_id": t.turn_id,
                    "e2e_ms": t.get_e2e_ms(),
                    "asr_ms": t.get_asr_ms(),
                    "llm_ms": t.get_llm_ms(),
                    "tts_ms": t.get_tts_ms(),
                    "asr_text_char_count": t.asr_text_char_count,
                    "asr_confidence": t.asr_confidence,
                    "speech_duration_ms": t.speech_duration_ms,
                    "vad_wait_ms": t.vad_wait_ms,
                    "is_short_text": t.is_short_text(),
                    "is_compliant": t.is_latency_compliant(),
                }
            )
        return cls(
            session_ids=[session_metrics.session_id],
            summary=session_metrics.get_detailed_summary(),
            turn_details=turn_details,
        )

    @classmethod
    def from_sessions(cls, sessions: list[SessionMetrics]) -> "BaselineReport":
        """从多个会话聚合基线报告"""
        all_turn_details = []
        all_session_ids = []
        all_e2e = []
        total_turns = 0
        total_interrupts = 0
        total_clarifications = 0

        for sm in sessions:
            all_session_ids.append(sm.session_id)
            total_turns += sm.total_turns
            total_interrupts += sm.total_interrupts
            total_clarifications += sm.total_clarifications
            all_e2e.extend(sm.get_e2e_latencies())
            for t in sm.turn_metrics_history:
                all_turn_details.append(
                    {
                        "session_id": sm.session_id,
                        "turn_id": t.turn_id,
                        "e2e_ms": t.get_e2e_ms(),
                        "asr_text_char_count": t.asr_text_char_count,
                        "asr_confidence": t.asr_confidence,
                        "speech_duration_ms": t.speech_duration_ms,
                        "vad_wait_ms": t.vad_wait_ms,
                        "is_short_text": t.is_short_text(),
                        "is_compliant": t.is_latency_compliant(),
                    }
                )

        # 聚合统计
        summary = {
            "session_count": len(sessions),
            "total_turns": total_turns,
            "total_interrupts": total_interrupts,
            "total_clarifications": total_clarifications,
        }
        if all_e2e:
            sorted_e2e = sorted(all_e2e)
            summary["e2e_p50_ms"] = sorted_e2e[int(len(sorted_e2e) * 0.50)]
            summary["e2e_p95_ms"] = sorted_e2e[min(int(len(sorted_e2e) * 0.95), len(sorted_e2e) - 1)]
            summary["e2e_avg_ms"] = round(sum(all_e2e) / len(all_e2e), 1)
            compliant = sum(1 for lat in all_e2e if lat <= DEFAULT_LATENCY_THRESHOLDS["e2e"])
            summary["compliance_rate"] = round(compliant / len(all_e2e), 4)

        return cls(
            session_ids=all_session_ids,
            summary=summary,
            turn_details=all_turn_details,
        )

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "session_ids": self.session_ids,
            "summary": self.summary,
            "turn_details": self.turn_details,
        }

    def save(self, path: str):
        """写入 JSON 文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
