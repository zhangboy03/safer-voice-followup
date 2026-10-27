"""会话管理模块"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from uuid import uuid4

from src.core.metrics import SessionMetrics, TurnMetrics


@dataclass
class InterruptContext:
    """打断恢复上下文 — 记录被打断时的播报状态"""

    full_response: str  # 系统原本要说的完整内容
    played_text: str  # 已播报部分（基于 played_chars 截取）
    truncated_text: str  # 被截断部分


@dataclass
class TTSState:
    """TTS 播放状态"""

    is_playing: bool = False
    current_text: str = ""
    total_chars: int = 0
    sent_audio_ms: int = 0
    chars_per_second: float = 5.0  # 平均语速（字/秒）

    @property
    def estimated_played_chars(self) -> int:
        """估算已播放的字符数"""
        if not self.is_playing:
            return self.total_chars
        played = int(self.sent_audio_ms / 1000 * self.chars_per_second)
        return min(played, self.total_chars)

    def add_audio_chunk(self, pcm_bytes: bytes, sample_rate: int = 16000) -> None:
        """添加音频块，更新已发送时长"""
        samples = len(pcm_bytes) // 2  # 16bit = 2 bytes per sample
        self.sent_audio_ms += int(samples * 1000 / sample_rate)

    def start_playback(self, text: str) -> None:
        """开始播放"""
        self.is_playing = True
        self.current_text = text
        self.total_chars = len(text)
        self.sent_audio_ms = 0

    def stop_playback(self) -> int:
        """停止播放，返回已播放字符数"""
        played = self.estimated_played_chars
        self.is_playing = False
        self.current_text = ""
        self.total_chars = 0
        self.sent_audio_ms = 0
        return played

    def reset(self) -> None:
        """重置状态"""
        self.is_playing = False
        self.current_text = ""
        self.total_chars = 0
        self.sent_audio_ms = 0


@dataclass
class Session:
    """会话实例"""

    session_id: str
    patient_id: str

    # 状态
    is_active: bool = True
    current_state: str = "idle"  # idle, listening, processing, speaking

    # 当前轮次
    current_turn_id: str | None = None
    turn_start_time: float | None = None
    current_transcript: str = ""

    # 累积文本（用于 transcript 消息）
    accumulated_text: str = ""

    # TTS 状态
    tts_state: TTSState = field(default_factory=TTSState)

    # 打断恢复上下文（最近一次打断时的播报状态）
    last_interrupt_context: InterruptContext | None = None

    # 指标（__post_init__ 保证非 None）
    session_metrics: SessionMetrics = field(init=False)
    current_turn_metrics: TurnMetrics | None = None

    def __post_init__(self) -> None:
        self.session_metrics = SessionMetrics(session_id=self.session_id)

    def on_local_speech_start(self, ts: float) -> str:
        """本地 VAD 检测到语音开始，生成 turn_id"""
        self.current_turn_id = f"turn-{int(ts * 1000)}-{uuid4().hex[:6]}"
        self.turn_start_time = time.monotonic()
        self.current_transcript = ""
        self.accumulated_text = ""
        self.current_turn_metrics = TurnMetrics(turn_id=self.current_turn_id)
        return self.current_turn_id

    def on_speech_end(self, ts: float) -> None:
        """用户停止说话（静音开始，VAD 等待前）

        这是 ASR 和 E2E 延迟计算的起点，包含 VAD 的等待时间。
        """
        if self.current_turn_metrics:
            self.current_turn_metrics.speech_end_time = ts
            # 计算发话时长
            if self.turn_start_time:
                self.current_turn_metrics.speech_duration_ms = int((ts - self.turn_start_time) * 1000)

    def on_asr_final(
        self,
        text: str,
        ts: float,
        confidence: float | None = None,
        begin_time_ms: int | None = None,
    ) -> None:
        """ASR 最终结果"""
        _ = begin_time_ms
        self.current_transcript = text
        if self.current_turn_metrics:
            self.current_turn_metrics.asr_final_time = ts
            self.current_turn_metrics.asr_confidence = confidence
            # 去标点空格后的字符数
            import re

            clean_text = re.sub(r"[，。？！,.?!\s]", "", text)
            self.current_turn_metrics.asr_text_char_count = len(clean_text)

    def on_rag_start(self, ts: float) -> None:
        """RAG 查询开始"""
        if self.current_turn_metrics:
            self.current_turn_metrics.rag_start_time = ts

    def on_rag_end(self, ts: float) -> None:
        """RAG 查询结束"""
        if self.current_turn_metrics:
            self.current_turn_metrics.rag_end_time = ts

    def on_safety_check_end(self, ts: float) -> None:
        """安全+转人工检测完成"""
        if self.current_turn_metrics:
            self.current_turn_metrics.safety_check_end_time = ts

    def on_analysis_start(self, ts: float) -> None:
        """SOP分析+RAG 并行任务开始"""
        if self.current_turn_metrics:
            self.current_turn_metrics.analysis_start_time = ts

    def on_analysis_end(self, ts: float) -> None:
        """SOP分析+RAG 并行任务全部完成"""
        if self.current_turn_metrics:
            self.current_turn_metrics.analysis_end_time = ts

    def on_llm_call_start(self, ts: float) -> None:
        """LLM 流式调用发起"""
        if self.current_turn_metrics:
            self.current_turn_metrics.llm_call_start_time = ts

    def on_llm_first_token(self, ts: float) -> None:
        """LLM 返回第一个 token"""
        if self.current_turn_metrics:
            self.current_turn_metrics.llm_first_token_time = ts

    def on_llm_first_sentence(self, ts: float) -> None:
        """LLM 返回第一个完整分句"""
        if self.current_turn_metrics:
            self.current_turn_metrics.llm_first_sentence_time = ts

    def on_tts_first_audio(self, ts: float) -> None:
        """TTS 返回第一个音频块（服务端收到）"""
        if self.current_turn_metrics:
            self.current_turn_metrics.tts_first_audio_time = ts

    def on_tts_play(self, ts: float) -> None:
        """TTS 首字节发送到前端（开始播放）"""
        if self.current_turn_metrics:
            self.current_turn_metrics.tts_play_time = ts

    def complete_turn(self) -> None:
        """完成当前轮次"""
        if self.current_turn_metrics:
            self.session_metrics.add_turn(self.current_turn_metrics)
        self.current_turn_id = None
        self.turn_start_time = None
        self.current_turn_metrics = None

    def record_interrupt(self) -> None:
        """记录打断"""
        self.session_metrics.record_interrupt()

    def record_clarification(self) -> None:
        """记录追问/澄清"""
        self.session_metrics.record_clarification()

    def set_state(self, state: str) -> None:
        """设置会话状态"""
        self.current_state = state

    def end_session(self) -> None:
        """结束会话"""
        self.is_active = False
        self.tts_state.reset()


def create_session(patient_id: str) -> Session:
    """创建新会话"""
    session_id = f"session-{int(time.time() * 1000)}-{uuid4().hex[:8]}"
    return Session(session_id=session_id, patient_id=patient_id)
