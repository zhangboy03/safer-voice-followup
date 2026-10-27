"""Study 1 barge-in / false-interrupt classification (pure, event-driven).

Pilot (2026-06-01) 暴露了两个口径问题，本模块把判定逻辑从 test_mvp.py 抽成可单测、
可离线重算的纯函数：

1. False interrupt 过敏 (#1)：旧口径 ``false_trigger = TTSInterruptedEvent 出现`` 把
   "背景声/短确认只是声学上让旧 TTS 暂停" 误判成 "业务打断成功"。新口径区分：
   - ``clean``：没有 TTS 停止，也没有新业务 turn。
   - ``acoustic_stop``：旧 TTS 被声学打断，但没有产生新业务 turn（允许的轻微 pause）。
   - ``business_false_trigger``：非目标输入触发了一个新业务 turn（真正的误打断）。
   头条指标 ``false_trigger`` 只统计 ``business_false_trigger``。

2. True barge-in 的成败口径 (#2)：核心成功是 "停旧线 + 答新意图"，不要求一定有硬切
   ``TTSInterruptedEvent``。"opener 播完" 的锚点必须是 *可听到* 的播放结束
   （``TTSPlayoutDrainedEvent`` = AudioSource.wait_for_playout 返回），**不是** generation-done
   的 ``TTSComplete`` / ``AgentTTSDone``——后者在 CosyVoice duplex 下约提前数秒，用它会把
   ASR 资格窗口砍在真打断之前（2026-06-03 排查实锤，等同 PlayoutGate 那个已证伪的坑）。
   口径区分：
   - ``interrupted``：窗口内出现 ``TTSInterruptedEvent``（硬切，发生在生成活跃的前 ~1s 内）。
   - ``soft_pivot``：合格打断、旧音频仍在可听窗口内，没有硬切但 agent 放弃原话起了新业务
     turn——机制上 "停了旧线"。``interrupted`` 与 ``soft_pivot`` 都满足 SIR 分子的机制前置，
     最终是否计入 SIR 由 judge 在 report 层 AND "答对新意图"。
   - ``stop_missing``：合格打断、旧音频仍在播，既无硬切也无新 turn——agent 真没停（兜底失败项）。
   - ``not_asr_qualified``：旧音频仍在播，但窗口内有效 ASR 字数 < 阈值（线上逻辑下不算打断）。
   - ``no_active_tts`` / ``detection_after_tts_complete``：检测到用户说话时 opener 已可听播完，
     属注入时机 / 检测延迟的测试设计诊断项，不计入 agent 失败。

判定全部基于 ``event_schema.md`` 里的可复现事件，不手工估计、不把缺失当成 0。
"""

from __future__ import annotations

import re
from typing import Any

# canonical event_type 名（runner 已在 aggregation 层做过映射）
_TTS_FIRST = "TTSFirstAudioEvent"
_TTS_COMPLETE = "TTSCompleteEvent"
_AGENT_TTS_DONE = "AgentTTSDoneEvent"
_TTS_DRAINED = "TTSPlayoutDrainedEvent"
_TTS_INTERRUPTED = "TTSInterruptedEvent"
_USER_STARTED = "UserStartedSpeakingEvent"
_TURN_METRICS = "TurnMetricsEvent"
_ASR_EVENTS = ("AsrInterimEvent", "AsrFinalEvent")

# opener 段落 *真正可听到* 播完的标志：TTSPlayoutDrainedEvent（AudioSource.wait_for_playout
# 返回 = 患者耳机里这段音频真的放完了）。CosyVoice duplex 下生成快于实时，
# TTSComplete / AgentTTSDone 只标 *生成* 完成（约落在一句 ~6s 话的第 ~1s），用它当
# "opener 播完" 会把 ASR 资格窗口砍掉数秒——正是真打断发生的那几秒。故 drain 优先，
# 仅当 drain 缺失（旧 run / 零音频）时退回 generation-done。详见 livekit_output._spawn_drain_watch。
_OPENER_DONE_EVENTS = (_TTS_COMPLETE, _AGENT_TTS_DONE)

# true barge-in 结果
INTERRUPT_INTERRUPTED = "interrupted"
INTERRUPT_SOFT_PIVOT = "soft_pivot"
INTERRUPT_STOP_MISSING = "stop_missing"
INTERRUPT_NOT_ASR_QUALIFIED = "not_asr_qualified"
INTERRUPT_DETECTION_AFTER_TTS_COMPLETE = "detection_after_tts_complete"
INTERRUPT_NO_ACTIVE_TTS = "no_active_tts"
INTERRUPT_NO_USER_SPEECH = "no_user_speech"

# false interrupt 结果
FALSE_CLEAN = "clean"
FALSE_ACOUSTIC_STOP = "acoustic_stop"
FALSE_BUSINESS_TRIGGER = "business_false_trigger"

_NON_INTERRUPT_CHARS = re.compile(r"[\s，。！？、；：,.!?;:\"'“”‘’（）()\[\]{}<>《》\-—…]+")


def _mono(event: dict[str, Any]) -> float | None:
    value = event.get("mono_ms")
    return float(value) if isinstance(value, (int, float)) else None


def _first(events: list[dict[str, Any]], event_type: str) -> dict[str, Any] | None:
    for event in events:
        if event.get("event_type") == event_type:
            return event
    return None


def _opener_tts_complete_ms(post_opener_events: list[dict[str, Any]], opener_turn_id: str | None) -> float | None:
    """opener 回复段 *真正可听到* 播完的时刻。

    ``post_opener_events`` 从 opener 的 ``TTSFirstAudioEvent`` 之后开始（test 用 cursor
    切分），所以 opener 段的 *起点* 不在列表里，但它的 drain / TTSComplete / AgentTTSDone 在。

    口径：优先 ``TTSPlayoutDrainedEvent``（音频真放完，audible end）；缺失时才退回
    generation-done（``TTSComplete`` / ``AgentTTSDone``，约提前数秒，会低估 opener 时长）。
    优先按 ``opener_turn_id`` 匹配；若 turn_id 缺失，则退化为 "第一个 TTSFirstAudio 之前
    出现的第一条对应事件"（即仍属于切分前那一段）。
    """

    if opener_turn_id is not None:
        for event in post_opener_events:
            if event.get("event_type") == _TTS_DRAINED and event.get("turn_id") == opener_turn_id:
                return _mono(event)
        for event in post_opener_events:
            if event.get("event_type") in _OPENER_DONE_EVENTS and event.get("turn_id") == opener_turn_id:
                return _mono(event)
        return None

    # turn_id 缺失的 legacy 退化路径：drain 在 audible end（数秒后）才触发，往往晚于
    # 下一段（恢复 turn）的 TTSFirstAudio，无法用 "首个 TTS_FIRST 之前" 切出，故这里只认
    # generation-done。新 run 一律带 opener_turn_id，走上面的 drain-优先分支。
    for event in post_opener_events:
        if event.get("event_type") == _TTS_FIRST:
            # 已经进入 opener 之后的新段，opener 段无法再界定
            return None
        if event.get("event_type") in _OPENER_DONE_EVENTS:
            return _mono(event)
    return None


def _opener_tts_interrupted_ms(post_opener_events: list[dict[str, Any]], opener_turn_id: str | None) -> float | None:
    if opener_turn_id is None:
        return None
    for event in post_opener_events:
        if event.get("event_type") == _TTS_INTERRUPTED and event.get("turn_id") == opener_turn_id:
            return _mono(event)
    return None


def _new_business_turn_ids(post_opener_events: list[dict[str, Any]], opener_turn_id: str | None) -> set[str]:
    """注入后产生过 TTS 的、不同于 opener 的 turn 集合（= 新业务/恢复 turn）。"""

    turns: set[str] = set()
    for event in post_opener_events:
        if event.get("event_type") not in (_TTS_FIRST, _TTS_COMPLETE):
            continue
        turn_id = event.get("turn_id")
        if turn_id and turn_id != opener_turn_id:
            turns.add(str(turn_id))
    return turns


def _followup_turn_metrics_count(post_opener_events: list[dict[str, Any]], opener_turn_id: str | None) -> int:
    count = 0
    for event in post_opener_events:
        if event.get("event_type") != _TURN_METRICS:
            continue
        turn_id = event.get("turn_id")
        ttfa = event.get("tts_ttfa_ms", -1)
        if turn_id and turn_id != opener_turn_id and isinstance(ttfa, (int, float)) and ttfa > 0:
            count += 1
    return count


def _interrupt_char_count(text: str) -> int:
    return len(_NON_INTERRUPT_CHARS.sub("", text or ""))


def _max_interrupt_asr_chars(
    post_opener_events: list[dict[str, Any]],
    *,
    opener_done_ms: float | None,
) -> int:
    max_chars = 0
    for event in post_opener_events:
        if event.get("event_type") not in _ASR_EVENTS:
            continue
        event_ms = _mono(event)
        if opener_done_ms is not None and event_ms is not None and event_ms >= opener_done_ms:
            continue
        max_chars = max(max_chars, _interrupt_char_count(str(event.get("text", ""))))
    return max_chars


def classify_barge_in(
    post_opener_events: list[dict[str, Any]],
    *,
    mode: str,
    opener_turn_id: str | None,
    injection_mono_ms: float | None = None,
    min_interrupt_chars: int = 3,
) -> dict[str, Any]:
    """从 opener 之后的事件流判定 barge-in / false-interrupt 行为口径。

    入参 ``post_opener_events`` 应为 opener ``TTSFirstAudioEvent`` 之后的事件快照
    （与 live harness 的 cursor 一致）。返回纯判定字段，绝对时延（detection_ms 等）仍由
    调用方结合注入时间计算。
    """

    user_started_evt = _first(post_opener_events, _USER_STARTED)
    tts_interrupted_evt = _first(post_opener_events, _TTS_INTERRUPTED)
    user_started = user_started_evt is not None
    tts_interrupted = tts_interrupted_evt is not None

    first_user_ms = _mono(user_started_evt) if user_started_evt else None
    opener_done_ms = _opener_tts_complete_ms(post_opener_events, opener_turn_id)
    opener_stop_ms = opener_done_ms or _opener_tts_interrupted_ms(post_opener_events, opener_turn_id)

    # 检测到用户说话时，opener 回复是否仍在播放
    active_opener_tts = first_user_ms is not None and opener_stop_ms is not None and first_user_ms < opener_stop_ms
    opener_overlap_ms = (
        round(opener_stop_ms - first_user_ms, 1)
        if (active_opener_tts and opener_stop_ms is not None and first_user_ms is not None)
        else 0.0
    )
    active_opener_tts_at_injection = (
        injection_mono_ms is not None and opener_stop_ms is not None and injection_mono_ms < opener_stop_ms
    )
    opener_injection_overlap_ms = (
        round(opener_stop_ms - injection_mono_ms, 1)
        if (active_opener_tts_at_injection and opener_stop_ms is not None and injection_mono_ms is not None)
        else 0.0
    )

    new_turns = _new_business_turn_ids(post_opener_events, opener_turn_id)
    new_business_turn = len(new_turns) > 0
    followup_turn_metrics_count = _followup_turn_metrics_count(post_opener_events, opener_turn_id)
    max_interrupt_asr_chars = _max_interrupt_asr_chars(post_opener_events, opener_done_ms=opener_done_ms)
    asr_interrupt_qualified = tts_interrupted or max_interrupt_asr_chars >= min_interrupt_chars
    # 旧线被停下的机制信号：硬切（TTSInterruptedEvent）或软转（agent 放弃原话、起了新业务 turn）。
    # 两者都满足口径里的 "停旧线"；是否 "答新意图" 由 judge 在 report 层再 AND。
    old_line_stopped = tts_interrupted or new_business_turn

    result: dict[str, Any] = {
        "user_started": user_started,
        "tts_interrupted": tts_interrupted,
        "asr_interrupt_min_chars": min_interrupt_chars,
        "max_interrupt_asr_chars": max_interrupt_asr_chars,
        "asr_interrupt_qualified": bool(asr_interrupt_qualified),
        "active_opener_tts_at_interrupt": bool(active_opener_tts),
        "opener_tts_overlap_ms": opener_overlap_ms,
        "active_opener_tts_at_injection": bool(active_opener_tts_at_injection),
        "opener_tts_injection_overlap_ms": opener_injection_overlap_ms,
        "new_business_turn": new_business_turn,
        "new_business_turn_count": len(new_turns),
        "followup_turn_metrics_count": followup_turn_metrics_count,
        "old_line_stopped": bool(old_line_stopped),
    }

    if mode == "interrupt":
        if not user_started:
            outcome = INTERRUPT_NO_USER_SPEECH
        elif tts_interrupted:
            outcome = INTERRUPT_INTERRUPTED
        elif active_opener_tts and not asr_interrupt_qualified:
            outcome = INTERRUPT_NOT_ASR_QUALIFIED
        elif active_opener_tts and new_business_turn:
            # 合格打断、旧音频仍在播，没有硬切但 agent 放弃原话起了新 turn = 软转打断。
            # 机制上 "停了旧线"；最终 SIR 分子还要 judge 判 "答对新意图"。
            outcome = INTERRUPT_SOFT_PIVOT
        elif active_opener_tts:
            # 合格打断、旧音频仍在播、既无硬切也无新 turn = agent 真的没停（兜底失败项）。
            outcome = INTERRUPT_STOP_MISSING
        elif active_opener_tts_at_injection:
            outcome = INTERRUPT_DETECTION_AFTER_TTS_COMPLETE
        else:
            outcome = INTERRUPT_NO_ACTIVE_TTS
        result["interrupt_outcome"] = outcome
        # behavior_pass = 机制层 "停了旧线"（硬切或软转）；judge 的 "答新意图" 在 report 层 AND。
        # no_active_tts / not_asr_qualified / detection_after_tts_complete / stop_missing 单独诊断。
        result["behavior_pass"] = outcome in (INTERRUPT_INTERRUPTED, INTERRUPT_SOFT_PIVOT)
        # 保持 false_trigger 字段存在（true barge-in 永远 False），便于统一 schema
        result["false_trigger"] = False
    else:
        if new_business_turn:
            outcome = FALSE_BUSINESS_TRIGGER
        elif tts_interrupted:
            outcome = FALSE_ACOUSTIC_STOP
        else:
            outcome = FALSE_CLEAN
        result["false_interrupt_outcome"] = outcome
        result["false_trigger"] = outcome == FALSE_BUSINESS_TRIGGER
        # clean / acoustic_stop 都算可接受（允许轻微声学 pause，不产生新业务 turn）
        result["behavior_pass"] = outcome != FALSE_BUSINESS_TRIGGER

    return result
