"""TurnDetectorProcessor — VAD-triggered SmartTurn single-gate.

VAD detects 200ms silence → UserStoppedSpeaking → SmartTurn runs once (~12ms).
SmartTurn is the sole turn-completion decision maker.  ASR sentence_end events
only update the transcript text — they do NOT trigger emission.

SmartTurn decides WHEN to emit; ASR final decides WHAT text to use.
If SmartTurn fires before ASR final arrives, we wait (up to 500ms timeout).

Three-layer defence:
  1. SmartTurn + ASR final (primary: SmartTurn gates, ASR final provides text)
  2. Timeout fallback for slow ASR final (500ms)
  3. Continuation-cancel: if user resumes speaking after emission,
     cancel in-flight LLM + TTS (handles SmartTurn false-positives)

State machine:
    IDLE -> LISTENING -> COMMITTED      (SmartTurn says complete)
                      -> IDLE           (SmartTurn says incomplete)

Layered interruption during bot speech:
    Bot speaking + user speaks → hold UserStartedSpeakingFrame
    ASR interim word count >= min_words_to_interrupt → release + InterruptionFrame
    UserStoppedSpeaking with insufficient words → suppress entirely
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum, auto

from src.core.config import settings
from src.core.logging import get_logger
from src.pipeline.events import SmartTurnEmitEvent
from src.pipeline.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    FrameDirection,
    InputAudioFrame,
    InterruptionFrame,
    InterruptionPolicyFrame,
    InterruptionReason,
    InterruptionScope,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    UserTurnCompleteFrame,
    _make_id,
)
from src.pipeline.processor import FrameProcessor
from src.processors.turn_detector.audio_ring import AudioRingBuffer
from src.processors.turn_detector.backends import TurnContext
from src.session.conversation_history import ConversationHistory

logger = get_logger(__name__)

# Max silence gap (seconds) between segments before clearing cross-turn accumulation.
# Natural intra-sentence pauses are typically < 1s; gaps > 2s indicate a new thought.
# 1.5s 兼顾两端：能捕获中文口语的自然停顿（0.5-1.2s），又不会把 >1.5s 的环境噪音误认为续说。
_ACCUMULATION_MAX_GAP_S = 1.5

# Recovery timeout after continuation-cancel: if no new emission happens within
# this window, re-emit the accumulated text to prevent the pipeline from stalling.
# Inspired by LiveKit's false_interruption_timeout and Pipecat's user_turn_stop_timeout.
_CONTINUATION_RECOVERY_S = 2.0


class _State(Enum):
    IDLE = auto()
    LISTENING = auto()
    COMMITTED = auto()


class TurnDetectorProcessor(FrameProcessor):
    """VAD-triggered SmartTurn turn detection.

    Parameters
    ----------
    name:
        Processor name for logging and topology.
    engine:
        TurnDecisionEngine for shadow/combined mode (runs all backends).
    smart_turn_backend:
        SmartTurnV3Backend called directly on UserStoppedSpeaking.
    history:
        Shared ConversationHistory for providing context to backends.
    turn_metrics:
        Optional TurnMetricsCollector for dashboard.
    final_timeout_ms:
        Maximum wait (ms) for SmartTurn + ASR after UserStoppedSpeaking.
    asr_final_grace_ms:
        A2: extra bounded wait (ms) for the ASR final when the timeout
        fallback would otherwise emit UNSETTLED partial-derived text.
        0 disables (= pre-A2 behavior, rollback switch).
    aec_warmup_s:
        Duration (s) to suppress turn detection after first bot greeting.
    min_words_during_bot:
        Minimum word count required for user speech during bot output.
    """

    def __init__(
        self,
        name: str,
        *,
        engine,
        smart_turn_backend=None,
        history: ConversationHistory,
        turn_metrics=None,
        final_timeout_ms: int = 500,
        asr_final_grace_ms: int = 0,
        aec_warmup_s: float = 0.8,
        min_words_during_bot: int = 2,
    ) -> None:
        super().__init__(name)
        self._engine = engine
        self._smart_turn_backend = smart_turn_backend
        self._history = history
        self._turn_metrics = turn_metrics
        self._final_timeout_ms = final_timeout_ms
        self._asr_final_grace_ms = asr_final_grace_ms
        self._aec_warmup_s = aec_warmup_s
        self._min_words_during_bot = min_words_during_bot

        # Config
        self._backchannel_blacklist = set(settings.backchannel_blacklist)
        self._min_words_to_interrupt = settings.min_words_to_interrupt

        # Interruption hold: bot speaking → hold UserStartedSpeaking until ASR words reach threshold
        self._interruption_held = False
        self._held_speaking_frame: UserStartedSpeakingFrame | None = None
        self._held_word_count = 0

        # State
        self._state = _State.IDLE
        self._audio_ring = AudioRingBuffer()

        # Current turn text — Aliyun ASR emits per-sentence finals (text does
        # NOT accumulate across sentences), so completed sentence finals are
        # appended in arrival order instead of stored in a single slot.
        self._finalized: list[str] = []
        self._partial_text: str = ""  # latest partial of the in-flight sentence
        # True while a sentence is in flight.  Intentionally NOT bool(_partial_text):
        # an empty-text partial must still hold the emit gate — dropping this flag
        # would reintroduce the stale-final shadowing bug.
        self._partial_pending: bool = False
        self._current_turn_id: str = ""
        self._speech_start_mono: float = 0.0
        self._speech_duration_ms: float = 0.0
        # Cross-turn text accumulation (for VAD-split utterances)
        self._accumulated_text: str = ""
        # Taint flag for _accumulated_text: accumulated text seeded from
        # non-final sources (in-flight partials, partial-derived emissions)
        # must never be laundered into a "final"-labeled emission.  Set at
        # the seeding sites, cleared only together with _accumulated_text
        # (via _reset_accumulation).
        self._accumulated_tainted: bool = False
        self._last_speech_end_mono: float = 0.0
        self._turn_suppressed: bool = False  # AEC/greeting/MinWords suppressed this turn

        # SmartTurn single-gate
        self._turn_complete: bool = False
        self._smart_turn_decision: bool | None = None

        # Emit tracking for late-final diff
        self._emitted_text: str = ""
        self._emitted_text_source: str = ""
        self._last_inference_ms: float = 0.0

        # Bot speaking state
        self._bot_speaking = False

        # AEC warmup — one-shot for first greeting
        self._aec_warmup_active = False
        self._aec_first_greeting_seen = False
        self._aec_warmup_task: asyncio.Task | None = None

        # Greeting mute — set externally, cleared on BotStoppedSpeaking
        self._greeting_muted = False

        # Interruption blocking — set via InterruptionPolicyFrame from LLMProcessor
        self._interruption_blocked: bool = False
        self._blocked_transcription: str = ""
        self._blocked_turn_texts: dict[str, str] = {}
        self._blocked_turn_order: list[str] = []

        # Async tasks
        self._engine_task: asyncio.Task | None = None
        self._timeout_task: asyncio.Task | None = None
        self._recovery_task: asyncio.Task | None = None

    # ── State transitions ────────────────────────────────────────────────────

    def _transition(self, new_state: _State) -> None:
        old = self._state
        self._state = new_state
        logger.debug("[%s] %s -> %s", self.name, old.name, new_state.name)

    @property
    def _final_settled(self) -> bool:
        """At least one sentence final arrived and no newer sentence is in flight."""
        return bool(self._finalized) and not self._partial_pending

    @property
    def _segment_text(self) -> str:
        """Current segment text: completed sentence finals + in-flight partial."""
        return "".join(self._finalized) + self._partial_text

    @property
    def _emit_text_source(self) -> str:
        """'final' only for settled ASR with untainted accumulation; else 'partial'."""
        return "final" if (self._final_settled and not self._accumulated_tainted) else "partial"

    def _get_full_text(self) -> str:
        """Current segment text with accumulated prefix from prior unemitted segments."""
        return self._accumulated_text + self._segment_text

    def _reset_accumulation(self) -> None:
        """Clear accumulated text AND its taint flag — always together, so they can't drift."""
        self._accumulated_text = ""
        self._accumulated_tainted = False

    def _audit(self, event: str, **kwargs) -> None:
        if settings.polaris_asr_audit_log:
            logger.info(f"{self.name}.{event}", **kwargs)

    # ── SystemFrame handling ─────────────────────────────────────────────────

    async def handle_system_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, InterruptionPolicyFrame):
            if frame.blocked:
                self._interruption_blocked = True
                self._clear_blocked_transcription()
                logger.info("turn_detector.interruption_blocked")
            else:
                self._interruption_blocked = False
                logger.info(
                    "turn_detector.interruption_unblocked",
                    extra={"buffered_text": self._blocked_transcription[:50]},
                )
                if self._blocked_transcription.strip():
                    await self._handle_deferred_turn(self._blocked_transcription)
                self._clear_blocked_transcription()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, UserStartedSpeakingFrame):
            if self._interruption_blocked:
                self._audit("suppressed", reason="interruption_blocked_start", turn_id=frame.turn_id)
                await self.push_frame(frame, direction)
                return
            # Layered interruption: bot speaking → hold frame, wait for ASR words
            if self._bot_speaking and self._min_words_to_interrupt > 0:
                self._interruption_held = True
                self._held_speaking_frame = frame
                self._held_word_count = 0
                self._audit(
                    "interruption_held",
                    turn_id=frame.turn_id,
                    min_words=self._min_words_to_interrupt,
                    aec_warmup_active=self._aec_warmup_active,
                )
                logger.debug(
                    "[%s] Interruption held: bot speaking, waiting for %d words",
                    self.name,
                    self._min_words_to_interrupt,
                )
                await self._on_user_started_speaking(frame)
                return  # Do NOT propagate
            await self._on_user_started_speaking(frame)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, UserStoppedSpeakingFrame):
            if self._interruption_blocked:
                self._audit("suppressed", reason="interruption_blocked_stop", turn_id=frame.turn_id)
                await self.push_frame(frame, direction)
                return
            # If interruption was held and never released → suppress
            if self._interruption_held:
                self._interruption_held = False
                text = self._segment_text
                self._audit(
                    "suppressed",
                    reason="interruption_min_words",
                    turn_id=frame.turn_id,
                    words=self._held_word_count,
                    min_words=self._min_words_to_interrupt,
                    text=(text or "")[:30],
                )
                logger.debug(
                    "[%s] Interruption suppressed: words=%d < %d, text='%s'",
                    self.name,
                    self._held_word_count,
                    self._min_words_to_interrupt,
                    (text or "")[:30],
                )
                self._cancel_pending_tasks()
                self._turn_suppressed = True
                self._last_speech_end_mono = time.monotonic()
                self._transition(_State.IDLE)
                await self.push_frame(frame, direction)
                return
            await self._on_user_stopped_speaking(frame)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, BotStartedSpeakingFrame):
            await self._on_bot_started_speaking(frame)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, BotStoppedSpeakingFrame):
            await self._on_bot_stopped_speaking(frame)
            await self.push_frame(frame, direction)
            return

        await super().handle_system_frame(frame, direction)

    # ── DataFrame handling ───────────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if self._interruption_blocked:
            if isinstance(frame, TranscriptionFrame):
                self._record_blocked_transcription(frame)
                await self.push_frame(frame, direction)
                return
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, InputAudioFrame):
            self._audio_ring.append(frame.audio)
            return

        if isinstance(frame, TranscriptionFrame):
            await self._on_transcription(frame)
            return

        await self.push_frame(frame, direction)

    # ── Event handlers ───────────────────────────────────────────────────────

    async def _on_user_started_speaking(self, frame: UserStartedSpeakingFrame) -> None:
        self._cancel_pending_tasks()

        # AEC warmup: suppress all speech events (likely echo from first greeting)
        if self._aec_warmup_active:
            self._audit("suppressed", reason="aec_warmup_start", turn_id=frame.turn_id)
            logger.debug("[%s] AEC warmup: ignoring speech start (likely echo)", self.name)
            self._turn_suppressed = True
            return

        # Cross-turn accumulation: preserve text from unemitted segments
        now = time.monotonic()
        if self._turn_suppressed:
            # Suppressed (AEC/greeting/MinWords/interruption); start fresh
            self._reset_accumulation()
        elif self._state == _State.COMMITTED:
            # Previous turn was emitted — check if this is a continuation
            gap = (now - self._last_speech_end_mono) if self._last_speech_end_mono > 0 else float("inf")
            if self._emitted_text and gap <= _ACCUMULATION_MAX_GAP_S:
                # Continuation: user keeps talking shortly after emission.
                # Cancel in-flight LLM inference + TTS — the emitted partial
                # text was incomplete.  The next turn will use accumulated
                # text (this segment + new speech) for a complete response.
                self._accumulated_text = self._emitted_text
                # Fresh seed, so plain `=` is safe: COMMITTED implies
                # _emit_turn_complete already ran _reset_accumulation().
                # Partial-derived emission seeds the accumulation → taint it;
                # `!= "final"` deliberately catches "recovered" and "" too.
                self._accumulated_tainted = self._emitted_text_source != "final"
                logger.info(
                    "[%s] Continuation after emission (gap=%.0fms): '%s' — cancelling inflight",
                    self.name,
                    gap * 1000,
                    self._emitted_text[:40],
                )
                await self.push_frame(
                    InterruptionFrame(
                        reason=InterruptionReason.USER_SPEECH,
                        source="continuation_cancel",
                    ),
                    FrameDirection.DOWNSTREAM,
                )
                self._start_recovery_timer()
            else:
                self._reset_accumulation()
        elif self._last_speech_end_mono > 0 and (now - self._last_speech_end_mono) > _ACCUMULATION_MAX_GAP_S:
            # Gap too long — treat as a new utterance
            if self._accumulated_text:
                logger.debug(
                    "[%s] Accumulation cleared: gap %.1fs > %.1fs",
                    self.name,
                    now - self._last_speech_end_mono,
                    _ACCUMULATION_MAX_GAP_S,
                )
            self._reset_accumulation()
        else:
            # SmartTurn said incomplete or was cancelled — preserve this segment's text
            prev_text = self._segment_text
            if prev_text and prev_text.strip():
                self._accumulated_text += prev_text
                # Carried text includes an in-flight partial → taint.
                # Sticky across +=: never un-taint here.  Checks content,
                # not _partial_pending: an empty in-flight partial adds no text.
                if self._partial_text:
                    self._accumulated_tainted = True
                logger.info(
                    "[%s] Accumulated unemitted segment: '%s' (total='%s')",
                    self.name,
                    prev_text[:40],
                    self._accumulated_text[:80],
                )

        self._turn_suppressed = False
        self._audio_ring.reset()
        self._finalized = []
        self._partial_text = ""
        self._partial_pending = False
        self._current_turn_id = frame.turn_id
        self._speech_start_mono = now
        self._emitted_text = ""
        self._emitted_text_source = ""
        self._turn_complete = False
        self._smart_turn_decision = None

        self._transition(_State.LISTENING)

    async def _on_user_stopped_speaking(self, frame: UserStoppedSpeakingFrame) -> None:
        self._speech_duration_ms = frame.speech_duration_ms
        self._last_speech_end_mono = time.monotonic()

        # Observer: emit turn_stop_anchor (e2e start anchor)
        mono_ts_ms = time.monotonic() * 1000
        self._emit("emit_latency", self.name, "turn_stop_anchor", 0.0, self._current_turn_id, mono_ts_ms)
        if self._turn_metrics is not None and self._current_turn_id:
            self._turn_metrics.record_user_stop(
                self._current_turn_id,
                vad_tail_wait_ms=frame.silence_duration_ms,
            )

        if self._state == _State.COMMITTED:
            return
        if self._state != _State.LISTENING:
            return

        # Transcript may still arrive after VAD stop for short utterances.
        # Defer the no-transcript decision to final-timeout instead of dropping
        # the turn immediately, otherwise late ASR finals get lost.
        if not self._finalized and not self._partial_text:
            self._audit("waiting_for_transcript", turn_id=frame.turn_id, timeout_ms=self._final_timeout_ms)
            logger.debug("[%s] No transcript yet at speech stop; waiting up to timeout", self.name)

        # AEC warmup suppression
        if self._aec_warmup_active:
            self._audit("suppressed", reason="aec_warmup_stop", turn_id=frame.turn_id)
            logger.debug("[%s] AEC warmup active, suppressing", self.name)
            self._turn_suppressed = True
            self._transition(_State.IDLE)
            return

        # Greeting mute suppression
        if self._greeting_muted:
            self._audit("suppressed", reason="greeting_muted", turn_id=frame.turn_id)
            logger.debug("[%s] Greeting muted, suppressing", self.name)
            self._turn_suppressed = True
            self._transition(_State.IDLE)
            return

        # MinWords check during bot speech
        text = self._get_full_text()
        if self._bot_speaking and self._count_words(text) < self._min_words_during_bot:
            self._audit(
                "suppressed",
                reason="min_words_during_bot",
                turn_id=frame.turn_id,
                text=text[:30],
                words=self._count_words(text),
                min_words=self._min_words_during_bot,
            )
            logger.debug("[%s] MinWords filter: '%s' during bot speech", self.name, text)
            self._turn_suppressed = True
            self._transition(_State.IDLE)
            return

        # Run SmartTurn (or full engine) once, with timeout fallback
        self._engine_task = asyncio.create_task(self._run_smart_turn_check())
        self._timeout_task = asyncio.create_task(self._run_final_timeout())

    async def _on_transcription(self, frame: TranscriptionFrame) -> None:
        if frame.is_final:
            # Per-sentence final: append (Aliyun text does not accumulate
            # across sentences) and mark no sentence in flight.
            self._finalized.append(frame.text)
            self._partial_pending = False
            self._partial_text = ""
        else:
            self._partial_text = frame.text
            self._partial_pending = True

        # Continuation recovery is a no-progress watchdog, not a speech deadline.
        # Long English sentences can outlast it; new ASR must keep it alive.
        if self._recovery_task is not None and not self._recovery_task.done():
            self._start_recovery_timer()

        # Interruption hold check: release when words reach threshold
        if self._interruption_held:
            text = self._segment_text
            word_count = self._count_words(text)
            self._held_word_count = word_count
            if word_count >= self._min_words_to_interrupt:
                self._interruption_held = False
                held = self._held_speaking_frame
                self._held_speaking_frame = None
                if held is not None:
                    self._audit(
                        "interruption_released",
                        turn_id=self._current_turn_id,
                        words=word_count,
                        min_words=self._min_words_to_interrupt,
                        text=(text or "")[:30],
                    )
                    logger.info(
                        "[%s] Interruption released: words=%d >= %d, text='%s'",
                        self.name,
                        word_count,
                        self._min_words_to_interrupt,
                        (text or "")[:30],
                    )
                    await self.push_frame(held, FrameDirection.DOWNSTREAM)
                    await self.push_frame(
                        InterruptionFrame(
                            reason=InterruptionReason.USER_SPEECH,
                            source=self.name,
                            scope=InterruptionScope.TTS_ONLY,
                            turn_id=self._current_turn_id,
                        ),
                        FrameDirection.DOWNSTREAM,
                    )

        # ASR final: update text, then try emit if SmartTurn already decided.
        # sentence_end alone never triggers emission — SmartTurn must have
        # set _turn_complete first.
        if frame.is_final and self._state == _State.LISTENING:
            logger.debug("[%s] ASR final arrived: '%s'", self.name, frame.text[:40])
            if self._turn_complete:
                await self._try_emit_smart_turn()

        # COMMITTED: late final tracking + metrics (for dashboard only)
        elif self._state == _State.COMMITTED and frame.is_final:
            if self._turn_metrics is not None:
                self._turn_metrics.record_asr_final(self._current_turn_id, frame.text)

    async def _on_bot_started_speaking(self, frame: BotStartedSpeakingFrame) -> None:
        self._bot_speaking = True
        if not self._aec_first_greeting_seen:
            self._aec_first_greeting_seen = True
            self._aec_warmup_active = True
            self._aec_warmup_task = asyncio.create_task(self._run_aec_warmup())

    async def _on_bot_stopped_speaking(self, frame: BotStoppedSpeakingFrame) -> None:
        self._bot_speaking = False
        if self._greeting_muted:
            self._greeting_muted = False
            logger.debug("[%s] Greeting mute cleared", self.name)
        if self._aec_warmup_active:
            self._aec_warmup_active = False
            if self._aec_warmup_task and not self._aec_warmup_task.done():
                self._aec_warmup_task.cancel()
            self._aec_warmup_task = None

    # ── SmartTurn single-gate emit logic ─────────────────────────────────────

    async def _try_emit_smart_turn(self) -> None:
        """Emit UserTurnCompleteFrame when SmartTurn says turn is complete AND
        ASR final text is available.

        SmartTurn decides WHEN to emit; ASR final decides WHAT text to use.
        If SmartTurn fires before ASR final, we wait (timeout fallback at 500ms).
        sentence_end alone never triggers emission — only SmartTurn can.
        """
        if self._state != _State.LISTENING:
            return

        if not self._turn_complete:
            return

        # Wait for ASR final — no final at all, or a newer sentence is still
        # in flight (last event was a partial).  The 500ms timeout fallback
        # still guarantees liveness if the final never arrives.
        if not self._final_settled:
            return

        text = self._get_full_text()
        if not text:
            return

        if self._is_backchannel(text):
            self._audit("suppressed", reason="backchannel", turn_id=self._current_turn_id, text=text[:30])
            logger.debug("[%s] Backchannel '%s' suppressed", self.name, text)
            self._cancel_pending_tasks()
            self._turn_suppressed = True
            self._transition(_State.IDLE)
            return

        await self._emit_turn_complete(
            text=text,
            source="smart_turn_v3",
            confidence=0.9,
            is_preemptive=False,
            text_source=self._emit_text_source,
        )
        if self._turn_metrics is not None:
            self._turn_metrics.record_asr_final(self._current_turn_id, text)

    # ── SmartTurn check (VAD-triggered, runs once) ───────────────────────────

    async def _run_smart_turn_check(self) -> None:
        """Run SmartTurn once after UserStoppedSpeaking. Sets _turn_complete flag.

        If SmartTurn says complete, emits immediately with best available text.
        """
        try:
            context = TurnContext(
                audio_buffer=self._audio_ring.get_snapshot(),
                transcript=self._get_full_text(),
                conversation_history=self._history.get_summary_context(),
                speech_duration_ms=self._speech_duration_ms,
                silence_duration_ms=0.0,
            )

            _t0 = time.monotonic()
            if self._smart_turn_backend is not None:
                prediction = await self._smart_turn_backend.predict(context)
            else:
                prediction = await self._engine.decide(context)
            inference_ms = (time.monotonic() - _t0) * 1000
            self._last_inference_ms = inference_ms
            self._smart_turn_decision = prediction.is_complete

            self._emit(
                "emit_latency",
                self.name,
                "smart_turn_check",
                inference_ms,
                self._current_turn_id,
                time.monotonic() * 1000,
            )

            if self._state != _State.LISTENING:
                return

            self._audit(
                "smart_turn_result",
                turn_id=self._current_turn_id,
                is_complete=prediction.is_complete,
                confidence=round(prediction.confidence, 4),
                source=prediction.source,
                transcript=self._get_full_text()[:40],
            )

            if prediction.is_complete:
                self._turn_complete = True
                logger.debug("[%s] SmartTurn: complete (%.1fms)", self.name, inference_ms)
                await self._try_emit_smart_turn()
            else:
                logger.debug(
                    "[%s] SmartTurn: incomplete (%.1fms), waiting for timeout/final",
                    self.name,
                    inference_ms,
                )

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("[%s] SmartTurn check error: %s", self.name, exc)
            # Fallback: treat as incomplete so timeout can still fire
            if self._state == _State.LISTENING:
                self._turn_complete = False
                # If no timeout is running, fall back to IDLE to avoid permanent stall
                if self._timeout_task is None or self._timeout_task.done():
                    self._cancel_pending_tasks()
                    self._transition(_State.IDLE)

    async def _run_final_timeout(self) -> None:
        """Fallback: if dual-gate hasn't fired within timeout, emit with best available text.

        Covers two cases:
        - SmartTurn complete but ASR final slow → emit with partial
        - SmartTurn slow → emit with whatever text + timeout source

        A2: if the emission would use UNSETTLED (partial-derived) text and
        ``asr_final_grace_ms`` > 0, wait one bounded grace period for the ASR
        final before falling back — production measured finals missing/late in
        31-66% of turns, and a Chinese ASR partial can diverge semantically
        from the final.  Settled-but-tainted text is emitted immediately:
        waiting cannot clear taint.
        """
        try:
            await asyncio.sleep(self._final_timeout_ms / 1000.0)
            if self._state == _State.LISTENING:
                text = self._get_full_text()
                should_emit = False
                text_source = ""
                emit_source = "timeout"

                if self._smart_turn_decision is False:
                    # SmartTurn false-negative guard: only recover if ASR final arrived.
                    if self._final_settled and text:
                        should_emit = True
                        text_source = self._emit_text_source
                        emit_source = "timeout_incomplete"
                elif text:
                    should_emit = True
                    text_source = self._emit_text_source

                incomplete_waiting_for_final = self._smart_turn_decision is False and bool(text)
                if (
                    (should_emit or incomplete_waiting_for_final)
                    and not self._final_settled
                    and self._asr_final_grace_ms > 0
                ):
                    # Give a slow streaming ASR final the same bounded grace
                    # even when SmartTurn said incomplete. Dropping to IDLE
                    # here would discard its later final and leave the caller
                    # waiting indefinitely. Incomplete turns still require a final.
                    turn_id_at_grace = self._current_turn_id
                    self._audit(
                        "final_grace_wait",
                        turn_id=turn_id_at_grace,
                        grace_ms=self._asr_final_grace_ms,
                    )
                    await asyncio.sleep(self._asr_final_grace_ms / 1000.0)
                    if self._state != _State.LISTENING or self._current_turn_id != turn_id_at_grace:
                        # Turn already emitted via the final-arrival path, was
                        # suppressed, or a NEW segment started — never emit
                        # stale text into a new turn.
                        return
                    # Re-evaluate: if the final landed during grace this now
                    # correctly emits "final"; otherwise fall back to partial.
                    # A SmartTurn decision that resolved False mid-grace is
                    # deliberately NOT re-checked — liveness wins at expiry.
                    text = self._get_full_text()
                    text_source = self._emit_text_source
                    should_emit = bool(text)
                    if incomplete_waiting_for_final and self._smart_turn_decision is False:
                        should_emit = should_emit and self._final_settled
                        emit_source = "timeout_incomplete"

                if should_emit:
                    await self._emit_turn_complete(
                        text=text,
                        source=emit_source,
                        confidence=0.5,
                        is_preemptive=False,
                        text_source=text_source,
                    )
                else:
                    self._audit("suppressed", reason="no_transcript_timeout", turn_id=self._current_turn_id)
                    self._cancel_pending_tasks()
                    self._transition(_State.IDLE)
        except asyncio.CancelledError:
            pass

    async def _run_aec_warmup(self) -> None:
        """One-shot AEC warmup timer."""
        try:
            await asyncio.sleep(self._aec_warmup_s)
        except asyncio.CancelledError:
            pass
        finally:
            self._aec_warmup_active = False

    # ── Emit helpers ─────────────────────────────────────────────────────────

    async def _emit_turn_complete(
        self,
        *,
        text: str,
        source: str,
        confidence: float,
        is_preemptive: bool,
        text_source: str = "",
    ) -> None:
        logger.info(
            "[%s] EMIT turn=%s text=%r source=%s text_type=%s preemptive=%s",
            self.name,
            self._current_turn_id,
            text[:40],
            source,
            text_source,
            is_preemptive,
        )

        self._cancel_recovery_timer()
        self._emitted_text = text
        self._emitted_text_source = text_source
        self._reset_accumulation()

        if self._turn_metrics is not None:
            self._turn_metrics.record_smart_turn_emit(self._current_turn_id, self._last_inference_ms, text, text_source)

        frame = UserTurnCompleteFrame(
            text=text,
            turn_id=self._current_turn_id,
            is_preemptive=is_preemptive,
            source=source,
            confidence=confidence,
            text_source=text_source,
        )
        # Atomicity invariant: double-emit safety relies on push_frame resolving
        # without truly suspending (no awaits in its body).  If push_frame ever
        # gains a real suspension point, every caller's check→emit→COMMITTED
        # sequence becomes a race window.
        await self.push_frame(frame, FrameDirection.DOWNSTREAM)

        if self._bus:
            self._bus.emit(
                SmartTurnEmitEvent(
                    turn_id=self._current_turn_id,
                    text=text,
                    text_type=text_source,
                    inference_ms=self._last_inference_ms,
                    source=source,
                )
            )

        self._cancel_pending_tasks()
        self._transition(_State.COMMITTED)

    async def _handle_deferred_turn(self, text: str) -> None:
        """Process buffered speech from interruption-blocked period as a new turn."""
        if not text.strip():
            return
        turn_id = _make_id()
        logger.info("turn_detector.deferred_turn", extra={"text": text[:80], "turn_id": turn_id})
        await self.push_frame(
            UserTurnCompleteFrame(
                text=text.strip(),
                turn_id=turn_id,
                source="deferred_block",
                confidence=0.8,
                is_preemptive=False,
                text_source="deferred",
            ),
            FrameDirection.DOWNSTREAM,
        )

    # ── Backchannel detection ────────────────────────────────────────────────

    def _is_backchannel(self, text: str) -> bool:
        """Check if text is a backchannel during bot speech."""
        if not self._bot_speaking:
            return False
        return text.strip() in self._backchannel_blacklist

    # ── Task cancellation ────────────────────────────────────────────────────

    def _cancel_pending_tasks(self) -> None:
        self._cancel_engine()
        self._cancel_timeout()

    def _cancel_engine(self) -> None:
        if self._engine_task and not self._engine_task.done():
            self._engine_task.cancel()
        self._engine_task = None

    def _cancel_timeout(self) -> None:
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()
        self._timeout_task = None

    # ── Continuation-cancel recovery ─────────────────────────────────────────

    def _start_recovery_timer(self) -> None:
        """Start a recovery timer after continuation-cancel.

        If no new emission happens within _CONTINUATION_RECOVERY_S, re-emit
        the accumulated text so the pipeline doesn't stall permanently.
        """
        self._cancel_recovery_timer()
        self._recovery_task = asyncio.create_task(self._run_recovery_timer())

    def _cancel_recovery_timer(self) -> None:
        if self._recovery_task and not self._recovery_task.done():
            self._recovery_task.cancel()
        self._recovery_task = None

    async def _run_recovery_timer(self) -> None:
        try:
            await asyncio.sleep(_CONTINUATION_RECOVERY_S)
            # Only recover if accumulated text exists and no emission happened
            if self._accumulated_text and self._state != _State.COMMITTED:
                logger.info(
                    "[%s] Continuation recovery: re-emitting '%s' after %.1fs timeout",
                    self.name,
                    self._accumulated_text[:40],
                    _CONTINUATION_RECOVERY_S,
                )
                await self._emit_turn_complete(
                    text=self._get_full_text(),
                    source="continuation_recovery",
                    confidence=0.8,
                    is_preemptive=False,
                    text_source="recovered",
                )
        except asyncio.CancelledError:
            pass

    def _record_blocked_transcription(self, frame: TranscriptionFrame) -> None:
        """Track per-turn latest transcript during interruption block.

        Avoids naive string concatenation which produces duplicate text
        (e.g. partial "我想" + final "我想问一下" → "我想我想问一下").
        """
        turn_id = frame.turn_id or "__blocked__"
        if turn_id not in self._blocked_turn_texts:
            self._blocked_turn_order.append(turn_id)
        self._blocked_turn_texts[turn_id] = frame.text
        self._blocked_transcription = " ".join(
            text.strip() for tid in self._blocked_turn_order if (text := self._blocked_turn_texts.get(tid, "")).strip()
        )

    def _clear_blocked_transcription(self) -> None:
        self._blocked_transcription = ""
        self._blocked_turn_texts.clear()
        self._blocked_turn_order.clear()

    # ── Flush override (B1) ──────────────────────────────────────────────────

    async def on_flush(self, frame: Frame) -> None:
        """Cancel all pending turn detection tasks on interruption/cancel."""
        self._cancel_pending_tasks()
        self._cancel_recovery_timer()
        self._interruption_held = False
        self._held_speaking_frame = None
        self._reset_accumulation()
        self._turn_suppressed = False
        self._clear_blocked_transcription()
        self._transition(_State.IDLE)
        await super().on_flush(frame)

    # ── Utilities ────────────────────────────────────────────────────────────

    @staticmethod
    def _count_words(text: str) -> int:
        """Count words in text. Handles both Chinese and space-delimited text."""
        if not text:
            return 0
        parts = text.split()
        if len(parts) > 1:
            return len(parts)
        return len(text.strip())
