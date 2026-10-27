"""VADProcessor: voice activity detection with pre-roll ring buffer.

Wraps any VADBackend and emits UserStartedSpeakingFrame / UserStoppedSpeakingFrame.
While not speaking, incoming audio is buffered in a ring buffer (max pre_roll_ms).
On speech start the buffered audio is replayed downstream with the new turn_id
injected so the STT processor receives complete context from before the onset.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from src.core.config import settings
from src.core.logging import get_logger
from src.pipeline.events import (
    UserStartedSpeakingEvent,
    UserStoppedSpeakingEvent,
    VADShadowFrameEvent,
    VADShadowSpeechEvent,
)
from src.pipeline.frames import (
    BotStoppedSpeakingFrame,
    Frame,
    FrameDirection,
    InputAudioFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from src.pipeline.processor import FrameProcessor

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class VADResult:
    is_speech: bool
    energy: float
    speech_duration_ms: float
    silence_duration_ms: float
    source: str = ""
    speech_prob: float | None = None
    threshold: float | None = None


class VADBackend(Protocol):
    def process(self, audio: bytes, sample_rate: int) -> VADResult: ...
    def reset(self) -> None: ...


class VADProcessor(FrameProcessor):
    """Detects voice activity and enriches audio frames with turn_id.

    Parameters
    ----------
    name:
        Processor name (used in tracing / logs).
    primary_backend:
        The VADBackend whose result drives UserStarted/UserStopped frames.
    shadow_backends:
        Optional list of additional VADBackends run per-frame for observability;
        their results are emitted to the bus only — never to the frame stream.
    speech_start_threshold_ms:
        Minimum consecutive speech duration before emitting UserStartedSpeakingFrame.
    speech_end_threshold_ms:
        Minimum consecutive silence duration before emitting UserStoppedSpeakingFrame.
    pre_roll_ms:
        How many milliseconds of audio to cache before speech start.
        These frames are replayed downstream (with turn_id) when speech is detected.
    enroll_pre_roll_ms:
        Pre-roll bytes passed to each shadow.on_speech_started for enrollment.
    """

    def __init__(
        self,
        name: str,
        *,
        primary_backend: VADBackend,
        shadow_backends: list[VADBackend] | None = None,
        speech_start_threshold_ms: int = 100,
        speech_end_threshold_ms: int = 200,
        pre_roll_ms: int = 200,
        enroll_pre_roll_ms: int = 100,
        pvad_gates_primary: bool = False,
    ) -> None:
        super().__init__(name)
        self._vad = primary_backend
        self._shadows: list[VADBackend] = list(shadow_backends or [])
        self._start_threshold = speech_start_threshold_ms
        self._end_threshold = speech_end_threshold_ms
        self._pre_roll_ms = pre_roll_ms
        self._enroll_pre_roll_ms = enroll_pre_roll_ms
        self._pvad_gates_primary = pvad_gates_primary
        self._is_speaking = False
        self._pending_pvad_confirm = False
        self._speech_start_mono = 0.0
        self._current_turn_id = ""
        self._ring_buffer: deque[InputAudioFrame] = deque()
        self._ring_buffer_duration_ms = 0.0
        self._audit_seq = 0
        self._last_result: VADResult | None = None
        self._first_bot_speech_done = False
        self._shadow_frame_sample_every = 50  # emit every N frames + on transition
        self._shadow_frame_counter = 0
        # Per-shadow tracking, keyed by id(shadow)
        self._shadow_prev_is_speech: dict[int, bool] = {}
        self._shadow_speech_started_mono: dict[int, float] = {}
        self._shadow_prev_state: dict[int, str] = {}

    # ── Frame helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _frame_duration_ms(frame: InputAudioFrame) -> float:
        """Return the audio duration in milliseconds for an InputAudioFrame."""
        # PCM 16-bit: 2 bytes per sample per channel
        samples = len(frame.audio) / (2 * frame.channels)
        return (samples / frame.sample_rate) * 1000.0

    def _enrich(self, frame: InputAudioFrame, turn_id: str) -> InputAudioFrame:
        """Return a copy of *frame* with turn_id set."""
        return InputAudioFrame(
            audio=frame.audio,
            sample_rate=frame.sample_rate,
            channels=frame.channels,
            session_id=frame.session_id,
            turn_id=turn_id,
        )

    # ── Ring buffer management ────────────────────────────────────────────────

    def _buffer_frame(self, frame: InputAudioFrame) -> None:
        """Add *frame* to the ring buffer, evicting oldest frames if over budget."""
        self._ring_buffer.append(frame)
        self._ring_buffer_duration_ms += self._frame_duration_ms(frame)
        while self._ring_buffer_duration_ms > self._pre_roll_ms and self._ring_buffer:
            evicted = self._ring_buffer.popleft()
            self._ring_buffer_duration_ms -= self._frame_duration_ms(evicted)

    def _clear_ring_buffer(self) -> None:
        self._ring_buffer.clear()
        self._ring_buffer_duration_ms = 0.0

    # ── System frame handling ─────────────────────────────────────────────────

    async def handle_system_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, BotStoppedSpeakingFrame) and not self._first_bot_speech_done:
            self._first_bot_speech_done = True
            for shadow in self._shadows:
                fn = getattr(shadow, "on_greeting_finished", None)
                if fn:
                    try:
                        fn()
                    except Exception as exc:
                        logger.warning(
                            "vad.shadow.greeting_callback_failed",
                            backend=type(shadow).__name__,
                            error=str(exc),
                        )
        await super().handle_system_frame(frame, direction)

    # ── Main processing ───────────────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if not isinstance(frame, InputAudioFrame):
            await self.push_frame(frame, direction)
            return

        result = self._vad.process(frame.audio, frame.sample_rate)
        self._last_result = result
        self._audit_seq += 1
        self._maybe_audit_probe(frame, result)

        # Run shadows isolated from main path
        shadow_results: list[tuple[VADBackend, VADResult | None]] = []
        for shadow in self._shadows:
            try:
                sres = shadow.process(frame.audio, frame.sample_rate)
                shadow_results.append((shadow, sres))
                self._track_shadow_transition(
                    shadow,
                    sres,
                    primary_is_speech=result.is_speech,
                )
            except Exception as exc:
                logger.warning(
                    "vad.shadow.process_failed",
                    backend=type(shadow).__name__,
                    error=str(exc),
                )
                shadow_results.append((shadow, None))
        self._emit_shadow_frames(shadow_results, primary_is_speech=result.is_speech)

        if not self._is_speaking:
            # Always buffer while silent (ring buffer enforces size limit)
            self._buffer_frame(frame)

            if self._pending_pvad_confirm:
                if not result.is_speech:
                    self._pending_pvad_confirm = False
                    self._audit_event(
                        "vad.pvad_gate.cancel_on_energy_stop",
                        silence_duration_ms=round(result.silence_duration_ms, 1),
                    )
                elif self._pvad_gate_allows_start(shadow_results):
                    self._pending_pvad_confirm = False
                    self._audit_event(
                        "vad.pvad_gate.release",
                        speech_duration_ms=round(result.speech_duration_ms, 1),
                    )
                    await self._on_speech_start(frame)
            elif result.is_speech and result.speech_duration_ms >= self._start_threshold:
                if self._pvad_gate_allows_start(shadow_results):
                    await self._on_speech_start(frame)
                else:
                    self._pending_pvad_confirm = True
                    self._audit_event(
                        "vad.pvad_gate.hold",
                        speech_duration_ms=round(result.speech_duration_ms, 1),
                    )
        elif self._is_speaking:
            if not result.is_speech and result.silence_duration_ms >= self._end_threshold:
                await self._on_speech_end(frame)
            else:
                # Still speaking — forward enriched frame
                await self.push_frame(
                    self._enrich(frame, self._current_turn_id),
                    FrameDirection.DOWNSTREAM,
                )

    async def _on_speech_start(self, trigger_frame: InputAudioFrame) -> None:
        """Handle transition from silence → speech."""
        self._is_speaking = True
        self._speech_start_mono = time.monotonic()
        self._current_turn_id = uuid4().hex[:12]
        self._audit_event(
            "vad.speech_start",
            turn_id=self._current_turn_id,
            energy=self._round(getattr(self._last_result, "energy", None) if self._last_result else None),
            speech_prob=self._round(getattr(self._last_result, "speech_prob", None) if self._last_result else None),
            threshold=self._round(getattr(self._last_result, "threshold", None) if self._last_result else None),
            source=getattr(self._last_result, "source", "") if self._last_result else "",
            pre_roll_ms=round(self._ring_buffer_duration_ms, 1),
            frame_sample_rate=trigger_frame.sample_rate,
        )

        # Emit speech-start signal
        await self.push_frame(
            UserStartedSpeakingFrame(
                session_id=trigger_frame.session_id,
                turn_id=self._current_turn_id,
            ),
            FrameDirection.DOWNSTREAM,
        )
        if self._bus:
            self._bus.emit(UserStartedSpeakingEvent(turn_id=self._current_turn_id))

        # Notify shadow enrollment hooks (NOT emitting shadow speech events —
        # those are tracked per-shadow against shadow's own is_speech transitions,
        # see _track_shadow_transition).
        pre_roll_for_enroll = self._slice_last_ms_bytes(self._enroll_pre_roll_ms)
        for shadow in self._shadows:
            fn = getattr(shadow, "on_speech_started", None)
            if fn:
                try:
                    fn(pre_roll_for_enroll)
                except Exception as exc:
                    logger.warning(
                        "vad.shadow.speech_started_callback_failed",
                        backend=type(shadow).__name__,
                        error=str(exc),
                    )

        # Replay pre-roll buffer (trigger_frame was already buffered)
        for buffered in self._ring_buffer:
            await self.push_frame(
                self._enrich(buffered, self._current_turn_id),
                FrameDirection.DOWNSTREAM,
            )
        self._clear_ring_buffer()

    async def _on_speech_end(self, silence_frame: InputAudioFrame) -> None:
        """Handle transition from speech → silence."""
        duration_ms = (time.monotonic() - self._speech_start_mono) * 1000.0
        silence_duration_ms = self._last_result.silence_duration_ms if self._last_result is not None else 0.0
        self._audit_event(
            "vad.speech_end",
            turn_id=self._current_turn_id,
            speech_duration_ms=round(duration_ms, 1),
            energy=self._round(getattr(self._last_result, "energy", None) if self._last_result else None),
            speech_prob=self._round(getattr(self._last_result, "speech_prob", None) if self._last_result else None),
            threshold=self._round(getattr(self._last_result, "threshold", None) if self._last_result else None),
            source=getattr(self._last_result, "source", "") if self._last_result else "",
            frame_sample_rate=silence_frame.sample_rate,
        )
        self._is_speaking = False
        await self.push_frame(
            UserStoppedSpeakingFrame(
                session_id=silence_frame.session_id,
                turn_id=self._current_turn_id,
                speech_duration_ms=duration_ms,
                silence_duration_ms=silence_duration_ms,
            ),
            FrameDirection.DOWNSTREAM,
        )
        if self._bus:
            self._bus.emit(
                UserStoppedSpeakingEvent(
                    turn_id=self._current_turn_id,
                    silence_duration_ms=silence_duration_ms,
                )
            )
        for shadow in self._shadows:
            fn = getattr(shadow, "on_speech_stopped", None)
            if fn:
                try:
                    fn()
                except Exception as exc:
                    logger.warning(
                        "vad.shadow.speech_stopped_callback_failed",
                        backend=type(shadow).__name__,
                        error=str(exc),
                    )

    # ── Flush / reset ─────────────────────────────────────────────────────────

    async def on_flush(self, frame: Frame) -> None:
        self._is_speaking = False
        self._pending_pvad_confirm = False
        self._current_turn_id = ""
        self._clear_ring_buffer()
        self._vad.reset()
        # Per-session reset: greeting gate, transition tracking, frame sampler.
        # Without this, a reused processor never re-fires on_greeting_finished
        # and stale prev_is_speech suppresses the first transition of the new session.
        self._first_bot_speech_done = False
        self._shadow_prev_is_speech.clear()
        self._shadow_speech_started_mono.clear()
        self._shadow_prev_state.clear()
        self._shadow_frame_counter = 0
        for shadow in self._shadows:
            try:
                shadow.reset()
            except Exception as exc:
                logger.warning(
                    "vad.shadow.reset_failed",
                    backend=type(shadow).__name__,
                    error=str(exc),
                )
        self._flush_mailbox()

    def _pvad_gate_allows_start(
        self,
        shadow_results: list[tuple[VADBackend, "VADResult | None"]],
    ) -> bool:
        """AND-gate: if any pvad-style shadow is in READY state, require its
        is_speech=True to allow emitting UserStartedSpeakingFrame.

        Pass-through (return True) when:
        - gating disabled by flag
        - no shadow has reached READY (state != "ready", e.g. enrollment ongoing)
        - shadow.process() failed this frame (sres is None)
        """
        if not self._pvad_gates_primary:
            return True
        for _shadow, sres in shadow_results:
            if sres is None:
                continue
            if getattr(sres, "source", "") != "ready":
                continue
            if not sres.is_speech:
                return False
        return True

    def _emit_shadow_frames(
        self,
        results: list[tuple[VADBackend, "VADResult | None"]],
        *,
        primary_is_speech: bool,
    ) -> None:
        if not self._bus or not results:
            return

        self._shadow_frame_counter += 1
        for shadow, sres in results:
            backend = getattr(shadow, "backend_name", type(shadow).__name__.lower())
            state = getattr(sres, "source", "unknown") if sres else "unavailable"
            speech_prob = getattr(sres, "speech_prob", None) if sres else None
            threshold = getattr(sres, "threshold", None) if sres else None
            is_speech = bool(sres.is_speech) if sres else False
            enroll_progress = getattr(shadow, "enroll_progress_ms", None)
            # Sample every N frames OR on transition (state change)
            prev = self._shadow_prev_state.get(id(shadow))
            transition = prev != state
            self._shadow_prev_state[id(shadow)] = state
            if not transition and (self._shadow_frame_counter % self._shadow_frame_sample_every) != 0:
                continue
            self._bus.emit(
                VADShadowFrameEvent(
                    session_id=self._current_turn_id or "",
                    backend=backend,
                    state=state,
                    is_speech=is_speech,
                    speech_prob=speech_prob,
                    threshold=threshold,
                    primary_is_speech=primary_is_speech,
                    enroll_progress_ms=enroll_progress,
                    ts_mono_ms=time.monotonic() * 1000.0,
                )
            )

    def _track_shadow_transition(
        self,
        shadow: VADBackend,
        sres: "VADResult | None",
        *,
        primary_is_speech: bool,
    ) -> None:
        """Emit VADShadowSpeechEvent when this shadow's is_speech flips.

        Only the READY state has meaningful is_speech — during enrollment,
        is_speech is always False and must NOT trigger transitions.
        """
        if not self._bus or sres is None:
            return
        # Gate on the shadow being in a "decidable" state (e.g. PVAD READY).
        state = getattr(sres, "source", "")
        if state != "ready":
            self._shadow_prev_is_speech[id(shadow)] = False
            return
        prev = self._shadow_prev_is_speech.get(id(shadow), False)
        curr = bool(sres.is_speech)
        if prev == curr:
            return
        self._shadow_prev_is_speech[id(shadow)] = curr
        now = time.monotonic() * 1000.0
        backend = getattr(shadow, "backend_name", type(shadow).__name__.lower())
        # event_type_ trailing underscore avoids collision with BaseEvent.event_type
        if curr:
            # 0 → 1: started
            self._shadow_speech_started_mono[id(shadow)] = now
            primary_marker = "started" if primary_is_speech else None
            self._bus.emit(
                VADShadowSpeechEvent(
                    session_id=self._current_turn_id or "",
                    backend=backend,
                    event_type_="started",
                    primary_event_type=primary_marker,
                    ts_mono_ms=now,
                    speech_duration_ms=None,
                )
            )
        else:
            # 1 → 0: stopped
            started = self._shadow_speech_started_mono.pop(id(shadow), now)
            duration = now - started
            primary_marker = "stopped" if not primary_is_speech else None
            self._bus.emit(
                VADShadowSpeechEvent(
                    session_id=self._current_turn_id or "",
                    backend=backend,
                    event_type_="stopped",
                    primary_event_type=primary_marker,
                    ts_mono_ms=now,
                    speech_duration_ms=duration,
                )
            )

    def _slice_last_ms_bytes(self, want_ms: int) -> bytes:
        """Return the last *want_ms* of audio from the ring buffer, oldest->newest."""
        if want_ms <= 0 or not self._ring_buffer:
            return b""
        acc: list[bytes] = []
        kept_ms = 0.0
        for frame in reversed(self._ring_buffer):
            dur = self._frame_duration_ms(frame)
            acc.append(frame.audio)
            kept_ms += dur
            if kept_ms >= want_ms:
                break
        acc.reverse()
        return b"".join(acc)

    def _maybe_audit_probe(self, frame: InputAudioFrame, result: VADResult) -> None:
        if not settings.polaris_asr_audit_log:
            return
        if not (
            self._audit_seq <= 5
            or self._audit_seq % 50 == 0
            or (not self._is_speaking and result.is_speech)
            or (self._is_speaking and not result.is_speech)
        ):
            return
        self._audit_event(
            "vad.frame_probe",
            seq=self._audit_seq,
            turn_id=self._current_turn_id,
            speaking=self._is_speaking,
            is_speech=result.is_speech,
            energy=self._round(result.energy),
            speech_prob=self._round(getattr(result, "speech_prob", None)),
            threshold=self._round(getattr(result, "threshold", None)),
            source=getattr(result, "source", ""),
            speech_ms=round(result.speech_duration_ms, 1),
            silence_ms=round(result.silence_duration_ms, 1),
            frame_sample_rate=frame.sample_rate,
        )

    def _audit_event(self, event: str, **kwargs) -> None:
        if settings.polaris_asr_audit_log:
            logger.info(event, **kwargs)

    @staticmethod
    def _round(value: float | None) -> float | None:
        if value is None:
            return None
        return round(value, 4)
