from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PlayoutSnapshot:
    context_id: str
    active: bool
    turn_id: str = ""
    playback_start_mono_ms: float | None = None
    playback_end_mono_ms: float | None = None
    # Total queued audio duration captured at mark_playout_done. With CosyVoice
    # duplex streaming, TTS pushes chunks into LiveKit's audio source faster
    # than realtime — `playback_end_mono_ms` reflects when generation completed,
    # not when the patient finishes hearing it. Actual playback drain ends at
    # `playback_start_mono_ms + audio_duration_ms`. Silent-rescue (ADR 0006)
    # uses this to anchor its silence timer to real playback end.
    audio_duration_ms: float | None = None
    interrupted: bool = False
    # ACTUAL audio-source playout end, set from rtc.AudioSource.wait_for_playout()
    # (not the `playback_start + audio_duration` estimate). Generation-done
    # (`active=False`) fires faster than realtime under CosyVoice duplex; the
    # patient's earpiece keeps playing until the source queue truly drains.
    # Silent-rescue (ADR 0006) starts its silence window only after `drained`.
    drained: bool = False
    drained_mono_ms: float | None = None


@dataclass(frozen=True)
class AppendDecision:
    allowed: bool
    reason: Literal["active", "unknown_context", "already_ended", "interrupted", "outside_grace"]
    gap_ms: float | None = None


class PlayoutGate:
    def __init__(self) -> None:
        self._snapshots: dict[str, PlayoutSnapshot] = {}

    def mark_started(self, context_id: str, *, turn_id: str, mono_ms: float) -> None:
        self._snapshots[context_id] = PlayoutSnapshot(
            context_id=context_id,
            turn_id=turn_id,
            active=True,
            playback_start_mono_ms=mono_ms,
        )

    def mark_playout_done(self, context_id: str, *, mono_ms: float, audio_duration_ms: float | None = None) -> None:
        current = self._snapshots.get(context_id)
        if current is None:
            return
        self._snapshots[context_id] = PlayoutSnapshot(
            context_id=context_id,
            turn_id=current.turn_id,
            active=False,
            playback_start_mono_ms=current.playback_start_mono_ms,
            playback_end_mono_ms=mono_ms,
            audio_duration_ms=audio_duration_ms,
            interrupted=current.interrupted,
        )

    def mark_interrupted(self, context_id: str, *, mono_ms: float) -> None:
        current = self._snapshots.get(context_id)
        if current is None:
            return
        self._snapshots[context_id] = PlayoutSnapshot(
            context_id=context_id,
            turn_id=current.turn_id,
            active=False,
            playback_start_mono_ms=current.playback_start_mono_ms,
            playback_end_mono_ms=mono_ms,
            audio_duration_ms=current.audio_duration_ms,
            interrupted=True,
        )

    def mark_drained(self, context_id: str, *, mono_ms: float) -> None:
        """The audio source has finished emitting every captured frame for this
        context (rtc.AudioSource.wait_for_playout returned). Preserves all other
        fields — notably `interrupted`, which silent-rescue keys off."""
        current = self._snapshots.get(context_id)
        if current is None:
            return
        self._snapshots[context_id] = PlayoutSnapshot(
            context_id=context_id,
            turn_id=current.turn_id,
            active=current.active,
            playback_start_mono_ms=current.playback_start_mono_ms,
            playback_end_mono_ms=current.playback_end_mono_ms,
            audio_duration_ms=current.audio_duration_ms,
            interrupted=current.interrupted,
            drained=True,
            drained_mono_ms=mono_ms,
        )

    def snapshot(self, context_id: str) -> PlayoutSnapshot:
        return self._snapshots.get(context_id, PlayoutSnapshot(context_id=context_id, active=False))

    def can_append(self, context_id: str, *, now_ms: float, grace_ms: int) -> AppendDecision:
        snap = self._snapshots.get(context_id)
        if snap is None:
            return AppendDecision(False, "unknown_context")
        if snap.interrupted:
            return AppendDecision(False, "interrupted")
        if snap.active:
            return AppendDecision(True, "active")
        if snap.playback_end_mono_ms is None:
            return AppendDecision(False, "already_ended")
        gap = now_ms - snap.playback_end_mono_ms
        if gap <= grace_ms:
            return AppendDecision(True, "already_ended", gap_ms=max(0, gap))
        return AppendDecision(False, "outside_grace", gap_ms=gap)
