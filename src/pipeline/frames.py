"""Frame type system for the voice pipeline.

Every unit of data flowing through the pipeline is a Frame.  Frames are
truly immutable: frozen dataclasses with deep-frozen metadata via
MappingProxyType.  Two priority levels exist:

  SystemFrame  – priority 0, dispatched immediately (state signals, lifecycle)
  DataFrame    – priority 1, queued for processing (audio, text, LLM output)
  ControlFrame – priority 1, queued (configuration / lifecycle control)

UninterruptibleMixin marks frames that must never be discarded during an
interruption flush (e.g. EndFrame).

FrameDirection indicates the direction a frame travels in the pipeline graph:
  DOWNSTREAM (1) – source → sink (normal flow)
  UPSTREAM   (2) – sink → source (feedback / backpressure)
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

# ── Utilities ─────────────────────────────────────────────────────────────────


def _deep_freeze(value: Any) -> Any:
    """Recursively convert dicts and lists into immutable equivalents.

    dict  → MappingProxyType (keys recursed)
    list  → tuple            (items recursed)
    Everything else is returned unchanged (primitives, tuples, etc.).
    """
    if isinstance(value, MappingProxyType):
        return value  # already frozen
    if isinstance(value, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _make_id(length: int = 12) -> str:
    """Return a URL-safe random string of *length* characters."""
    return secrets.token_urlsafe(length)[:length]


# ── Direction ─────────────────────────────────────────────────────────────────


class FrameDirection(IntEnum):
    DOWNSTREAM = 1
    UPSTREAM = 2


# ── Mixin ─────────────────────────────────────────────────────────────────────


class UninterruptibleMixin:
    """Marker: this frame must never be discarded during an interruption flush."""


# ── Base frame ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True, kw_only=True)
class Frame:
    """Base class for all pipeline frames.

    Fields
    ------
    id                  12-character random URL-safe string (auto-generated)
    session_id          WebSocket / call session identifier
    turn_id             Current conversational turn identifier
    trace_id            Distributed tracing span ID
    causal_parent_ids   IDs of frames that caused this frame to be produced
    created_at_monotonic time.monotonic() at construction
    pts                 Presentation timestamp in seconds (optional)
    metadata            Arbitrary key/value bag; deep-frozen at construction
    """

    id: str = field(default_factory=lambda: _make_id(12))
    session_id: str = ""
    turn_id: str = ""
    trace_id: str = ""
    causal_parent_ids: Tuple[str, ...] = ()
    created_at_monotonic: float = field(default_factory=time.monotonic)
    pts: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Deep-freeze metadata after the frozen dataclass has been built.
        # We must use object.__setattr__ because the dataclass is frozen.
        if not self.metadata:
            return
        if not isinstance(self.metadata, MappingProxyType):
            object.__setattr__(self, "metadata", _deep_freeze(self.metadata))


# ── Abstract frame categories ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True, kw_only=True)
class SystemFrame(Frame):
    """Priority-0 frames: dispatched immediately, bypass queues."""


@dataclass(frozen=True, slots=True, kw_only=True)
class DataFrame(Frame):
    """Priority-1 frames: queued for normal pipeline processing."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ControlFrame(Frame):
    """Priority-1 frames: configuration / lifecycle control."""


# ── Audio frames ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True, kw_only=True)
class InputAudioFrame(DataFrame):
    """Raw audio captured from the user / microphone."""

    audio: bytes = b""
    sample_rate: int = 16000
    channels: int = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class OutputAudioFrame(DataFrame):
    """Audio destined for playback / transmission to the user."""

    audio: bytes = b""
    sample_rate: int = 16000
    channels: int = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class TTSAudioFrame(OutputAudioFrame):
    """TTS-synthesised audio chunk, linked to its originating text span."""

    text_span: str = ""
    context_id: str = ""
    sentence_id: str = ""
    seq: int = 0


# ── Speaking-state frames ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True, kw_only=True)
class UserStartedSpeakingFrame(SystemFrame):
    """VAD detected that the user began speaking."""


@dataclass(frozen=True, slots=True, kw_only=True)
class UserStoppedSpeakingFrame(SystemFrame):
    """VAD detected that the user stopped speaking."""

    speech_duration_ms: float = 0.0
    silence_duration_ms: float = 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class BotStartedSpeakingFrame(SystemFrame):
    """The bot has started emitting audio to the user."""


@dataclass(frozen=True, slots=True, kw_only=True)
class BotStoppedSpeakingFrame(SystemFrame):
    """The bot has finished emitting audio to the user."""

    context_id: str = ""


# ── Text frames ───────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True, kw_only=True)
class TranscriptionFrame(DataFrame):
    """ASR transcription result for a single utterance segment."""

    text: str = ""
    is_final: bool = False
    confidence: float = 1.0


@dataclass(frozen=True, slots=True, kw_only=True)
class UserTurnCompleteFrame(DataFrame):
    """Full transcription of the user's turn, after end-of-utterance detection."""

    text: str = ""
    is_preemptive: bool = False
    source: str = ""
    confidence: float = 1.0
    # ""|"final"|"partial"|"recovered"|"deferred" — non-"final" means the text
    # may diverge from what ASR would finalize.  "" = unlabeled producer;
    # consumers must treat it as non-final.  Distinct from `source`, which
    # records what triggered the emit; text_source records text provenance.
    text_source: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class LLMTextFrame(DataFrame):
    """A streaming text chunk produced by the LLM."""

    text: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class LLMResponseEndFrame(DataFrame):
    """Signals that the LLM has finished streaming its response."""

    context_id: str = ""


# ── Playback frames ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True, kw_only=True)
class PlaybackQueuedFrame(SystemFrame):
    """Notifies that a TTS chunk has been queued for playback."""

    context_id: str = ""
    text_queued: str = ""
    chars_queued: int = 0
    audio_duration_ms: float = 0.0


# ── Interruption frames ───────────────────────────────────────────────────────


class InterruptionReason(str, Enum):
    USER_SPEECH = "user_speech"
    PIPELINE_ERROR = "pipeline_error"
    TIMEOUT = "timeout"
    EXTERNAL = "external"


class InterruptionScope(str, Enum):
    ALL_DOWNSTREAM = "all_downstream"
    CURRENT_TURN = "current_turn"
    TTS_ONLY = "tts_only"
    AFTER_PROCESSOR = "after_processor"


@dataclass(frozen=True, slots=True, kw_only=True)
class InterruptionFrame(SystemFrame):
    """Signals that ongoing bot output should be flushed/cancelled."""

    reason: InterruptionReason = InterruptionReason.USER_SPEECH
    source: str = ""
    scope: InterruptionScope = InterruptionScope.ALL_DOWNSTREAM
    interrupt_id: str = field(default_factory=lambda: _make_id(8))
    after_processor: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class InterruptionPolicyFrame(SystemFrame):
    """Signals TurnDetector to block/unblock user interruptions.

    SystemFrame so it gets priority routing through handle_system_frame().
    Propagates UPSTREAM from LLMProcessor to TurnDetector.
    """

    blocked: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class CancelFrame(SystemFrame):
    """Request immediate cancellation of a pipeline operation."""

    reason: str = ""


# ── TTS playout frames ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True, kw_only=True)
class TTSPlayoutDoneFrame(Frame):
    """Signals that TTS audio generation for a context is complete.

    Pushed DOWNSTREAM by TTSDuplexProcessor after _recv_loop ends.
    OutputTransport uses it to trigger commit + BotStoppedSpeakingFrame
    on the natural completion path (no interruption, no context switch).
    """

    context_id: str = ""


# ── Lifecycle frames ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True, kw_only=True)
class StartFrame(ControlFrame):
    """Sent once at the start of a session to initialise processors."""

    sample_rate_in: int = 16000
    sample_rate_out: int = 16000
    allow_interruptions: bool = True


@dataclass(frozen=True, slots=True, kw_only=True)
class EndFrame(SystemFrame, UninterruptibleMixin):
    """Sent to cleanly shut down the pipeline; never discarded during flush."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionBootstrapFrame(ControlFrame):
    """Signals that the session is ready for the initial node greeting.

    Emitted once after outbound call connects + audio track is ready.
    LLMProcessor consumes this to trigger prepare_node_entry() for the
    initial node, generating the opening greeting without user input.

    Uses ControlFrame (not DataFrame) — this is a lifecycle signal, not data.
    """

    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class ErrorFrame(SystemFrame):
    """Carries a serialisable error report (no live exception object).

    We deliberately avoid storing the exception instance so that ErrorFrame
    remains trivially serialisable and can be safely passed across thread or
    process boundaries.
    """

    error_type: str = ""
    message: str = ""
    stack: str = ""
