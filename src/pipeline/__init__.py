"""Pipeline framework for voice AI applications."""

from src.pipeline.events import BaseEvent, EventBus
from src.pipeline.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    ControlFrame,
    DataFrame,
    EndFrame,
    ErrorFrame,
    # Base types
    Frame,
    FrameDirection,
    # Audio
    InputAudioFrame,
    # Interruption
    InterruptionFrame,
    InterruptionReason,
    InterruptionScope,
    LLMResponseEndFrame,
    LLMTextFrame,
    OutputAudioFrame,
    # Playback
    PlaybackQueuedFrame,
    # Lifecycle
    StartFrame,
    SystemFrame,
    # Text
    TranscriptionFrame,
    TTSAudioFrame,
    UninterruptibleMixin,
    # Speaking state
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    UserTurnCompleteFrame,
)
from src.pipeline.observer import LoggingObserver, PipelineObserver
from src.pipeline.pipeline import Pipeline
from src.pipeline.processor import FrameProcessor
from src.pipeline.runner import PipelineRunner

__all__ = [
    # Core
    "Frame",
    "SystemFrame",
    "DataFrame",
    "ControlFrame",
    "UninterruptibleMixin",
    "FrameDirection",
    "FrameProcessor",
    "Pipeline",
    "PipelineRunner",
    "PipelineObserver",
    "LoggingObserver",
    "BaseEvent",
    "EventBus",
    # Frame types
    "InputAudioFrame",
    "OutputAudioFrame",
    "TTSAudioFrame",
    "UserStartedSpeakingFrame",
    "UserStoppedSpeakingFrame",
    "BotStartedSpeakingFrame",
    "BotStoppedSpeakingFrame",
    "TranscriptionFrame",
    "UserTurnCompleteFrame",
    "LLMTextFrame",
    "LLMResponseEndFrame",
    "PlaybackQueuedFrame",
    "InterruptionFrame",
    "InterruptionReason",
    "InterruptionScope",
    "CancelFrame",
    "StartFrame",
    "EndFrame",
    "ErrorFrame",
]
