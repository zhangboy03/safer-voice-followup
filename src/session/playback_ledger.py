"""PlaybackLedger: playout-side text accounting for a voice AI pipeline.

Tracks what text was sent to TTS, how much audio was received from TTS, and
how much audio was actually played out to the user.  Provides the single
authoritative commit entry point for assistant history.

Typical call sequence
---------------------
1. TTSDuplexProcessor._writer_task calls register_text() for each chunk sent.
2. TTSDuplexProcessor._recv_loop calls register_audio_received() for each
   audio chunk returned by TTS.
3. OutputTransport.capture_frame calls register_audio_played() for each frame
   written to the speaker.
4. OutputTransport (or a pipeline observer) calls commit_to_history() once
   playout ends — passing interrupted=True when a barge-in cut it short.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src.session.conversation_history import ConversationHistory
from src.session.playout_gate import PlayoutGate


@dataclass
class _ContextState:
    text_sent: str = ""
    audio_bytes_received: int = 0
    audio_bytes_played: int = 0
    audio_duration_played_ms: float = 0.0


class PlaybackLedger:
    """Runtime source of truth for what the user actually heard.

    Args:
        history: Shared :class:`ConversationHistory` instance.  All history
            mutations go exclusively through :meth:`commit_to_history`.
    """

    def __init__(
        self,
        history: ConversationHistory,
        node_id_resolver: Callable[[str], str] | None = None,
        playout_gate: PlayoutGate | None = None,
    ) -> None:
        self._history = history
        self._contexts: dict[str, _ContextState] = {}
        self._node_id_resolver = node_id_resolver or (lambda _ctx_id: "")
        self._playout_gate = playout_gate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, context_id: str) -> _ContextState:
        if context_id not in self._contexts:
            self._contexts[context_id] = _ContextState()
        return self._contexts[context_id]

    # ------------------------------------------------------------------
    # Registration (called during active playout)
    # ------------------------------------------------------------------

    def register_text(self, context_id: str, text: str) -> None:
        """Called by _writer_task after a successful send to TTS.

        Multiple calls for the same context accumulate (concatenate) text.
        """
        ctx = self._get_or_create(context_id)
        ctx.text_sent += text

    def register_audio_received(self, context_id: str, audio_bytes: int) -> None:
        """Called by _recv_loop for each audio chunk returned by TTS.

        Cumulative: each call adds to the total bytes received.
        """
        ctx = self._get_or_create(context_id)
        ctx.audio_bytes_received += audio_bytes

    def register_audio_played(self, context_id: str, audio_bytes: int, duration_ms: float) -> None:
        """Called by OutputTransport after capture_frame.

        Cumulative: each call adds to the running totals.  This means callers
        pass the bytes / duration of the *current* frame, not the running sum.
        """
        ctx = self._get_or_create(context_id)
        ctx.audio_bytes_played += audio_bytes
        ctx.audio_duration_played_ms += duration_ms

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_full_text(self, context_id: str) -> str:
        """Return all text sent to TTS for this context, or '' if unknown."""
        ctx = self._contexts.get(context_id)
        if ctx is None:
            return ""
        return ctx.text_sent

    def get_heard_text(self, context_id: str) -> str:
        """Approximate prefix of text_sent the user actually heard.

        Estimation method: ratio = audio_bytes_played / audio_bytes_received.
        Heard text = text_sent[:round(ratio * len(text_sent))].

        Returns '' when:
        - context_id is unknown
        - no audio has been received yet (avoids division-by-zero)
        """
        ctx = self._contexts.get(context_id)
        if ctx is None:
            return ""
        if ctx.audio_bytes_received == 0:
            return ""
        ratio = ctx.audio_bytes_played / ctx.audio_bytes_received
        # Clamp to [0, 1] so over-run audio doesn't exceed the text length.
        ratio = min(1.0, max(0.0, ratio))
        chars = round(ratio * len(ctx.text_sent))
        return ctx.text_sent[:chars]

    # ------------------------------------------------------------------
    # History commit
    # ------------------------------------------------------------------

    def commit_to_history(self, context_id: str, interrupted: bool) -> None:
        """THE ONLY commit entry point for assistant history.

        Normal (interrupted=False): commits full text_sent.
        Interrupted (interrupted=True): commits the heard portion only.

        If the context is unknown or text is empty, this is a no-op to avoid
        inserting empty assistant messages.
        """
        ctx = self._contexts.get(context_id)
        if ctx is None:
            return

        if interrupted:
            text = self.get_heard_text(context_id)
        else:
            text = ctx.text_sent

        if not text:
            return

        node_id = self._node_id_resolver(context_id)
        self._history.append_assistant(text, node_id=node_id)
