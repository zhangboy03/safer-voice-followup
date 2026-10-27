"""Session-scoped conversation history with provisional/commit/rollback lifecycle.

Shared between TurnDetectorProcessor (read-only) and LLMProcessor (read-write).
All mutations are synchronous; callers are responsible for concurrency gating.

C-stage: internal storage migrated from flat list[dict] to list[HistoryEntry]
(turn journal model).  External API is backward-compatible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from src.flows.models import EntryType, HistoryEntry


@dataclass
class ConversationHistory:
    """Manages the OpenAI-format message list for a single call session.

    State model
    -----------
    _journal            : list of HistoryEntry — the durable turn journal.
    _current_turn_index : monotonically increasing turn counter set by the
                          flow layer via set_turn_index().
    _provisional_user   : pending user text, visible via build_messages() but
        not yet durable.
    _provisional_assistant : pending assistant text, set while streaming; only
        moves to _journal on commit_provisional().
    """

    system_prompt: str
    max_turns: int = 20

    # Turn journal — replaces the old _committed flat list.
    _journal: list[HistoryEntry] = field(default_factory=list, init=False, repr=False)

    # Current turn index, set by the flow layer.
    _current_turn_index: int = field(default=0, init=False, repr=False)

    # Provisional slots – both are None when no turn is in flight.
    _provisional_user: Optional[str] = field(default=None, init=False, repr=False)
    _provisional_assistant: Optional[str] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Turn index
    # ------------------------------------------------------------------

    def set_turn_index(self, idx: int) -> None:
        """Set the current turn index (called by the flow layer)."""
        self._current_turn_index = idx

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def build_messages(self) -> list[dict]:
        """Return OpenAI-format messages: [system, ...journal, ?provisional_user].

        Backward-compatible with B-stage callers.
        """
        msgs: list[dict] = [{"role": "system", "content": self.system_prompt}]
        for entry in self._journal:
            msgs.append(entry.message)
        if self._provisional_user is not None:
            msgs.append({"role": "user", "content": self._provisional_user})
        return msgs

    def get_summary_context(self, last_n: int = 3, *, exclude_turn_index: int | None = None) -> list[dict]:
        """Return the last *last_n* committed turns worth of messages.

        A "turn" is identified by turn_index.  We find the last *last_n*
        distinct turn indices and return all messages from those turns.
        """
        entries = [entry for entry in self._journal if entry.turn_index != exclude_turn_index]
        if not entries:
            return []
        # Collect distinct turn indices in order of appearance (last N).
        seen: list[int] = []
        for entry in reversed(entries):
            if entry.turn_index not in seen:
                seen.append(entry.turn_index)
            if len(seen) >= last_n:
                break
        if not seen:
            return []
        cutoff = min(seen)
        return [e.message for e in entries if e.turn_index >= cutoff]

    @property
    def current_turn_index(self) -> int:
        return self._current_turn_index

    def get_messages_from(self, turn_index: int) -> list[dict]:
        """Return message dicts where entry.turn_index >= *turn_index*."""
        return [e.message for e in self._journal if e.turn_index >= turn_index]

    def get_journal_entries(self) -> list[HistoryEntry]:
        """Return all journal entries (for testing / debugging)."""
        return list(self._journal)

    def get_journal_messages(self) -> list[dict]:
        """Return all committed journal messages as dicts (for PostCallSnapshot)."""
        return [e.message for e in self._journal]

    def serialize_for_storage(self) -> list[dict]:
        """Return journal messages with PII scrubbed, for any persistence path.

        Call this (rather than ``get_journal_messages``) when writing the
        conversation to disk / DB / any downstream consumer outside the
        live in-memory session.  Runs the full ``PIIScrubber`` (regex +
        jieba) on each ``content`` field — acceptable because
        persistence is rare (end-of-call) and recall matters more than
        latency.
        """
        # Local import keeps the session module free of the jieba
        # dependency on the hot path.
        from src.core.pii import PIIScrubber

        scrubber = PIIScrubber()
        out: list[dict] = []
        for entry in self._journal:
            msg = dict(entry.message)
            content = msg.get("content")
            if isinstance(content, str) and content:
                msg["content"], _ = scrubber.scrub(content)
            out.append(msg)
        return out

    def get_last_assistant_text(self) -> str | None:
        """Return the most recent assistant text from the journal."""
        for entry in reversed(self._journal):
            if entry.message.get("role") == "assistant":
                return entry.message.get("content", "")
        return None

    def has_committed_user_message(self, text: str, turn_index: int | None = None) -> bool:
        """Return whether this user text is already committed for the target turn."""
        target_turn = self._current_turn_index if turn_index is None else turn_index
        for entry in reversed(self._journal):
            if entry.turn_index < target_turn:
                break
            if entry.turn_index != target_turn:
                continue
            if entry.entry_type != EntryType.USER:
                continue
            if entry.message.get("content", "") == text:
                return True
        return False

    # ------------------------------------------------------------------
    # Write – user side
    # ------------------------------------------------------------------

    def append_user(self, text: str, provisional: bool = False) -> None:
        """Add a user message.

        provisional=False  → creates a HistoryEntry and appends to journal.
        provisional=True   → stored in the provisional slot; visible in
                             build_messages() until commit or rollback.
        """
        if provisional:
            self._provisional_user = text
            self._provisional_assistant = None  # reset any stale assistant draft
        else:
            entry = HistoryEntry(
                message={"role": "user", "content": text},
                turn_index=self._current_turn_index,
                entry_type=EntryType.USER,
                node_id="",
                timestamp=time.time(),
            )
            self._journal.append(entry)
            self.truncate_to_safe_limit()

    def update_provisional(self, text: str) -> None:
        """Overwrite the provisional user text (e.g. as STT transcription grows)."""
        self._provisional_user = text

    # ------------------------------------------------------------------
    # Write – assistant side
    # ------------------------------------------------------------------

    def set_provisional_assistant(self, text: str) -> None:
        """Set (or overwrite) the provisional assistant response."""
        self._provisional_assistant = text

    def append_assistant(self, text: str, node_id: str = "") -> None:
        """Directly commit a finalised assistant message.

        Used by PlaybackLedger.commit_to_history after audio playout confirms
        what the user actually heard.  The caller is responsible for ensuring
        the preceding user turn has already been committed.
        """
        entry = HistoryEntry(
            message={"role": "assistant", "content": text},
            turn_index=self._current_turn_index,
            entry_type=EntryType.ASSISTANT_TEXT,
            node_id=node_id,
            timestamp=time.time(),
        )
        self._journal.append(entry)
        self.truncate_to_safe_limit()

    # ------------------------------------------------------------------
    # Write – tool interactions
    # ------------------------------------------------------------------

    def record_tool_interaction(
        self,
        tool_call: dict,
        tool_result: dict,
        node_id: str,
    ) -> None:
        """Record an ASSISTANT_TOOL_CALL + TOOL_RESULT pair in the journal."""
        now = time.time()
        self._journal.append(
            HistoryEntry(
                message={"role": "assistant", "tool_calls": [tool_call]},
                turn_index=self._current_turn_index,
                entry_type=EntryType.ASSISTANT_TOOL_CALL,
                node_id=node_id,
                timestamp=now,
            )
        )
        self._journal.append(
            HistoryEntry(
                message={"role": "tool", **tool_result},
                turn_index=self._current_turn_index,
                entry_type=EntryType.TOOL_RESULT,
                node_id=node_id,
                timestamp=now,
            )
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def commit_provisional(self) -> None:
        """Move provisional user (+ optional assistant) into the journal.

        Two cases handled:
        1. _provisional_user is set  → commit user then assistant (if any).
        2. _provisional_user is None → user was already committed directly via
           append_user(provisional=False); only flush _provisional_assistant.

        After commit, truncation to max_turns is applied.
        """
        now = time.time()
        if self._provisional_user is not None:
            self._journal.append(
                HistoryEntry(
                    message={"role": "user", "content": self._provisional_user},
                    turn_index=self._current_turn_index,
                    entry_type=EntryType.USER,
                    node_id="",
                    timestamp=now,
                )
            )
            self._provisional_user = None

        if self._provisional_assistant is not None:
            self._journal.append(
                HistoryEntry(
                    message={"role": "assistant", "content": self._provisional_assistant},
                    turn_index=self._current_turn_index,
                    entry_type=EntryType.ASSISTANT_TEXT,
                    node_id="",
                    timestamp=now,
                )
            )
            self._provisional_assistant = None

        self.truncate_to_safe_limit()

    def rollback_provisional(self) -> None:
        """Discard provisional user and assistant without touching the journal."""
        self._provisional_user = None
        self._provisional_assistant = None

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------

    def truncate_to_safe_limit(self) -> None:
        """Hard-truncate journal entries to the most recent max_turns turns.

        Keeps entries whose turn_index >= (current_turn_index - max_turns + 1).
        Uses _current_turn_index directly instead of scanning journal for max.
        """
        if not self._journal:
            return
        cutoff = self._current_turn_index - self.max_turns + 1
        if cutoff <= 0:
            return
        self._journal = [e for e in self._journal if e.turn_index >= cutoff]
