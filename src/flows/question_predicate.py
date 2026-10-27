"""Single source of truth for "did this Speech utterance ask a question?".

Deterministic, zero-inference. Shared by stage1-followup's det_asked AND-gate
and the advance-over-open-question guard. Keep ONE definition — do not write two.
"""

from __future__ import annotations

# Unambiguous sentence-final interrogative particles. 吧 is intentionally excluded:
# it is ambiguous with assents ("好吧" / "行吧"), which would cause false blocks.
_INTERROGATIVE_SUFFIXES = ("吗", "呢")
# Strong interrogative constructions that signal a question even without 吗/呢/？.
_INTERROGATIVE_MARKERS = ("有没有", "是不是", "能不能", "可不可以", "对不对", "是否")
# Trailing punctuation/whitespace stripped before the suffix check.
_TRAILING = "。.！!~～… \t\n\r　"


def speech_asked_question(text: str | None, *, interrogatives: bool = False) -> bool:
    """True iff the utterance is a question.

    Default (`interrogatives=False`): only a full-width/ASCII question mark counts
    — the original behavior, preserved for stage1-followup's det_asked AND-gate.

    `interrogatives=True`: additionally treat unpunctuated Chinese questions as
    questions (trailing 吗/呢, or any strong interrogative marker). Used by the
    advance-over-open-question guard, where fail-safe over-detection is desired.
    """
    if not text:
        return False
    if "？" in text or "?" in text:
        return True
    if not interrogatives:
        return False
    stripped = text.rstrip(_TRAILING)
    if stripped.endswith(_INTERROGATIVE_SUFFIXES):
        return True
    return any(marker in text for marker in _INTERROGATIVE_MARKERS)
