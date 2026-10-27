"""Shared PII scrubber for logs and any persistence path.

Extracted from ``scripts/extract_personas_from_calls.py`` so that both the
persona-extraction pipeline (offline, heavy, jieba-backed) and the
production logging / storage paths (hot path, fast-regex-only) can share
a single well-tested core.

Two entry points:

* ``PIIScrubber.scrub(text)``           — full scrub: regex + jieba 'nr'
  name detection. Suitable for offline batches where jieba's cost is
  acceptable (~1–5 ms per call on short utterances).

* ``redact_text(text)``                 — fast regex-only scrub (no
  jieba). Suitable for the logging processor where we want < 0.1 ms
  per log record.  Context-anchored Chinese-name regex still runs so
  patterns like ``您是张三先生`` are masked.

Both produce text that contains no mobile phone, no national ID, no
device SN, and (for high-recall mode) no recognised Chinese full name.

This module is importable from anywhere — it has no project-internal
dependencies beyond the stdlib + jieba (optional).
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

# ──────────────────────────────────────────────────────────────────────
# Regex primitives
# ──────────────────────────────────────────────────────────────────────
# Mobile phone: 1[3-9] + 9 digits, must not be part of a larger number.
# We use look-arounds rather than \b because \b doesn't behave well
# around CJK characters (which count as \w in Unicode mode).
PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")

# National ID: 18 digits (or 17 digits + X) or legacy 15 digits.
# Apply BEFORE phone regex so the 11-digit substring of an 18-digit ID
# is not accidentally consumed by PHONE_RE.
ID_CARD_RE = re.compile(r"(?<!\d)(?:\d{17}[\dXx]|\d{15})(?!\d)")

# Device serial numbers like "ABC-12345".
DEVICE_SN_RE = re.compile(r"(?<![A-Za-z])[A-Z]{2,4}-\d{4,8}(?!\d)")

# Chinese name anchored to a surrounding context marker. High-precision.
NAME_CONTEXT_RE = re.compile(
    r"(?:您是|我是|叫|姓)([^\s,，。！？]{2,4}?)(?:先生|女士|阿姨|大爷|叔叔|老师|吗|的|，|。|$)"
)

# Context markers used by the jieba-backed fallback to gate low-precision
# 'nr' tags (jieba will tag things like "小姐" or "王者" as 'nr').
_NAME_CONTEXT_MARKERS = (
    "您是",
    "我是",
    "叫",
    "姓",
    "是",
    "先生",
    "女士",
    "阿姨",
    "大爷",
    "叔叔",
    "老师",
)


def _apply_regex_masks(text: str, audit: dict[str, Any]) -> str:
    """Apply ID + phone + device SN regex masks (order matters).

    Mutates ``audit`` in-place.
    """
    # ID first so 18-digit id isn't split by the 11-digit phone rule.
    text = ID_CARD_RE.sub(lambda _m: _bump(audit, "ids") or "<ID>", text)
    text = PHONE_RE.sub(lambda _m: _bump(audit, "phones") or "<PHONE>", text)
    text = DEVICE_SN_RE.sub(lambda _m: _bump(audit, "devices") or "<SN>", text)
    return text


def _mask_context_names(text: str, audit: dict[str, Any]) -> str:
    """Mask names that appear with context markers (regex-only, fast)."""
    seen: set[str] = set()
    for m in NAME_CONTEXT_RE.finditer(text):
        name = m.group(1).strip()
        if len(name) >= 2 and name not in seen:
            seen.add(name)
            audit["names"].append(name)

    # Longest-first prevents fragment leakage like '<NAME>丰' when the
    # corpus contains both '张三丰' and '张三'.
    for name in sorted(seen, key=len, reverse=True):
        text = text.replace(name, "<NAME>")
    return text


def _bump(audit: dict[str, Any], key: str) -> None:
    """Helper for re.sub lambdas: increments the counter, returns None so
    the lambda expression evaluates to the replacement string via
    short-circuit ``_bump(...) or '<MASK>'`` pattern."""
    audit[key] += 1


# ──────────────────────────────────────────────────────────────────────
# Fast path — regex-only, safe for the logging hot path
# ──────────────────────────────────────────────────────────────────────


def redact_text(text: str) -> str:
    """Fast, regex-only redaction. O(n) in text length, no jieba.

    Guarantees:
      - No 11-digit Chinese mobile numbers remain.
      - No 18-digit (or 17+X) Chinese national IDs remain.
      - No ``ABCD-12345``-style device SNs remain.
      - Names appearing with surrounding context markers
        (``您是张三先生``) are masked; bare names are NOT masked
        (that's the slow path's job).

    Suitable for calling on every log record.  Target overhead: < 0.1 ms
    per short (<1 KB) string.
    """
    if not text or not isinstance(text, str):
        return text
    # Cheap short-circuit: if nothing looks like a digit run OR a
    # context marker, skip the whole pipeline.  Saves ~90 % of log
    # records which are pure ASCII diagnostic messages.
    if not _LIKELY_PII_RE.search(text):
        return text
    audit: dict[str, Any] = {"phones": 0, "ids": 0, "devices": 0, "names": []}
    text = _apply_regex_masks(text, audit)
    text = _mask_context_names(text, audit)
    return text


# Cheap pre-filter so most log records skip the full regex pipeline.
# Triggers on: 2+ consecutive digits, OR any Chinese name-context marker.
_LIKELY_PII_RE = re.compile(r"\d{2}|您是|我是|[叫姓]")


# ──────────────────────────────────────────────────────────────────────
# Full path — regex + jieba 'nr' fallback for offline / batch use
# ──────────────────────────────────────────────────────────────────────


class PIIScrubber:
    """Deidentify PII from text.

    The full ``scrub()`` pipeline adds a jieba-based Chinese name
    detector on top of ``redact_text()``.  Use this for offline dataset
    preparation where recall matters more than latency.
    """

    def scrub(self, text: str) -> tuple[str, dict[str, Any]]:
        """Return ``(scrubbed_text, audit_dict)``."""
        audit: dict[str, Any] = {
            "phones": 0,
            "ids": 0,
            "devices": 0,
            "names": [],
        }
        if not isinstance(text, str) or not text:
            return text, audit

        text = _apply_regex_masks(text, audit)

        seen: set[str] = set()

        def _add(name: str) -> None:
            if len(name) >= 2 and name not in seen:
                seen.add(name)
                audit["names"].append(name)

        # High-precision context-anchored names.
        for match in NAME_CONTEXT_RE.finditer(text):
            _add(match.group(1).strip())

        # Lower-precision jieba 'nr' tags, gated to cut false positives
        # like '王者' / '小姐' which jieba sometimes labels as names.
        try:
            import jieba.posseg as pseg  # type: ignore

            for word, flag in pseg.cut(text):
                if flag != "nr" or len(word) < 2:
                    continue
                if (
                    len(word) >= 3
                    or any(f"{m}{word}" in text for m in _NAME_CONTEXT_MARKERS)
                    or any(f"{word}{m}" in text for m in _NAME_CONTEXT_MARKERS)
                ):
                    _add(word)
        except ImportError:
            pass

        for name in sorted(audit["names"], key=len, reverse=True):
            text = text.replace(name, "<NAME>")

        return text, audit

    @staticmethod
    def scrub_filename(filename: str) -> str:
        """Hash phone numbers in filenames (keeps the rest intact)."""
        return PHONE_RE.sub(
            lambda m: hashlib.md5(m.group().encode()).hexdigest()[:8],
            filename,
        )


# ──────────────────────────────────────────────────────────────────────
# Convenience wrappers used by audit_04 / callers outside structlog
# ──────────────────────────────────────────────────────────────────────


def mask(text: str) -> str:
    """Alias for ``redact_text`` — kept so privacy_audit.audit_04 can
    import ``mask`` or ``redact`` from this module and confirm presence
    of a masking implementation."""
    return redact_text(text)


def redact(text: str) -> str:
    """Alias for ``redact_text``."""
    return redact_text(text)


__all__ = [
    "PIIScrubber",
    "PHONE_RE",
    "ID_CARD_RE",
    "DEVICE_SN_RE",
    "NAME_CONTEXT_RE",
    "redact_text",
    "mask",
    "redact",
]
