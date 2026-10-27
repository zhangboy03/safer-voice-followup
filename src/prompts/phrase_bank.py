"""Frozen phrase bank for knowledge_qa fixed-string outputs.

Per spec §5 deploy contract: phrases are loaded at process boot from a single
markdown file, frozen into a MappingProxyType so application code cannot
mutate them at runtime, and the source file's sha256 hash is logged at
startup so an operator can verify the running process has the expected
content. NOT hot-reloadable — process restart required to pick up edits.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

logger = logging.getLogger(__name__)

PHRASE_TYPE_1 = "type_1"
PHRASE_TYPE_2 = "type_2"
PHRASE_TYPE_3 = "type_3"
PHRASE_TYPE_E1 = "type_e1"
PHRASE_TYPE_E2 = "type_e2"

KnowledgePhraseBank = Mapping[str, str]

_REFUSE_MD = Path("config/prompts/packs/sop-v1/speech/knowledge_qa_refuse.md")

_HEADING_TO_KEY = {
    "Type 1": PHRASE_TYPE_1,
    "Type 2": PHRASE_TYPE_2,
    "Type 3": PHRASE_TYPE_3,
    "Type E1": PHRASE_TYPE_E1,
    "Type E2": PHRASE_TYPE_E2,
}

_HEADING_RE = re.compile(r"^##\s+(Type (?:1|2|3|E1|E2))\b")


def load_knowledge_phrases(language: str = "zh") -> KnowledgePhraseBank:
    """Parse the .md file and return a frozen mapping of phrase-id → phrase.

    Side effect: logs the source-file sha256 hash at INFO level for operator
    verification.
    """
    path = _REFUSE_MD.with_name("knowledge_qa_refuse_en.md") if language == "en" else _REFUSE_MD
    text = path.read_text(encoding="utf-8")
    hash_hex = hashlib.sha256(text.encode("utf-8")).hexdigest()

    phrases: dict[str, str] = {}
    current_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        m = _HEADING_RE.match(line)
        if m:
            current_key = _HEADING_TO_KEY[m.group(1)]
            continue
        if current_key is None or not line or line.startswith("#"):
            continue
        # First non-empty line after a heading is the phrase
        phrases[current_key] = line
        current_key = None

    missing = set(_HEADING_TO_KEY.values()) - set(phrases.keys())
    assert not missing, f"phrase bank missing entries: {missing}"

    logger.info(
        "knowledge_phrase_bank.loaded path=%s sha256:%s entries=%d",
        path,
        hash_hex,
        len(phrases),
    )
    return MappingProxyType(phrases)
