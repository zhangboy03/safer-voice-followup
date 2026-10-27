"""Judge-vs-human agreement helpers (spec success criterion #3).

The coverage judge must clear an agreement threshold (≥85% recommended) on a
human-labeled subset before its numbers back any conclusion. These helpers
sample field verdicts for human labeling and compute agreement after backfill.
"""

from __future__ import annotations

import random
from typing import Any


def sample_field_verdicts(verdicts: list[dict[str, Any]], *, k: int, seed: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if k >= len(verdicts):
        return list(verdicts)
    idx = sorted(rng.sample(range(len(verdicts)), k))
    return [verdicts[i] for i in idx]


def agreement_rate(judge: list[dict[str, Any]], human: list[dict[str, Any]], *, key: str) -> float:
    human_by_field = {h["field"]: h for h in human}
    matched = 0
    total = 0
    for j in judge:
        h = human_by_field.get(j["field"])
        if h is None or key not in h:
            continue
        total += 1
        if j.get(key) == h.get(key):
            matched += 1
    return matched / total if total else 0.0
