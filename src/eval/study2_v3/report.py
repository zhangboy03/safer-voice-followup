"""Aggregate trial rows into the Study 2 comparison outputs (spec §8).

- bootstrap_ci: deterministic (seeded) 95% bootstrap CI.
- build_pareto_points: one point per (model, arm): x=TTFT median, y=compliance
  (completeness × accuracy), with CI.
- per_card_table: completeness/accuracy per (model, arm, card) — the contrast
  lives on the hard cards, so we never collapse cards into a single mean.
"""

from __future__ import annotations

import random
import statistics
from typing import Any


def bootstrap_ci(values: list[float], *, iters: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters) - 1]
    return (lo, hi)


def _group(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


def build_pareto_points(rows: list[dict[str, Any]], *, seed: int = 0) -> list[dict[str, Any]]:
    points = []
    for (model, arm), group in _group(rows, ("agent_model", "arm")).items():
        compliance = [r["completeness"] * r["accuracy"] for r in group]
        ttfts = [r["ttft_ms_median"] for r in group if isinstance(r.get("ttft_ms_median"), (int, float))]
        points.append(
            {
                "agent_model": model,
                "arm": arm,
                "compliance": sum(compliance) / len(compliance),
                "compliance_ci": bootstrap_ci(compliance, seed=seed),
                "ttft_ms": statistics.median(ttfts) if ttfts else None,
                "n": len(group),
            }
        )
    return points


def per_card_table(rows: list[dict[str, Any]], *, seed: int = 0) -> list[dict[str, Any]]:
    table = []
    for (model, arm, case_id), group in _group(rows, ("agent_model", "arm", "case_id")).items():
        comp = [r["completeness"] for r in group]
        acc = [r["accuracy"] for r in group]
        refusal = [r["refusal_compliance"] for r in group]
        table.append(
            {
                "agent_model": model,
                "arm": arm,
                "case_id": case_id,
                "case_type": group[0].get("case_type"),
                "completeness_mean": sum(comp) / len(comp),
                "completeness_ci": bootstrap_ci(comp, seed=seed),
                "accuracy_mean": sum(acc) / len(acc),
                "accuracy_ci": bootstrap_ci(acc, seed=seed),
                "refusal_compliance_mean": sum(refusal) / len(refusal),
                "refusal_compliance_ci": bootstrap_ci(refusal, seed=seed),
                "n": len(group),
            }
        )
    return table
