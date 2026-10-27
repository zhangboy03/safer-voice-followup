# src/eval/study3/report.py
"""Render the Study 3 mini three tables (markdown) from scored dicts. Pure."""

from __future__ import annotations

from typing import Any


def _fmt_rate(d: dict[str, Any]) -> str:
    if d["n"] == 0 or d["rate"] is None:
        return "—"
    lo, hi = d["ci"]
    return f"{d['k']}/{d['n']} ({d['rate'] * 100:.0f}%, [{lo * 100:.0f},{hi * 100:.0f}])"


def _fmt_failures(failures: dict[str, int] | None) -> str:
    if not failures:
        return "—"
    return ", ".join(f"{k}:{v}" for k, v in sorted(failures.items()))


def render_routing_table(scored: list[dict]) -> str:
    head = "| System | N | Failures | High/severe under-deflection | Direct-advice viol. | Over-deflection | Gold-answerable success |"
    sep = "| --- | ---: | --- | --- | --- | --- | --- |"
    rows = [head, sep]
    for s in scored:
        rows.append(
            f"| {s['system']} | {s['n_cases']} | {_fmt_failures(s.get('failures'))} | "
            f"{_fmt_rate(s['under_deflection'])} | "
            f"{_fmt_rate(s['direct_advice_violation'])} | {_fmt_rate(s['over_deflection'])} | "
            f"{_fmt_rate(s['gold_answerable_success'])} |"
        )
    return "\n".join(rows) + "\n"


def _num(x: Any) -> str:
    return "—" if x is None else f"{x:.2f}".rstrip("0").rstrip(".")


def render_answer_table(scored: list[dict]) -> str:
    head = "| System | N_answered | AQ failures | Groundedness | Completeness | Unsupported-claim |"
    sep = "| --- | ---: | --- | ---: | ---: | ---: |"
    rows = [head, sep]
    for s in scored:
        aq = s["answer_quality"]
        rows.append(
            f"| {s['system']} | {aq['n_answered']} | {_fmt_failures(aq.get('failures'))} | "
            f"{_num(aq['groundedness'])} | "
            f"{_num(aq['completeness'])} | {_num(aq['unsupported_claim_rate'])} |"
        )
    return "\n".join(rows)


def render_confusion(cm: dict[str, dict[str, int]]) -> str:
    cols = ("ANSWER", "CLARIFY", "DEFLECT")
    head = "| gold \\ pred | " + " | ".join(cols) + " |"
    sep = "| --- | " + " | ".join(["---:"] * len(cols)) + " |"
    rows = [head, sep]
    for g in cols:
        rows.append(f"| {g} | " + " | ".join(str(cm[g][p]) for p in cols) + " |")
    return "\n".join(rows)
