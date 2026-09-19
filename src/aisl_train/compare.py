# SPDX-License-Identifier: Apache-2.0
"""Model-neutral comparison of BASE / PROMPTED / TRAINED conditions."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from aisl_train.errors import ExperimentError
from aisl_train.experiment import read_json, write_json
from aisl_train.results import read_jsonl, summarize_records, write_markdown_report

CONDITION_DIRS = {
    "base": "baseline",
    "prompted": "prompted",
    "trained": "trained-eval",
}


def compare_experiment(experiment_dir: Path) -> dict[str, Any]:
    if not experiment_dir.is_dir():
        raise ExperimentError(f"experiment directory not found: {experiment_dir}")
    conditions: dict[str, list[dict[str, Any]]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for name, subdir in CONDITION_DIRS.items():
        results_path = experiment_dir / subdir / "results.jsonl"
        if not results_path.is_file():
            continue
        conditions[name] = read_jsonl(results_path)
        summary_path = experiment_dir / subdir / "summary.json"
        metadata[name] = read_json(summary_path) if summary_path.is_file() else {}

    if not conditions:
        raise ExperimentError(
            f"{experiment_dir} has no baseline/, prompted/, or trained-eval/ results.jsonl"
        )

    overall = {name: summarize_records(rows) for name, rows in conditions.items()}
    by_id = _index_by_id(conditions)
    deltas = _pairwise_deltas(by_id)
    breakdowns = {
        "task": _breakdown(conditions, "task"),
        "difficulty": _breakdown(conditions, "difficulty"),
        "language": _breakdown(conditions, "language"),
        "category": _breakdown(conditions, "category"),
    }
    report = {
        "experiment": str(experiment_dir),
        "model": _first_present(metadata, "model"),
        "conditions_present": sorted(conditions),
        "overall": overall,
        "breakdowns": breakdowns,
        "improvements": deltas["improvements"],
        "regressions": deltas["regressions"],
        "unchanged": deltas["unchanged"],
        "missing_conditions": [name for name in CONDITION_DIRS if name not in conditions],
    }
    out_dir = experiment_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "comparison.json", report)
    write_markdown_report(
        out_dir / "REPORT.md",
        title="Experiment comparison",
        summary={
            "condition": ",".join(sorted(conditions)),
            "model": report["model"],
            "metrics": overall.get("trained") or next(iter(overall.values())),
        },
    )
    _write_compare_markdown(out_dir / "COMPARE.md", report)
    return report


def _first_present(metadata: Mapping[str, Mapping[str, Any]], key: str) -> Any:
    for payload in metadata.values():
        if payload.get(key):
            return payload[key]
    return None


def _index_by_id(
    conditions: Mapping[str, list[dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for name, rows in conditions.items():
        for row in rows:
            item_id = str(row.get("id") or "")
            if not item_id:
                continue
            by_id[item_id][name] = row
    return by_id


def _hit(row: Mapping[str, Any] | None) -> bool | None:
    if row is None:
        return None
    score = row.get("score") or {}
    if score.get("exact_canonical") is not None:
        return bool(score.get("exact_canonical"))
    if score.get("surface_exact") is not None:
        return bool(score.get("surface_exact"))
    return None


def _pairwise_deltas(by_id: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    improvements: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    unchanged: list[dict[str, Any]] = []
    pairs = (("base", "prompted"), ("base", "trained"), ("prompted", "trained"))
    for item_id, conds in by_id.items():
        for left, right in pairs:
            a = _hit(conds.get(left))
            b = _hit(conds.get(right))
            if a is None or b is None:
                continue
            payload = {"id": item_id, "from": left, "to": right, "was": a, "now": b}
            if a is False and b is True:
                improvements.append(payload)
            elif a is True and b is False:
                regressions.append(payload)
            elif a == b:
                unchanged.append(payload)
    return {
        "improvements": improvements,
        "regressions": regressions,
        "unchanged": unchanged,
    }


def _breakdown(
    conditions: Mapping[str, list[dict[str, Any]]],
    field_name: str,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for cond, rows in conditions.items():
        for row in rows:
            key = row.get(field_name)
            if not key:
                continue
            groups[str(key)][cond].append(row)
    return {
        label: {cond: summarize_records(rows) for cond, rows in conds.items()}
        for label, conds in sorted(groups.items())
    }


def _write_compare_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Comparison",
        "",
        f"Model: {report.get('model') or 'from experiment metadata'}",
        f"Conditions: {', '.join(report.get('conditions_present') or [])}",
        "",
        "## Overall",
        "",
    ]
    for name, metrics in (report.get("overall") or {}).items():
        lines.append(
            f"- {name}: surface {metrics.get('surface_exact')}/{metrics.get('surface_n')} "
            f"({metrics.get('surface_exact_pct')})"
        )
    lines.extend(["", "## Improvements / regressions", ""])
    lines.append(f"- improvements: {len(report.get('improvements') or [])}")
    lines.append(f"- regressions: {len(report.get('regressions') or [])}")
    lines.append("")
    lines.append("BASE, PROMPTED, and TRAINED remain distinct conditions.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
