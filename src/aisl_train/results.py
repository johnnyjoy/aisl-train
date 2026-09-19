# SPDX-License-Identifier: Apache-2.0
"""Evaluation result records and Markdown reports."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from aisl_train.errors import ExperimentError
from aisl_train.experiment import write_json


@dataclass
class GenerationRecord:
    id: str
    input: str
    raw_generation: str
    extracted_answer: str
    expected: str | None
    score: dict[str, Any] = field(default_factory=dict)
    token_counts: dict[str, int] = field(default_factory=dict)
    latency_ms: float | None = None
    prompt_mode: str = "none"
    system_prompt: str = ""
    model: str | None = None
    adapter: str | None = None
    task: str | None = None
    language: str | None = None
    difficulty: str | None = None
    category: str | None = None
    condition: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def surface_score(expected: str | None, predicted: str) -> dict[str, Any]:
    if expected is None:
        return {"surface_exact": None, "available": False}
    return {
        "surface_exact": expected.strip() == predicted.strip(),
        "available": True,
    }


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ExperimentError(f"results file not found: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExperimentError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ExperimentError(f"{path}:{line_number}: expected object")
        rows.append(raw)
    return rows


def summarize_records(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    scored = [row for row in rows if (row.get("score") or {}).get("available")]
    exact = [row for row in scored if (row.get("score") or {}).get("surface_exact")]
    semantic_rows = [
        row
        for row in rows
        if (row.get("score") or {}).get("semantic_f1") is not None
        or (row.get("score") or {}).get("exact_canonical") is not None
    ]
    semantic_hits = [
        row
        for row in semantic_rows
        if (row.get("score") or {}).get("exact_canonical")
        or ((row.get("score") or {}).get("semantic_f1") or 0) >= 0.999
    ]
    latencies = [row["latency_ms"] for row in rows if isinstance(row.get("latency_ms"), (int, float))]
    return {
        "n": total,
        "surface_n": len(scored),
        "surface_exact": len(exact),
        "surface_exact_pct": _pct(len(exact), len(scored)),
        "semantic_n": len(semantic_rows),
        "semantic_exact": len(semantic_hits),
        "semantic_exact_pct": _pct(len(semantic_hits), len(semantic_rows)),
        "mean_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
    }


def _pct(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return 100.0 * num / den


def write_eval_outputs(
    output_dir: Path,
    *,
    rows: list[GenerationRecord],
    metadata: Mapping[str, Any],
    report_title: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serialized = [row.to_dict() for row in rows]
    write_jsonl(output_dir / "results.jsonl", serialized)
    write_jsonl(output_dir / "raw_generations.jsonl", [{"id": r.id, "raw": r.raw_generation} for r in rows])
    summary = {
        **dict(metadata),
        "metrics": summarize_records(serialized),
    }
    write_json(output_dir / "summary.json", summary)
    write_markdown_report(output_dir / "REPORT.md", title=report_title, summary=summary, rows=serialized)


def write_markdown_report(
    path: Path,
    *,
    title: str,
    summary: Mapping[str, Any],
    rows: list[Mapping[str, Any]] | None = None,
) -> None:
    metrics = summary.get("metrics") or {}
    lines = [
        f"# {title}",
        "",
        f"Condition: {summary.get('condition') or 'unspecified'}",
        f"Items: {metrics.get('n', 0)}",
        "",
        "## Metrics",
        "",
        f"- surface exact: {metrics.get('surface_exact')} / {metrics.get('surface_n')} "
        f"({_fmt_pct(metrics.get('surface_exact_pct'))})",
        f"- semantic exact (when present): {metrics.get('semantic_exact')} / {metrics.get('semantic_n')} "
        f"({_fmt_pct(metrics.get('semantic_exact_pct'))})",
        f"- mean latency ms: {metrics.get('mean_latency_ms')}",
        "",
        "## Settings",
        "",
        f"- model: {summary.get('model')}",
        f"- adapter: {summary.get('adapter')}",
        f"- prompt mode: {summary.get('prompt_mode')}",
        "",
        "JSON/JSONL files in this directory are authoritative. This report is a summary.",
        "",
    ]
    if rows:
        misses = [row for row in rows if (row.get("score") or {}).get("surface_exact") is False]
        lines.extend(["## Surface misses (first 20)", ""])
        if not misses:
            lines.append("None.")
        else:
            for row in misses[:20]:
                lines.append(f"- `{row.get('id')}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f}%"


def aisl_runner_prediction(row: GenerationRecord) -> dict[str, Any]:
    """External AISL runner-contract prediction. Scoring stays in aisl."""
    return {
        "id": row.id,
        "output": row.raw_generation,
        "aisl": row.extracted_answer,
        "text": row.extracted_answer,
        "answer": row.extracted_answer,
        "rejected": False,
        "invented": False,
        "error": "",
    }
