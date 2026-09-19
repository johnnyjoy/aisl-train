# SPDX-License-Identifier: Apache-2.0
"""Run model generations. AISL owns semantic scoring."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from aisl_train.config import TrainingConfig
from aisl_train.dataset import LoadedDataset
from aisl_train.errors import EvaluationError
from aisl_train.experiment import write_json
from aisl_train.inference import generate_one
from aisl_train.results import (
    GenerationRecord,
    aisl_runner_prediction,
    surface_score,
    write_eval_outputs,
    write_jsonl,
)


def load_prompt_text(config: TrainingConfig) -> str:
    if config.prompt_mode == "none":
        if config.prompt_file:
            return Path(config.prompt_file).expanduser().read_text(encoding="utf-8").strip()
        if config.aisl_root:
            base = Path(config.aisl_root).expanduser() / "eval" / "prompts" / "base.txt"
            if base.is_file():
                return base.read_text(encoding="utf-8").strip()
        return ""
    if config.prompt_file:
        path = Path(config.prompt_file).expanduser()
        if not path.is_file():
            raise EvaluationError(f"prompt file not found: {path}")
        return path.read_text(encoding="utf-8").rstrip()
    if config.aisl_root:
        path = Path(config.aisl_root).expanduser() / "eval" / "prompts" / "minimal.txt"
        if path.is_file():
            return path.read_text(encoding="utf-8").rstrip()
    raise EvaluationError(
        "prompt-mode minimal requires --prompt-file or --aisl-root pointing at aisl/eval/prompts/minimal.txt"
    )


def condition_name(config: TrainingConfig) -> str:
    if config.adapter_path:
        return "trained"
    if config.prompt_mode == "minimal":
        return "prompted"
    return "base"


def run_generation_eval(
    *,
    model: Any,
    tokenizer: Any,
    dataset: LoadedDataset,
    config: TrainingConfig,
    output_dir: Path,
    model_id: str,
) -> dict[str, Any]:
    system_prompt = load_prompt_text(config)
    condition = condition_name(config)
    rows: list[GenerationRecord] = []
    last_generation = {}
    for record in dataset.records:
        generated = generate_one(
            model,
            tokenizer,
            record.user_text(),
            settings=config.generation,
            system_prompt=system_prompt or None,
        )
        last_generation = generated["generation"]
        extracted = generated["extracted"]
        score = surface_score(record.expected, extracted)
        rows.append(
            GenerationRecord(
                id=record.record_id or f"line:{record.line_number}",
                input=record.user_text(),
                raw_generation=generated["raw"],
                extracted_answer=extracted,
                expected=record.expected,
                score=score,
                token_counts={
                    "prompt": generated["prompt_tokens"],
                    "completion": generated["completion_tokens"],
                },
                latency_ms=generated["latency_ms"],
                prompt_mode=config.prompt_mode,
                system_prompt=system_prompt,
                model=model_id,
                adapter=config.adapter_path,
                task=record.task,
                language=record.language,
                difficulty=record.difficulty,
                category=record.category,
                condition=condition,
            )
        )
    metadata = {
        "condition": condition,
        "model": model_id,
        "adapter": config.adapter_path,
        "prompt_mode": config.prompt_mode,
        "system_prompt": system_prompt,
        "generation": last_generation,
        "dataset": str(dataset.path),
        "n": len(rows),
    }
    write_eval_outputs(output_dir, rows=rows, metadata=metadata, report_title=f"{condition} evaluation")
    write_jsonl(output_dir / "aisl_predictions.jsonl", [aisl_runner_prediction(row) for row in rows])
    semantic = maybe_score_with_aisl(output_dir, dataset, config)
    if semantic:
        metadata["aisl_scoring"] = semantic
        write_json(output_dir / "aisl_scoring.json", semantic)
    return metadata


def maybe_score_with_aisl(
    output_dir: Path,
    dataset: LoadedDataset,
    config: TrainingConfig,
) -> dict[str, Any] | None:
    """Invoke aisl's scorer if an aisl checkout is explicitly provided."""
    root = config.aisl_root or os.environ.get("AISL_ROOT")
    if not root:
        return None
    script = Path(root).expanduser() / "eval" / "scripts" / "score_file.py"
    if not script.is_file():
        return {"ran": False, "reason": f"aisl scorer not found: {script}"}
    if dataset.kind != "eval_case":
        return {
            "ran": False,
            "reason": "semantic scoring is invoked for AISL eval-case files; generations were saved for later scoring",
        }
    pred = output_dir / "aisl_predictions.jsonl"
    gold = dataset.path
    task = next((r.task for r in dataset.records if r.task), None)
    if not task:
        return {"ran": False, "reason": "eval cases have no task id"}
    try:
        result = subprocess.run(
            [sys.executable, str(script), "--task", str(task), "--pred", str(pred), "--gold", str(gold)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return {"ran": False, "reason": str(exc)}
    payload: dict[str, Any] = {
        "ran": result.returncode == 0,
        "returncode": result.returncode,
        "stderr": result.stderr[-2000:],
    }
    try:
        payload["stdout_json"] = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload["stdout"] = result.stdout[-2000:]
    return payload


def attach_record_metadata(dataset: LoadedDataset, records_path: Path | None) -> None:
    if records_path is None or not records_path.is_file():
        return
    by_input: dict[str, dict[str, Any]] = {}
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if isinstance(raw, dict) and raw.get("input"):
            by_input[str(raw["input"])] = raw
    for record in dataset.records:
        extra = by_input.get(record.user_text())
        if not extra:
            continue
        record.record_id = record.record_id or extra.get("id")
        record.semantic_case_id = record.semantic_case_id or extra.get("semantic_case_id")
        record.task = record.task or extra.get("task")
        record.language = record.language or extra.get("language")
        record.difficulty = record.difficulty or extra.get("difficulty")
        record.category = record.category or extra.get("category")
        record.split = record.split or extra.get("split")
