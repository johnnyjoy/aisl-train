# SPDX-License-Identifier: Apache-2.0
"""aisl-train command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

from aisl_train import __version__
from aisl_train.compare import compare_experiment
from aisl_train.config import DEFAULT_CONTAINER_MODEL, TrainingConfig, load_training_config
from aisl_train.environment import doctor_report, format_doctor_text, snapshot_environment
from aisl_train.errors import AislTrainError
from aisl_train.inspect import format_inspection_text, inspect_model_path
from aisl_train.packaged import apply_packaged_defaults
from aisl_train.profile import resolve_profile

PROG = "aisl-train"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Generic model-training and experiment runtime for AISL datasets.",
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="report environment and dependency readiness")
    _add_common_paths(doctor, model=True, dataset=True, output=True)
    doctor.add_argument("--json", action="store_true", help="print JSON")
    doctor.add_argument("--require-gpu", action="store_true", help="fail if CUDA is unavailable")

    inspect_p = sub.add_parser("inspect", help="inspect a model without training")
    _add_common_paths(inspect_p, model=True, required_model=False)
    inspect_p.add_argument("--profile", help="profile id or JSON path")
    inspect_p.add_argument("--json", action="store_true", help="print JSON")

    smoke = sub.add_parser("smoke", help="GPU readiness: load, LoRA, forward, backward, save, reload")
    _add_train_shared(smoke)
    smoke.add_argument("--dataset", help="tiny JSONL dataset (alias of --train)")

    baseline = sub.add_parser("baseline", help="inference without training")
    _add_eval_shared(baseline)
    baseline.add_argument(
        "--prompt-mode",
        choices=["none", "minimal"],
        default="none",
        help="none = no AISL bootstrap; minimal = short AISL reminder",
    )

    train = sub.add_parser("train", help="generic SFT / LoRA / QLoRA training")
    _add_train_shared(train)
    train.add_argument("--train", help="training JSONL")
    train.add_argument("--dataset", help="alias of --train")
    train.add_argument("--validation", help="validation JSONL")
    train.add_argument("--resume", help="latest or a checkpoint path")
    train.add_argument("--force-resume", action="store_true")
    train.add_argument("--allow-high-truncation", action="store_true")
    train.add_argument("--loss-mask", choices=["assistant", "completion", "full"])

    evaluate = sub.add_parser("evaluate", help="evaluate a base model plus optional adapter")
    _add_eval_shared(evaluate)
    evaluate.add_argument("--adapter", help="trained adapter directory")
    evaluate.add_argument(
        "--prompt-mode",
        choices=["none", "minimal"],
        default="none",
    )

    compare = sub.add_parser("compare", help="compare BASE / PROMPTED / TRAINED results")
    compare.add_argument("--experiment", required=True, help="experiment directory")

    return parser


def _add_common_paths(
    parser: argparse.ArgumentParser,
    *,
    model: bool = False,
    dataset: bool = False,
    output: bool = False,
    required_model: bool = False,
) -> None:
    if model:
        parser.add_argument("--model", required=required_model, help="local model directory or Hub id")
    if dataset:
        parser.add_argument("--dataset", help="dataset JSONL")
    if output:
        parser.add_argument("--output", help="output directory")


def _add_profile_and_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="training JSON config")
    parser.add_argument("--profile", help="profile id or JSON path")
    parser.add_argument("--aisl-root", help="optional aisl checkout for prompts/scorer/provenance")
    parser.add_argument("--manifest", help="AISL corpus manifest.json")
    parser.add_argument("--records", help="AISL records.jsonl for metadata join")
    parser.add_argument("--allow-hub", action="store_true", help="allow downloading a Hub model id")
    parser.add_argument("--limit", type=int, help="optional record cap")
    parser.add_argument("--seed", type=int)


def _add_train_shared(parser: argparse.ArgumentParser) -> None:
    _add_common_paths(parser, model=True, required_model=True, output=True)
    _add_profile_and_config(parser)
    parser.add_argument("--device-strategy", dest="device_strategy")
    parser.add_argument("--memory-headroom-gib", dest="memory_headroom_gib", type=float)


def _add_eval_shared(parser: argparse.ArgumentParser) -> None:
    _add_common_paths(parser, model=True, required_model=True, output=True)
    parser.add_argument("--eval", help="evaluation JSONL (default: packaged AISL test set)")
    parser.add_argument("--prompt-file", help="system prompt file (AISL eval/prompts/*.txt)")
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int)
    _add_profile_and_config(parser)


def _config_from_args(args: argparse.Namespace) -> TrainingConfig:
    config_path = Path(args.config).expanduser() if getattr(args, "config", None) else None
    overrides: dict[str, Any] = {
        "model_path": getattr(args, "model", None),
        "profile": getattr(args, "profile", None),
        "output_path": getattr(args, "output", None),
        "aisl_root": getattr(args, "aisl_root", None),
        "manifest_path": getattr(args, "manifest", None),
        "records_path": getattr(args, "records", None),
        "allow_hub": True if getattr(args, "allow_hub", False) else None,
        "limit": getattr(args, "limit", None),
        "seed": getattr(args, "seed", None),
        "device_strategy": getattr(args, "device_strategy", None),
        "memory_headroom_gib": getattr(args, "memory_headroom_gib", None),
        "resume": getattr(args, "resume", None),
        "force_resume": True if getattr(args, "force_resume", False) else None,
        "allow_high_truncation": True if getattr(args, "allow_high_truncation", False) else None,
        "loss_mask": getattr(args, "loss_mask", None),
        "prompt_mode": getattr(args, "prompt_mode", None),
        "prompt_file": getattr(args, "prompt_file", None),
        "adapter_path": getattr(args, "adapter", None),
        "eval_path": getattr(args, "eval", None),
        "validation_path": getattr(args, "validation", None),
    }
    train_path = getattr(args, "train", None) or getattr(args, "dataset", None)
    if train_path:
        overrides["train_path"] = train_path
    if getattr(args, "max_new_tokens", None):
        overrides["generation"] = {"max_new_tokens": args.max_new_tokens}
    return apply_packaged_defaults(load_training_config(config_path, overrides=overrides))


def _load_runtime():
    # Training stack is optional at parse time so `aisl-train --help` works
    # without torch. This is the single dispatch exception.
    from aisl_train import runtime

    return runtime


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    try:
        return _dispatch(args)
    except AislTrainError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code


def _dispatch(args: argparse.Namespace) -> int:
    command = args.command
    if command == "doctor":
        return _cmd_doctor(args)
    if command == "inspect":
        return _cmd_inspect(args)
    if command == "compare":
        report = compare_experiment(Path(args.experiment).expanduser())
        print(
            json.dumps(
                {"conditions": report["conditions_present"], "overall": report["overall"]},
                indent=2,
            )
        )
        return 0
    if command == "smoke":
        runtime = _load_runtime()
        cfg = _config_from_args(args)
        if not cfg.output_path:
            raise AislTrainError("smoke requires --output")
        return runtime.run_smoke(cfg)
    if command == "train":
        runtime = _load_runtime()
        cfg = _config_from_args(args)
        if not cfg.output_path:
            raise AislTrainError("train requires --output")
        return runtime.run_train(cfg)
    if command == "baseline":
        runtime = _load_runtime()
        cfg = _config_from_args(args)
        if not cfg.output_path:
            raise AislTrainError("baseline requires --output")
        return runtime.run_eval_command(cfg, kind="baseline")
    if command == "evaluate":
        runtime = _load_runtime()
        cfg = _config_from_args(args)
        if not cfg.output_path:
            raise AislTrainError("evaluate requires --output")
        return runtime.run_eval_command(cfg, kind="evaluate")
    raise AislTrainError(f"unknown command: {command}")


def _cmd_doctor(args: argparse.Namespace) -> int:
    report = doctor_report(
        model_path=Path(args.model).expanduser() if args.model else None,
        dataset_path=Path(args.dataset).expanduser() if args.dataset else None,
        output_path=Path(args.output).expanduser() if args.output else None,
        require_gpu=args.require_gpu,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_doctor_text(report), end="")
    return 0 if report["ok"] else 1


def _cmd_inspect(args: argparse.Namespace) -> int:
    model = args.model or DEFAULT_CONTAINER_MODEL
    inspection = inspect_model_path(model)
    profile = resolve_profile(args.profile, inspection=inspection.to_dict())
    if profile:
        inspection.matched_profile = profile.id
        inspection.profile_overrides = profile.overrides()
    env = snapshot_environment()
    payload = {
        "inspection": inspection.to_dict(),
        "matched_profile": inspection.matched_profile,
        "profile_overrides": inspection.profile_overrides,
        "environment": env,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_inspection_text(inspection), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
