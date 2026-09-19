# SPDX-License-Identifier: Apache-2.0
"""TRL-backed SFT. Swap this module later; CLI and experiment metadata stay."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from aisl_train.config import TrainingConfig
from aisl_train.errors import MissingDependencyError, OutOfMemoryError, TrainingError
from aisl_train.devices import snapshot_memory


def run_supervised_finetune(
    *,
    model: Any,
    tokenizer: Any,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]] | None,
    config: TrainingConfig,
    output_dir: Path,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    """First trainer backend: TRL SFTTrainer."""
    try:
        from datasets import Dataset
        from transformers import TrainerCallback
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:
        raise MissingDependencyError(f"TRL/transformers/datasets are required to train: {exc}") from exc

    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(eval_rows) if eval_rows else None
    use_bf16 = config.precision in {"auto", "bf16"} and _can_bf16()
    use_fp16 = (config.precision == "fp16") or (config.precision == "auto" and not use_bf16)
    if config.precision == "fp32":
        use_bf16 = False
        use_fp16 = False

    sft_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.micro_batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": config.gradient_accumulation,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "logging_steps": config.logging_steps,
        "save_total_limit": config.save_total_limit,
        "seed": config.seed,
        "bf16": use_bf16,
        "fp16": use_fp16 and not use_bf16,
        "gradient_checkpointing": config.gradient_checkpointing,
        "report_to": [],
        "remove_unused_columns": False,
    }
    if eval_ds is not None:
        sft_kwargs["eval_strategy"] = "steps" if config.eval_steps else "epoch"
        if config.eval_steps:
            sft_kwargs["eval_steps"] = config.eval_steps
    if config.save_steps:
        sft_kwargs["save_strategy"] = "steps"
        sft_kwargs["save_steps"] = config.save_steps
    else:
        sft_kwargs["save_strategy"] = "epoch"

    _assign_if_supported(SFTConfig, sft_kwargs, "max_length", config.max_seq_length)
    _assign_if_supported(SFTConfig, sft_kwargs, "max_seq_length", config.max_seq_length)
    if config.loss_mask == "assistant":
        _assign_if_supported(SFTConfig, sft_kwargs, "assistant_only_loss", True)
    if config.loss_mask in {"assistant", "completion"}:
        _assign_if_supported(SFTConfig, sft_kwargs, "completion_only_loss", True)

    args = SFTConfig(**sft_kwargs)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
    }
    if eval_ds is not None:
        trainer_kwargs["eval_dataset"] = eval_ds
    _assign_if_supported(SFTTrainer, trainer_kwargs, "processing_class", tokenizer)
    if "processing_class" not in trainer_kwargs:
        trainer_kwargs["tokenizer"] = tokenizer

    class NanLossCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[no-untyped-def]
            if not logs:
                return
            loss = logs.get("loss")
            if loss is not None and loss != loss:
                raise TrainingError(f"NaN loss at step {getattr(state, 'global_step', '?')}")

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.add_callback(NanLossCallback())

    train_kwargs: dict[str, Any] = {}
    if resume_from is not None:
        train_kwargs["resume_from_checkpoint"] = str(resume_from)
    try:
        result = trainer.train(**train_kwargs)
    except Exception as exc:
        _reraise_train_failure(exc, config)
    metrics = getattr(result, "metrics", {}) or {}
    trainer.save_model(str(output_dir.parent / "adapter"))
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(output_dir.parent / "adapter"))
    return {
        "metrics": dict(metrics),
        "bf16": use_bf16,
        "fp16": use_fp16 and not use_bf16,
        "backend": "trl.SFTTrainer",
    }


def _assign_if_supported(cls: type, dest: dict[str, Any], name: str, value: Any) -> None:
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        params = {}
    if name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        if name in params:
            dest[name] = value


def _can_bf16() -> bool:
    try:
        from aisl_train.devices import bf16_supported

        return bf16_supported()
    except Exception:
        return False


def _reraise_train_failure(exc: Exception, config: TrainingConfig) -> None:
    if "out of memory" in str(exc).lower() or exc.__class__.__name__ in {"OutOfMemoryError", "CUDAOutOfMemoryError"}:
        raise OutOfMemoryError(
            "CUDA OOM during training. "
            f"model={config.model_path} quantization={config.quantization.scheme} "
            f"seq={config.max_seq_length} micro_batch={config.micro_batch_size} "
            f"lora_rank={config.lora.rank} targets={config.lora.target_modules} "
            f"device_map={config.device_strategy} "
            f"memory={snapshot_memory('train_oom')}"
        ) from exc
    if isinstance(exc, TrainingError):
        raise exc
    raise TrainingError(f"training failed: {exc}") from exc
