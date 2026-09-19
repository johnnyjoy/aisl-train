# SPDX-License-Identifier: Apache-2.0
"""Generic LoRA / QLoRA attachment. No architecture-specific module lists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aisl_train.config import LoRASettings, TrainingConfig
from aisl_train.errors import MissingDependencyError, PeftError
from aisl_train.profile import ModelProfile


@dataclass
class AdapterStats:
    total_parameters: int
    trainable_parameters: int
    trainable_pct: float
    target_strategy: str
    matched_categories: list[str]
    frozen_base: bool


def build_lora_config(settings: LoRASettings, profile: ModelProfile | None = None) -> Any:
    try:
        from peft import LoraConfig, TaskType
    except Exception as exc:
        raise MissingDependencyError(f"peft is required for LoRA/QLoRA: {exc}") from exc

    peft_over = dict(profile.peft) if profile else {}
    target_modules = peft_over.get("target_modules", settings.target_modules)
    exclude_modules = peft_over.get("exclude_modules", settings.exclude_modules)
    target_parameters = peft_over.get("target_parameters", settings.target_parameters)
    modules_to_save = peft_over.get("modules_to_save", settings.modules_to_save)

    kwargs: dict[str, Any] = {
        "r": settings.rank,
        "lora_alpha": settings.alpha,
        "lora_dropout": settings.dropout,
        "bias": settings.bias,
        "task_type": getattr(TaskType, settings.task_type, settings.task_type),
        "target_modules": target_modules,
    }
    if exclude_modules:
        kwargs["exclude_modules"] = exclude_modules
    if target_parameters:
        kwargs["target_parameters"] = target_parameters
    if modules_to_save:
        kwargs["modules_to_save"] = modules_to_save
    return LoraConfig(**kwargs)


def attach_lora(
    model: Any,
    config: TrainingConfig,
    *,
    profile: ModelProfile | None = None,
) -> tuple[Any, AdapterStats]:
    try:
        from peft import get_peft_model, prepare_model_for_kbit_training
    except Exception as exc:
        raise MissingDependencyError(f"peft is required for LoRA/QLoRA: {exc}") from exc

    if config.method == "qlora" or config.quantization.enabled:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )
    elif config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    lora_config = build_lora_config(config.lora, profile)
    model = get_peft_model(model, lora_config)
    stats = adapter_stats(model, config)
    if stats.trainable_parameters == 0:
        raise PeftError("PEFT attachment produced zero trainable parameters")
    if stats.trainable_pct > 15.0:
        raise PeftError(
            f"trainable parameter share {stats.trainable_pct:.2f}% is inconsistent with LoRA/QLoRA"
        )
    if not stats.frozen_base:
        raise PeftError("base parameters are not frozen after LoRA attachment")
    return model, stats


def adapter_stats(model: Any, config: TrainingConfig) -> AdapterStats:
    total = 0
    trainable = 0
    trainable_names: list[str] = []
    base_trainable = 0
    for name, param in model.named_parameters():
        n = int(param.numel())
        total += n
        if param.requires_grad:
            trainable += n
            trainable_names.append(name)
            if "lora_" not in name and "modules_to_save" not in name:
                base_trainable += n
    if total == 0:
        raise PeftError("model has zero parameters")
    categories = sorted({_category(name) for name in trainable_names})
    return AdapterStats(
        total_parameters=total,
        trainable_parameters=trainable,
        trainable_pct=100.0 * trainable / total,
        target_strategy=str(config.lora.target_modules),
        matched_categories=categories,
        frozen_base=base_trainable == 0,
    )


def _category(name: str) -> str:
    parts = name.split(".")
    for part in reversed(parts):
        if part.startswith("lora_") or part in {"weight", "bias"}:
            continue
        return part
    return name


def load_adapter(model: Any, adapter_path: str) -> Any:
    try:
        from peft import PeftModel
    except Exception as exc:
        raise MissingDependencyError(f"peft is required to load an adapter: {exc}") from exc
    return PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
