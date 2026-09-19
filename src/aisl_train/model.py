# SPDX-License-Identifier: Apache-2.0
"""Generic causal-LM load/save. Model names do not belong here."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aisl_train.config import TrainingConfig
from aisl_train.devices import device_map_for, max_memory_map, resolve_compute_dtype
from aisl_train.errors import MissingDependencyError, UnsupportedModelError
from aisl_train.inspect import ModelInspection, inspect_model_path, reject_if_unsupported
from aisl_train.quantization import build_quantization_config


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    inspection: ModelInspection
    compute_dtype_name: str
    device_map: Any
    local_files_only: bool
    use_cache_original: bool | None


def local_files_flag(model_path: Path, config: TrainingConfig) -> bool:
    if model_path.exists():
        return True
    if config.allow_hub:
        return False
    return bool(config.local_files_only)


def load_tokenizer(model_path: Path, *, local_files_only: bool) -> Any:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise MissingDependencyError(f"transformers is required to load a tokenizer: {exc}") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=local_files_only,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(
    config: TrainingConfig,
    *,
    inspection: ModelInspection | None = None,
    for_training: bool = True,
) -> LoadedModel:
    try:
        from transformers import AutoModelForCausalLM
    except Exception as exc:
        raise MissingDependencyError(f"transformers is required to load a model: {exc}") from exc

    model_path = config.resolved_model_path()
    inspected = inspection or inspect_model_path(model_path)
    reject_if_unsupported(inspected)
    offline = local_files_flag(model_path, config)
    compute_dtype, compute_name = resolve_compute_dtype(config.precision, config.quantization.compute_dtype)
    quant_config = None
    if config.method == "qlora" or config.quantization.enabled:
        quant_config = build_quantization_config(config.quantization, compute_dtype)
    device_map = device_map_for(config.device_strategy, headroom_gib=config.memory_headroom_gib)
    max_memory = max_memory_map(config.memory_headroom_gib) if device_map == "auto" else None

    load_kwargs: dict[str, Any] = {
        "local_files_only": offline,
        "trust_remote_code": False,
        "device_map": device_map,
        "torch_dtype": compute_dtype,
    }
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory

    try:
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
    except Exception as exc:
        _reraise_load_failure(exc, inspected, config, compute_name, device_map)

    tokenizer = load_tokenizer(model_path, local_files_only=offline)
    use_cache_original = getattr(getattr(model, "config", None), "use_cache", None)
    if for_training and config.gradient_checkpointing:
        if hasattr(model, "config"):
            model.config.use_cache = False
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        inspection=inspected,
        compute_dtype_name=compute_name,
        device_map=device_map,
        local_files_only=offline,
        use_cache_original=use_cache_original,
    )


def restore_use_cache(loaded: LoadedModel) -> None:
    if loaded.use_cache_original is None:
        return
    if hasattr(loaded.model, "config"):
        loaded.model.config.use_cache = loaded.use_cache_original


def _reraise_load_failure(
    exc: Exception,
    inspection: ModelInspection,
    config: TrainingConfig,
    compute_name: str,
    device_map: Any,
) -> None:
    message = str(exc)
    if "out of memory" in message.lower() or exc.__class__.__name__ == "OutOfMemoryError":
        from aisl_train.errors import OutOfMemoryError
        from aisl_train.devices import snapshot_memory

        raise OutOfMemoryError(
            "CUDA OOM while loading the model. "
            f"model={inspection.model_path} quantization={config.quantization.scheme} "
            f"seq={config.max_seq_length} micro_batch={config.micro_batch_size} "
            f"lora_rank={config.lora.rank} targets={config.lora.target_modules} "
            f"device_map={device_map} compute={compute_name} "
            f"memory={snapshot_memory('load_oom')}"
        ) from exc
    raise UnsupportedModelError(f"failed to load causal LM from {inspection.model_path}: {exc}") from exc
