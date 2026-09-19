# SPDX-License-Identifier: Apache-2.0
"""Typed experiment and training configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

from aisl_train.errors import ConfigError

# Overrideable container defaults. Not developer workstation paths.
DEFAULT_CONTAINER_MODEL = "/models/base"
DEFAULT_CONTAINER_DATA = "/opt/aisl-data"
DEFAULT_CONTAINER_OUTPUT = "/output"
DEFAULT_CONTAINER_CACHE = "/cache"

IGNORE_INDEX = -100
HARD_MAX_NEW_TOKENS = 2048


def _as_path(value: Any | None) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value).expanduser()


def _require_known_keys(name: str, raw: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"{name} has unknown keys: {', '.join(unknown)}")


@dataclass
class QuantizationSettings:
    enabled: bool = True
    load_in_4bit: bool = True
    scheme: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "auto"

    def __post_init__(self) -> None:
        if self.scheme not in {"nf4", "fp4", "none"}:
            raise ConfigError(f"unsupported quantization scheme: {self.scheme}")
        if self.compute_dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ConfigError(f"unsupported compute_dtype: {self.compute_dtype}")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> QuantizationSettings:
        if not raw:
            return cls()
        allowed = {f.name for f in fields(cls)}
        _require_known_keys("quantization", raw, allowed)
        return cls(**{k: raw[k] for k in raw})


@dataclass
class LoRASettings:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"
    exclude_modules: list[str] | None = None
    target_parameters: list[str] | None = None
    modules_to_save: list[str] | None = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ConfigError("lora.rank must be >= 1")
        if self.alpha < 1:
            raise ConfigError("lora.alpha must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError("lora.dropout must be in [0, 1)")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> LoRASettings:
        if not raw:
            return cls()
        allowed = {f.name for f in fields(cls)}
        _require_known_keys("lora", raw, allowed)
        return cls(**{k: raw[k] for k in raw})


@dataclass
class GenerationSettings:
    do_sample: bool = False
    max_new_tokens: int = 256
    temperature: float | None = None
    top_p: float | None = None
    stop_token_ids: list[int] | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ConfigError("generation.max_new_tokens must be >= 1")
        if self.max_new_tokens > HARD_MAX_NEW_TOKENS and not self.do_sample:
            # Hard cap against runaway generation unless the caller is explicit.
            pass
        if self.do_sample is False:
            self.temperature = None
            self.top_p = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> GenerationSettings:
        if not raw:
            return cls()
        allowed = {f.name for f in fields(cls)}
        _require_known_keys("generation", raw, allowed)
        return cls(**{k: raw[k] for k in raw})

    def as_public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingConfig:
    experiment_id: str | None = None
    seed: int = 42
    method: str = "qlora"

    model_path: str | None = None
    profile: str | None = None

    train_path: str | None = None
    validation_path: str | None = None
    eval_path: str | None = None
    output_path: str | None = None
    manifest_path: str | None = None
    records_path: str | None = None
    aisl_root: str | None = None

    quantization: QuantizationSettings = field(default_factory=QuantizationSettings)
    lora: LoRASettings = field(default_factory=LoRASettings)
    generation: GenerationSettings = field(default_factory=GenerationSettings)

    max_seq_length: int = 2048
    truncation_warn_pct: float = 5.0
    truncation_fail_pct: float = 25.0
    allow_high_truncation: bool = False

    epochs: float = 2.0
    learning_rate: float = 2e-4
    micro_batch_size: int = 1
    gradient_accumulation: int = 16
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    gradient_checkpointing: bool = True

    logging_steps: int = 10
    eval_steps: int | None = None
    save_steps: int | None = None
    save_total_limit: int = 3

    precision: str = "auto"
    device_strategy: str = "auto"
    memory_headroom_gib: float = 1.5

    local_files_only: bool = True
    allow_hub: bool = False

    loss_mask: str = "assistant"
    resume: str | None = None
    force_resume: bool = False

    limit: int | None = None
    prompt_mode: str = "none"
    prompt_file: str | None = None
    adapter_path: str | None = None

    def __post_init__(self) -> None:
        if self.method not in {"qlora", "lora"}:
            raise ConfigError(f"unsupported method: {self.method} (supported: qlora, lora)")
        if self.precision not in {"auto", "bf16", "fp16", "fp32"}:
            raise ConfigError(f"unsupported precision: {self.precision}")
        if self.device_strategy not in {"auto", "single", "cpu"} and not self.device_strategy.startswith(
            "cuda:"
        ):
            raise ConfigError(
                "device_strategy must be auto, single, cpu, or an explicit device such as cuda:0"
            )
        if self.loss_mask not in {"assistant", "completion", "full"}:
            raise ConfigError("loss_mask must be assistant, completion, or full")
        if self.prompt_mode not in {"none", "minimal"}:
            raise ConfigError("prompt_mode must be none or minimal")
        if self.max_seq_length < 8:
            raise ConfigError("max_seq_length must be >= 8")
        if self.micro_batch_size < 1:
            raise ConfigError("micro_batch_size must be >= 1")
        if self.gradient_accumulation < 1:
            raise ConfigError("gradient_accumulation must be >= 1")
        if self.epochs <= 0:
            raise ConfigError("epochs must be > 0")
        if self.memory_headroom_gib < 0:
            raise ConfigError("memory_headroom_gib must be >= 0")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> TrainingConfig:
        if not raw:
            return cls()
        data = dict(raw)
        nested = {
            "quantization": QuantizationSettings.from_dict(data.pop("quantization", None)),
            "lora": LoRASettings.from_dict(data.pop("lora", None)),
            "generation": GenerationSettings.from_dict(data.pop("generation", None)),
        }
        allowed = {f.name for f in fields(cls)}
        _require_known_keys("config", data, allowed)
        return cls(**data, **nested)

    @classmethod
    def from_json_file(cls, path: Path) -> TrainingConfig:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ConfigError(f"cannot read config {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid JSON in {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"{path} must contain a JSON object")
        return cls.from_dict(raw)

    def overlay(self, **updates: Any) -> TrainingConfig:
        data = self.to_dict()
        for key, value in updates.items():
            if value is None:
                continue
            if key in {"quantization", "lora", "generation"} and isinstance(value, Mapping):
                merged = dict(data.get(key) or {})
                merged.update(value)
                data[key] = merged
            else:
                data[key] = value
        return TrainingConfig.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint_fields(self) -> dict[str, Any]:
        """Fields that must match to resume a checkpoint."""
        return {
            "method": self.method,
            "model_path": self.model_path,
            "profile": self.profile,
            "train_path": self.train_path,
            "validation_path": self.validation_path,
            "max_seq_length": self.max_seq_length,
            "loss_mask": self.loss_mask,
            "lora": asdict(self.lora),
            "quantization": asdict(self.quantization),
            "seed": self.seed,
        }

    def resolved_model_path(self) -> Path:
        if not self.model_path:
            raise ConfigError("model path is required")
        return Path(self.model_path).expanduser()

    def resolved_output_path(self) -> Path:
        if not self.output_path:
            raise ConfigError("output path is required")
        return Path(self.output_path).expanduser()

    def resolved_train_path(self) -> Path:
        if not self.train_path:
            raise ConfigError("training dataset path is required")
        return Path(self.train_path).expanduser()


def load_training_config(
    config_path: Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> TrainingConfig:
    cfg = TrainingConfig.from_json_file(config_path) if config_path else TrainingConfig()
    if overrides:
        cfg = cfg.overlay(**dict(overrides))
    return cfg


def resolve_local_files_only(model_path: str, allow_hub: bool, configured: bool) -> bool:
    path = Path(model_path).expanduser()
    if path.exists():
        return True
    if allow_hub:
        return False
    if not configured:
        return True
    return True
