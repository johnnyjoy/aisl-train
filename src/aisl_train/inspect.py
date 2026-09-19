# SPDX-License-Identifier: Apache-2.0
"""Generic model inspection from filesystem metadata.

Weight loading is optional. Inspect must not train.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from aisl_train.errors import UnsupportedModelError
from aisl_train.profile import ModelProfile

try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING as _CONFIG_MAPPING
except Exception:  # transformers is optional for filesystem inspect
    _CONFIG_MAPPING = None


@dataclass
class LinearCensus:
    unique_suffixes: list[str] = field(default_factory=list)
    weight_tensors: int = 0
    linear_like_tensors: int = 0
    parameter_like_anomalies: list[str] = field(default_factory=list)


@dataclass
class ModelInspection:
    model_path: str
    model_name: str | None
    architecture: str | None
    architectures: list[str]
    model_type: str | None
    is_encoder_decoder: bool | None
    tie_word_embeddings: bool | None
    torch_dtype: str | None
    max_position_embeddings: int | None
    vocab_size: int | None
    hidden_size: int | None
    num_hidden_layers: int | None
    parameter_count: int | None
    parameter_count_source: str | None
    tokenizer_class: str | None
    chat_template_present: bool
    chat_template_has_generation_block: bool
    local_files_only: bool
    transformers_recognized: bool
    supported_class: bool
    linear: LinearCensus
    matched_profile: str | None = None
    profile_overrides: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    raw_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def inspect_model_path(model_path: str | Path, *, profile: ModelProfile | None = None) -> ModelInspection:
    path = Path(model_path).expanduser()
    if not path.exists():
        raise UnsupportedModelError(f"model path does not exist: {path}")
    raw_config = _read_json(path / "config.json") or {}
    tokenizer_config = _read_json(path / "tokenizer_config.json") or {}
    generation_config = _read_json(path / "generation_config.json") or {}
    index = _read_json(path / "model.safetensors.index.json") or _read_json(path / "pytorch_model.bin.index.json")

    architectures = list(raw_config.get("architectures") or [])
    model_type = raw_config.get("model_type")
    is_encoder_decoder = raw_config.get("is_encoder_decoder")
    if is_encoder_decoder is None:
        is_encoder_decoder = any("EncoderDecoder" in str(a) or "Seq2Seq" in str(a) for a in architectures)
    supported = _is_supported_causal_lm(raw_config, architectures, is_encoder_decoder)

    chat_template = tokenizer_config.get("chat_template")
    if not chat_template:
        chat_file = path / "chat_template.jinja"
        if chat_file.is_file():
            chat_template = chat_file.read_text(encoding="utf-8")
    chat_present = bool(chat_template)
    generation_block = bool(chat_template) and (
        "{% generation %}" in str(chat_template) or "{%- generation %}" in str(chat_template)
    )

    param_count, param_source = _parameter_count(raw_config, index)
    linear = census_from_weight_map((index or {}).get("weight_map") if index else None)
    recognized = _transformers_recognizes(model_type)

    notes: list[str] = []
    if not supported:
        notes.append("not a supported decoder-only causal LM according to config metadata")
    if is_encoder_decoder:
        notes.append("encoder-decoder models are rejected by this runtime")
    if not recognized:
        notes.append(
            "Transformers may not recognize this model_type; inspect used raw config.json"
        )
    if not chat_present:
        notes.append("no chat template found in tokenizer_config.json or chat_template.jinja")
    elif not generation_block:
        notes.append("chat template lacks {% generation %}; assistant-only masking may be unavailable")

    inspection = ModelInspection(
        model_path=str(path),
        model_name=path.name,
        architecture=architectures[0] if architectures else None,
        architectures=architectures,
        model_type=str(model_type) if model_type is not None else None,
        is_encoder_decoder=bool(is_encoder_decoder) if is_encoder_decoder is not None else None,
        tie_word_embeddings=raw_config.get("tie_word_embeddings"),
        torch_dtype=_dtype_name(raw_config.get("torch_dtype") or generation_config.get("torch_dtype")),
        max_position_embeddings=_as_int(
            raw_config.get("max_position_embeddings") or raw_config.get("max_sequence_length")
        ),
        vocab_size=_as_int(raw_config.get("vocab_size")),
        hidden_size=_as_int(raw_config.get("hidden_size")),
        num_hidden_layers=_as_int(raw_config.get("num_hidden_layers")),
        parameter_count=param_count,
        parameter_count_source=param_source,
        tokenizer_class=tokenizer_config.get("tokenizer_class"),
        chat_template_present=chat_present,
        chat_template_has_generation_block=generation_block,
        local_files_only=True,
        transformers_recognized=recognized,
        supported_class=supported,
        linear=linear,
        notes=notes,
        raw_config=_public_config(raw_config),
    )
    if profile:
        inspection.matched_profile = profile.id
        inspection.profile_overrides = profile.overrides()
    return inspection


def reject_if_unsupported(inspection: ModelInspection) -> None:
    if inspection.is_encoder_decoder:
        raise UnsupportedModelError(
            f"{inspection.model_path} looks encoder-decoder; this runtime supports decoder-only causal LMs"
        )
    if not inspection.supported_class:
        raise UnsupportedModelError(
            f"{inspection.model_path} is not a supported Hugging Face causal LM "
            f"(model_type={inspection.model_type!r}, architectures={inspection.architectures})"
        )


def census_from_weight_map(weight_map: Mapping[str, Any] | None) -> LinearCensus:
    if not weight_map:
        return LinearCensus()
    suffixes: set[str] = set()
    linear_like = 0
    anomalies: list[str] = []
    for name in weight_map:
        if not str(name).endswith(".weight"):
            continue
        module = str(name)[: -len(".weight")]
        suffix = module.rsplit(".", 1)[-1]
        suffixes.add(suffix)
        if suffix.endswith("proj") or suffix in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"}:
            linear_like += 1
        if "expert" in module.lower() and "weight" in name:
            if suffix not in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}:
                anomalies.append(module)
    unique = sorted(suffixes)
    return LinearCensus(
        unique_suffixes=unique,
        weight_tensors=len(weight_map),
        linear_like_tensors=linear_like,
        parameter_like_anomalies=sorted(set(anomalies))[:32],
    )


def format_inspection_text(
    inspection: ModelInspection,
    *,
    capabilities: Mapping[str, Any] | None = None,
    devices: Mapping[str, Any] | None = None,
) -> str:
    lines = [
        f"model: {inspection.model_path}",
        f"name: {inspection.model_name}",
        f"model_type: {inspection.model_type}",
        f"architectures: {', '.join(inspection.architectures) or 'unknown'}",
        f"supported causal LM: {inspection.supported_class}",
        f"parameter count: {inspection.parameter_count} ({inspection.parameter_count_source})",
        f"tokenizer class: {inspection.tokenizer_class}",
        f"chat template: {inspection.chat_template_present}",
        f"generation mask block: {inspection.chat_template_has_generation_block}",
        f"context: {inspection.max_position_embeddings}",
        f"dtype: {inspection.torch_dtype}",
        f"matched profile: {inspection.matched_profile or 'none'}",
        f"linear-like tensors: {inspection.linear.linear_like_tensors}",
        f"module suffixes: {', '.join(inspection.linear.unique_suffixes[:24])}",
    ]
    if inspection.linear.parameter_like_anomalies:
        lines.append(
            "PEFT anomalies: " + ", ".join(inspection.linear.parameter_like_anomalies[:8])
        )
    if inspection.profile_overrides:
        lines.append(f"profile overrides: {json.dumps(inspection.profile_overrides, sort_keys=True)}")
    if devices:
        lines.append(f"CUDA: {devices.get('cuda_available')}")
        lines.append(f"GPU count: {devices.get('gpu_count')}")
        for gpu in devices.get("gpus") or []:
            lines.append(
                f"GPU {gpu.get('index')}: {gpu.get('name')} {gpu.get('total_gib')} GiB"
            )
        lines.append(f"BF16: {devices.get('bf16')}")
    if capabilities:
        lines.append("capabilities: " + ", ".join(f"{k}={v}" for k, v in capabilities.items() if k != "notes"))
    for note in inspection.notes:
        lines.append(f"note: {note}")
    return "\n".join(lines) + "\n"


def _is_supported_causal_lm(
    raw_config: Mapping[str, Any],
    architectures: list[str],
    is_encoder_decoder: Any,
) -> bool:
    if is_encoder_decoder:
        return False
    if architectures:
        if any(name.endswith("ForCausalLM") or name.endswith("ForConditionalGeneration") for name in architectures):
            return any(name.endswith("ForCausalLM") for name in architectures)
        if any("Seq2Seq" in name or "EncoderDecoder" in name for name in architectures):
            return False
    model_type = raw_config.get("model_type")
    if raw_config.get("is_encoder_decoder") is False:
        return True
    if model_type and architectures:
        return True
    return bool(model_type)


def _parameter_count(
    raw_config: Mapping[str, Any],
    index: Mapping[str, Any] | None,
) -> tuple[int | None, str | None]:
    for key in ("num_parameters", "n_params"):
        if key in raw_config:
            return int(raw_config[key]), f"config.{key}"
    if index and isinstance(index.get("metadata"), dict):
        total_size = index["metadata"].get("total_size")
        if isinstance(total_size, int):
            # bytes on disk, not a parameter count
            return None, f"index.total_size_bytes={total_size}"
    hidden = _as_int(raw_config.get("hidden_size"))
    layers = _as_int(raw_config.get("num_hidden_layers"))
    vocab = _as_int(raw_config.get("vocab_size"))
    if hidden and layers and vocab:
        # Rough decoder-only estimate, labeled as estimate.
        approx = layers * (12 * hidden * hidden) + vocab * hidden
        return int(approx), "estimate_from_config"
    return None, None


def _transformers_recognizes(model_type: Any) -> bool:
    if not model_type or _CONFIG_MAPPING is None:
        return False
    try:
        return str(model_type) in _CONFIG_MAPPING
    except Exception:
        return False


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    return None


def _dtype_name(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).replace("torch.", "")


def _public_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "model_type",
        "architectures",
        "torch_dtype",
        "is_encoder_decoder",
        "tie_word_embeddings",
        "max_position_embeddings",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_theta",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
    )
    return {key: raw[key] for key in keys if key in raw}
