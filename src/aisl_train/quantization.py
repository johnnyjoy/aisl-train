# SPDX-License-Identifier: Apache-2.0
"""Generic bitsandbytes 4-bit configuration. No custom quantization math."""

from __future__ import annotations

from typing import Any

from aisl_train.config import QuantizationSettings, TrainingConfig
from aisl_train.errors import MissingCapabilityError, MissingDependencyError


def bitsandbytes_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        return False
    return True


def build_quantization_config(
    settings: QuantizationSettings,
    compute_dtype: Any,
) -> Any | None:
    if not settings.enabled or settings.scheme == "none":
        return None
    if not bitsandbytes_available():
        raise MissingDependencyError("bitsandbytes is required for 4-bit / QLoRA loading")
    try:
        from transformers import BitsAndBytesConfig
    except Exception as exc:
        raise MissingDependencyError(f"transformers BitsAndBytesConfig is unavailable: {exc}") from exc

    if settings.scheme not in {"nf4", "fp4"}:
        raise MissingCapabilityError(f"unsupported quantization scheme: {settings.scheme}")

    return BitsAndBytesConfig(
        load_in_4bit=settings.load_in_4bit,
        bnb_4bit_quant_type=settings.scheme,
        bnb_4bit_use_double_quant=settings.double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def quantization_metadata(config: TrainingConfig, actual_compute_dtype: str) -> dict[str, Any]:
    return {
        "enabled": config.quantization.enabled,
        "scheme": config.quantization.scheme,
        "load_in_4bit": config.quantization.load_in_4bit,
        "double_quant": config.quantization.double_quant,
        "configured_compute_dtype": config.quantization.compute_dtype,
        "actual_compute_dtype": actual_compute_dtype,
    }
