# SPDX-License-Identifier: Apache-2.0
"""Explicit capability object. Do not scatter capability checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Iterable, Mapping

from aisl_train.errors import MissingCapabilityError


@dataclass
class Capabilities:
    causal_lm: bool = False
    chat_template: bool = False
    bf16: bool = False
    fp16: bool = False
    bnb_4bit: bool = False
    nf4: bool = False
    double_quant: bool = False
    gradient_checkpointing: bool = False
    peft_lora: bool = False
    all_linear_lora: bool = False
    multi_gpu: bool = False
    generation: bool = False
    notes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def require(self, names: Iterable[str]) -> None:
        missing = [name for name in names if not bool(getattr(self, name, False))]
        if missing:
            detail = "; ".join(f"{name}: {self.notes.get(name, 'unavailable')}" for name in missing)
            raise MissingCapabilityError(f"required capability missing: {', '.join(missing)} ({detail})")

    def optional_status(self, name: str) -> tuple[bool, str]:
        available = bool(getattr(self, name, False))
        return available, self.notes.get(name, "available" if available else "unavailable")


CAPABILITY_NAMES = tuple(f.name for f in fields(Capabilities) if f.name != "notes")


def capabilities_from_probes(probes: Mapping[str, Any]) -> Capabilities:
    notes = {str(k): str(v) for k, v in (probes.get("notes") or {}).items()}
    values = {name: bool(probes.get(name, False)) for name in CAPABILITY_NAMES}
    return Capabilities(**values, notes=notes)
