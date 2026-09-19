# SPDX-License-Identifier: Apache-2.0
"""Device placement and GPU memory snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from aisl_train.errors import MissingCapabilityError, MissingDependencyError

GIB = 1024**3

try:
    import torch
except Exception as exc:  # pragma: no cover - imported only on GPU/train paths
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def _require_torch() -> None:
    if torch is None:
        raise MissingDependencyError(f"torch is required for device operations: {_TORCH_IMPORT_ERROR}")


@dataclass
class GPUMemory:
    index: int
    name: str
    total_bytes: int
    free_bytes: int | None
    allocated_bytes: int | None
    reserved_bytes: int | None
    peak_allocated_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["total_gib"] = round(self.total_bytes / GIB, 2)
        if self.free_bytes is not None:
            payload["free_gib"] = round(self.free_bytes / GIB, 2)
        return payload


def cuda_available() -> bool:
    _require_torch()
    return bool(torch.cuda.is_available())


def gpu_count() -> int:
    _require_torch()
    return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0


def bf16_supported() -> bool:
    _require_torch()
    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def require_cuda(device_strategy: str) -> None:
    if device_strategy == "cpu":
        return
    if not cuda_available():
        raise MissingCapabilityError("CUDA is required for GPU training; device_strategy=cpu was not selected")


def snapshot_memory(label: str) -> dict[str, Any]:
    _require_torch()
    devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            free: int | None
            total: int
            try:
                free, total = torch.cuda.mem_get_info(index)
            except Exception:
                free, total = None, int(props.total_memory)
            devices.append(
                GPUMemory(
                    index=index,
                    name=props.name,
                    total_bytes=int(total if total else props.total_memory),
                    free_bytes=int(free) if free is not None else None,
                    allocated_bytes=int(torch.cuda.memory_allocated(index)),
                    reserved_bytes=int(torch.cuda.memory_reserved(index)),
                    peak_allocated_bytes=int(torch.cuda.max_memory_allocated(index)),
                ).to_dict()
            )
    return {"label": label, "cuda": bool(torch.cuda.is_available()), "devices": devices}


def reset_peak_memory() -> None:
    _require_torch()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def device_map_for(strategy: str, *, headroom_gib: float) -> str | dict[str, int]:
    """One logical model. Never start one replica per GPU."""
    if strategy == "cpu":
        return "cpu"
    if strategy == "single":
        return "cuda:0" if cuda_available() else "cpu"
    if strategy.startswith("cuda:"):
        return strategy
    return "auto"


def max_memory_map(headroom_gib: float) -> dict[int, str] | None:
    _require_torch()
    if not torch.cuda.is_available():
        return None
    mapping: dict[int, str] = {}
    for index in range(torch.cuda.device_count()):
        try:
            free, _total = torch.cuda.mem_get_info(index)
        except Exception:
            free = torch.cuda.get_device_properties(index).total_memory
        usable_gib = max(int((int(free) - int(headroom_gib * GIB)) / GIB), 1)
        mapping[index] = f"{usable_gib}GiB"
    return mapping


def resolve_compute_dtype(precision: str, compute_dtype: str) -> tuple[Any, str]:
    _require_torch()
    requested = compute_dtype if compute_dtype != "auto" else precision
    if requested in {"auto", "bf16", "bfloat16"}:
        if bf16_supported():
            return torch.bfloat16, "bfloat16"
        if requested in {"bf16", "bfloat16"}:
            raise MissingCapabilityError("BF16 was required but torch.cuda.is_bf16_supported() is false")
        return torch.float16, "float16"
    if requested in {"fp16", "float16"}:
        return torch.float16, "float16"
    if requested in {"fp32", "float32"}:
        return torch.float32, "float32"
    raise MissingCapabilityError(f"unresolved compute dtype: {precision}/{compute_dtype}")
