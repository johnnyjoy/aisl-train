# SPDX-License-Identifier: Apache-2.0
"""Environment snapshot and doctor checks."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from aisl_train.errors import MissingCapabilityError, MissingDependencyError

TRAIN_PACKAGES = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "bitsandbytes",
    "accelerate",
    "datasets",
    "safetensors",
    "huggingface_hub",
)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def snapshot_environment() -> dict[str, Any]:
    versions = {name: package_version(name) for name in TRAIN_PACKAGES}
    versions["python"] = sys.version.split()[0]
    versions["aisl-train"] = package_version("aisl-train")
    cuda = _cuda_probe()
    return {
        "versions": versions,
        "cuda": cuda,
        "env": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
    }


def _cuda_probe() -> dict[str, Any]:
    if not module_available("torch"):
        return {"available": False, "reason": "torch not installed"}
    import torch

    gpus: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_bytes": int(props.total_memory),
                    "total_gib": round(props.total_memory / (1024**3), 2),
                }
            )
    return {
        "available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "runtime": getattr(torch.version, "cuda", None),
        "bf16": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "gpus": gpus,
    }


def doctor_report(
    *,
    model_path: Path | None = None,
    dataset_path: Path | None = None,
    output_path: Path | None = None,
    require_gpu: bool = False,
) -> dict[str, Any]:
    env = snapshot_environment()
    checks: list[dict[str, Any]] = []
    ok = True

    def add(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        checks.append({"name": name, "ok": passed, "detail": detail})
        if not passed:
            ok = False

    add("python", True, env["versions"]["python"])
    for pkg in TRAIN_PACKAGES:
        version = env["versions"].get(pkg)
        add(pkg, version is not None, version or "not installed")

    cuda = env["cuda"]
    add("cuda_available", bool(cuda.get("available")), str(cuda))
    if require_gpu and not cuda.get("available"):
        add("gpu_required", False, "CUDA is required for this check")

    if model_path is not None:
        add("model_path", model_path.exists(), str(model_path))
    if dataset_path is not None:
        add("dataset_path", dataset_path.exists(), str(dataset_path))
    if output_path is not None:
        writable = _writable(output_path)
        add("output_writable", writable, str(output_path))

    return {"ok": ok, "checks": checks, "environment": env}


def assert_doctor(report: dict[str, Any], *, require: list[str] | None = None) -> None:
    required = require or ["torch", "transformers", "trl", "peft"]
    failed = [c for c in report["checks"] if c["name"] in required and not c["ok"]]
    if failed:
        names = ", ".join(c["name"] for c in failed)
        raise MissingDependencyError(f"doctor failed required checks: {names}")
    if not report["ok"] and require is None:
        raise MissingCapabilityError("doctor reported one or more failed checks")


def _writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".doctor-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def format_doctor_text(report: dict[str, Any]) -> str:
    lines = ["aisl-train doctor", ""]
    for check in report["checks"]:
        mark = "ok" if check["ok"] else "FAIL"
        lines.append(f"{mark:4} {check['name']}: {check['detail']}")
    lines.append("")
    lines.append("overall: " + ("ok" if report["ok"] else "FAIL"))
    return "\n".join(lines) + "\n"
