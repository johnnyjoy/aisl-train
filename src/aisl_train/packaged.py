# SPDX-License-Identifier: Apache-2.0
"""Packaged AISL corpus defaults from the training image.

These paths are container defaults. Explicit CLI paths always win.
A sibling aisl checkout is not required.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from aisl_train.config import DEFAULT_CONTAINER_CACHE, DEFAULT_CONTAINER_MODEL, DEFAULT_CONTAINER_OUTPUT, TrainingConfig

DEFAULT_PACKAGED_DATA = Path("/opt/aisl-data")
DEFAULT_PACKAGED_ROOT = Path("/workspace/aisl")
AISL_REF_FILE = Path("/workspace/AISL_REF")
AISL_COMMIT_FILE = Path("/workspace/AISL_COMMIT")
AISL_TRAIN_COMMIT_FILE = Path("/opt/aisl-train/AISL_TRAIN_COMMIT")
IMAGE_METADATA_FILE = Path("/opt/aisl-train/image-metadata.json")

REQUIRED_EXPORTS = (
    "train.chat.jsonl",
    "validation.chat.jsonl",
    "test.chat.jsonl",
    "manifest.json",
)


def packaged_data_dir() -> Path:
    return Path(os.environ.get("AISL_PACKAGED_DATA", str(DEFAULT_PACKAGED_DATA)))


def packaged_aisl_root() -> Path:
    return Path(os.environ.get("AISL_ROOT", str(DEFAULT_PACKAGED_ROOT)))


def default_train_path() -> Path:
    return packaged_data_dir() / "train.chat.jsonl"


def default_validation_path() -> Path:
    return packaged_data_dir() / "validation.chat.jsonl"


def default_eval_path() -> Path:
    return packaged_data_dir() / "test.chat.jsonl"


def default_manifest_path() -> Path:
    return packaged_data_dir() / "manifest.json"


def default_records_path() -> Path:
    return packaged_data_dir() / "records.jsonl"


def apply_packaged_defaults(config: TrainingConfig) -> TrainingConfig:
    """Fill omitted dataset/eval paths with packaged AISL locations."""
    root = packaged_aisl_root()
    updates: dict[str, Any] = {}
    if not config.train_path:
        updates["train_path"] = str(default_train_path())
    if not config.validation_path:
        updates["validation_path"] = str(default_validation_path())
    if not config.eval_path:
        updates["eval_path"] = str(default_eval_path())
    if not config.manifest_path:
        updates["manifest_path"] = str(default_manifest_path())
    if not config.records_path:
        updates["records_path"] = str(default_records_path())
    if not config.aisl_root and (root / "eval").is_dir():
        updates["aisl_root"] = str(root)
    if not config.model_path and Path(DEFAULT_CONTAINER_MODEL).exists():
        updates["model_path"] = DEFAULT_CONTAINER_MODEL
    if not updates:
        return config
    return config.overlay(**updates)


def load_aisl_build_metadata() -> dict[str, Any]:
    image = _read_json(IMAGE_METADATA_FILE)
    return {
        "aisl_ref_requested": _read_text(AISL_REF_FILE) or os.environ.get("AISL_REF"),
        "aisl_commit": _read_text(AISL_COMMIT_FILE),
        "aisl_train_commit": _read_text(AISL_TRAIN_COMMIT_FILE),
        "aisl_root": str(packaged_aisl_root()) if packaged_aisl_root().exists() else None,
        "packaged_data": str(packaged_data_dir()),
        "image_metadata": image,
    }


def packaged_corpus_status() -> dict[str, Any]:
    root = packaged_data_dir()
    files = {name: (root / name).is_file() for name in REQUIRED_EXPORTS}
    extra = {
        "records.jsonl": (root / "records.jsonl").is_file(),
        "stats.json": (root / "stats.json").is_file(),
        "train.prompt.jsonl": (root / "train.prompt.jsonl").is_file(),
    }
    return {
        "directory": str(root),
        "present": root.is_dir(),
        "required": files,
        "optional": extra,
        "ok": root.is_dir() and all(files.values()),
    }


def in_training_container() -> bool:
    return Path("/.dockerenv").exists() or DEFAULT_PACKAGED_DATA.is_dir()


def container_runtime_paths() -> dict[str, Path]:
    return {
        "model": Path(DEFAULT_CONTAINER_MODEL),
        "output": Path(DEFAULT_CONTAINER_OUTPUT),
        "cache": Path(DEFAULT_CONTAINER_CACHE),
    }


def _read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None
