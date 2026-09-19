# SPDX-License-Identifier: Apache-2.0
"""Experiment directories, provenance, and resume selection."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from aisl_train.errors import CheckpointError, ExperimentError

PACKAGE_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_SUBDIRS = (
    "baseline",
    "prompted",
    "smoke",
    "checkpoints",
    "adapter",
    "training",
    "trained-eval",
    "comparison",
)


@dataclass
class ExperimentPaths:
    root: Path
    baseline: Path
    prompted: Path
    smoke: Path
    checkpoints: Path
    adapter: Path
    training: Path
    trained_eval: Path
    comparison: Path

    def to_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


@dataclass
class Provenance:
    aisl_train_version: str
    aisl_train_commit: str | None
    aisl_version: str | None = None
    aisl_corpus_version: str | None = None
    aisl_corpus_content_hash: str | None = None
    aisl_dictionary_versions: list[str] = field(default_factory=list)
    aisl_repository_commit: str | None = None
    model_path: str | None = None
    model_type: str | None = None
    model_revision: str | None = None
    profile_id: str | None = None
    profile_hash: str | None = None
    seed: int | None = None
    created_at: str = ""
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_commit(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ExperimentError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ExperimentError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ExperimentError(f"{path} must contain a JSON object")
    return raw


def prepare_experiment_dir(root: Path, *, create_subdirs: bool = True) -> ExperimentPaths:
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise ExperimentError(f"output directory is not writable: {root}: {exc}") from exc
    paths = ExperimentPaths(
        root=root,
        baseline=root / "baseline",
        prompted=root / "prompted",
        smoke=root / "smoke",
        checkpoints=root / "checkpoints",
        adapter=root / "adapter",
        training=root / "training",
        trained_eval=root / "trained-eval",
        comparison=root / "comparison",
    )
    if create_subdirs:
        for name in EXPERIMENT_SUBDIRS:
            (root / name).mkdir(parents=True, exist_ok=True)
    return paths


def config_fingerprint(fields: Mapping[str, Any]) -> str:
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_provenance(
    *,
    aisl_train_version: str,
    model_path: str | None = None,
    model_type: str | None = None,
    model_revision: str | None = None,
    profile_id: str | None = None,
    profile_hash: str | None = None,
    seed: int | None = None,
    manifest: Mapping[str, Any] | None = None,
    aisl_root: Path | None = None,
) -> Provenance:
    missing: list[str] = []
    commit = git_commit(PACKAGE_ROOT)
    if not commit:
        missing.append("aisl_train_commit")
    aisl_commit = git_commit(aisl_root) if aisl_root else None
    if aisl_root and not aisl_commit:
        missing.append("aisl_repository_commit")
    aisl_version = None
    corpus_version = None
    content_hash = None
    dictionaries: list[str] = []
    if manifest:
        aisl_version = _as_str(manifest.get("aisl_version"))
        corpus_version = _as_str(manifest.get("dataset_version"))
        content_hash = _as_str(manifest.get("content_hash"))
        raw_dicts = manifest.get("dictionary_versions") or []
        if isinstance(raw_dicts, list):
            dictionaries = [str(item) for item in raw_dicts]
    else:
        missing.extend(
            [
                "aisl_version",
                "aisl_corpus_version",
                "aisl_corpus_content_hash",
                "aisl_dictionary_versions",
            ]
        )
    if not model_path:
        missing.append("model_path")
    if not model_revision:
        missing.append("model_revision")
    if not profile_id:
        missing.append("profile_id")
    return Provenance(
        aisl_train_version=aisl_train_version,
        aisl_train_commit=commit,
        aisl_version=aisl_version,
        aisl_corpus_version=corpus_version,
        aisl_corpus_content_hash=content_hash,
        aisl_dictionary_versions=dictionaries,
        aisl_repository_commit=aisl_commit,
        model_path=model_path,
        model_type=model_type,
        model_revision=model_revision,
        profile_id=profile_id,
        profile_hash=profile_hash,
        seed=seed,
        created_at=utc_now(),
        missing=missing,
    )


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def warn_if_missing_baseline(experiment_root: Path) -> str | None:
    baseline_summary = experiment_root / "baseline" / "summary.json"
    if baseline_summary.is_file():
        return None
    return (
        f"experiment {experiment_root} has no baseline/summary.json; "
        "the recommended workflow is baseline (none) → baseline (minimal) → train"
    )


def list_checkpoints(checkpoints_dir: Path) -> list[Path]:
    if not checkpoints_dir.is_dir():
        return []
    found = [p for p in checkpoints_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    def sort_key(path: Path) -> tuple[int, str]:
        suffix = path.name.split("-", 1)[-1]
        return (int(suffix), path.name) if suffix.isdigit() else (-1, path.name)

    return sorted(found, key=sort_key)


def select_resume_path(spec: str, checkpoints_dir: Path) -> Path:
    if spec == "latest":
        checkpoints = list_checkpoints(checkpoints_dir)
        if not checkpoints:
            raise CheckpointError(f"no checkpoints in {checkpoints_dir}")
        return checkpoints[-1]
    path = Path(spec).expanduser()
    if not path.exists():
        raise CheckpointError(f"resume path does not exist: {path}")
    return path


def assert_resume_compatible(
    *,
    current: Mapping[str, Any],
    saved: Mapping[str, Any],
    force: bool = False,
) -> None:
    mismatches: list[str] = []
    for key in (
        "model_path",
        "train_path",
        "method",
        "max_seq_length",
        "loss_mask",
        "seed",
    ):
        if current.get(key) != saved.get(key):
            mismatches.append(f"{key}: saved={saved.get(key)!r} current={current.get(key)!r}")
    if current.get("lora") != saved.get("lora"):
        mismatches.append("lora settings differ")
    if current.get("quantization") != saved.get("quantization"):
        mismatches.append("quantization settings differ")
    if not mismatches:
        return
    detail = "; ".join(mismatches)
    if force:
        return
    raise CheckpointError(
        "resume checkpoint is incompatible with the current configuration "
        f"({detail}); pass --force-resume to override"
    )
