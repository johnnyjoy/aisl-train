# SPDX-License-Identifier: Apache-2.0
"""Model profiles: overrides only, never the normal source of truth."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from aisl_train.errors import ConfigError

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILES_DIR = PACKAGE_ROOT / "profiles"


@dataclass
class ModelProfile:
    id: str
    version: str = "0.1.0"
    match: dict[str, Any] = field(default_factory=dict)
    peft: dict[str, Any] = field(default_factory=dict)
    tokenizer: dict[str, Any] = field(default_factory=dict)
    precision: dict[str, Any] = field(default_factory=dict)
    quantization: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    runtime_validation: str = "pending"
    source_path: str | None = None
    content_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def overrides(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "peft": self.peft,
                "tokenizer": self.tokenizer,
                "precision": self.precision,
                "quantization": self.quantization,
            }.items()
            if value
        }


def profile_hash(path: Path, raw: Mapping[str, Any]) -> str:
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return digest


def load_profile(path: Path) -> ModelProfile:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read profile {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid profile JSON {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a JSON object")
    profile_id = raw.get("id")
    if not profile_id:
        raise ConfigError(f"{path} is missing required field 'id'")
    allowed = {
        "id",
        "version",
        "match",
        "peft",
        "tokenizer",
        "precision",
        "quantization",
        "notes",
        "runtime_validation",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"profile {path} has unknown keys: {', '.join(unknown)}")
    return ModelProfile(
        id=str(profile_id),
        version=str(raw.get("version", "0.1.0")),
        match=dict(raw.get("match") or {}),
        peft=dict(raw.get("peft") or {}),
        tokenizer=dict(raw.get("tokenizer") or {}),
        precision=dict(raw.get("precision") or {}),
        quantization=dict(raw.get("quantization") or {}),
        notes=str(raw.get("notes") or ""),
        runtime_validation=str(raw.get("runtime_validation") or "pending"),
        source_path=str(path),
        content_hash=profile_hash(path, raw),
    )


def list_profiles(directory: Path | None = None) -> list[ModelProfile]:
    root = directory or DEFAULT_PROFILES_DIR
    if not root.is_dir():
        return []
    profiles: list[ModelProfile] = []
    for path in sorted(root.glob("*.json")):
        if path.name.upper() == "README.JSON":
            continue
        profiles.append(load_profile(path))
    return profiles


def resolve_profile(
    spec: str | None,
    *,
    inspection: Mapping[str, Any] | None = None,
    profiles_dir: Path | None = None,
) -> ModelProfile | None:
    """Resolve an explicit profile path/id, else match inspection, else None."""
    if spec:
        path = Path(spec).expanduser()
        if path.is_file():
            return load_profile(path)
        for profile in list_profiles(profiles_dir):
            if profile.id == spec:
                return profile
        raise ConfigError(f"model profile not found: {spec}")

    if not inspection:
        return None
    matches = [p for p in list_profiles(profiles_dir) if profile_matches(p, inspection)]
    if len(matches) > 1:
        ids = ", ".join(p.id for p in matches)
        raise ConfigError(f"multiple profiles match this model: {ids}; pass --profile")
    return matches[0] if matches else None


def profile_matches(profile: ModelProfile, inspection: Mapping[str, Any]) -> bool:
    match = profile.match
    if not match:
        return False
    model_type = inspection.get("model_type")
    if "model_type" in match and match["model_type"] != model_type:
        return False
    wanted_arch = match.get("architectures")
    if wanted_arch:
        have = set(inspection.get("architectures") or [])
        if not have.intersection(wanted_arch):
            return False
    name = str(inspection.get("model_path") or inspection.get("model_name") or "")
    needles = match.get("name_contains") or []
    if needles and not any(str(n).lower() in name.lower() for n in needles):
        return False
    return True


def merge_resolution(
    *,
    cli_overrides: Mapping[str, Any] | None = None,
    profile: ModelProfile | None = None,
    inspected: Mapping[str, Any] | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Resolution order:

    explicit CLI override
            ↓
    explicit model profile
            ↓
    automatic model inspection
            ↓
    safe generic defaults
    """
    resolved: dict[str, Any] = dict(defaults or {})
    sources: dict[str, str] = {key: "default" for key in resolved}

    if inspected:
        for key, value in inspected.items():
            if value is not None:
                resolved[key] = value
                sources[key] = "inspection"
    if profile:
        for section, payload in profile.overrides().items():
            resolved[section] = payload
            sources[section] = f"profile:{profile.id}"
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None:
                resolved[key] = value
                sources[key] = "cli"
    resolved["_sources"] = sources
    return resolved
