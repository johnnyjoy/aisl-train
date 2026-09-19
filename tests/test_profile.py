# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from aisl_train.errors import ConfigError
from aisl_train.profile import merge_resolution, profile_matches, resolve_profile

ROOT = Path(__file__).resolve().parents[1]


def test_qwen_profile_matches_name_not_invented_type() -> None:
    profile = resolve_profile("qwen3.8-27b", profiles_dir=ROOT / "profiles")
    assert profile is not None
    assert profile.peft.get("target_modules") == "all-linear"
    assert "model_type" not in profile.match
    assert profile_matches(profile, {"model_path": "/models/Qwen3.8-27B"})
    assert not profile_matches(profile, {"model_path": "/models/other-model"})


def test_explicit_path_profile() -> None:
    profile = resolve_profile(str(ROOT / "profiles" / "qwen3.8-27b.json"))
    assert profile is not None
    assert profile.id == "qwen3.8-27b"


def test_missing_profile_fails() -> None:
    with pytest.raises(ConfigError, match="not found"):
        resolve_profile("does-not-exist", profiles_dir=ROOT / "profiles")


def test_resolution_order() -> None:
    resolved = merge_resolution(
        defaults={"target_modules": "all-linear", "compute_dtype": "auto"},
        inspected={"compute_dtype": "bfloat16"},
        profile=resolve_profile("qwen3.8-27b", profiles_dir=ROOT / "profiles"),
        cli_overrides={"compute_dtype": "float16"},
    )
    assert resolved["compute_dtype"] == "float16"
    assert resolved["_sources"]["compute_dtype"] == "cli"
