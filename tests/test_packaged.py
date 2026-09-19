# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from aisl_train.config import DEFAULT_CONTAINER_DATA, TrainingConfig, load_training_config
from aisl_train.packaged import (
    apply_packaged_defaults,
    default_eval_path,
    default_train_path,
    default_validation_path,
    load_aisl_build_metadata,
    packaged_corpus_status,
    packaged_data_dir,
)

FIXTURES = Path(__file__).resolve().parents[1] / "testdata" / "fixtures" / "packaged"


def test_container_data_default_is_packaged() -> None:
    assert DEFAULT_CONTAINER_DATA == "/opt/aisl-data"


def test_env_overrides_packaged_dir(monkeypatch) -> None:
    monkeypatch.setenv("AISL_PACKAGED_DATA", str(FIXTURES))
    assert packaged_data_dir() == FIXTURES
    assert default_train_path() == FIXTURES / "train.chat.jsonl"
    assert default_validation_path() == FIXTURES / "validation.chat.jsonl"
    assert default_eval_path() == FIXTURES / "test.chat.jsonl"
    assert packaged_corpus_status()["ok"] is True


def test_apply_defaults_then_explicit_override(monkeypatch) -> None:
    monkeypatch.setenv("AISL_PACKAGED_DATA", str(FIXTURES))
    cfg = apply_packaged_defaults(TrainingConfig())
    assert cfg.train_path == str(FIXTURES / "train.chat.jsonl")
    assert cfg.validation_path == str(FIXTURES / "validation.chat.jsonl")
    assert cfg.eval_path == str(FIXTURES / "test.chat.jsonl")
    assert cfg.manifest_path == str(FIXTURES / "manifest.json")
    overridden = apply_packaged_defaults(
        load_training_config(overrides={"train_path": "/custom/train.jsonl", "eval_path": "/custom/test.jsonl"})
    )
    assert overridden.train_path == "/custom/train.jsonl"
    assert overridden.eval_path == "/custom/test.jsonl"
    assert overridden.validation_path == str(FIXTURES / "validation.chat.jsonl")


def test_aisl_commit_metadata_missing_outside_image() -> None:
    meta = load_aisl_build_metadata()
    assert "aisl_commit" in meta
    assert meta["aisl_commit"] is None or isinstance(meta["aisl_commit"], str)


def test_source_has_no_host_aisl_mount_contract() -> None:
    src = Path(__file__).resolve().parents[1] / "src" / "aisl_train"
    banned = ("AISL_DATA_HOST_PATH", "../aisl", "/srv/aisl-training", "/home/james")
    for path in src.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in banned:
            assert needle not in text, f"{path} contains {needle}"
