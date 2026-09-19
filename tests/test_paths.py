# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from aisl_train.config import DEFAULT_CONTAINER_DATA, TrainingConfig
from aisl_train.dataset import discover_manifest

SRC = Path(__file__).resolve().parents[1] / "src" / "aisl_train"


def test_container_defaults_are_overrideable() -> None:
    cfg = TrainingConfig(train_path="/custom/train.chat.jsonl", output_path="/custom/out", model_path="/custom/model")
    assert str(cfg.resolved_train_path()) == "/custom/train.chat.jsonl"
    assert DEFAULT_CONTAINER_DATA == "/data/aisl"


def test_manifest_discovery_is_relative_to_dataset(tmp_path: Path) -> None:
    data = tmp_path / "exports"
    data.mkdir()
    (data / "train.chat.jsonl").write_text("{}\n", encoding="utf-8")
    (data / "manifest.json").write_text("{}", encoding="utf-8")
    found = discover_manifest(data / "train.chat.jsonl")
    assert found == data / "manifest.json"


def test_runtime_source_has_no_developer_paths() -> None:
    banned = ("/home/james", "../aisl", "/models/Qwen")
    for path in SRC.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in banned:
            assert needle not in text, f"{path} contains {needle}"
