# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from aisl_train.errors import CheckpointError
from aisl_train.experiment import assert_resume_compatible, list_checkpoints, select_resume_path


def test_select_latest_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    (root / "checkpoint-10").mkdir(parents=True)
    (root / "checkpoint-200").mkdir()
    (root / "checkpoint-30").mkdir()
    assert select_resume_path("latest", root).name == "checkpoint-200"
    assert [p.name for p in list_checkpoints(root)] == [
        "checkpoint-10",
        "checkpoint-30",
        "checkpoint-200",
    ]


def test_resume_mismatch_fails() -> None:
    current = {
        "model_path": "/models/a",
        "train_path": "/data/train.chat.jsonl",
        "method": "qlora",
        "max_seq_length": 2048,
        "loss_mask": "assistant",
        "seed": 42,
        "lora": {"rank": 16},
        "quantization": {"scheme": "nf4"},
    }
    saved = dict(current)
    saved["model_path"] = "/models/b"
    with pytest.raises(CheckpointError, match="incompatible"):
        assert_resume_compatible(current=current, saved=saved, force=False)
    assert_resume_compatible(current=current, saved=saved, force=True)


def test_missing_latest_fails(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="no checkpoints"):
        select_resume_path("latest", tmp_path / "empty")
