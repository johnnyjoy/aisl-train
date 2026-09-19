# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from aisl_train.config import TrainingConfig, load_training_config
from aisl_train.errors import ConfigError

ROOT = Path(__file__).resolve().parents[1]


def test_pilot_config_loads() -> None:
    cfg = TrainingConfig.from_json_file(ROOT / "configs" / "pilot-qlora.json")
    assert cfg.method == "qlora"
    assert cfg.quantization.scheme == "nf4"
    assert cfg.quantization.double_quant is True
    assert cfg.lora.rank == 16
    assert cfg.lora.target_modules == "all-linear"
    assert cfg.max_seq_length == 2048
    assert cfg.micro_batch_size == 1
    assert cfg.gradient_accumulation == 16
    assert cfg.epochs == 2
    assert cfg.learning_rate == 2e-4
    assert cfg.seed == 42


def test_cli_overrides_win() -> None:
    cfg = load_training_config(
        ROOT / "configs" / "pilot-qlora.json",
        overrides={"learning_rate": 1e-4, "max_seq_length": 1024},
    )
    assert cfg.learning_rate == 1e-4
    assert cfg.max_seq_length == 1024
    assert cfg.lora.rank == 16


def test_unknown_key_fails() -> None:
    with pytest.raises(ConfigError, match="unknown keys"):
        TrainingConfig.from_dict({"not_a_field": 1})


def test_unsupported_method_fails() -> None:
    with pytest.raises(ConfigError, match="unsupported method"):
        TrainingConfig.from_dict({"method": "dpo"})
