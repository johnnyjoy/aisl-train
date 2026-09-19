# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from aisl_train.errors import UnsupportedModelError
from aisl_train.inspect import inspect_model_path, reject_if_unsupported

FIXTURES = Path(__file__).resolve().parents[1] / "testdata" / "fixtures"


def test_inspect_tiny_causal_model() -> None:
    inspection = inspect_model_path(FIXTURES / "tiny-model")
    assert inspection.model_type == "llama"
    assert inspection.supported_class is True
    assert inspection.chat_template_present is True
    assert inspection.chat_template_has_generation_block is True
    assert "q_proj" in inspection.linear.unique_suffixes
    reject_if_unsupported(inspection)


def test_inspect_rejects_encoder_decoder() -> None:
    inspection = inspect_model_path(FIXTURES / "tiny-encoder-decoder")
    assert inspection.supported_class is False
    with pytest.raises(UnsupportedModelError, match="encoder-decoder"):
        reject_if_unsupported(inspection)
