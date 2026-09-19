# SPDX-License-Identifier: Apache-2.0
import pytest

from aisl_train.capabilities import Capabilities
from aisl_train.errors import MissingCapabilityError


def test_require_fails_clearly() -> None:
    caps = Capabilities(causal_lm=True, bnb_4bit=False, notes={"bnb_4bit": "not installed"})
    with pytest.raises(MissingCapabilityError, match="bnb_4bit"):
        caps.require(["causal_lm", "bnb_4bit"])


def test_optional_status() -> None:
    caps = Capabilities(bf16=False, notes={"bf16": "no CUDA"})
    ok, note = caps.optional_status("bf16")
    assert ok is False
    assert "CUDA" in note
