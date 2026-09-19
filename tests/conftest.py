# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "testdata" / "fixtures"


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES
