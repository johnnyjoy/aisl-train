# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from aisl_train.dataset import (
    analyze_sequence_lengths,
    check_split_disjointness,
    enforce_truncation_policy,
    load_jsonl_records,
)
from aisl_train.errors import DatasetError

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "testdata" / "fixtures"


def test_load_chat_fixture() -> None:
    ds = load_jsonl_records(FIXTURES / "sample.chat.jsonl")
    assert ds.kind == "chat"
    assert len(ds.records) == 3
    assert ds.records[0].assistant_text() == "Fact"


def test_load_prompt_fixture() -> None:
    ds = load_jsonl_records(FIXTURES / "sample.prompt.jsonl")
    assert ds.kind == "prompt_completion"
    assert ds.records[0].completion == "Fact"


def test_malformed_reports_line_number() -> None:
    with pytest.raises(DatasetError, match="malformed.chat.jsonl:1"):
        load_jsonl_records(FIXTURES / "malformed.chat.jsonl")


def test_missing_dataset() -> None:
    with pytest.raises(DatasetError, match="not found"):
        load_jsonl_records(FIXTURES / "does-not-exist.jsonl")


def test_leakage_detection() -> None:
    train = load_jsonl_records(FIXTURES / "leakage" / "train.jsonl")
    test = load_jsonl_records(FIXTURES / "leakage" / "test.jsonl")
    with pytest.raises(DatasetError, match="leakage"):
        check_split_disjointness(train, test)


def test_sequence_length_stats_and_policy() -> None:
    stats = analyze_sequence_lengths([10, 20, 30, 40, 100], max_seq_length=50)
    assert stats.minimum == 10
    assert stats.maximum == 100
    assert stats.truncated_count == 1
    assert stats.median == 30
    warning = enforce_truncation_policy(
        stats, warn_pct=1.0, fail_pct=50.0, allow_high_truncation=False
    )
    assert warning is not None
    with pytest.raises(DatasetError, match="exceed max_seq_length"):
        enforce_truncation_policy(
            stats, warn_pct=1.0, fail_pct=10.0, allow_high_truncation=False
        )
