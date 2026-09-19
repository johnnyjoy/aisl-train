# SPDX-License-Identifier: Apache-2.0
"""Demonstrate exactly which token positions are trained vs ignored."""

from aisl_train.config import IGNORE_INDEX
from aisl_train.masking import apply_span_mask, completion_only_mask, mask_from_assistant_token_flags

# Tiny deterministic vocabulary for the unit test.
#   1 = BOS   2 = EOS   3 = USER   4 = ASSISTANT
#   10='A' 11='B'   20='C' 21='D'
USER = [3, 10, 11]  # USER A B
ASSISTANT = [4, 20, 21]  # ASSISTANT C D
BOS = [1]
EOS = [2]


def test_assistant_spans_are_the_only_trained_positions() -> None:
    """
    Conversation: user "AB" / assistant "CD"

    Token stream:
        0:BOS  1:USER  2:A  3:B  4:ASSISTANT  5:C  6:D  7:EOS

    Trainable assistant content is C, D, and EOS (span [5, 8)).
    USER tokens and the assistant role marker stay at -100.
    """
    input_ids = BOS + USER + ASSISTANT + EOS
    masked = apply_span_mask(input_ids, [(5, 8)])
    assert masked.input_ids == [1, 3, 10, 11, 4, 20, 21, 2]
    assert masked.labels == [
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        20,
        21,
        2,
    ]
    assert masked.assistant_positions == [5, 6, 7]
    assert masked.ignored_positions == [0, 1, 2, 3, 4]
    assert all(masked.labels[i] != IGNORE_INDEX for i in masked.assistant_positions)
    assert all(masked.labels[i] == IGNORE_INDEX for i in masked.ignored_positions)


def test_completion_only_mask() -> None:
    prompt = [1, 3, 10]
    completion = [20, 21, 2]
    masked = completion_only_mask(prompt, completion)
    assert masked.labels == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 20, 21, 2]


def test_flags_to_spans() -> None:
    input_ids = [1, 2, 3, 4, 5]
    flags = [0, 0, 1, 1, 0]
    masked = mask_from_assistant_token_flags(input_ids, flags)
    assert masked.labels == [IGNORE_INDEX, IGNORE_INDEX, 3, 4, IGNORE_INDEX]
