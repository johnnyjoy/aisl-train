# SPDX-License-Identifier: Apache-2.0
"""Assistant/completion-only loss labels.

The trainer uses current TRL mechanisms at runtime. This module is the
testable definition of which token positions should receive a real label
versus IGNORE_INDEX (-100).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from aisl_train.config import IGNORE_INDEX
from aisl_train.errors import MaskingError


@dataclass(frozen=True)
class MaskedSequence:
    input_ids: list[int]
    labels: list[int]
    assistant_positions: list[int]
    ignored_positions: list[int]


def apply_span_mask(
    input_ids: Sequence[int],
    trainable_spans: Sequence[tuple[int, int]],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> MaskedSequence:
    """Keep labels only on half-open token spans ``[start, end)``."""
    labels = [ignore_index] * len(input_ids)
    if not trainable_spans:
        raise MaskingError("assistant/completion mask is empty; refuse silent full-sequence loss")
    for start, end in trainable_spans:
        if start < 0 or end > len(input_ids) or start >= end:
            raise MaskingError(f"invalid trainable span [{start}, {end}) for length {len(input_ids)}")
        for index in range(start, end):
            labels[index] = int(input_ids[index])
    assistant_positions = [i for i, label in enumerate(labels) if label != ignore_index]
    ignored_positions = [i for i, label in enumerate(labels) if label == ignore_index]
    if not assistant_positions:
        raise MaskingError("derived mask has zero trainable positions")
    return MaskedSequence(
        input_ids=list(input_ids),
        labels=labels,
        assistant_positions=assistant_positions,
        ignored_positions=ignored_positions,
    )


def completion_only_mask(
    prompt_ids: Sequence[int],
    completion_ids: Sequence[int],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> MaskedSequence:
    if not completion_ids:
        raise MaskingError("completion is empty; cannot derive a completion-only mask")
    input_ids = list(prompt_ids) + list(completion_ids)
    start = len(prompt_ids)
    return apply_span_mask(input_ids, [(start, len(input_ids))], ignore_index=ignore_index)


def mask_from_assistant_token_flags(
    input_ids: Sequence[int],
    assistant_mask: Sequence[int],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> MaskedSequence:
    if len(input_ids) != len(assistant_mask):
        raise MaskingError(
            f"assistant token mask length {len(assistant_mask)} != input length {len(input_ids)}"
        )
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(assistant_mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(assistant_mask)))
    return apply_span_mask(input_ids, spans, ignore_index=ignore_index)


def require_mask_or_fail(loss_mask: str, derived: MaskedSequence | None) -> MaskedSequence:
    if derived is not None:
        return derived
    if loss_mask == "full":
        raise MaskingError("internal error: full-sequence loss should not use require_mask_or_fail")
    raise MaskingError(
        "assistant-only loss could not be derived for this tokenizer/template; "
        "pass --loss-mask full only if you intentionally want to train on the entire sequence"
    )
