# SPDX-License-Identifier: Apache-2.0
import pytest

from aisl_train.errors import MaskingError
from aisl_train.formatting import resolve_format_plan, tokenizer_has_generation_mask, validate_loss_mask_plan
from aisl_train.profile import ModelProfile


def test_prompt_completion_skips_chat_template() -> None:
    plan = resolve_format_plan(
        dataset_kind="prompt_completion",
        tokenizer_chat_template=None,
        loss_mask="assistant",
    )
    assert plan.chat_template_source == "none"
    assert plan.loss_mask == "completion"


def test_tokenizer_template_is_preferred() -> None:
    template = "{% generation %}x{% endgeneration %}"
    plan = resolve_format_plan(
        dataset_kind="chat",
        tokenizer_chat_template=template,
        profile=ModelProfile(id="p", tokenizer={"chat_template": "tokenizer"}),
    )
    assert plan.chat_template_source == "tokenizer"
    validate_loss_mask_plan(plan)


def test_missing_generation_block_fails_assistant_mask() -> None:
    plan = resolve_format_plan(
        dataset_kind="chat",
        tokenizer_chat_template="{{ messages }}",
    )
    assert not tokenizer_has_generation_mask(plan.template)
    with pytest.raises(MaskingError, match="generation"):
        validate_loss_mask_plan(plan)


def test_full_mask_override_is_explicit() -> None:
    plan = resolve_format_plan(
        dataset_kind="chat",
        tokenizer_chat_template="{{ messages }}",
        loss_mask="full",
    )
    validate_loss_mask_plan(plan)
