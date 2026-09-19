# SPDX-License-Identifier: Apache-2.0
"""Chat-template and prompt/completion formatting plan."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from aisl_train.dataset import DatasetKind
from aisl_train.errors import MaskingError
from aisl_train.profile import ModelProfile


@dataclass
class FormatPlan:
    dataset_kind: DatasetKind
    chat_template_source: str
    loss_mask: str
    template: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_format_plan(
    *,
    dataset_kind: DatasetKind,
    tokenizer_chat_template: str | None,
    profile: ModelProfile | None = None,
    loss_mask: str = "assistant",
    cli_template: str | None = None,
) -> FormatPlan:
    if dataset_kind == "prompt_completion":
        chosen_mask = "completion" if loss_mask == "assistant" else loss_mask
        return FormatPlan(
            dataset_kind=dataset_kind,
            chat_template_source="none",
            loss_mask=chosen_mask,
            template=None,
        )

    if cli_template:
        return FormatPlan(
            dataset_kind=dataset_kind,
            chat_template_source="cli",
            loss_mask=loss_mask,
            template=cli_template,
        )

    profile_source = None
    profile_template = None
    if profile and profile.tokenizer:
        profile_source = profile.tokenizer.get("chat_template")
        profile_template = profile.tokenizer.get("template")
        if profile_source == "inline" and profile_template:
            return FormatPlan(
                dataset_kind=dataset_kind,
                chat_template_source="profile",
                loss_mask=loss_mask,
                template=str(profile_template),
            )

    if tokenizer_chat_template:
        return FormatPlan(
            dataset_kind=dataset_kind,
            chat_template_source="tokenizer",
            loss_mask=loss_mask,
            template=tokenizer_chat_template,
        )

    if profile_source == "tokenizer":
        raise MaskingError(
            "profile requests tokenizer chat template, but the tokenizer does not provide one"
        )
    raise MaskingError(
        "no chat template available; supply a tokenizer template, a profile template, "
        "or use a prompt/completion dataset"
    )


def tokenizer_has_generation_mask(chat_template: str | None) -> bool:
    if not chat_template:
        return False
    return "{% generation %}" in chat_template or "{%- generation %}" in chat_template


def validate_loss_mask_plan(plan: FormatPlan) -> None:
    if plan.loss_mask == "full":
        return
    if plan.dataset_kind == "prompt_completion":
        return
    if plan.loss_mask == "assistant" and not tokenizer_has_generation_mask(plan.template):
        raise MaskingError(
            "assistant-only loss requires a chat template with a {% generation %} block "
            "so assistant token positions can be derived; use a profile template override "
            "or --loss-mask full if you intentionally accept training on the whole conversation"
        )


def format_eval_messages(
    user_text: str,
    *,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    return messages


def extract_generation_text(raw: str) -> str:
    """Preserve raw output elsewhere; this is a light surface extraction."""
    return raw.strip()


def format_plan_from_mapping(raw: Mapping[str, Any]) -> FormatPlan:
    return FormatPlan(
        dataset_kind=raw["dataset_kind"],  # type: ignore[arg-type]
        chat_template_source=str(raw["chat_template_source"]),
        loss_mask=str(raw["loss_mask"]),
        template=raw.get("template"),
    )
