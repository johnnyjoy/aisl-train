# SPDX-License-Identifier: Apache-2.0
"""Deterministic generation for evaluation."""

from __future__ import annotations

import time
from typing import Any

from aisl_train.config import GenerationSettings, HARD_MAX_NEW_TOKENS
from aisl_train.errors import EvaluationError
from aisl_train.formatting import extract_generation_text, format_eval_messages


def generation_kwargs(settings: GenerationSettings, tokenizer: Any) -> dict[str, Any]:
    max_new = min(settings.max_new_tokens, HARD_MAX_NEW_TOKENS)
    eos = settings.stop_token_ids
    if eos is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        eos = [eos_token_id] if eos_token_id is not None else None
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new,
        "do_sample": bool(settings.do_sample),
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if eos:
        kwargs["eos_token_id"] = eos if len(eos) != 1 else eos[0]
    if settings.do_sample:
        if settings.temperature is not None:
            kwargs["temperature"] = settings.temperature
        if settings.top_p is not None:
            kwargs["top_p"] = settings.top_p
    return kwargs


def render_prompt(tokenizer: Any, user_text: str, system_prompt: str | None) -> str:
    messages = format_eval_messages(user_text, system_prompt=system_prompt)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if system_prompt:
        return f"{system_prompt}\n\n{user_text}"
    return user_text


def generate_one(
    model: Any,
    tokenizer: Any,
    user_text: str,
    *,
    settings: GenerationSettings,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    prompt = render_prompt(tokenizer, user_text, system_prompt)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    device = _model_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_len = int(encoded["input_ids"].shape[-1])
    kwargs = generation_kwargs(settings, tokenizer)
    started = time.perf_counter()
    try:
        output = model.generate(**encoded, **kwargs)
    except Exception as exc:
        if "out of memory" in str(exc).lower() or exc.__class__.__name__ == "OutOfMemoryError":
            from aisl_train.errors import OutOfMemoryError
            from aisl_train.devices import snapshot_memory

            raise OutOfMemoryError(
                f"CUDA OOM during generation. max_new_tokens={kwargs.get('max_new_tokens')} "
                f"memory={snapshot_memory('generate_oom')}"
            ) from exc
        raise EvaluationError(f"generation failed: {exc}") from exc
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    sequence = output[0]
    completion_ids = sequence[prompt_len:]
    raw = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return {
        "prompt": prompt,
        "raw": raw,
        "extracted": extract_generation_text(raw),
        "prompt_tokens": prompt_len,
        "completion_tokens": int(completion_ids.shape[-1]),
        "latency_ms": elapsed_ms,
        "generation": {
            "do_sample": kwargs.get("do_sample"),
            "max_new_tokens": kwargs.get("max_new_tokens"),
            "temperature": kwargs.get("temperature"),
            "top_p": kwargs.get("top_p"),
            "eos_token_id": kwargs.get("eos_token_id"),
        },
    }


def _model_device(model: Any) -> Any:
    if hasattr(model, "device"):
        return model.device
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise EvaluationError("model has no parameters") from exc
