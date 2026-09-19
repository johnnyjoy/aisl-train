# SPDX-License-Identifier: Apache-2.0
"""Trainer-neutral JSONL loading and validation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping

from aisl_train.errors import DatasetError

DatasetKind = Literal["chat", "prompt_completion", "eval_case"]
ALLOWED_ROLES = {"system", "user", "assistant"}


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class DatasetRecord:
    line_number: int
    kind: DatasetKind
    messages: list[ChatMessage] | None = None
    prompt: str | None = None
    completion: str | None = None
    record_id: str | None = None
    semantic_case_id: str | None = None
    split: str | None = None
    task: str | None = None
    language: str | None = None
    difficulty: str | None = None
    category: str | None = None
    expected: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def user_text(self) -> str:
        if self.kind == "prompt_completion":
            return self.prompt or ""
        if self.kind == "eval_case":
            return _eval_case_input_text(self.raw)
        if not self.messages:
            return ""
        parts = [m.content for m in self.messages if m.role == "user"]
        return "\n".join(parts)

    def assistant_text(self) -> str:
        if self.kind == "prompt_completion":
            return self.completion or ""
        if self.expected:
            return self.expected
        if not self.messages:
            return ""
        parts = [m.content for m in self.messages if m.role == "assistant"]
        return "\n".join(parts)

    def as_trainer_row(self) -> dict[str, Any]:
        if self.kind == "chat":
            return {
                "messages": [{"role": m.role, "content": m.content} for m in self.messages or []]
            }
        if self.kind == "prompt_completion":
            return {"prompt": self.prompt, "completion": self.completion}
        return {"prompt": self.user_text(), "completion": self.assistant_text()}


@dataclass
class LoadedDataset:
    path: Path
    kind: DatasetKind
    records: list[DatasetRecord]

    def trainer_rows(self) -> list[dict[str, Any]]:
        return [r.as_trainer_row() for r in self.records]

    def ids(self) -> list[str]:
        return [r.record_id or f"line:{r.line_number}" for r in self.records]


@dataclass
class SequenceLengthStats:
    count: int
    minimum: int
    median: float
    p90: float
    p95: float
    p99: float
    maximum: int
    max_seq_length: int
    truncated_count: int
    truncated_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_jsonl_records(path: Path, *, limit: int | None = None) -> LoadedDataset:
    if not path.is_file():
        raise DatasetError(f"dataset not found: {path}")
    records: list[DatasetRecord] = []
    kinds: set[str] = set()
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if limit is not None and len(records) >= limit:
                    break
                text = line.strip()
                if not text:
                    continue
                try:
                    raw = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise DatasetError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                record = parse_record(raw, line_number=line_number, path=path)
                kinds.add(record.kind)
                records.append(record)
    except OSError as exc:
        raise DatasetError(f"cannot read dataset {path}: {exc}") from exc
    if not records:
        raise DatasetError(f"{path}: dataset is empty")
    if len(kinds) != 1:
        raise DatasetError(f"{path}: mixed record kinds {sorted(kinds)}")
    kind = records[0].kind
    return LoadedDataset(path=path, kind=kind, records=records)


def parse_record(raw: Any, *, line_number: int, path: Path) -> DatasetRecord:
    loc = f"{path}:{line_number}"
    if not isinstance(raw, dict):
        raise DatasetError(f"{loc}: record must be a JSON object")

    if "messages" in raw:
        messages = _parse_messages(raw["messages"], loc)
        assistant = [m for m in messages if m.role == "assistant"]
        if not assistant:
            raise DatasetError(f"{loc}: chat record needs at least one assistant message")
        return DatasetRecord(
            line_number=line_number,
            kind="chat",
            messages=messages,
            record_id=_optional_str(raw, "id"),
            semantic_case_id=_optional_str(raw, "semantic_case_id"),
            split=_optional_str(raw, "split"),
            task=_optional_str(raw, "task"),
            language=_optional_str(raw, "language"),
            difficulty=_optional_str(raw, "difficulty"),
            category=_optional_str(raw, "category"),
            expected=assistant[-1].content,
            raw=raw,
        )

    if "prompt" in raw and "completion" in raw:
        prompt = raw.get("prompt")
        completion = raw.get("completion")
        if not isinstance(prompt, str) or not prompt.strip():
            raise DatasetError(f"{loc}: prompt must be a non-empty string")
        if not isinstance(completion, str) or not completion.strip():
            raise DatasetError(f"{loc}: completion must be a non-empty string")
        return DatasetRecord(
            line_number=line_number,
            kind="prompt_completion",
            prompt=prompt,
            completion=completion,
            record_id=_optional_str(raw, "id"),
            semantic_case_id=_optional_str(raw, "semantic_case_id"),
            split=_optional_str(raw, "split"),
            task=_optional_str(raw, "task"),
            language=_optional_str(raw, "language"),
            difficulty=_optional_str(raw, "difficulty"),
            category=_optional_str(raw, "category"),
            expected=completion,
            raw=raw,
        )

    if "task" in raw and "input" in raw and ("expected" in raw or "expected_output_type" in raw):
        return _parse_eval_case(raw, line_number=line_number, loc=loc)

    raise DatasetError(
        f"{loc}: unsupported record; expected chat messages, prompt/completion, or an AISL eval case"
    )


def _parse_eval_case(raw: dict[str, Any], *, line_number: int, loc: str) -> DatasetRecord:
    expected = raw.get("expected") or {}
    expected_text = ""
    if isinstance(expected, dict):
        expected_text = str(
            expected.get("aisl")
            or expected.get("answer")
            or expected.get("text")
            or expected.get("output")
            or ""
        )
    return DatasetRecord(
        line_number=line_number,
        kind="eval_case",
        record_id=_optional_str(raw, "id"),
        semantic_case_id=_optional_str(raw, "source_case") or _optional_str(raw, "semantic_case_id"),
        split=_optional_str(raw, "split"),
        task=str(raw.get("task") or ""),
        language=_optional_str(raw, "language"),
        difficulty=_optional_str(raw, "difficulty"),
        category=_optional_str(raw, "category"),
        expected=expected_text,
        raw=raw,
    )


def _eval_case_input_text(raw: dict[str, Any]) -> str:
    payload = raw.get("input")
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        parts = []
        for key in ("text", "aisl", "question"):
            value = payload.get(key)
            if value:
                parts.append(str(value))
        return "\n".join(parts)
    return ""


def _parse_messages(raw: Any, loc: str) -> list[ChatMessage]:
    if not isinstance(raw, list) or not raw:
        raise DatasetError(f"{loc}: messages must be a non-empty list")
    messages: list[ChatMessage] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise DatasetError(f"{loc}: messages[{index}] must be an object")
        role = item.get("role")
        content = item.get("content")
        if role not in ALLOWED_ROLES:
            raise DatasetError(f"{loc}: messages[{index}] has invalid role {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise DatasetError(f"{loc}: messages[{index}] content must be a non-empty string")
        messages.append(ChatMessage(role=str(role), content=content))
    return messages


def _optional_str(raw: Mapping[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    return str(value)


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DatasetError(f"cannot read manifest {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetError(f"invalid manifest JSON {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise DatasetError(f"{path}: manifest must be a JSON object")
    return raw


def discover_manifest(dataset_path: Path, explicit: Path | None = None) -> Path | None:
    if explicit:
        return explicit
    candidates = [
        dataset_path.parent / "manifest.json",
        dataset_path.parent.parent / "manifest.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def discover_records(dataset_path: Path, explicit: Path | None = None) -> Path | None:
    if explicit:
        return explicit
    candidates = [
        dataset_path.parent / "records.jsonl",
        dataset_path.parent.parent / "records.jsonl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def check_split_disjointness(*datasets: LoadedDataset) -> None:
    """Fail if the same semantic_case_id appears in more than one split."""
    by_split: dict[str, set[str]] = {}
    for dataset in datasets:
        for record in dataset.records:
            if not record.semantic_case_id or not record.split:
                continue
            by_split.setdefault(record.split, set()).add(record.semantic_case_id)
    splits = sorted(by_split)
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            overlap = sorted(by_split[left] & by_split[right])
            if overlap:
                preview = ", ".join(overlap[:8])
                raise DatasetError(
                    f"dataset leakage: {len(overlap)} semantic_case_id values appear in both "
                    f"{left} and {right} (e.g. {preview})"
                )


def percentile(values: list[int], q: float) -> float:
    if not values:
        raise DatasetError("cannot compute percentiles of an empty list")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * q
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def analyze_sequence_lengths(
    lengths: Iterable[int],
    *,
    max_seq_length: int,
) -> SequenceLengthStats:
    values = [int(v) for v in lengths]
    if not values:
        raise DatasetError("no sequence lengths to analyze")
    truncated = sum(1 for v in values if v > max_seq_length)
    return SequenceLengthStats(
        count=len(values),
        minimum=min(values),
        median=percentile(values, 0.50),
        p90=percentile(values, 0.90),
        p95=percentile(values, 0.95),
        p99=percentile(values, 0.99),
        maximum=max(values),
        max_seq_length=max_seq_length,
        truncated_count=truncated,
        truncated_pct=100.0 * truncated / len(values),
    )


def enforce_truncation_policy(
    stats: SequenceLengthStats,
    *,
    warn_pct: float,
    fail_pct: float,
    allow_high_truncation: bool,
) -> str | None:
    if stats.truncated_pct <= warn_pct:
        return None
    message = (
        f"{stats.truncated_count}/{stats.count} sequences ({stats.truncated_pct:.1f}%) "
        f"exceed max_seq_length={stats.max_seq_length} "
        f"(max observed {stats.maximum})"
    )
    if stats.truncated_pct > fail_pct and not allow_high_truncation:
        raise DatasetError(
            message + "; pass allow_high_truncation or raise max_seq_length"
        )
    return message


LengthFn = Callable[[DatasetRecord], int]


def record_lengths(dataset: LoadedDataset, length_fn: LengthFn) -> list[int]:
    return [length_fn(record) for record in dataset.records]
