# SPDX-License-Identifier: Apache-2.0
"""Command orchestration. Imports the training stack.

CLI parses arguments without importing this module so ``--help`` works
without torch.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from aisl_train import __version__
from aisl_train.capabilities import Capabilities
from aisl_train.compare import compare_experiment
from aisl_train.config import TrainingConfig
from aisl_train.dataset import (
    analyze_sequence_lengths,
    check_split_disjointness,
    discover_manifest,
    discover_records,
    enforce_truncation_policy,
    load_jsonl_records,
    load_manifest,
)
from aisl_train.devices import (
    bf16_supported,
    cuda_available,
    gpu_count,
    require_cuda,
    reset_peak_memory,
    snapshot_memory,
)
from aisl_train.environment import doctor_report, format_doctor_text, snapshot_environment
from aisl_train.errors import AislTrainError, MaskingError, OutOfMemoryError, PeftError
from aisl_train.evaluation import attach_record_metadata, run_generation_eval
from aisl_train.experiment import (
    assert_resume_compatible,
    build_provenance,
    config_fingerprint,
    prepare_experiment_dir,
    select_resume_path,
    warn_if_missing_baseline,
    write_json,
)
from aisl_train.formatting import resolve_format_plan, validate_loss_mask_plan
from aisl_train.inference import generate_one
from aisl_train.inspect import format_inspection_text, inspect_model_path, reject_if_unsupported
from aisl_train.model import load_causal_lm, load_tokenizer, restore_use_cache
from aisl_train.peft import attach_lora, load_adapter
from aisl_train.profile import resolve_profile
from aisl_train.quantization import bitsandbytes_available, quantization_metadata
from aisl_train.trainer import run_supervised_finetune

log = logging.getLogger("aisl_train")


def detect_capabilities(inspection: Any | None = None) -> Capabilities:
    chat = bool(inspection and inspection.chat_template_present)
    causal = bool(inspection.supported_class) if inspection else True
    notes = {}
    bnb = bitsandbytes_available()
    notes["bnb_4bit"] = "bitsandbytes importable" if bnb else "bitsandbytes not importable"
    try:
        import peft  # noqa: F401

        peft_ok = True
    except Exception:
        peft_ok = False
        notes["peft_lora"] = "peft not importable"
    try:
        cuda = cuda_available()
        bf16 = bf16_supported()
        gpus = gpu_count()
    except Exception as exc:
        cuda = False
        bf16 = False
        gpus = 0
        notes["generation"] = str(exc)
    return Capabilities(
        causal_lm=causal,
        chat_template=chat,
        bf16=bf16,
        fp16=cuda,
        bnb_4bit=bnb,
        nf4=bnb,
        double_quant=bnb,
        gradient_checkpointing=True,
        peft_lora=peft_ok,
        all_linear_lora=peft_ok,
        multi_gpu=gpus > 1,
        generation=cuda or True,
        notes=notes,
    )


def device_report() -> dict[str, Any]:
    try:
        return {
            "cuda_available": cuda_available(),
            "gpu_count": gpu_count(),
            "bf16": bf16_supported(),
            "gpus": snapshot_memory("doctor").get("devices", []),
        }
    except Exception as exc:
        return {"cuda_available": False, "gpu_count": 0, "bf16": False, "gpus": [], "error": str(exc)}


def run_inspect(config: TrainingConfig, *, as_json: bool = False) -> int:
    model_path = config.resolved_model_path()
    inspection = inspect_model_path(model_path)
    profile = resolve_profile(config.profile, inspection=inspection.to_dict())
    if profile:
        inspection.matched_profile = profile.id
        inspection.profile_overrides = profile.overrides()
    capabilities = detect_capabilities(inspection)
    devices = device_report()
    payload = {
        "inspection": inspection.to_dict(),
        "capabilities": capabilities.to_dict(),
        "devices": devices,
        "candidate_peft": {
            "target_modules": (profile.peft.get("target_modules") if profile and profile.peft else "all-linear"),
            "anomalies": inspection.linear.parameter_like_anomalies,
        },
    }
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_inspection_text(inspection, capabilities=capabilities.to_dict(), devices=devices))
    return 0


def run_doctor(config: TrainingConfig, *, as_json: bool = False, require_gpu: bool = False) -> int:
    report = doctor_report(
        model_path=Path(config.model_path).expanduser() if config.model_path else None,
        dataset_path=Path(config.train_path or config.eval_path).expanduser()
        if (config.train_path or config.eval_path)
        else None,
        output_path=Path(config.output_path).expanduser() if config.output_path else None,
        require_gpu=require_gpu,
    )
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_doctor_text(report), end="")
    return 0 if report["ok"] else 1


def _load_profile(config: TrainingConfig, inspection_dict: dict[str, Any] | None = None):
    return resolve_profile(config.profile, inspection=inspection_dict)


def _sequence_stats(dataset, tokenizer, config: TrainingConfig) -> dict[str, Any]:
    def length_fn(record) -> int:
        if record.kind == "chat":
            text = tokenizer.apply_chat_template(
                record.as_trainer_row()["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            text = f"{record.prompt}{record.completion}"
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    stats = analyze_sequence_lengths(
        (length_fn(record) for record in dataset.records),
        max_seq_length=config.max_seq_length,
    )
    warning = enforce_truncation_policy(
        stats,
        warn_pct=config.truncation_warn_pct,
        fail_pct=config.truncation_fail_pct,
        allow_high_truncation=config.allow_high_truncation,
    )
    if warning:
        log.warning(warning)
    return stats.to_dict()


def run_train(config: TrainingConfig) -> int:
    require_cuda(config.device_strategy)
    output = config.resolved_output_path()
    paths = prepare_experiment_dir(output)
    inspection = inspect_model_path(config.resolved_model_path())
    reject_if_unsupported(inspection)
    profile = _load_profile(config, inspection.to_dict())
    capabilities = detect_capabilities(inspection)
    capabilities.require(["causal_lm", "peft_lora"])
    if config.method == "qlora":
        capabilities.require(["bnb_4bit"])

    train_ds = load_jsonl_records(config.resolved_train_path(), limit=config.limit)
    val_ds = None
    if config.validation_path:
        val_ds = load_jsonl_records(Path(config.validation_path).expanduser(), limit=config.limit)
        check_split_disjointness(train_ds, val_ds)

    tokenizer = load_tokenizer(config.resolved_model_path(), local_files_only=True)
    plan = resolve_format_plan(
        dataset_kind=train_ds.kind,
        tokenizer_chat_template=getattr(tokenizer, "chat_template", None),
        profile=profile,
        loss_mask=config.loss_mask,
    )
    validate_loss_mask_plan(plan)
    if plan.template and plan.chat_template_source == "profile":
        tokenizer.chat_template = plan.template

    seq_stats = _sequence_stats(train_ds, tokenizer, config)
    baseline_warning = warn_if_missing_baseline(output)
    if baseline_warning:
        log.warning(baseline_warning)

    manifest_path = discover_manifest(
        config.resolved_train_path(),
        Path(config.manifest_path).expanduser() if config.manifest_path else None,
    )
    manifest = load_manifest(manifest_path) if manifest_path else None
    provenance = build_provenance(
        aisl_train_version=__version__,
        model_path=str(config.resolved_model_path()),
        model_type=inspection.model_type,
        profile_id=profile.id if profile else None,
        profile_hash=profile.content_hash if profile else None,
        seed=config.seed,
        manifest=manifest,
        aisl_root=Path(config.aisl_root).expanduser() if config.aisl_root else None,
    )
    fingerprint = config_fingerprint(config.fingerprint_fields())
    experiment_file = paths.root / "experiment.json"
    if config.resume and experiment_file.is_file():
        previous = json.loads(experiment_file.read_text(encoding="utf-8"))
        assert_resume_compatible(
            current=config.fingerprint_fields(),
            saved=previous.get("config") or {},
            force=config.force_resume,
        )
    write_json(experiment_file, {
        "id": config.experiment_id or output.name,
        "config": config.to_dict(),
        "fingerprint": fingerprint,
        "format": plan.to_dict(),
        "provenance": provenance.to_dict(),
    })
    write_json(paths.root / "environment.json", snapshot_environment())
    write_json(paths.root / "model-inspection.json", inspection.to_dict())
    write_json(paths.root / "dataset-stats.json", {
        "train": {"path": str(train_ds.path), "n": len(train_ds.records), "kind": train_ds.kind},
        "validation": {"path": str(val_ds.path), "n": len(val_ds.records), "kind": val_ds.kind} if val_ds else None,
        "sequence_lengths": seq_stats,
    })

    resume_from = None
    if config.resume:
        resume_from = select_resume_path(config.resume, paths.checkpoints)

    memories = [snapshot_memory("before_model_load")]
    reset_peak_memory()
    loaded = load_causal_lm(config, inspection=inspection, for_training=True)
    memories.append(snapshot_memory("after_model_load"))
    model, stats = attach_lora(loaded.model, config, profile=profile)
    memories.append(snapshot_memory("after_peft"))
    write_json(paths.training / "adapter-stats.json", {
        "total_parameters": stats.total_parameters,
        "trainable_parameters": stats.trainable_parameters,
        "trainable_pct": stats.trainable_pct,
        "target_strategy": stats.target_strategy,
        "matched_categories": stats.matched_categories,
        "frozen_base": stats.frozen_base,
    })
    write_json(paths.root / "quantization.json", quantization_metadata(config, loaded.compute_dtype_name))

    result = run_supervised_finetune(
        model=model,
        tokenizer=loaded.tokenizer,
        train_rows=train_ds.trainer_rows(),
        eval_rows=val_ds.trainer_rows() if val_ds else None,
        config=config,
        output_dir=paths.checkpoints,
        resume_from=resume_from,
    )
    memories.append(snapshot_memory("peak_training"))
    write_json(paths.training / "metrics.json", result)
    write_json(paths.training / "memory.json", {"snapshots": memories})
    write_json(paths.adapter / "experiment.json", {
        "provenance": provenance.to_dict(),
        "config": config.to_dict(),
        "adapter_stats": {
            "trainable_parameters": stats.trainable_parameters,
            "trainable_pct": stats.trainable_pct,
        },
    })
    log.info(
        "training complete trainable=%s (%.4f%%) adapter=%s",
        stats.trainable_parameters,
        stats.trainable_pct,
        paths.adapter,
    )
    return 0


def run_smoke(config: TrainingConfig) -> int:
    require_cuda(config.device_strategy)
    output = config.resolved_output_path()
    paths = prepare_experiment_dir(output)
    smoke_dir = paths.smoke if output.name != "smoke" else output
    smoke_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_jsonl_records(Path(config.train_path or config.eval_path).expanduser(), limit=config.limit or 4)
    inspection = inspect_model_path(config.resolved_model_path())
    reject_if_unsupported(inspection)
    profile = _load_profile(config, inspection.to_dict())
    capabilities = detect_capabilities(inspection)
    capabilities.require(["causal_lm", "peft_lora", "bnb_4bit"] if config.method == "qlora" else ["causal_lm", "peft_lora"])

    memories = [snapshot_memory("before_model_load")]
    loaded = load_causal_lm(config, inspection=inspection, for_training=True)
    memories.append(snapshot_memory("after_model_load"))
    model, stats = attach_lora(loaded.model, config, profile=profile)
    memories.append(snapshot_memory("after_peft"))
    if stats.trainable_parameters == 0:
        raise PeftError("smoke: zero trainable parameters")

    tokenizer = loaded.tokenizer
    plan = resolve_format_plan(
        dataset_kind=dataset.kind,
        tokenizer_chat_template=getattr(tokenizer, "chat_template", None),
        profile=profile,
        loss_mask=config.loss_mask,
    )
    if plan.loss_mask != "full":
        try:
            validate_loss_mask_plan(plan)
        except MaskingError as exc:
            log.warning("smoke format note: %s", exc)

    row = dataset.records[0]
    if row.kind == "chat":
        text = tokenizer.apply_chat_template(row.as_trainer_row()["messages"], tokenize=False, add_generation_prompt=False)
    else:
        text = f"{row.prompt}{row.completion}"
    encoded = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    labels = encoded["input_ids"].clone()
    try:
        outputs = model(**encoded, labels=labels)
        memories.append(snapshot_memory("after_first_forward"))
        loss = outputs.loss
        loss.backward()
        memories.append(snapshot_memory("after_first_backward"))
        import torch as _torch

        optimizer = _torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    except Exception as exc:
        if "out of memory" in str(exc).lower():
            raise OutOfMemoryError(f"smoke OOM: {snapshot_memory('smoke_oom')}") from exc
        raise

    adapter_dir = smoke_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    reload_cfg = config.overlay(adapter_path=str(adapter_dir))
    reloaded = load_causal_lm(reload_cfg, inspection=inspection, for_training=False)
    reloaded.model = load_adapter(reloaded.model, str(adapter_dir))
    restore_use_cache(reloaded)
    generated = generate_one(
        reloaded.model,
        reloaded.tokenizer,
        row.user_text(),
        settings=config.generation,
        system_prompt=None,
    )
    write_json(smoke_dir / "smoke.json", {
        "ok": True,
        "adapter_stats": {
            "total_parameters": stats.total_parameters,
            "trainable_parameters": stats.trainable_parameters,
            "trainable_pct": stats.trainable_pct,
            "frozen_base": stats.frozen_base,
            "matched_categories": stats.matched_categories,
        },
        "loss": float(loss.detach().cpu()),
        "generation_preview": generated["raw"][:200],
        "memory": memories,
        "compute_dtype": loaded.compute_dtype_name,
    })
    log.info("smoke passed trainable=%s loss=%s", stats.trainable_parameters, float(loss.detach().cpu()))
    return 0


def run_eval_command(config: TrainingConfig, *, kind: str) -> int:
    require_cuda(config.device_strategy)
    output = config.resolved_output_path()
    output.mkdir(parents=True, exist_ok=True)
    eval_path = Path(config.eval_path or "").expanduser()
    if not eval_path:
        raise AislTrainError("evaluation dataset path is required")
    dataset = load_jsonl_records(eval_path, limit=config.limit)
    records_path = discover_records(
        eval_path,
        Path(config.records_path).expanduser() if config.records_path else None,
    )
    attach_record_metadata(dataset, records_path)
    inspection = inspect_model_path(config.resolved_model_path())
    reject_if_unsupported(inspection)
    loaded = load_causal_lm(config, inspection=inspection, for_training=False)
    restore_use_cache(loaded)
    if config.adapter_path:
        loaded.model = load_adapter(loaded.model, config.adapter_path)
    if kind == "baseline" and config.adapter_path:
        raise AislTrainError("baseline must not load an adapter; use evaluate for trained adapters")
    metadata = run_generation_eval(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        dataset=dataset,
        config=config,
        output_dir=output,
        model_id=str(config.resolved_model_path()),
    )
    log.info("%s wrote %s items to %s", metadata.get("condition"), metadata.get("n"), output)
    return 0


def run_compare(experiment: Path) -> int:
    report = compare_experiment(experiment)
    print(json.dumps({"conditions": report["conditions_present"], "overall": report["overall"]}, indent=2))
    return 0
