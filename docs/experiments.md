# Experiments

Every run writes under `--output`. Nothing valuable belongs in the
container filesystem.

```text
/output/pilot-001/
├── experiment.json
├── environment.json
├── model-inspection.json
├── dataset-stats.json
├── baseline/
├── prompted/
├── smoke/
├── checkpoints/
├── adapter/
├── training/
├── trained-eval/
└── comparison/
```

JSON/JSONL files are authoritative. Markdown reports are summaries.

## Conditions

| Directory | Condition | Meaning |
| --- | --- | --- |
| `baseline/` | BASE | model, no AISL bootstrap |
| `prompted/` | PROMPTED | model + persisted minimal prompt |
| `trained-eval/` | TRAINED | same base + adapter |

Do not collapse these into one score.

## Reproducing a run later

A completed experiment should record:

```text
aisl-train commit
AISL version / corpus version / content hash / dictionaries
model path and inspectable config
profile id/hash
training config and seed
Python, PyTorch, CUDA, library versions
GPU names and VRAM
```

`experiment.json` lists `provenance.missing` when a field could
not be copied.

## Splits

Train on `train`. Intermediate trainer evaluation on `validation`.
Final comparison on `test`. If `semantic_case_id` values are
present, overlapping IDs across splits fail the run.
