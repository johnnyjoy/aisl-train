# Architecture

```text
aisl
    produces semantic truth + datasets + eval definitions
        │
        ▼
aisl-train
    consumes them (configurable filesystem / mount paths)
        │
        ▼
inspection + capability detection
        │
        ▼
optional model profile (deltas only)
        │
        ▼
generic load / PEFT / SFT / evaluate / compare
        │
        ▼
adapter / checkpoints / results under --output
```

## Ownership

| Concern | Owner |
| --- | --- |
| AISL grammar, dictionaries, canonical form | aisl |
| Semantic source cases and generated JSONL | aisl |
| Held-out eval cases and semantic scorers | aisl |
| Model load, device, LoRA/QLoRA, checkpoints | aisl-train |
| Experiment orchestration | aisl-train |

aisl-train must not redefine AISL semantics, modify held-out truth,
or require AISL as an installed Python package.

## Internal boundaries

| Module | Responsibility |
| --- | --- |
| `inspect` / `capabilities` / `profile` | what the model is, what it can do, overrides |
| `model` / `quantization` / `devices` | load and place one logical model |
| `peft` | attach LoRA; count trainable parameters |
| `trainer` | TRL `SFTTrainer` backend |
| `inference` / `evaluation` | generation + result files |
| `compare` / `experiment` | metadata and BASE/PROMPTED/TRAINED diffs |

TRL is the first trainer backend. Replacing it should not require
rewriting the CLI or `experiment.json` schema.

## Resolution order

```text
explicit CLI override
        ↓
explicit model profile
        ↓
automatic model inspection
        ↓
safe generic defaults
        ↓
clear failure if unresolved
```

## Device model

One training process. `device_map=auto` may shard one model across
GPUs. The runtime does not start one replica per GPU and does not
require `accelerate launch`.

## Provenance

Copy from an AISL `manifest.json` when provided or discovered next
to the dataset:

```text
AISL version
dataset_version
content_hash
dictionary_versions
```

Add runtime fields: model identity, profile, training config,
aisl-train commit, Python/PyTorch/CUDA/library versions, GPUs.

## Dataset paths

The training image packages AISL exports at `/opt/aisl-data`.
Omitted `--train` / `--validation` / `--eval` use those files.
Explicit paths always override. A sibling aisl checkout is not
required.

Evaluation prompts and scorers come from the cloned tree at
`/workspace/aisl`.
