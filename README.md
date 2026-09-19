# aisl-train

**SPDX-License-Identifier: Apache-2.0**

aisl-train is a generic model-training and experiment runtime for
datasets produced by [aisl](https://github.com/johnnyjoy/aisl).

It inspects a Hugging Face causal language model, trains a LoRA or
QLoRA adapter, runs baseline and trained evaluation, and compares
results. It does not define AISL semantics, dictionaries, or
held-out evaluation truth.

## What this is not

- not a Qwen-specific trainer
- not a full-parameter / DPO / RLHF / GRPO trainer
- not an AISL language implementation
- not a corpus generator

Qwen3.8-27B is the first model used to validate the runtime. Another
compatible decoder-only model should use the same commands with a
different `--model` (and a profile only if inspection is not enough).

## Relationship to aisl

```text
aisl
    semantic truth, trainer-neutral JSONL, eval definitions
        │
        ▼
aisl-train
    inspect / QLoRA / baseline / evaluate / compare
        │
        ▼
arbitrary supported causal LM
```

Dataset contract: aisl `docs/training-interface.md`.
Paths are always caller-supplied. Library code does not assume
`../aisl`.

## Supported training approach

Hugging Face Transformers decoder-only causal LMs.

Default pilot method: bitsandbytes 4-bit NF4 + PEFT LoRA with
`target_modules="all-linear"` + TRL `SFTTrainer`.

Model differences belong in inspection, capability detection, and
thin JSON profiles — not in `train_<family>.py` programs.

## CLI

```text
aisl-train doctor
aisl-train inspect
aisl-train smoke
aisl-train baseline
aisl-train train
aisl-train evaluate
aisl-train compare
```

See [docs/cli.md](docs/cli.md). Every command listed here implements
`--help`.

## Docker-first workflow

The GPU host is expected to have Docker, the NVIDIA driver, and
the NVIDIA Container Toolkit. It does not need host Python, pip,
or Hugging Face CLI.

```bash
cp .env.example .env
# edit MODEL_HOST_PATH, AISL_DATA_HOST_PATH, OUTPUT_HOST_PATH, CACHE_HOST_PATH

docker compose build

docker compose run --rm trainer \
  aisl-train doctor --model /models/base --dataset /data/aisl/train.chat.jsonl --output /output

docker compose run --rm trainer \
  aisl-train inspect --model /models/base

docker compose run --rm trainer \
  aisl-train smoke \
  --model /models/base \
  --dataset /data/aisl/train.chat.jsonl \
  --output /output/smoke

docker compose run --rm trainer \
  aisl-train baseline \
  --model /models/base \
  --eval /data/aisl/test.chat.jsonl \
  --prompt-mode none \
  --output /output/pilot-001/baseline

docker compose run --rm trainer \
  aisl-train baseline \
  --model /models/base \
  --eval /data/aisl/test.chat.jsonl \
  --prompt-mode minimal \
  --aisl-root /data/aisl-src \
  --output /output/pilot-001/prompted

docker compose run --rm trainer \
  aisl-train train \
  --model /models/base \
  --train /data/aisl/train.chat.jsonl \
  --validation /data/aisl/validation.chat.jsonl \
  --config configs/pilot-qlora.json \
  --output /output/pilot-001

docker compose run --rm trainer \
  aisl-train evaluate \
  --model /models/base \
  --adapter /output/pilot-001/adapter \
  --eval /data/aisl/test.chat.jsonl \
  --output /output/pilot-001/trained-eval

docker compose run --rm trainer \
  aisl-train compare \
  --experiment /output/pilot-001
```

`--prompt-mode minimal` needs AISL's prompt file. Either pass
`--prompt-file /data/aisl-src/eval/prompts/minimal.txt` or mount
the aisl checkout and pass `--aisl-root`.

Do not use `accelerate launch` or `python train_qwen.py`.

## Experiment lifecycle

1. doctor
2. inspect
3. smoke
4. baseline without AISL bootstrap (`BASE`)
5. baseline with minimal AISL bootstrap (`PROMPTED`)
6. train QLoRA adapter
7. evaluate trained adapter (`TRAINED`)
8. compare

These three conditions stay distinct. Train uses the train split;
trainer eval uses validation; final scoring uses the held-out test
split.

## Local unit tests

```bash
python -m pip install -e .
python -m pip install pytest
python -m pytest
aisl-train --help
```

Unit tests do not download models and do not need a GPU.

| State | Meaning |
| --- | --- |
| unit tests passed | Level B |
| GPU smoke passed | Level D (`aisl-train smoke` on real hardware) |
| full experiment completed | Level E (baseline + train + evaluate + compare) |

## Current status

Generic runtime and CLI are implemented. Unit tests are the
verification available in ordinary development. GPU smoke and the
first Qwen experiment have not been run from this checkout.

See [STATUS.md](STATUS.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
