# aisl-train

**SPDX-License-Identifier: Apache-2.0**

aisl-train is a generic model-training and experiment runtime for
datasets produced by [aisl](https://github.com/johnnyjoy/aisl).

The GPU host needs Docker, an NVIDIA driver, Compose, a local
model directory, and writable output/cache. It does not need
host Python, PyTorch, Hugging Face CLI, or an AISL git checkout.

AISL is cloned into the image at build time. The model stays on
the host and is mounted read-only.

## What this is not

- not a Qwen-specific trainer
- not a full-parameter / DPO / RLHF / GRPO trainer
- not an AISL language implementation
- not a corpus generator

Qwen3.8-27B is the first mounted model used to validate the
runtime. Another compatible decoder-only model uses the same
commands with a different `MODEL_HOST_PATH`.

## Quick start

```bash
git clone https://github.com/johnnyjoy/aisl-train.git
cd aisl-train

cp .env.example .env
# Set MODEL_HOST_PATH to the existing model directory.

mkdir -p output cache

docker compose build

docker compose run --rm trainer aisl-train doctor

docker compose run --rm trainer \
  aisl-train inspect --model /models/base

docker compose run --rm trainer \
  aisl-train smoke \
  --model /models/base \
  --output /output/smoke
```

After smoke succeeds:

```bash
docker compose run --rm trainer \
  aisl-train baseline \
  --model /models/base \
  --prompt-mode none \
  --output /output/pilot-001/baseline

docker compose run --rm trainer \
  aisl-train baseline \
  --model /models/base \
  --prompt-mode minimal \
  --output /output/pilot-001/prompted

docker compose run --rm trainer \
  aisl-train train \
  --model /models/base \
  --config configs/pilot-qlora.json \
  --output /output/pilot-001

docker compose run --rm trainer \
  aisl-train evaluate \
  --model /models/base \
  --adapter /output/pilot-001/adapter \
  --output /output/pilot-001/trained-eval

docker compose run --rm trainer \
  aisl-train compare \
  --experiment /output/pilot-001
```

Omitted `--train` / `--validation` / `--eval` use the packaged
AISL corpus at `/opt/aisl-data`. Explicit paths still override.

Do not use `accelerate launch` or `python train_qwen.py`.

## Host mounts

| Variable | Container |
| --- | --- |
| `MODEL_HOST_PATH` | `/models/base` (read-only) |
| `OUTPUT_HOST_PATH` | `/output` |
| `CACHE_HOST_PATH` | `/cache` |

There is no AISL host mount. A sibling `aisl` checkout is not
required.

## Future release workflow

Not published yet. After a tag exists:

```bash
docker pull tigersmile/aisl-train:<tag>
```

Set `AISL_TRAIN_IMAGE=tigersmile/aisl-train:<tag>` in `.env` and
run the same Compose commands without a source checkout of AISL.

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

## Local unit tests

```bash
python -m pip install -e .
python -m pip install pytest
python -m pytest
aisl-train --help
```

Unit tests do not download models and do not need a GPU or Docker.

## Current status

See [STATUS.md](STATUS.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
