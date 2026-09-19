# Troubleshooting

## doctor fails on a desktop without torch

Expected. `aisl-train doctor` checks the training stack. Unit
tests and `--help` do not need those packages. On the GPU server,
run doctor inside the Compose service.

## inspect cannot load a new model_type

Inspect still reads raw `config.json`. Training will fail later if
the installed Transformers release cannot construct the model.
Record the `model_type` and upgrade Transformers or add a verified
profile exception. Do not invent module names.

## assistant-only loss refused

The chat template needs a `{% generation %}` block, or the dataset
must be prompt/completion, or you must pass `--loss-mask full`.
Silent full-sequence training is not the default.

## zero trainable parameters

PEFT targeting matched nothing. Inspect linear-module suffixes and
any `parameter_like_anomalies` (common on some MoE checkpoints).
Override with profile `target_modules` / `target_parameters`.

## CUDA OOM

The process exits. The error includes model path, quantization,
sequence length, micro-batch, LoRA rank, target strategy, device
map, and a memory snapshot. Change the config explicitly. The
runtime will not silently shrink the experiment.

## NaN loss

Training stops. Check the data row IDs and the last logged step.

## resume rejected

The checkpoint was produced with a different model path, corpus
path, method, sequence length, loss mask, seed, LoRA, or
quantization. Use `--force-resume` only when that is intentional.

## semantic scores missing

aisl-train always keeps raw generations. In the training image,
`--aisl-root` defaults to the cloned `/workspace/aisl`. Semantic
scoring runs when that scorer is present and the eval file is an
AISL eval-case set.

## Docker build

If `docker compose build` cannot resolve the pinned torch or CUDA
base tag, stop and record the resolver error. Do not silently
switch to a CPU wheel.

If `AISL_REF` cannot be fetched, or required exports are missing
from that revision, the build fails. There is no fallback branch.

## doctor reports packaged AISL missing on a desktop

Expected outside the image. Run doctor via
`docker compose run --rm trainer aisl-train doctor`.
