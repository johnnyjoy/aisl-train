# CLI

Implemented commands. Each supports `--help`.

```text
aisl-train doctor
aisl-train inspect [--model PATH] [--json] [--profile ID]
aisl-train smoke --model PATH --output PATH [--config FILE]
aisl-train baseline --model PATH --output PATH --prompt-mode none|minimal
aisl-train train --model PATH --output PATH [--config FILE]
aisl-train evaluate --model PATH --output PATH [--adapter PATH]
aisl-train compare --experiment PATH
```

When dataset flags are omitted, the packaged AISL files are used:

```text
train       /opt/aisl-data/train.chat.jsonl
validation  /opt/aisl-data/validation.chat.jsonl
eval        /opt/aisl-data/test.chat.jsonl
```

`--train` / `--dataset` / `--validation` / `--eval` override those
defaults. `--dataset` is an alias of `--train` on `train` and `smoke`.

## doctor

Reports Python, package versions, CUDA, GPU names/VRAM, packaged
AISL corpus, AISL ref/commit, `/models/base`, and output/cache
writability inside the container. Exit status is non-zero when a
listed check fails.

## inspect

Reads model metadata from disk. Does not train. `--json` prints
the structured report. `--model` defaults to `/models/base`.

## smoke

GPU gate using a handful of packaged training records unless
`--dataset` is set.

## baseline / evaluate

`baseline` is inference without an adapter. `--prompt-mode none`
is BASE. `--prompt-mode minimal` uses the packaged AISL prompt at
`/workspace/aisl/eval/prompts/minimal.txt`.

`evaluate` loads the same base model plus `--adapter`.

Both write `results.jsonl`, `raw_generations.jsonl`,
`aisl_predictions.jsonl`, `summary.json`, and `REPORT.md`.

## train

Generic SFT/QLoRA. Warns if the experiment directory has no
baseline results. `--resume latest` or `--resume PATH`.

## compare

Reads `baseline/`, `prompted/`, and `trained-eval/` under the
experiment directory.
