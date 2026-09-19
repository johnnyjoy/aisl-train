# CLI

Implemented commands. Each supports `--help`.

```text
aisl-train doctor
aisl-train inspect --model PATH [--json] [--profile ID]
aisl-train smoke --model PATH --dataset PATH --output PATH [--config FILE]
aisl-train baseline --model PATH --eval PATH --output PATH --prompt-mode none|minimal
aisl-train train --model PATH --train PATH --validation PATH --output PATH [--config FILE]
aisl-train evaluate --model PATH --eval PATH --output PATH [--adapter PATH]
aisl-train compare --experiment PATH
```

`--dataset` is an alias of `--train` on `train` and `smoke`.

## doctor

Reports Python, package versions, CUDA, GPU names/VRAM, and
optional path accessibility. Exit status is non-zero when a listed
check fails.

## inspect

Reads model metadata from disk. Does not train. `--json` prints
the structured report.

## smoke

GPU gate: load tokenizer and quantized model, attach LoRA, check
trainable parameters, forward, backward, optimizer step, save
adapter, reload adapter, generate. A handful of records only.

## baseline / evaluate

`baseline` is inference without an adapter. `--prompt-mode none`
is BASE; `--prompt-mode minimal` is PROMPTED and requires
`--prompt-file` or `--aisl-root`.

`evaluate` loads the same base model plus `--adapter`.

Both write `results.jsonl`, `raw_generations.jsonl`,
`aisl_predictions.jsonl`, `summary.json`, and `REPORT.md`.

## train

Generic SFT/QLoRA. Warns if the experiment directory has no
baseline results. `--resume latest` or `--resume PATH`.
Incompatible checkpoint/config pairs fail unless `--force-resume`.

## compare

Reads `baseline/`, `prompted/`, and `trained-eval/` under the
experiment directory. Reports overall metrics plus breakdowns by
task, difficulty, language, and category when those fields exist.
Highlights improvements and regressions.
