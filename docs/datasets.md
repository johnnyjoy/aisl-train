# Datasets

aisl-train consumes aisl exports. It does not generate them.

Authoritative contract: aisl `docs/training-interface.md`.

## Formats

Chat JSONL:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Prompt/completion JSONL:

```json
{"prompt": "...", "completion": "..."}
```

AISL eval-case JSONL (`eval/cases/task-*.jsonl`) is accepted for
evaluation. Malformed rows fail with `path:line`.

## Loss

Chat + tokenizer `{% generation %}` block: assistant-only loss
(`--loss-mask assistant`, the default).

Prompt/completion: completion-only loss.

If an assistant mask cannot be derived, the run fails unless
`--loss-mask full` is set explicitly.

## Provenance files

If `manifest.json` sits next to or one directory above the JSONL,
it is copied into experiment metadata. `records.jsonl` can be
joined for task/language/difficulty/category breakdowns.

Pass `--manifest` / `--records` / `--aisl-root` when discovery is
not enough.

## Sequence length

Before training, token lengths are measured. Default
`max_seq_length` is 2048. A large truncated fraction fails unless
`--allow-high-truncation` is set.
