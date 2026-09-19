# Model profiles

A profile records only architecture-specific exceptions the generic
path cannot infer safely.

Inspection of `config.json`, tokenizer files, and weight maps is
the normal source of truth. A profile overrides that result.

## Location

JSON files under `profiles/`. Pass `--profile <id>` or a path.

## Fields

```text
id
version
match.model_type          optional exact config.model_type
match.architectures       optional overlap
match.name_contains       optional path/name substrings
peft.target_modules       default remains all-linear
peft.exclude_modules
peft.target_parameters    for expert weights that are not nn.Linear
peft.modules_to_save
tokenizer.chat_template   tokenizer | inline
tokenizer.template        only when chat_template=inline
precision.preferred_compute_dtype
runtime_validation        pending | verified
```

Do not add a field merely to justify the profile mechanism.

## First profile: qwen3.8-27b

`profiles/qwen3.8-27b.json` is minimal: generic `all-linear`,
tokenizer chat template, `compute_dtype=auto`. It matches on
directory-name fragments because no local checkpoint `config.json`
was available when the file was written. `runtime_validation` is
`pending` until `aisl-train inspect --model /models/base` is run
against the real weights.

If that inspect shows a `model_type` that needs an override, add
it then. Do not invent one now.
