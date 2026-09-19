# Docker

The GPU host is container-only. Host Python and an AISL checkout
are not part of the workflow.

## Image

`Dockerfile` uses `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`.

1. Install the training stack into `/opt/venv`.
2. Clone `https://github.com/johnnyjoy/aisl.git` at `AISL_REF`
   into `/workspace/aisl`. The build fails if the ref cannot be
   resolved or required exports are missing.
3. Symlink trainer-neutral artifacts to `/opt/aisl-data`.
4. `COPY` this aisl-train checkout to `/workspace/aisl-train`.

aisl-train is never cloned from GitHub during the build.

Intended pins (change only with a recorded reason):

```text
torch          2.11.0  (cu128 index)
transformers   5.17.0
bitsandbytes   0.50.2
peft           0.21.0
trl            1.13.0
```

Build metadata is written to `/workspace/AISL_REF`,
`/workspace/AISL_COMMIT`, and
`/opt/aisl-train/image-metadata.json`. `aisl-train doctor`
reports those values. `pip freeze` is stored at
`/opt/aisl-train/pip-freeze.txt`.

## Compose mounts

| Host variable | Container |
| --- | --- |
| `MODEL_HOST_PATH` | `/models/base` (read-only) |
| `OUTPUT_HOST_PATH` | `/output` |
| `CACHE_HOST_PATH` | `/cache` |

All GPUs are exposed (`gpus: all`). Shared memory is 16g.

`AISL_REF` is a build argument, not a runtime mount.

## Offline model use

Compose sets `HF_HUB_OFFLINE=1` by default. Local checkpoints are
loaded with `local_files_only=True`.

Optional future work (not implemented): Hugging Face download,
model cache management, authenticated Hub access.
