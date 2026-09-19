# Docker

The GPU server is container-only. Host Python is not part of the
workflow.

## Image

`Dockerfile` uses `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`
and installs the training stack into `/opt/venv`. The build context
is this git checkout (`COPY . /workspace/aisl-train`). The image
is not cloned from GitHub.

Intended pins (change only with a recorded reason):

```text
torch          2.11.0  (cu128 index)
transformers   5.17.0
bitsandbytes   0.50.2
peft           0.21.0
trl            1.13.0
```

Exact resolved versions are written to
`/opt/aisl-train/pip-freeze.txt` at build time.

## Compose mounts

| Host variable | Container |
| --- | --- |
| `MODEL_HOST_PATH` | `/models/base` (read-only) |
| `AISL_DATA_HOST_PATH` | `/data/aisl` (read-only) |
| `OUTPUT_HOST_PATH` | `/output` |
| `CACHE_HOST_PATH` | `/cache` |

All GPUs are exposed (`gpus: all`). Shared memory is 16g.

Copy `.env.example` to `.env` on the GPU host. The example data
path `../aisl/training/data/exports` is a local convenience, not a
Python default.

## Offline model use

Compose sets `HF_HUB_OFFLINE=1` by default. Local checkpoints are
loaded with `local_files_only=True`. Hub ids require `--allow-hub`
and turning offline mode off.
