# SPDX-License-Identifier: Apache-2.0
# GPU training image. Build context is this repository checkout.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    HF_DATASETS_CACHE=/cache/datasets \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

# Previously functioning GPU-host stack. Change only with a documented reason.
# torch 2.11.0+cu128, CUDA userspace 12.8, transformers 5.17.0,
# bitsandbytes 0.50.2, peft 0.21.0, trl 1.13.0
RUN pip install --no-cache-dir \
        "torch==2.11.0" \
        --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir \
        "transformers==5.17.0" \
        "bitsandbytes==0.50.2" \
        "peft==0.21.0" \
        "trl==1.13.0" \
        "accelerate>=1.10" \
        "datasets>=4.0" \
        "safetensors>=0.5" \
        "sentencepiece>=0.2" \
        "huggingface_hub>=0.30"

COPY . /workspace/aisl-train
WORKDIR /workspace/aisl-train

RUN pip install --no-cache-dir --no-deps -e . \
    && mkdir -p /opt/aisl-train \
    && pip freeze > /opt/aisl-train/pip-freeze.txt

ENTRYPOINT []
CMD ["aisl-train", "--help"]
