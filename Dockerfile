# SPDX-License-Identifier: Apache-2.0
# GPU training image. Build context is this aisl-train checkout.
# AISL is cloned from GitHub during the build and is not a host mount.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    HF_DATASETS_CACHE=/cache/datasets \
    AISL_ROOT=/workspace/aisl \
    AISL_PACKAGED_DATA=/opt/aisl-data \
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

# Declared after the ML install so changing AISL_REF does not bust that layer.
ARG AISL_REF=main
ENV AISL_REF=${AISL_REF}

RUN set -eux; \
    git init /workspace/aisl; \
    git -C /workspace/aisl remote add origin https://github.com/johnnyjoy/aisl.git; \
    git -C /workspace/aisl fetch --depth 1 origin "${AISL_REF}"; \
    git -C /workspace/aisl checkout --detach FETCH_HEAD; \
    printf '%s\n' "${AISL_REF}" > /workspace/AISL_REF; \
    git -C /workspace/aisl rev-parse HEAD > /workspace/AISL_COMMIT; \
    EXPORTS=/workspace/aisl/training/data/exports; \
    DATA=/workspace/aisl/training/data; \
    mkdir -p /opt/aisl-data; \
    for f in train.chat.jsonl validation.chat.jsonl test.chat.jsonl \
             train.prompt.jsonl validation.prompt.jsonl test.prompt.jsonl; do \
        if [ ! -f "${EXPORTS}/${f}" ]; then \
            echo "AISL export missing after checkout of ${AISL_REF}: ${EXPORTS}/${f}" >&2; \
            exit 1; \
        fi; \
        ln -s "${EXPORTS}/${f}" "/opt/aisl-data/${f}"; \
    done; \
    if [ ! -f "${DATA}/manifest.json" ]; then \
        echo "AISL manifest.json missing after checkout of ${AISL_REF}" >&2; \
        exit 1; \
    fi; \
    ln -s "${DATA}/manifest.json" /opt/aisl-data/manifest.json; \
    if [ -f "${DATA}/stats.json" ]; then ln -s "${DATA}/stats.json" /opt/aisl-data/stats.json; fi; \
    if [ -f "${DATA}/records.jsonl" ]; then ln -s "${DATA}/records.jsonl" /opt/aisl-data/records.jsonl; fi; \
    if [ ! -d /workspace/aisl/eval ]; then \
        echo "AISL eval/ directory missing after checkout of ${AISL_REF}" >&2; \
        exit 1; \
    fi

COPY . /workspace/aisl-train
WORKDIR /workspace/aisl-train

ARG AISL_TRAIN_COMMIT=unknown
RUN pip install --no-cache-dir --no-deps -e . \
    && mkdir -p /opt/aisl-train \
    && pip freeze > /opt/aisl-train/pip-freeze.txt \
    && printf '%s\n' "${AISL_TRAIN_COMMIT}" > /opt/aisl-train/AISL_TRAIN_COMMIT \
    && python /workspace/aisl-train/scripts/write-image-metadata.py

ENTRYPOINT []
CMD ["aisl-train", "--help"]
