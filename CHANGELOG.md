# Changelog

## Unreleased

* Clone AISL during the image build; no host AISL mount.
* Packaged corpus defaults at `/opt/aisl-data`.
* Compose mounts only model, output, and cache.

## 0.1.0

First functional generic AISL-Train runtime.

* CLI: doctor, inspect, smoke, baseline, train, evaluate, compare
* Filesystem model inspection and capability object
* JSON model profiles (override only)
* Chat and prompt/completion JSONL loading with row-level errors
* Assistant/completion loss-mask helpers and truncation policy
* Generic QLoRA/SFT path (bitsandbytes + PEFT `all-linear` + TRL)
* Experiment metadata, baseline/prompted/trained comparison
* Docker image and Compose mounts
* Unit tests that do not require a GPU or a 27B model

No prior trainer release existed in this repository.
