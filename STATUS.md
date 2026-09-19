# Status

```text
generic runtime implemented
unit tested (44 passed)
Qwen profile implemented (minimal; runtime inspection pending)
Docker environment implemented
docker compose config: passed
docker compose build: passed (aisl-train:local)
GPU smoke test not yet run
full Qwen training not yet run
```

This checkout is a generic inspect → dataset-load → QLoRA/SFT →
evaluate → compare runtime. It is not a claim that a 27B model has
been trained.

Do not treat unit tests as GPU readiness. `aisl-train smoke` on
the GPU server is the hardware gate.
