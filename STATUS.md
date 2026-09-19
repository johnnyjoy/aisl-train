# Status

```text
container architecture implemented
AISL build-time acquisition implemented
container build tested                 passed (aisl-train:local)
GPU model inspection tested            not tested
GPU smoke tested                       not tested
full training tested                   not tested
```

Image build cloned AISL `main` at
`698bdd1abc6fd7bf645568ac6e88c932d226b397` and packaged
`/opt/aisl-data`. `aisl-train doctor` inside the image saw the
corpus and ML stack. CUDA/model checks failed on this host because
no GPU and no mounted checkpoint were present. That is not a GPU
smoke result.

Do not treat unit tests as GPU readiness.
