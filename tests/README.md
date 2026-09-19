# tests

Unit tests run without a GPU and without a 27B model.

```bash
python -m pip install -e .
python -m pip install pytest
python -m pytest
```

They cover config, profiles, datasets, loss masks, inspect of tiny
fixtures, experiment metadata, resume selection, comparison, and
CLI `--help`.

`aisl-train smoke` on a GPU host is a separate gate. It is not
part of ordinary `pytest`.
