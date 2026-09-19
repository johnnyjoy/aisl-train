# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from aisl_train import __version__
from aisl_train.experiment import build_provenance, prepare_experiment_dir, warn_if_missing_baseline


def test_experiment_layout(tmp_path: Path) -> None:
    paths = prepare_experiment_dir(tmp_path / "pilot-001")
    assert paths.adapter.is_dir()
    assert paths.checkpoints.is_dir()
    assert paths.baseline.is_dir()
    assert paths.trained_eval.name == "trained-eval"


def test_provenance_reports_missing_fields() -> None:
    prov = build_provenance(aisl_train_version=__version__, seed=42)
    assert prov.aisl_train_version == __version__
    assert "aisl_version" in prov.missing
    assert "model_path" in prov.missing


def test_provenance_copies_manifest() -> None:
    prov = build_provenance(
        aisl_train_version=__version__,
        model_path="/models/base",
        profile_id="example",
        manifest={
            "aisl_version": "0.2.0",
            "dataset_version": "0.1.0",
            "content_hash": "abc",
            "dictionary_versions": ["core@0.2.0"],
        },
    )
    assert prov.aisl_version == "0.2.0"
    assert prov.aisl_corpus_content_hash == "abc"
    assert "aisl_version" not in prov.missing


def test_baseline_warning(tmp_path: Path) -> None:
    root = tmp_path / "exp"
    prepare_experiment_dir(root)
    assert warn_if_missing_baseline(root)
    (root / "baseline" / "summary.json").write_text("{}", encoding="utf-8")
    assert warn_if_missing_baseline(root) is None
