# SPDX-License-Identifier: Apache-2.0
import shutil
from pathlib import Path

from aisl_train.compare import compare_experiment

FIXTURES = Path(__file__).resolve().parents[1] / "testdata" / "fixtures"


def test_compare_identifies_improvements_and_regressions(tmp_path: Path) -> None:
    experiment = tmp_path / "pilot-001"
    shutil.copytree(FIXTURES / "experiment", experiment)
    report = compare_experiment(experiment)
    assert set(report["conditions_present"]) == {"base", "prompted", "trained"}
    assert report["overall"]["base"]["surface_exact"] == 2
    assert report["overall"]["prompted"]["surface_exact"] == 2
    assert report["overall"]["trained"]["surface_exact"] == 2
    improved_ids = {row["id"] for row in report["improvements"] if row["from"] == "base" and row["to"] == "prompted"}
    assert "c1" in improved_ids
    trained_reg = {row["id"] for row in report["regressions"] if row["from"] == "base" and row["to"] == "trained"}
    assert "c2" in trained_reg
    assert "aisl_to_answer" in report["breakdowns"]["task"]
    assert "L0" in report["breakdowns"]["difficulty"]
    assert "en" in report["breakdowns"]["language"]
    assert report["model"] == "fixture-model"
