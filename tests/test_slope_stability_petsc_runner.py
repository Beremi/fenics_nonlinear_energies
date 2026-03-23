from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def test_slope_stability_petsc_minimizer_sweep_runner_writes_winner(tmp_path: Path) -> None:
    out_dir = tmp_path / "sweep"
    subprocess.run(
        [
            str(PYTHON),
            "experiments/runners/run_slope_stability_petsc_minimizer_sweep.py",
            "--out-dir",
            str(out_dir),
            "--no-resume",
            "--rep-case",
            "1:1",
            "--verification-case",
            "1:2",
            "--stages",
            "A",
            "--max-candidates-per-stage",
            "1",
            "--verify-top-k",
            "1",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["runner"] == "slope_stability_petsc_minimizer_sweep"
    assert summary["lambda_target"] == 1.2
    assert len(summary["stage_a_ranked"]) == 1
    assert summary["winner"]["candidate"] == summary["stage_a_ranked"][0]["candidate"]
    assert summary["verification_rows"][0]["verification_passed"] is True


def test_slope_stability_petsc_final_suite_runner_writes_summary(tmp_path: Path) -> None:
    out_dir = tmp_path / "final_suite"
    subprocess.run(
        [
            str(PYTHON),
            "experiments/runners/run_slope_stability_petsc_final_suite.py",
            "--out-dir",
            str(out_dir),
            "--no-resume",
            "--levels",
            "1",
            "--nprocs",
            "1",
            "2",
            "--min-elements-per-rank",
            "32",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    rows = summary["rows"]
    assert summary["runner"] == "slope_stability_petsc_final_suite"
    assert summary["lambda_target"] == 1.2
    assert len(rows) == 2
    assert {(row["level"], row["nprocs"]) for row in rows} == {(1, 1), (1, 2)}
    assert all(row["result"] == "completed" for row in rows)


def test_slope_stability_l2_p4_frozen_pmat_runner_writes_summary(tmp_path: Path) -> None:
    out_dir = tmp_path / "frozen_pmat"
    subprocess.run(
        [
            str(PYTHON),
            "experiments/runners/run_slope_stability_l2_p4_frozen_pmat_bench.py",
            "--out-dir",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    rows = summary["rows"]
    assert summary["runner"] == "slope_stability_l2_p4_frozen_pmat_bench"
    assert summary["lambda_target"] == 1.0
    assert len(rows) == 3
    assert {row["name"] for row in rows} == {
        "baseline_assembled_legacy_full",
        "matfree_legacy_pmg_elastic_frozen",
        "matfree_legacy_pmg_initial_tangent_frozen",
    }
    assert summary["matrix_comparison"]["same_pattern"] is True


def test_slope_stability_l2_p4_staggered_pmat_runner_writes_summary(tmp_path: Path) -> None:
    out_dir = tmp_path / "staggered_pmat"
    subprocess.run(
        [
            str(PYTHON),
            "experiments/runners/run_slope_stability_l2_p4_staggered_pmat_bench.py",
            "--out-dir",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    rows = summary["rows"]
    assert summary["runner"] == "slope_stability_l2_p4_staggered_pmat_bench"
    assert summary["lambda_target"] == 1.0
    assert len(rows) == 4
    assert {row["name"] for row in rows} == {
        "baseline_assembled_legacy_full",
        "matfree_legacy_pmg_staggered_whole",
        "matfree_explicit_pmg_staggered_smoother_only_fixed",
        "matfree_explicit_pmg_staggered_smoother_only_refresh_attempt",
    }
