from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from experiments.analysis import merge_plaplace_u3_thesis_chunks as thesis_merge
from experiments.analysis.plaplace_u3_thesis_docs import timing_metadata_for_row
from experiments.runners import run_plaplace_u3_thesis_suite as thesis_suite
from src.core.serial.minimizers import golden_section_search
from src.problems.plaplace_u3.thesis import solver_oa as thesis_solver_oa
from src.problems.plaplace_u3.thesis import solver_rmpa as thesis_solver_rmpa
from src.problems.plaplace_u3.thesis.solver_common import build_problem
from src.problems.plaplace_u3.thesis.solver_mpa import run_mpa


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
THESIS_CLI = "src/problems/plaplace_u3/thesis/scripts/solve_case.py"
THESIS_RUNNER = "experiments/runners/run_plaplace_u3_thesis_suite.py"
THESIS_REPORT = "experiments/analysis/generate_plaplace_u3_thesis_report.py"
THESIS_MERGE = "experiments/analysis/merge_plaplace_u3_thesis_chunks.py"
THESIS_DOC_PAGE = "experiments/analysis/generate_plaplace_u3_thesis_problem_page.py"
THESIS_SECTION = "experiments/analysis/materialize_plaplace_u3_thesis_section.py"
THESIS_DOC_COMMANDS = "experiments/runners/run_plaplace_u3_thesis_problem_doc_commands.py"


def _run_json(command: list[str], output_path: Path) -> dict[str, object]:
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def test_thesis_cli_methods_smoke(tmp_path: Path):
    methods = {
        "rmpa": ["--dimension", "2", "--level", "2", "--p", "2.0", "--direction", "d_vh", "--epsilon", "1e-3", "--maxit", "3"],
        "oa1": ["--dimension", "2", "--level", "2", "--p", "2.0", "--direction", "d_vh", "--epsilon", "1e-3", "--maxit", "3"],
        "oa2": ["--dimension", "2", "--level", "2", "--p", "2.0", "--direction", "d_vh", "--epsilon", "1e-3", "--maxit", "2"],
        "mpa": ["--dimension", "2", "--level", "1", "--p", "2.0", "--direction", "d_vh", "--epsilon", "1e-2", "--maxit", "2", "--num-nodes", "10"],
    }
    for method, extra in methods.items():
        out_path = tmp_path / method / "output.json"
        state_path = tmp_path / method / "state.npz"
        payload = _run_json(
            [
                str(PYTHON),
                "-u",
                THESIS_CLI,
                "--method",
                method,
                "--out",
                str(out_path),
                "--state-out",
                str(state_path),
                *extra,
            ],
            out_path,
        )
        assert payload["method"] == method
        assert payload["direction"] == "d_vh"
        assert payload["status"] in {"completed", "maxit", "failed"}
        assert np.isfinite(payload["J"])
        assert np.isfinite(payload["I"])
        assert state_path.exists()


def test_thesis_square_hole_oa2_smoke(tmp_path: Path):
    out_path = tmp_path / "square_hole" / "output.json"
    payload = _run_json(
        [
            str(PYTHON),
            "-u",
            THESIS_CLI,
            "--method",
            "oa2",
            "--dimension",
            "2",
            "--geometry",
            "square_hole_pi",
            "--init-mode",
            "abs_sine_y2",
            "--level",
            "3",
            "--p",
            "2.0",
            "--direction",
            "d_vh",
            "--epsilon",
            "1e-3",
            "--maxit",
            "2",
            "--out",
            str(out_path),
        ],
        out_path,
    )
    assert payload["geometry"] == "square_hole_pi"
    assert payload["method"] == "oa2"
    assert np.isfinite(payload["J"])


def test_mpa_reports_last_evaluated_peak_when_hitting_maxit():
    payload = run_mpa(
        build_problem(
            dimension=2,
            level=2,
            p=2.0,
            geometry="square_pi",
            init_mode="sine",
            seed=0,
        ),
        direction="d_vh",
        epsilon=1.0e-8,
        maxit=2,
        num_nodes=10,
        rho=1.0,
        segment_tol_factor=0.125,
    )
    assert payload["status"] == "maxit"
    assert payload["history"]
    assert abs(float(payload["J"]) - float(payload["history"][-1]["J"])) < 1.0e-12
    assert payload["configured_maxit"] == 2
    assert payload["accepted_step_count"] == 2
    assert payload["best_stop_measure"] is not None
    assert payload["best_stop_outer_it"] is not None
    assert payload["max_halves"] >= 0
    assert payload["final_halves"] >= 0
    assert payload["refinement_count"] >= 0
    assert isinstance(payload["distinct_peak_nodes"], int)
    assert isinstance(payload["peak_cycle_detected"], bool)


def test_row_from_result_propagates_solver_diagnostics():
    case = thesis_suite.Case(
        table="table_5_2",
        method="rmpa",
        direction="d",
        dimension=1,
        geometry="interval_pi",
        level=5,
        p=2.0,
        epsilon=1.0e-4,
        init_mode="sine",
    )
    problem = build_problem(
        dimension=1,
        level=5,
        p=2.0,
        geometry="interval_pi",
        init_mode="sine",
        seed=0,
    )
    result = {
        "status": "completed",
        "message": "ok",
        "J": 0.5,
        "I": 0.25,
        "c": 1.0,
        "outer_iterations": 7,
        "direction_solves": 7,
        "configured_maxit": 500,
        "best_stop_measure": 1.0e-5,
        "best_stop_outer_it": 6,
        "accepted_step_count": 7,
        "max_halves": 3,
        "final_halves": 1,
        "objective_name": "J",
        "delta0": 1.0,
        "step_search": "golden",
    }

    row = thesis_suite._row_from_result(
        case,
        problem,
        result,
        result_path=Path("/tmp/fake_output.json"),
        solve_time_s=1.23,
    )

    assert row["configured_maxit"] == 500
    assert row["best_stop_measure"] == pytest.approx(1.0e-5)
    assert row["best_stop_outer_it"] == 6
    assert row["accepted_step_count"] == 7
    assert row["max_halves"] == 3
    assert row["final_halves"] == 1
    assert row["objective_name"] == "J"
    assert row["delta0"] == pytest.approx(1.0)
    assert row["step_search"] == "golden"


def test_merge_row_score_prefers_current_budget_repo_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_repo_root = tmp_path / "repo"
    fake_repo_root.mkdir()
    monkeypatch.setattr(thesis_merge, "REPO_ROOT", fake_repo_root)

    source_path = tmp_path / "source_summary.json"
    source_path.write_text("{}", encoding="utf-8")

    repo_result = fake_repo_root / "artifacts" / "raw_results" / "fresh" / "output.json"
    repo_result.parent.mkdir(parents=True, exist_ok=True)
    repo_result.write_text("{}", encoding="utf-8")

    stale_result = tmp_path / "stale_output.json"
    stale_result.write_text("{}", encoding="utf-8")

    fresh_row = {
        "method": "rmpa",
        "configured_maxit": 500,
        "result_path": str(repo_result),
    }
    stale_row = {
        "method": "rmpa",
        "configured_maxit": 200,
        "result_path": str(stale_result),
    }

    assert thesis_merge._row_score(fresh_row, source_path=source_path, source_priority=1) > thesis_merge._row_score(
        stale_row,
        source_path=source_path,
        source_priority=0,
    )


def test_merge_row_score_prefers_publishable_timing_over_zero_time_chunk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_repo_root = tmp_path / "repo"
    fake_repo_root.mkdir()
    monkeypatch.setattr(thesis_merge, "REPO_ROOT", fake_repo_root)

    chunk_summary = fake_repo_root / "artifacts" / "raw_results" / "plaplace_u3_thesis_chunks" / "chunk0" / "summary.json"
    chunk_summary.parent.mkdir(parents=True, exist_ok=True)
    chunk_summary.write_text("{}", encoding="utf-8")
    fresh_summary = fake_repo_root / "artifacts" / "raw_results" / "plaplace_u3_thesis_refresh" / "summary.json"
    fresh_summary.parent.mkdir(parents=True, exist_ok=True)
    fresh_summary.write_text("{}", encoding="utf-8")

    chunk_result = fake_repo_root / "artifacts" / "raw_results" / "plaplace_u3_thesis_chunks" / "chunk0" / "output.json"
    chunk_result.write_text("{}", encoding="utf-8")
    fresh_result = fake_repo_root / "artifacts" / "raw_results" / "plaplace_u3_thesis_refresh" / "output.json"
    fresh_result.write_text("{}", encoding="utf-8")

    stale_zero_time = {
        "table": "table_5_13",
        "method": "rmpa",
        "configured_maxit": 500,
        "status": "completed",
        "solve_time_s": 0.0,
        "result_path": str(chunk_result),
    }
    fresh_publishable = {
        "table": "table_5_13",
        "method": "rmpa",
        "configured_maxit": 500,
        "status": "completed",
        "solve_time_s": 1.23,
        "result_path": str(fresh_result),
    }

    assert thesis_merge._row_score(
        fresh_publishable,
        source_path=fresh_summary,
        source_priority=1,
    ) > thesis_merge._row_score(
        stale_zero_time,
        source_path=chunk_summary,
        source_priority=0,
    )


def test_merge_row_score_prefers_stable_repo_artifact_over_temp_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_repo_root = tmp_path / "repo"
    fake_repo_root.mkdir()
    monkeypatch.setattr(thesis_merge, "REPO_ROOT", fake_repo_root)

    summary_path = fake_repo_root / "artifacts" / "raw_results" / "overlay" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("{}", encoding="utf-8")

    stable_result = fake_repo_root / "artifacts" / "raw_results" / "overlay" / "output.json"
    stable_result.write_text("{}", encoding="utf-8")
    temp_result = tmp_path / "tmp_output.json"
    temp_result.write_text("{}", encoding="utf-8")

    stable_row = {
        "method": "rmpa",
        "configured_maxit": 500,
        "status": "completed",
        "solve_time_s": 1.0,
        "result_path": str(stable_result),
    }
    temp_row = {
        "method": "rmpa",
        "configured_maxit": 500,
        "status": "completed",
        "solve_time_s": 1.0,
        "result_path": str(temp_result),
    }

    assert thesis_merge._row_score(stable_row, source_path=summary_path, source_priority=1) > thesis_merge._row_score(
        temp_row,
        source_path=summary_path,
        source_priority=1,
    )


def test_timing_metadata_distinguishes_unavailable_from_non_completed():
    unavailable = timing_metadata_for_row(
        {
            "table": "table_5_13",
            "method": "rmpa",
            "direction": "d",
            "p": 2.0,
            "status": "completed",
            "outer_iterations": 8,
            "solve_time_s": 0.0,
            "launcher": "serial python",
            "process_count": 1,
        },
        "table_5_13",
    )
    assert unavailable["timing_status"] == "timing unavailable"
    assert "timing propagation bug" in str(unavailable["timing_reason"])
    assert unavailable["repo_time_s"] is None
    assert unavailable["repo_iterations"] == 8

    non_completed = timing_metadata_for_row(
        {
            "table": "table_5_12",
            "method": "mpa",
            "p": 2.0,
            "status": "maxit",
            "outer_iterations": 1000,
            "configured_maxit": 1000,
            "solve_time_s": 45.0,
            "launcher": "serial python",
            "process_count": 1,
        },
        "table_5_12",
    )
    assert non_completed["timing_status"] == "non-completed"
    assert "maxit=1000" in str(non_completed["timing_reason"])
    assert non_completed["repo_time_s"] is None
    assert non_completed["raw_repo_time_s"] == pytest.approx(45.0)


def test_golden_section_search_keeps_boundary_minimizer():
    alpha, n_evals = golden_section_search(lambda x: float(x), 0.0, 1.0, 1.0e-5)
    assert alpha == pytest.approx(0.0, abs=1.0e-12)
    assert n_evals >= 4


def test_rmpa_halving_search_tries_the_true_half_step(monkeypatch: pytest.MonkeyPatch):
    attempted: list[float] = []

    class _Stats:
        def __init__(self, j: float) -> None:
            self.J = j
            self.I = j
            self.c = 1.0
            self.scale_to_solution = 1.0

    class _Problem:
        params: dict[str, object] = {}
        u_init = np.zeros(1, dtype=np.float64)

        def stats(self, u_free: np.ndarray) -> _Stats:
            return _stats_for(u_free)

    def _stats_for(u_free: np.ndarray) -> _Stats:
        x = float(np.asarray(u_free, dtype=np.float64)[0])
        attempted.append(x)
        return _Stats(0.0 if abs(x - 0.5) <= 1.0e-12 else 1.0)

    class _Directions:
        def compute(self, current: np.ndarray, direction_kind: str):
            return type(
                "DirResult",
                (),
                {
                    "direction": np.array([1.0], dtype=np.float64),
                    "stop_measure": 1.0,
                    "stop_name": "(5.7)",
                    "descent_value": -1.0,
                    "direction_solves": 1,
                },
            )()

    monkeypatch.setattr(thesis_solver_rmpa, "build_objective_bundle", lambda problem, objective: object())
    monkeypatch.setattr(thesis_solver_rmpa, "build_direction_context", lambda problem, objective: _Directions())
    monkeypatch.setattr(thesis_solver_rmpa, "compute_state_stats_free", lambda params, u_free: _stats_for(u_free))
    monkeypatch.setattr(
        thesis_solver_rmpa,
        "build_result_payload",
        lambda **kwargs: {"status": kwargs["status"], "history": kwargs["history"], "message": kwargs["message"]},
    )

    result = thesis_solver_rmpa.run_rmpa(
        _Problem(),
        direction="d_vh",
        epsilon=1.0e-4,
        maxit=1,
        delta0=1.0,
        step_search="halving",
    )

    assert any(abs(x - 0.5) <= 1.0e-12 for x in attempted)
    assert result["history"][0]["accepted"] is True
    assert result["history"][0]["alpha"] == pytest.approx(0.5)


def test_oa1_halving_search_tries_the_true_half_step(monkeypatch: pytest.MonkeyPatch):
    attempted: list[float] = []

    class _Stats:
        def __init__(self, value: float) -> None:
            self.I = value
            self.c = 1.0
            self.J = value
            self.seminorm_p = 1.0
            self.scale_to_solution = 1.0

    class _Problem:
        u_init = np.zeros(1, dtype=np.float64)

        def stats(self, u_free: np.ndarray) -> _Stats:
            x = float(np.asarray(u_free, dtype=np.float64)[0])
            attempted.append(x)
            return _Stats(0.0 if abs(x - 0.5) <= 1.0e-12 else 1.0)

    class _Objective:
        @staticmethod
        def value(u_free: np.ndarray) -> float:
            x = float(np.asarray(u_free, dtype=np.float64)[0])
            attempted.append(x)
            return 0.0 if abs(x - 0.5) <= 1.0e-12 else 1.0

    class _Directions:
        def compute(self, current: np.ndarray, direction_kind: str):
            return type(
                "DirResult",
                (),
                {
                    "direction": np.array([1.0], dtype=np.float64),
                    "stop_measure": 1.0,
                    "stop_name": "(5.7)",
                    "descent_value": -1.0,
                    "direction_solves": 1,
                },
            )()

    monkeypatch.setattr(thesis_solver_oa, "build_objective_bundle", lambda problem, objective: _Objective())
    monkeypatch.setattr(thesis_solver_oa, "build_direction_context", lambda problem, objective: _Directions())
    monkeypatch.setattr(
        thesis_solver_oa,
        "build_result_payload",
        lambda **kwargs: {"status": kwargs["status"], "history": kwargs["history"], "message": kwargs["message"]},
    )

    result = thesis_solver_oa.run_oa(
        _Problem(),
        variant="oa1",
        direction="d_vh",
        epsilon=1.0e-4,
        maxit=1,
        delta_hat=1.0,
    )

    assert any(abs(x - 0.5) <= 1.0e-12 for x in attempted)
    assert result["history"][0]["accepted"] is True
    assert result["history"][0]["alpha"] == pytest.approx(0.5)


def test_thesis_quick_runner_and_report(tmp_path: Path):
    out_dir = tmp_path / "raw"
    subprocess.run(
        [
            str(PYTHON),
            "-u",
            THESIS_RUNNER,
            "--quick",
            "--skip-reference",
            "--out-dir",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["quick"] is True
    assert len(summary["rows"]) == 5
    assert {"table", "method", "J", "I", "outer_iterations"} <= set(summary["rows"][0].keys())

    report_path = tmp_path / "report" / "README.md"
    subprocess.run(
        [
            str(PYTHON),
            "-u",
            THESIS_REPORT,
            "--summary",
            str(summary_path),
            "--summary-label",
            "artifacts/raw_results/plaplace_u3_thesis_full/summary.json",
            "--out",
            str(report_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    text = report_path.read_text(encoding="utf-8")
    assert "pLaplaceU3 Thesis Reproduction Report" in text
    assert "## Thesis Problem And Functionals" in text
    assert "## Assignment Stage Map" in text
    assert "## What Works" in text
    assert "artifacts/raw_results/plaplace_u3_thesis_full/summary.json" in text
    assert (report_path.parent / "quick_sample.png").exists()


def test_thesis_runner_only_table_filter(tmp_path: Path):
    out_dir = tmp_path / "filtered"
    subprocess.run(
        [
            str(PYTHON),
            "-u",
            THESIS_RUNNER,
            "--quick",
            "--skip-reference",
            "--only-table",
            "quick",
            "--out-dir",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["rows"]) == 5
    assert {row["table"] for row in summary["rows"]} == {"quick"}


def test_thesis_problem_page_generator_smoke(tmp_path: Path):
    out_path = tmp_path / "docs" / "pLaplace_u3_thesis_replications.md"
    asset_dir = tmp_path / "assets"
    subprocess.run(
        [
            str(PYTHON),
            "-u",
            THESIS_DOC_PAGE,
            "--summary",
            "artifacts/raw_results/plaplace_u3_thesis_full/summary.json",
            "--out",
            str(out_path),
            "--asset-dir",
            str(asset_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    text = out_path.read_text(encoding="utf-8")
    assert "Michaela Bailová" in text
    assert "## Implementation Map" in text
    assert "src/problems/plaplace_u3/thesis/scripts/solve_case.py" in text
    assert "## RMPA Square Principal-Branch Replication" in text
    assert "## Rebuild The Canonical Thesis Packet And This Page" in text
    assert "../assets/plaplace_u3_thesis/" not in text
    assert text.count("```bash") >= 7
    assert "secondary target" not in text
    assert "- unresolved rows:" in text
    assert "### What is low impact" in text
    assert "## Convergence Diagnostics" in text
    assert (asset_dir / "plaplace_u3_sample_state.png").exists()
    assert (asset_dir / "plaplace_u3_sample_state.pdf").exists()
    assert (asset_dir / "square_multibranch_panel.png").exists()
    assert (asset_dir / "square_hole_panel.png").exists()


def test_thesis_report_generator_includes_stage_c_timing_summary(tmp_path: Path):
    out_path = tmp_path / "report" / "README.md"
    subprocess.run(
        [
            str(PYTHON),
            "-u",
            THESIS_REPORT,
            "--summary",
            "artifacts/raw_results/plaplace_u3_thesis_full/summary.json",
            "--summary-label",
            "artifacts/raw_results/plaplace_u3_thesis_full/summary.json",
            "--out",
            str(out_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    text = out_path.read_text(encoding="utf-8")
    assert "## Stage C Timing Summary" in text
    assert "timing complete" in text
    assert "timing unavailable" in text
    assert "non-completed" in text
    assert "## Convergence Diagnostics" in text
    assert "solver status" in text
    assert "maxit=1000" in text
    assert "1 proc, serial python, JAX + SciPy + PyAMG helper solves" in text
    assert "thesis Table 5.12 timings are surfaced alongside the current local timings" in text
    assert "timing comparison remains blocked" not in text


def test_thesis_section_materializer_smoke(tmp_path: Path):
    out_dir = tmp_path / "section"
    subprocess.run(
        [
            str(PYTHON),
            "-u",
            THESIS_SECTION,
            "--summary",
            "artifacts/raw_results/plaplace_u3_thesis_full/summary.json",
            "--only-table",
            "table_5_8",
            "--only-table",
            "table_5_9",
            "--out-dir",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["selected_tables"] == ["table_5_8", "table_5_9"]
    assert summary["num_rows"] == 60
    assert {row["table"] for row in summary["rows"]} == {"table_5_8", "table_5_9"}
    assert (out_dir / "README.md").exists()


def test_thesis_problem_doc_command_runner_smoke(tmp_path: Path):
    doc_path = tmp_path / "page.md"
    out_a = tmp_path / "section_a"
    out_b = tmp_path / "section_b"
    summary_out = tmp_path / "summary_report.md"
    doc_out = tmp_path / "generated.md"
    assets_out = tmp_path / "assets"
    doc_path.write_text(
        "\n".join(
            [
                "```bash",
                f"./.venv/bin/python -u {THESIS_SECTION} \\",
                "  --summary artifacts/raw_results/plaplace_u3_thesis_full/summary.json \\",
                "  --only-table table_5_8 \\",
                f"  --out-dir {out_a}",
                "```",
                "",
                "```bash",
                f"./.venv/bin/python -u {THESIS_SECTION} \\",
                "  --summary artifacts/raw_results/plaplace_u3_thesis_full/summary.json \\",
                "  --only-table table_5_14 \\",
                f"  --out-dir {out_b}",
                "```",
                "",
                "```bash",
                f"./.venv/bin/python -u {THESIS_REPORT} \\",
                "  --summary artifacts/raw_results/plaplace_u3_thesis_full/summary.json \\",
                "  --summary-label artifacts/raw_results/plaplace_u3_thesis_full/summary.json \\",
                f"  --out {summary_out}",
                "```",
                "",
                "```bash",
                f"./.venv/bin/python -u {THESIS_DOC_PAGE} \\",
                "  --summary artifacts/raw_results/plaplace_u3_thesis_full/summary.json \\",
                f"  --out {doc_out} \\",
                f"  --asset-dir {assets_out}",
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            str(PYTHON),
            "-u",
            THESIS_DOC_COMMANDS,
            "--doc",
            str(doc_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (out_a / "summary.json").exists()
    assert (out_b / "summary.json").exists()
    assert summary_out.exists()
    assert doc_out.exists()


def test_thesis_calibration_handles_missing_level_table_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fake_run_case(case, **kwargs):
        target_j = thesis_suite._published_target_j(case) or 1.0
        target_iterations = thesis_suite.TABLE_5_12_ITERATIONS.get(float(case.p), {}).get(case.method, 0)
        return {
            "table": case.table,
            "method": case.method,
            "p": case.p,
            "J": target_j,
            "outer_iterations": target_iterations,
        }

    monkeypatch.setattr(thesis_suite, "_run_case", fake_run_case)
    monkeypatch.setattr(thesis_suite, "THESIS_CANDIDATE_RMPA_DELTA0", (1.0,))
    monkeypatch.setattr(thesis_suite, "THESIS_CANDIDATE_OA_DELTA_HAT", (1.0,))
    monkeypatch.setattr(thesis_suite, "THESIS_CANDIDATE_MPA_SEGMENT_TOL_FACTORS", (0.125,))

    payload = thesis_suite._run_calibration(tmp_path / "calibration")
    assert {"best", "score", "rows", "constraint_satisfied"} <= set(payload.keys())
    assert {"rmpa_delta0", "oa_delta_hat", "mpa_segment_tol_factor"} <= set(payload["best"].keys())
    assert payload["rows"]
    assert payload["rows"][0]["assignment_stage"] == "Calibration"


def test_thesis_report_handles_full_table_rows(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "suite": "plaplace_u3_thesis_full",
                "quick": False,
                "skip_reference": False,
                "constants": {
                    "rmpa_delta0": 1.0,
                    "oa_delta_hat": 1.0,
                    "mpa_segment_tol_factor": 0.125,
                },
                "assignment_overview": {
                    "problem_statement": "test problem",
                    "functional_summary": "test functional",
                    "geometry_summary": "test geometry",
                    "discretization_summary": "test discretisation",
                    "seed_summary": "test seeds",
                    "legend": {"exact": "exact", "proxy": "proxy", "unmatched": "unmatched"},
                    "stage_details": {
                        "Stage A": "a",
                        "Stage B": "b",
                        "Stage C": "c",
                    },
                    "method_to_tables": {"RMPA": ["Table 5.9"], "OA1": ["Table 5.11"], "MPA": ["Table 5.7"]},
                    "overall": {"pass": 3, "fail": 0, "unknown": 0},
                    "by_stage": {
                        "Stage A": {"pass": 1, "fail": 0, "unknown": 0, "total": 1},
                        "Stage B": {"pass": 1, "fail": 0, "unknown": 0, "total": 1},
                        "Stage C": {"pass": 1, "fail": 0, "unknown": 0, "total": 1},
                    },
                    "by_table": {
                        "table_5_7": {"pass": 1, "fail": 0, "unknown": 0, "total": 1},
                        "table_5_9": {"pass": 1, "fail": 0, "unknown": 0, "total": 1},
                        "table_5_11": {"pass": 1, "fail": 0, "unknown": 0, "total": 1},
                    },
                },
                "rows": [
                    {
                        "table": "table_5_9",
                        "method": "rmpa",
                        "direction": "d_vh",
                        "geometry": "square_pi",
                        "level": 6,
                        "p": 2.0,
                        "epsilon": 1.0e-4,
                        "init_mode": "sine",
                        "J": 3.8324,
                        "I": 1.9787,
                        "outer_iterations": 8,
                        "status": "completed",
                        "thesis_J": 3.83,
                        "delta_J": 0.0024,
                        "thesis_error": 1.23e-3,
                        "reference_error_w1p": 1.24e-3,
                        "assignment_stage": "Stage A",
                        "assignment_section": "Section 14.2 / Table 5.9",
                        "assignment_target": "RMPA principal branch by tolerance",
                        "assignment_primary": True,
                        "assignment_acceptance_pass": True,
                    },
                    {
                        "table": "table_5_11",
                        "method": "oa1",
                        "direction": "d_vh",
                        "geometry": "square_pi",
                        "level": 6,
                        "p": 2.0,
                        "epsilon": 1.0e-4,
                        "init_mode": "sine",
                        "J": 3.8325,
                        "I": 1.9788,
                        "outer_iterations": 7,
                        "status": "completed",
                        "thesis_J": 3.83,
                        "delta_J": 0.0025,
                        "thesis_error": 1.23e-3,
                        "reference_error_w1p": 1.22e-3,
                        "assignment_stage": "Stage B",
                        "assignment_section": "Section 15.2 / Table 5.11",
                        "assignment_target": "OA1 principal branch by tolerance",
                        "assignment_primary": False,
                        "assignment_acceptance_pass": None,
                    },
                    {
                        "table": "table_5_7",
                        "method": "mpa",
                        "direction": "d_vh",
                        "geometry": "square_pi",
                        "level": 6,
                        "p": 2.0,
                        "epsilon": 1.0e-4,
                        "init_mode": "sine",
                        "J": 3.8331,
                        "I": 1.9790,
                        "outer_iterations": 68,
                        "status": "completed",
                        "thesis_J": 3.83,
                        "delta_J": 0.0031,
                        "thesis_error": 1.23e-3,
                        "reference_error_w1p": 1.25e-3,
                        "assignment_stage": "Stage C",
                        "assignment_section": "Section 16.1 / Table 5.7",
                        "assignment_target": "MPA principal-branch tolerance cross-check",
                        "assignment_primary": True,
                        "assignment_acceptance_pass": True,
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_path = tmp_path / "full_report" / "README.md"
    subprocess.run(
        [
            str(PYTHON),
            "-u",
            THESIS_REPORT,
            "--summary",
            str(summary_path),
            "--summary-label",
            "artifacts/raw_results/plaplace_u3_thesis_full/summary.json",
            "--out",
            str(report_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    text = report_path.read_text(encoding="utf-8")
    assert "## Thesis Problem And Functionals" in text
    assert "## Table-By-Table Status Matrix" in text
    assert "## What Is Low Impact" in text
    assert "## What Needs Context" in text
    assert "## What Does Not Yet Match" in text
    assert "## Convergence Diagnostics" in text
    assert "artifacts/raw_results/plaplace_u3_thesis_full/summary.json" in text


def test_canonical_summary_promotes_refreshed_fail_family_rows():
    rows = json.loads((REPO_ROOT / "artifacts" / "raw_results" / "plaplace_u3_thesis_full" / "summary.json").read_text(encoding="utf-8"))["rows"]

    def pick(table: str, level: int, p: float, epsilon: float) -> dict[str, object]:
        for row in rows:
            if (
                str(row["table"]) == table
                and int(row["level"]) == level
                and abs(float(row["p"]) - p) <= 1.0e-12
                and abs(float(row["epsilon"]) - epsilon) <= 1.0e-14
            ):
                return row
        raise AssertionError((table, level, p, epsilon))

    table58_l6 = pick("table_5_8", 6, 1.5, 1.0e-5)
    assert table58_l6["status"] == "completed"
    assert table58_l6["configured_maxit"] == 500
    assert "table_5_8" in str(table58_l6["result_path"])
    assert "p1p5" in str(table58_l6["result_path"])
    assert (
        "plaplace_u3_fail_refresh_20260324" in str(table58_l6["result_path"])
        or "plaplace_u3_thesis_sections/rmpa_square" in str(table58_l6["result_path"])
    )

    table58_l7 = pick("table_5_8", 7, 1.5, 1.0e-5)
    assert table58_l7["status"] == "completed"
    assert table58_l7["configured_maxit"] == 500
    assert "table_5_8" in str(table58_l7["result_path"])
    assert "p1p5" in str(table58_l7["result_path"])
    assert (
        "plaplace_u3_fail_refresh_20260324" in str(table58_l7["result_path"])
        or "plaplace_u3_thesis_sections/rmpa_square" in str(table58_l7["result_path"])
    )

    table510_l7 = pick("table_5_10", 7, 1.5, 1.0e-5)
    assert table510_l7["status"] == "completed"
    assert table510_l7["configured_maxit"] == 500
    assert "table_5_10" in str(table510_l7["result_path"])
    assert "p1p5" in str(table510_l7["result_path"])
    assert (
        "plaplace_u3_fail_refresh_20260324" in str(table510_l7["result_path"])
        or "plaplace_u3_thesis_sections/oa1_square" in str(table510_l7["result_path"])
    )

    drn = pick("table_5_2_drn_sanity", 11, 2.0, 1.0e-3)
    assert drn["status"] == "completed"
    assert drn["configured_maxit"] == 500
    assert "Stopping criterion (5.8) satisfied" in str(drn["message"])

    table52_lowp = pick("table_5_2", 11, 1.5, 1.0e-4)
    assert table52_lowp["status"] == "failed"
    assert table52_lowp["configured_maxit"] == 500
    assert "ray maximum" in str(table52_lowp["message"])
    assert (
        "plaplace_u3_fail_refresh_20260324/one_dimensional" in str(table52_lowp["result_path"])
        or "plaplace_u3_thesis_sections/one_dimensional" in str(table52_lowp["result_path"])
    )

    table53_lowp = pick("table_5_3", 11, 1.5, 1.0e-4)
    assert table53_lowp["status"] == "failed"
    assert table53_lowp["configured_maxit"] == 500
    assert "ray maximum" in str(table53_lowp["message"])
    assert (
        "plaplace_u3_fail_refresh_20260324/one_dimensional" in str(table53_lowp["result_path"])
        or "plaplace_u3_thesis_sections/one_dimensional" in str(table53_lowp["result_path"])
    )

    table59_lowp = pick("table_5_9", 6, 1.5, 1.0e-5)
    assert table59_lowp["status"] == "completed"
    assert table59_lowp["configured_maxit"] == 500
    assert float(table59_lowp["solve_time_s"]) > 0.0
    assert "plaplace_u3_repair_20260324/table_5_9_fresh" in str(table59_lowp["result_path"])

    table513_rmpa = next(
        row
        for row in rows
        if str(row["table"]) == "table_5_13"
        and str(row["method"]) == "rmpa"
        and str(row["direction"]) == "d"
        and int(row["level"]) == 6
        and abs(float(row["p"]) - 2.0) <= 1.0e-12
        and abs(float(row["epsilon"]) - 1.0e-4) <= 1.0e-14
    )
    assert table513_rmpa["status"] == "completed"
    assert table513_rmpa["configured_maxit"] == 500
    assert float(table513_rmpa["solve_time_s"]) > 0.0
    assert "plaplace_u3_repair_20260324/table_5_13_fresh" in str(table513_rmpa["result_path"])

    table56_l7 = pick("table_5_6", 7, 10.0 / 6.0, 1.0e-4)
    assert table56_l7["status"] == "maxit"
    if table56_l7.get("configured_maxit") is not None:
        assert table56_l7["configured_maxit"] == 1000
    if table56_l7.get("peak_cycle_detected") is not None:
        assert table56_l7["peak_cycle_detected"] is True

    table514_oa2 = next(
        row
        for row in rows
        if str(row["table"]) == "table_5_14"
        and str(row["method"]) == "oa2"
        and str(row["init_mode"]) == "sine"
    )
    assert table514_oa2["status"] == "completed"
    assert float(table514_oa2["solve_time_s"]) > 0.0
    assert "plaplace_u3_table_5_14_refresh" in str(table514_oa2["result_path"])


def test_thesis_chunk_merge_smoke(tmp_path: Path):
    chunks_dir = tmp_path / "chunks"
    for idx, row in enumerate(
        [
            {
                "table": "quick",
                "method": "rmpa",
                "direction": "d_vh",
                "dimension": 2,
                "geometry": "square_pi",
                "level": 2,
                "p": 2.0,
                "epsilon": 1.0e-3,
                "init_mode": "sine",
                "seed": 0,
                "reference_level": None,
                "h": 0.1,
                "solve_time_s": 0.1,
                "result_path": "a.json",
                "state_path": None,
                "status": "completed",
                "message": "ok",
                "J": 1.0,
                "I": 2.0,
                "c": 0.5,
                "reference_error_w1p": None,
                "outer_iterations": 3,
                "direction_solves": 3,
            },
            {
                "table": "quick",
                "method": "oa1",
                "direction": "d_vh",
                "dimension": 2,
                "geometry": "square_pi",
                "level": 2,
                "p": 2.0,
                "epsilon": 1.0e-3,
                "init_mode": "sine",
                "seed": 0,
                "reference_level": None,
                "h": 0.1,
                "solve_time_s": 0.2,
                "result_path": "b.json",
                "state_path": None,
                "status": "completed",
                "message": "ok",
                "J": 1.1,
                "I": 2.1,
                "c": 0.49,
                "reference_error_w1p": None,
                "outer_iterations": 4,
                "direction_solves": 4,
            },
            {
                "table": "quick",
                "method": "oa2",
                "direction": "d_vh",
                "dimension": 2,
                "geometry": "square_pi",
                "level": 3,
                "p": 2.0,
                "epsilon": 1.0e-3,
                "init_mode": "sine_x2",
                "seed": 0,
                "reference_level": None,
                "h": 0.05,
                "solve_time_s": 0.3,
                "result_path": "c.json",
                "state_path": None,
                "status": "completed",
                "message": "ok",
                "J": 3.0,
                "I": 2.5,
                "c": 0.4,
                "reference_error_w1p": None,
                "outer_iterations": 5,
                "direction_solves": 5,
            },
            {
                "table": "quick",
                "method": "oa2",
                "direction": "d_vh",
                "dimension": 2,
                "geometry": "square_hole_pi",
                "level": 3,
                "p": 2.0,
                "epsilon": 1.0e-3,
                "init_mode": "abs_sine_y2",
                "seed": 0,
                "reference_level": None,
                "h": 0.05,
                "solve_time_s": 0.4,
                "result_path": "d.json",
                "state_path": None,
                "status": "completed",
                "message": "ok",
                "J": 4.0,
                "I": 3.5,
                "c": 0.3,
                "reference_error_w1p": None,
                "outer_iterations": 6,
                "direction_solves": 6,
            },
            {
                "table": "quick",
                "method": "rmpa",
                "direction": "d_vh",
                "dimension": 1,
                "geometry": "interval_pi",
                "level": 5,
                "p": 2.0,
                "epsilon": 1.0e-3,
                "init_mode": "sine",
                "seed": 0,
                "reference_level": None,
                "h": 0.05,
                "solve_time_s": 0.5,
                "result_path": "e.json",
                "state_path": None,
                "status": "completed",
                "message": "ok",
                "J": 0.5,
                "I": 1.2,
                "c": 0.8,
                "reference_error_w1p": None,
                "outer_iterations": 5,
                "direction_solves": 5,
                "thesis_J": 4.47,
                "delta_J": 0.40723798882084505,
                "thesis_error": 5.24e-4,
                "delta_error": 8.1e-2,
                "assignment_verdict": "fail",
            },
        ]
    ):
        chunk_dir = chunks_dir / f"chunk{idx}"
        chunk_dir.mkdir(parents=True)
        (chunk_dir / "summary.json").write_text(
            json.dumps(
                {
                    "suite": "plaplace_u3_thesis_quick",
                    "quick": True,
                    "skip_reference": True,
                    "chunk_index": idx,
                    "chunk_count": 5,
                    "constants": {
                        "rmpa_delta0": 1.0,
                        "oa_delta_hat": 1.0,
                        "mpa_segment_tol_factor": 0.125,
                    },
                    "rows": [row],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    out_path = tmp_path / "merged" / "summary.json"
    subprocess.run(
        [
            str(PYTHON),
            "-u",
            THESIS_MERGE,
            "--chunks-dir",
            str(chunks_dir),
            "--out",
            str(out_path),
            "--quick",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    merged = json.loads(out_path.read_text(encoding="utf-8"))
    assert merged["chunked"] is True
    assert merged["chunk_count"] == 5
    assert len(merged["rows"]) == 5
    refreshed = next(row for row in merged["rows"] if row["result_path"] == "e.json")
    assert "thesis_J" not in refreshed
    assert refreshed["assignment_verdict"] == "secondary"
