from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from experiments.runners import run_plaplace_u3_thesis_suite as thesis_suite


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
THESIS_CLI = "src/problems/plaplace_u3/thesis/solve_plaplace_u3_thesis.py"
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
    assert "## RMPA Square Principal-Branch Replication" in text
    assert "## Rebuild The Canonical Thesis Packet And This Page" in text
    assert "../assets/plaplace_u3_thesis/" not in text
    assert text.count("```bash") >= 7
    assert (asset_dir / "plaplace_u3_sample_state.png").exists()
    assert (asset_dir / "plaplace_u3_sample_state.pdf").exists()
    assert (asset_dir / "square_multibranch_panel.png").exists()
    assert (asset_dir / "square_hole_panel.png").exists()


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
    assert "## What Partially Works" in text
    assert "## What Does Not Yet Match" in text
    assert "artifacts/raw_results/plaplace_u3_thesis_full/summary.json" in text
    assert "Square Principal Branch By Tolerance" in text
    assert "iteration-order passes" in text


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
