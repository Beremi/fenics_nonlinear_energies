#!/usr/bin/env python3
"""Curated staged L6 P4 PMG progression search on 8 MPI ranks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.problems.slope_stability.support import ensure_same_mesh_case_hdf5


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
SOLVER = (
    REPO_ROOT
    / "src"
    / "problems"
    / "slope_stability"
    / "jax_petsc"
    / "solve_slope_stability_dof.py"
)
OUTPUT_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "slope_stability_l6_p4_progression_search_lambda1_np8_staged"
)
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
RANKS = 8

COMMON_ARGS = [
    "--level",
    "6",
    "--elem_degree",
    "4",
    "--lambda-target",
    "1.0",
    "--profile",
    "performance",
    "--pc_type",
    "mg",
    "--mg_variant",
    "legacy_pmg",
    "--ksp_type",
    "fgmres",
    "--ksp_rtol",
    "1e-2",
    "--ksp_max_it",
    "100",
    "--benchmark_mode",
    "warmup_once_then_solve",
    "--save-history",
    "--save-linear-timing",
    "--quiet",
    "--no-use_trust_region",
    "--distribution_strategy",
    "overlap_p2p",
    "--problem_build_mode",
    "root_bcast",
    "--mg_level_build_mode",
    "root_bcast",
    "--mg_transfer_build_mode",
    "owned_rows",
    "--mg_p4_smoother_ksp_type",
    "richardson",
    "--mg_p4_smoother_pc_type",
    "sor",
    "--mg_p4_smoother_steps",
    "3",
    "--mg_p2_smoother_ksp_type",
    "richardson",
    "--mg_p2_smoother_pc_type",
    "sor",
    "--mg_p2_smoother_steps",
    "3",
    "--mg_p1_smoother_ksp_type",
    "richardson",
    "--mg_p1_smoother_pc_type",
    "sor",
    "--mg_p1_smoother_steps",
    "3",
    "--mg_coarse_backend",
    "hypre",
    "--mg_coarse_ksp_type",
    "cg",
    "--mg_coarse_pc_type",
    "hypre",
    "--mg_coarse_hypre_nodal_coarsen",
    "6",
    "--mg_coarse_hypre_vec_interp_variant",
    "3",
    "--mg_coarse_hypre_strong_threshold",
    "0.5",
    "--mg_coarse_hypre_coarsen_type",
    "HMIS",
    "--mg_coarse_hypre_max_iter",
    "4",
    "--mg_coarse_hypre_tol",
    "0.0",
    "--mg_coarse_hypre_relax_type_all",
    "symmetric-SOR/Jacobi",
]


def _spec_string(specs: list[tuple[int, int]]) -> str:
    return ",".join(f"{int(level)}:{int(degree)}" for level, degree in specs)


def _p1_tail_specs(*, start_level: int, include_p2: bool) -> list[tuple[int, int]]:
    specs = [(level, 1) for level in range(int(start_level), 7)]
    if include_p2:
        specs.append((6, 2))
    specs.append((6, 4))
    return specs


def _uniform_chain_specs(*, start_level: int, degree: int) -> list[tuple[int, int]]:
    specs = [(level, int(degree)) for level in range(int(start_level), 7)]
    if int(degree) != 4:
        specs.append((6, 4))
    return specs


STAGE_A_CASES = [
    {
        "name": "screen_same_mesh_only",
        "label": "Same-mesh P4->P2->P1",
        "family": "same_mesh",
        "stage": "screen",
        "maxit": 1,
        "args": ["--mg_strategy", "same_mesh_p4_p2_p1"],
    },
    {
        "name": "screen_short_p1_tail",
        "label": "Short P1 tail (L5 P1 + same-mesh P4->P2->P1)",
        "family": "p1_tail",
        "stage": "screen",
        "maxit": 1,
        "args": ["--mg_strategy", "same_mesh_p4_p2_p1_lminus1_p1"],
    },
    {
        "name": "screen_full_p1_tail",
        "label": "Full P1 tail to L1",
        "family": "p1_tail",
        "stage": "screen",
        "maxit": 1,
        "args": [
            "--mg_strategy",
            "custom_mixed",
            "--mg_custom_hierarchy",
            _spec_string(_p1_tail_specs(start_level=1, include_p2=True)),
        ],
    },
    {
        "name": "screen_full_p1_tail_no_p2",
        "label": "Full P1 tail to L1, no P2",
        "family": "p1_tail_no_p2",
        "stage": "screen",
        "maxit": 1,
        "args": [
            "--mg_strategy",
            "custom_mixed",
            "--mg_custom_hierarchy",
            _spec_string(_p1_tail_specs(start_level=1, include_p2=False)),
        ],
    },
    {
        "name": "screen_full_p2_chain",
        "label": "Full P2 chain to L1",
        "family": "p2_chain",
        "stage": "screen",
        "maxit": 1,
        "args": [
            "--mg_strategy",
            "custom_mixed",
            "--mg_custom_hierarchy",
            _spec_string(_uniform_chain_specs(start_level=1, degree=2)),
        ],
    },
    {
        "name": "screen_full_p4_chain",
        "label": "Full P4 chain to L1",
        "family": "p4_chain",
        "stage": "screen",
        "maxit": 1,
        "args": [
            "--mg_strategy",
            "custom_mixed",
            "--mg_custom_hierarchy",
            _spec_string(_uniform_chain_specs(start_level=1, degree=4)),
        ],
    },
]


FOLLOWUP_CASES_BY_FAMILY = {
    "p1_tail": [
        {
            "name": "screen_p1_tail_to_l4",
            "label": "P1 tail to L4",
            "family": "p1_tail",
            "stage": "screen_followup",
            "maxit": 1,
            "args": [
                "--mg_strategy",
                "custom_mixed",
                "--mg_custom_hierarchy",
                _spec_string(_p1_tail_specs(start_level=4, include_p2=True)),
            ],
        },
        {
            "name": "screen_p1_tail_to_l3",
            "label": "P1 tail to L3",
            "family": "p1_tail",
            "stage": "screen_followup",
            "maxit": 1,
            "args": [
                "--mg_strategy",
                "custom_mixed",
                "--mg_custom_hierarchy",
                _spec_string(_p1_tail_specs(start_level=3, include_p2=True)),
            ],
        },
        {
            "name": "screen_p1_tail_to_l2",
            "label": "P1 tail to L2",
            "family": "p1_tail",
            "stage": "screen_followup",
            "maxit": 1,
            "args": [
                "--mg_strategy",
                "custom_mixed",
                "--mg_custom_hierarchy",
                _spec_string(_p1_tail_specs(start_level=2, include_p2=True)),
            ],
        },
    ],
    "p1_tail_no_p2": [
        {
            "name": "screen_p1_tail_no_p2_to_l4",
            "label": "P1 tail to L4, no P2",
            "family": "p1_tail_no_p2",
            "stage": "screen_followup",
            "maxit": 1,
            "args": [
                "--mg_strategy",
                "custom_mixed",
                "--mg_custom_hierarchy",
                _spec_string(_p1_tail_specs(start_level=4, include_p2=False)),
            ],
        },
        {
            "name": "screen_p1_tail_no_p2_to_l3",
            "label": "P1 tail to L3, no P2",
            "family": "p1_tail_no_p2",
            "stage": "screen_followup",
            "maxit": 1,
            "args": [
                "--mg_strategy",
                "custom_mixed",
                "--mg_custom_hierarchy",
                _spec_string(_p1_tail_specs(start_level=3, include_p2=False)),
            ],
        },
        {
            "name": "screen_p1_tail_no_p2_to_l2",
            "label": "P1 tail to L2, no P2",
            "family": "p1_tail_no_p2",
            "stage": "screen_followup",
            "maxit": 1,
            "args": [
                "--mg_strategy",
                "custom_mixed",
                "--mg_custom_hierarchy",
                _spec_string(_p1_tail_specs(start_level=2, include_p2=False)),
            ],
        },
    ],
    "p2_chain": [
        {
            "name": "screen_p2_chain_to_l5",
            "label": "P2 chain to L5",
            "family": "p2_chain",
            "stage": "screen_followup",
            "maxit": 1,
            "args": [
                "--mg_strategy",
                "custom_mixed",
                "--mg_custom_hierarchy",
                _spec_string(_uniform_chain_specs(start_level=5, degree=2)),
            ],
        },
        {
            "name": "screen_p2_chain_to_l4",
            "label": "P2 chain to L4",
            "family": "p2_chain",
            "stage": "screen_followup",
            "maxit": 1,
            "args": [
                "--mg_strategy",
                "custom_mixed",
                "--mg_custom_hierarchy",
                _spec_string(_uniform_chain_specs(start_level=4, degree=2)),
            ],
        },
        {
            "name": "screen_p2_chain_to_l3",
            "label": "P2 chain to L3",
            "family": "p2_chain",
            "stage": "screen_followup",
            "maxit": 1,
            "args": [
                "--mg_strategy",
                "custom_mixed",
                "--mg_custom_hierarchy",
                _spec_string(_uniform_chain_specs(start_level=3, degree=2)),
            ],
        },
    ],
    "p4_chain": [
        {
            "name": "screen_p4_chain_to_l5",
            "label": "P4 chain to L5",
            "family": "p4_chain",
            "stage": "screen_followup",
            "maxit": 1,
            "args": [
                "--mg_strategy",
                "custom_mixed",
                "--mg_custom_hierarchy",
                _spec_string(_uniform_chain_specs(start_level=5, degree=4)),
            ],
        },
        {
            "name": "screen_p4_chain_to_l4",
            "label": "P4 chain to L4",
            "family": "p4_chain",
            "stage": "screen_followup",
            "maxit": 1,
            "args": [
                "--mg_strategy",
                "custom_mixed",
                "--mg_custom_hierarchy",
                _spec_string(_uniform_chain_specs(start_level=4, degree=4)),
            ],
        },
        {
            "name": "screen_p4_chain_to_l3",
            "label": "P4 chain to L3",
            "family": "p4_chain",
            "stage": "screen_followup",
            "maxit": 1,
            "args": [
                "--mg_strategy",
                "custom_mixed",
                "--mg_custom_hierarchy",
                _spec_string(_uniform_chain_specs(start_level=3, degree=4)),
            ],
        },
    ],
}


def _case_output_dir(name: str) -> Path:
    return OUTPUT_ROOT / name


def _command(case: dict[str, object], out: Path) -> list[str]:
    return [
        "mpiexec",
        "-n",
        str(RANKS),
        str(PYTHON),
        "-u",
        str(SOLVER),
        *COMMON_ARGS,
        "--maxit",
        str(int(case.get("maxit", 20))),
        *[str(arg) for arg in case["args"]],
        "--out",
        str(out),
    ]


def _extract_result(case: dict[str, object], payload: dict[str, object]) -> dict[str, object]:
    metadata = payload.get("metadata", {})
    linear_solver = metadata.get("linear_solver", {}) if isinstance(metadata, dict) else {}
    timings = payload.get("timings", {}) if isinstance(payload, dict) else {}
    result = payload.get("result", {}) if isinstance(payload, dict) else {}
    steps = result.get("steps", []) if isinstance(result, dict) else []
    last_step = steps[-1] if steps else {}
    linear_summary = last_step.get("linear_summary", {}) if isinstance(last_step, dict) else {}
    return {
        "name": str(case["name"]),
        "label": str(case["label"]),
        "family": str(case["family"]),
        "stage": str(case.get("stage", "final")),
        "maxit": int(case.get("maxit", 20)),
        "mg_strategy": str(linear_solver.get("mg_strategy", "")),
        "mg_custom_hierarchy": linear_solver.get("mg_custom_hierarchy"),
        "solver_success": bool(result.get("solver_success", False)),
        "status": str(result.get("status", "")),
        "message": str(last_step.get("message", "")),
        "newton_iterations": int(len(steps)),
        "linear_iterations": int(sum(int(step.get("linear_iters", 0)) for step in steps)),
        "end_to_end_total_time_sec": float(result.get("total_time", 0.0)),
        "steady_state_total_time_sec": float(result.get("steady_state_total_time", 0.0)),
        "solve_time_sec": float(result.get("solve_time_total", 0.0)),
        "energy": float(last_step.get("energy", 0.0)),
        "omega": float(last_step.get("omega", 0.0)),
        "u_max": float(last_step.get("u_max", 0.0)),
        "worst_true_relative_residual": float(
            linear_summary.get("worst_true_relative_residual", 0.0)
        ),
        "last_reason_name": str(linear_summary.get("last_reason_name", "")),
        "problem_build_time_sec": float(timings.get("problem_build_time", 0.0)),
        "assembler_setup_time_sec": float(timings.get("assembler_setup_time", 0.0)),
        "mg_hierarchy_build_time_sec": float(
            (timings.get("solver_bootstrap_breakdown", {}) or {}).get("mg_hierarchy_build_time", 0.0)
        ),
    }


def _run_case(case: dict[str, object]) -> dict[str, object]:
    out_dir = _case_output_dir(str(case["name"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    payload_path = out_dir / "result.json"
    stdout_path = out_dir / "stdout.txt"
    stderr_path = out_dir / "stderr.txt"
    if not payload_path.exists():
        command = _command(case, payload_path)
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0 and not payload_path.exists():
            raise RuntimeError(
                f"Case {case['name']} failed with exit code {proc.returncode}; "
                f"see {stderr_path}"
            )
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    return _extract_result(case, payload)


def _best_success(rows: list[dict[str, object]]) -> dict[str, object] | None:
    successes = [row for row in rows if bool(row["solver_success"])]
    if not successes:
        return None
    return min(successes, key=lambda row: float(row["end_to_end_total_time_sec"]))


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for level in range(1, 7):
        for degree in (1, 2, 4):
            ensure_same_mesh_case_hdf5(level, degree)

    rows: list[dict[str, object]] = []
    seen: set[str] = set()

    def run_cases(cases: list[dict[str, object]]) -> None:
        for case in cases:
            name = str(case["name"])
            if name in seen:
                continue
            rows.append(_run_case(case))
            seen.add(name)
            SUMMARY_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    run_cases(STAGE_A_CASES)
    best_stage_a = _best_success([row for row in rows if str(row["stage"]).startswith("screen")])
    if best_stage_a is not None:
        best_family = str(best_stage_a["family"])
        baseline = next((row for row in rows if row["name"] == "screen_short_p1_tail"), None)
        if baseline is not None:
            baseline_time = float(baseline["steady_state_total_time_sec"])
            best_time = float(best_stage_a["end_to_end_total_time_sec"])
            if (
                best_family in FOLLOWUP_CASES_BY_FAMILY
                and float(best_stage_a["steady_state_total_time_sec"]) <= 1.10 * baseline_time
            ):
                run_cases(FOLLOWUP_CASES_BY_FAMILY[best_family])

    screen_rows = [row for row in rows if str(row["stage"]).startswith("screen") and bool(row["solver_success"])]
    screen_rows_sorted = sorted(screen_rows, key=lambda row: float(row["steady_state_total_time_sec"]))
    finalists: list[dict[str, object]] = []
    baseline_screen = next((row for row in screen_rows_sorted if row["name"] == "screen_short_p1_tail"), None)
    if baseline_screen is not None:
        finalists.append(
            {
                "name": "final_short_p1_tail",
                "label": "Final short P1 tail",
                "family": str(baseline_screen["family"]),
                "stage": "final",
                "maxit": 20,
                "args": ["--mg_strategy", "same_mesh_p4_p2_p1_lminus1_p1"],
            }
        )
    for row in screen_rows_sorted:
        if row["name"] == "screen_short_p1_tail":
            continue
        if len(finalists) >= 3:
            break
        args = ["--mg_strategy", str(row["mg_strategy"])]
        if row.get("mg_custom_hierarchy"):
            args = [
                "--mg_strategy",
                "custom_mixed",
                "--mg_custom_hierarchy",
                str(row["mg_custom_hierarchy"]),
            ]
        finalists.append(
            {
                "name": f"final_{row['name'].removeprefix('screen_')}",
                "label": f"Final {row['label']}",
                "family": str(row["family"]),
                "stage": "final",
                "maxit": 20,
                "args": args,
            }
        )

    run_cases(finalists)

    SUMMARY_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
