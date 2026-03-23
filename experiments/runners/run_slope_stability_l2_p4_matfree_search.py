#!/usr/bin/env python3
"""Search matrix-free P4 fine-level PETSc options on L2 same-mesh P4->P2->P1."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path


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
    REPO_ROOT / "artifacts" / "raw_results" / "slope_stability_l2_p4_matfree_search_lambda1"
)
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"


COMMON_ARGS = [
    "--level",
    "2",
    "--elem_degree",
    "4",
    "--lambda-target",
    "1.0",
    "--profile",
    "performance",
    "--ksp_type",
    "fgmres",
    "--ksp_rtol",
    "1e-2",
    "--ksp_max_it",
    "100",
    "--save-linear-timing",
    "--quiet",
]

MATFREE_COMMON_ARGS = [
    "--operator_mode",
    "matfree_overlap",
    "--mg_strategy",
    "same_mesh_p4_p2_p1",
    "--no-use_trust_region",
]


def _run_case(name: str, args: list[str]) -> dict[str, object]:
    case_dir = OUTPUT_ROOT / name
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    stdout_path = case_dir / "stdout.txt"
    stderr_path = case_dir / "stderr.txt"

    command = ["mpiexec", "-n", "1", str(PYTHON), "-u", str(SOLVER), *args, "--out", str(result_path)]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    payload = None
    if completed.returncode == 0 and result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))

    return {
        "name": name,
        "args": list(args),
        "returncode": int(completed.returncode),
        "result_path": str(result_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "payload": payload,
    }


def _summarize_run(run: dict[str, object], *, stage: str, description: str) -> dict[str, object]:
    payload = run.get("payload")
    row = {
        "stage": str(stage),
        "name": str(run["name"]),
        "description": str(description),
        "returncode": int(run["returncode"]),
        "result_json": str(run["result_path"]),
        "stdout_path": str(run["stdout_path"]),
        "stderr_path": str(run["stderr_path"]),
    }
    if payload is None:
        row.update(
            {
                "solver_success": False,
                "status": "error",
                "message": "solver process failed",
                "fine_p4_assembly_zero": False,
            }
        )
        return row

    step = payload["result"]["steps"][0]
    linear_summary = dict(step.get("linear_summary", {}))
    linear_records = list(step.get("linear_timing", []))
    first_record = linear_records[0] if linear_records else {}
    fine_zero = bool(all(float(record.get("assemble_total_time", 0.0)) == 0.0 for record in linear_records))
    any_true_residual_accept = bool(
        any(bool(record.get("ksp_accepted_via_true_residual", False)) for record in linear_records)
    )
    row.update(
        {
            "solver_success": bool(payload["result"]["solver_success"]),
            "status": str(payload["result"]["status"]),
            "message": str(step.get("message", "")),
            "energy": float(step.get("energy", math.nan)),
            "omega": float(step.get("omega", math.nan)),
            "u_max": float(step.get("u_max", math.nan)),
            "newton_iterations": int(step.get("nit", 0)),
            "linear_iterations": int(step.get("linear_iters", 0)),
            "all_ksp_converged": bool(linear_summary.get("all_converged", False)),
            "n_linear_solves": int(linear_summary.get("n_solves", 0)),
            "n_linear_failed": int(linear_summary.get("n_failed", 0)),
            "worst_true_relative_residual": float(
                linear_summary.get("worst_true_relative_residual", math.nan)
            ),
            "last_true_relative_residual": float(
                linear_summary.get("last_true_relative_residual", math.nan)
            ),
            "first_ksp_reason_name": str(first_record.get("ksp_reason_name", "")),
            "operator_mode": str(payload["metadata"]["linear_solver"]["operator_mode"]),
            "pc_type": str(payload["metadata"]["linear_solver"]["pc_type"]),
            "mg_variant": str(payload["metadata"]["linear_solver"]["mg_variant"]),
            "mg_lower_operator_policy": str(
                payload["metadata"]["linear_solver"]["mg_lower_operator_policy"]
            ),
            "mg_fine_down_ksp_type": str(
                payload["metadata"]["linear_solver"]["mg_fine_down"]["ksp_type"]
            ),
            "mg_fine_down_pc_type": str(
                payload["metadata"]["linear_solver"]["mg_fine_down"]["pc_type"]
            ),
            "mg_fine_down_steps": int(
                payload["metadata"]["linear_solver"]["mg_fine_down"]["steps"]
            ),
            "mg_fine_python_pc_variant": str(
                payload["metadata"]["linear_solver"].get("mg_fine_python_pc_variant", "none")
            ),
            "python_pc_variant": str(
                payload["metadata"]["linear_solver"].get("python_pc_variant", "none")
            ),
            "solve_time_sec": float(payload["timings"]["solve_time"]),
            "setup_time_sec": float(payload["timings"]["setup_time"]),
            "total_time_sec": float(payload["timings"]["total_time"]),
            "operator_prepare_time_sec": float(
                sum(float(record.get("operator_prepare_total_time", 0.0)) for record in linear_records)
            ),
            "operator_apply_time_sec": float(
                sum(float(record.get("operator_apply_total_time", 0.0)) for record in linear_records)
            ),
            "operator_diagonal_prepare_time_sec": float(
                sum(float(record.get("operator_diagonal_prepare_total", 0.0)) for record in linear_records)
            ),
            "lower_level_assembly_time_sec": float(
                sum(float(record.get("pc_operator_assemble_total_time", 0.0)) for record in linear_records)
            ),
            "python_pc_prepare_time_sec": float(
                sum(float(record.get("python_pc_prepare_total_time", 0.0)) for record in linear_records)
            ),
            "python_pc_apply_time_sec": float(
                sum(float(record.get("python_pc_apply_total_time", 0.0)) for record in linear_records)
            ),
            "ksp_solve_time_sec": float(
                sum(float(record.get("solve_time", 0.0)) for record in linear_records)
            ),
            "fine_p4_assembly_zero": bool(fine_zero),
            "operator_diagonal_source": str(first_record.get("operator_diagonal_source", "")),
            "any_true_residual_accept": bool(any_true_residual_accept),
        }
    )
    return row


def _built_in_candidates() -> list[tuple[str, list[str], str]]:
    specs = [
        ("richardson", "none", 2, "fixed_setup", "jacobi"),
        ("richardson", "jacobi", 4, "fixed_setup", "jacobi"),
        ("richardson", "jacobi", 4, "fixed_setup", "sor"),
        ("chebyshev", "jacobi", 4, "fixed_setup", "jacobi"),
        ("chebyshev", "jacobi", 8, "fixed_setup", "sor"),
        ("gmres", "jacobi", 4, "fixed_setup", "sor"),
        ("gcr", "jacobi", 4, "fixed_setup", "sor"),
        ("fgmres", "jacobi", 4, "fixed_setup", "sor"),
        ("chebyshev", "jacobi", 4, "refresh_each_newton", "sor"),
        ("fgmres", "jacobi", 4, "refresh_each_newton", "sor"),
    ]
    candidates: list[tuple[str, list[str], str]] = []
    for fine_ksp, fine_pc, steps, lower_policy, degree2_pc in specs:
        name = (
            f"screen_{fine_ksp}_{fine_pc}_s{steps}_"
            f"{lower_policy}_p2{degree2_pc}"
        )
        args = [
            *COMMON_ARGS,
            "--pc_type",
            "mg",
            *MATFREE_COMMON_ARGS,
            "--mg_variant",
            "explicit_pmg",
            "--mg_lower_operator_policy",
            lower_policy,
            "--mg_fine_ksp_type",
            fine_ksp,
            "--mg_fine_pc_type",
            fine_pc,
            "--mg_fine_steps",
            str(steps),
            "--mg_degree2_pc_type",
            degree2_pc,
            "--mg_degree1_pc_type",
            "jacobi",
            "--maxit",
            "1",
        ]
        desc = (
            f"Built-in explicit MG screen: fine {fine_ksp}+{fine_pc}, "
            f"steps={steps}, lower={lower_policy}, P2={degree2_pc}"
        )
        candidates.append((name, args, desc))
    return candidates


def _full_stage1_args(screen_row: dict[str, object]) -> list[str]:
    args = list(screen_row["args"])
    if "--maxit" in args:
        idx = args.index("--maxit")
        del args[idx : idx + 2]
    return args


def _passes_stage0_gate(row: dict[str, object], baseline_energy_abs: float) -> bool:
    return (
        bool(row.get("fine_p4_assembly_zero"))
        and bool(row.get("all_ksp_converged"))
        and float(row.get("worst_true_relative_residual", 1.0)) < 2.0e-1
        and abs(float(row.get("energy", 0.0))) >= 1.0e-1 * float(baseline_energy_abs)
    )


def _passes_final_threshold(
    row: dict[str, object],
    *,
    baseline_linear_iters: int,
) -> bool:
    return (
        bool(row.get("solver_success"))
        and bool(row.get("all_ksp_converged"))
        and not bool(row.get("any_true_residual_accept"))
        and bool(row.get("fine_p4_assembly_zero"))
        and int(row.get("linear_iterations", 10**9)) <= int(math.ceil(1.25 * baseline_linear_iters))
    )


def _sort_screen_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            0 if row.get("all_ksp_converged") else 1,
            float(row.get("worst_true_relative_residual", math.inf)),
            float(row.get("solve_time_sec", math.inf)),
        ),
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    stage0_cases = [
        (
            "stage0_assembled_legacy_control",
            [
                *COMMON_ARGS,
                "--pc_type",
                "mg",
                "--operator_mode",
                "assembled",
                "--mg_strategy",
                "same_mesh_p4_p2_p1",
                "--mg_variant",
                "legacy_pmg",
                "--maxit",
                "1",
            ],
            "Assembled legacy PCMG control on the first Newton step",
        ),
        (
            "stage0_assembled_explicit_control",
            [
                *COMMON_ARGS,
                "--pc_type",
                "mg",
                "--operator_mode",
                "assembled",
                "--mg_strategy",
                "same_mesh_p4_p2_p1",
                "--mg_variant",
                "explicit_pmg",
                "--mg_lower_operator_policy",
                "refresh_each_newton",
                "--maxit",
                "1",
            ],
            "Assembled explicit same-mesh hierarchy control on the first Newton step",
        ),
        (
            "stage0_matfree_explicit_control",
            [
                *COMMON_ARGS,
                "--pc_type",
                "mg",
                *MATFREE_COMMON_ARGS,
                "--mg_variant",
                "explicit_pmg",
                "--mg_lower_operator_policy",
                "refresh_each_newton",
                "--maxit",
                "1",
            ],
            "Matrix-free explicit same-mesh hierarchy control on the first Newton step",
        ),
    ]
    for name, args, desc in stage0_cases:
        run = _run_case(name, args)
        row = _summarize_run(run, stage="stage0", description=desc)
        row["args"] = list(args)
        summary_rows.append(row)

    legacy_stage0 = next(row for row in summary_rows if row["name"] == "stage0_assembled_legacy_control")
    baseline_energy_abs = abs(float(legacy_stage0.get("energy", 0.0)))

    baseline_cases = [
        (
            "baseline_assembled_legacy_full",
            [
                *COMMON_ARGS,
                "--pc_type",
                "mg",
                "--operator_mode",
                "assembled",
                "--mg_strategy",
                "same_mesh_p4_p2_p1",
                "--mg_variant",
                "legacy_pmg",
                "--no-use_trust_region",
            ],
            "Primary assembled legacy baseline on the full L2 solve",
        ),
        (
            "baseline_assembled_explicit_full",
            [
                *COMMON_ARGS,
                "--pc_type",
                "mg",
                "--operator_mode",
                "assembled",
                "--mg_strategy",
                "same_mesh_p4_p2_p1",
                "--mg_variant",
                "explicit_pmg",
                "--mg_lower_operator_policy",
                "refresh_each_newton",
                "--no-use_trust_region",
            ],
            "Assembled explicit hierarchy isolation control on the full L2 solve",
        ),
    ]
    for name, args, desc in baseline_cases:
        run = _run_case(name, args)
        row = _summarize_run(run, stage="baseline", description=desc)
        row["args"] = list(args)
        summary_rows.append(row)

    baseline_full = next(row for row in summary_rows if row["name"] == "baseline_assembled_legacy_full")
    baseline_linear_iters = int(baseline_full["linear_iterations"])

    stage1_screen_rows: list[dict[str, object]] = []
    for name, args, desc in _built_in_candidates():
        run = _run_case(name, args)
        row = _summarize_run(run, stage="stage1_screen", description=desc)
        row["args"] = list(args)
        row["passes_stage0_gate"] = _passes_stage0_gate(row, baseline_energy_abs)
        summary_rows.append(row)
        stage1_screen_rows.append(row)

    promoted_stage1 = [
        row for row in _sort_screen_rows(stage1_screen_rows) if bool(row.get("passes_stage0_gate"))
    ][:8]

    winner: dict[str, object] | None = None
    stage1_full_rows: list[dict[str, object]] = []
    for row in promoted_stage1:
        name = str(row["name"]).replace("screen_", "stage1_full_")
        run = _run_case(name, _full_stage1_args(row))
        full_row = _summarize_run(
            run,
            stage="stage1_full",
            description=f"Promoted built-in candidate from {row['name']}",
        )
        full_row["args"] = _full_stage1_args(row)
        full_row["source_screen"] = str(row["name"])
        full_row["passes_final_threshold"] = _passes_final_threshold(
            full_row,
            baseline_linear_iters=baseline_linear_iters,
        )
        summary_rows.append(full_row)
        stage1_full_rows.append(full_row)
        if bool(full_row["passes_final_threshold"]) and winner is None:
            winner = full_row
            break

    if winner is None:
        stage2_seed = stage1_full_rows if stage1_full_rows else promoted_stage1[:4]
        promoted_stage2 = _sort_screen_rows(stage2_seed)[:4]
        for seed in promoted_stage2:
            for inner_ksp in ("fgmres", "gcr"):
                name = f"stage2_outer_pcksp_{seed['name']}_{inner_ksp}"
                args = _full_stage1_args(seed)
                args = [
                    arg
                    for arg in args
                    if arg not in {"--mg_variant", "explicit_pmg"}
                ]
                if "--mg_lower_operator_policy" in args:
                    pass
                args.extend(
                    [
                        "--mg_variant",
                        "outer_pcksp",
                        "--outer_pcksp_inner_ksp_type",
                        inner_ksp,
                    ]
                )
                run = _run_case(name, args)
                row = _summarize_run(
                    run,
                    stage="stage2",
                    description=f"Outer PCKSP candidate derived from {seed['name']} with inner {inner_ksp}",
                )
                row["args"] = list(args)
                row["source_candidate"] = str(seed["name"])
                row["passes_final_threshold"] = _passes_final_threshold(
                    row,
                    baseline_linear_iters=baseline_linear_iters,
                )
                summary_rows.append(row)
                if bool(row["passes_final_threshold"]) and winner is None:
                    winner = row
                    break
            if winner is not None:
                break

    if winner is None:
        stage3_cases = [
            (
                "stage3_explicit_python_pc_fixed",
                [
                    *COMMON_ARGS,
                    "--pc_type",
                    "mg",
                    *MATFREE_COMMON_ARGS,
                    "--mg_variant",
                    "explicit_pmg",
                    "--mg_lower_operator_policy",
                    "fixed_setup",
                    "--mg_fine_ksp_type",
                    "richardson",
                    "--mg_fine_pc_type",
                    "python",
                    "--mg_fine_python_pc_variant",
                    "overlap_lu",
                    "--mg_fine_steps",
                    "1",
                    "--mg_degree2_pc_type",
                    "jacobi",
                    "--mg_degree1_pc_type",
                    "jacobi",
                ],
                "Custom Python fine-level overlap LU inside explicit PCMG",
            ),
            (
                "stage3_explicit_python_pc_refresh",
                [
                    *COMMON_ARGS,
                    "--pc_type",
                    "mg",
                    *MATFREE_COMMON_ARGS,
                    "--mg_variant",
                    "explicit_pmg",
                    "--mg_lower_operator_policy",
                    "refresh_each_newton",
                    "--mg_fine_ksp_type",
                    "richardson",
                    "--mg_fine_pc_type",
                    "python",
                    "--mg_fine_python_pc_variant",
                    "overlap_lu",
                    "--mg_fine_steps",
                    "1",
                    "--mg_degree2_pc_type",
                    "jacobi",
                    "--mg_degree1_pc_type",
                    "jacobi",
                ],
                "Custom Python fine-level overlap LU inside explicit PCMG with refreshed lower levels",
            ),
            (
                "stage3_python_pc_standalone_fgmres",
                [
                    *COMMON_ARGS,
                    "--pc_type",
                    "python",
                    "--python_pc_variant",
                    "overlap_lu",
                    "--operator_mode",
                    "matfree_overlap",
                    "--no-use_trust_region",
                ],
                "Standalone Python overlap-LU preconditioner under outer FGMRES",
            ),
            (
                "stage3_python_pc_standalone_gcr",
                [
                    *COMMON_ARGS,
                    "--pc_type",
                    "python",
                    "--python_pc_variant",
                    "overlap_lu",
                    "--operator_mode",
                    "matfree_overlap",
                    "--ksp_type",
                    "gcr",
                    "--no-use_trust_region",
                ],
                "Standalone Python overlap-LU preconditioner under outer GCR",
            ),
        ]
        for name, args, desc in stage3_cases:
            run = _run_case(name, args)
            row = _summarize_run(run, stage="stage3", description=desc)
            row["args"] = list(args)
            row["passes_final_threshold"] = _passes_final_threshold(
                row,
                baseline_linear_iters=baseline_linear_iters,
            )
            summary_rows.append(row)
            if bool(row["passes_final_threshold"]) and winner is None:
                winner = row
                break

    if winner is not None:
        sanity_cases = [
            (
                "sanity_winner_default_newton",
                [arg for arg in winner["args"] if arg != "--no-use_trust_region"],
                "Winner rerun with default nonlinear settings",
            ),
            (
                "sanity_assembled_legacy_default_newton",
                [
                    arg
                    for arg in next(
                        row["args"]
                        for row in summary_rows
                        if row["name"] == "baseline_assembled_legacy_full"
                    )
                    if arg != "--no-use_trust_region"
                ],
                "Assembled legacy rerun with default nonlinear settings",
            ),
        ]
        for name, args, desc in sanity_cases:
            run = _run_case(name, args)
            row = _summarize_run(run, stage="sanity", description=desc)
            row["args"] = list(args)
            summary_rows.append(row)

    SUMMARY_PATH.write_text(json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
