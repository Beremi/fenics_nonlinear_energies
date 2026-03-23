#!/usr/bin/env python3
"""Run the L5 confirmation matrix for the accepted P4 PMG variant."""

from __future__ import annotations

import argparse
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
LEVEL = 5
DEGREE = 4
RANKS = [1, 2, 4, 8]

PMG_VARIANTS = {
    "original_baseline": {
        "label": "Original baseline",
        "args": [
            "--distribution_strategy",
            "overlap_allgather",
            "--problem_build_mode",
            "replicated",
            "--mg_level_build_mode",
            "replicated",
            "--mg_transfer_build_mode",
            "replicated",
            "--mg_coarse_backend",
            "jacobi",
            "--mg_coarse_ksp_type",
            "cg",
            "--mg_coarse_pc_type",
            "jacobi",
            "--line_search",
            "golden_fixed",
        ],
    },
    "coarse_hypre_only": {
        "label": "Coarse Hypre only",
        "args": [
            "--distribution_strategy",
            "overlap_allgather",
            "--problem_build_mode",
            "replicated",
            "--mg_level_build_mode",
            "replicated",
            "--mg_transfer_build_mode",
            "replicated",
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
            "--line_search",
            "golden_fixed",
        ],
    },
    "distributed_setup_hypre": {
        "label": "Distributed setup + coarse Hypre",
        "args": [
            "--distribution_strategy",
            "overlap_p2p",
            "--problem_build_mode",
            "root_bcast",
            "--mg_level_build_mode",
            "root_bcast",
            "--mg_transfer_build_mode",
            "owned_rows",
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
            "--line_search",
            "golden_fixed",
        ],
    },
    "distributed_setup_hypre_armijo": {
        "label": "Distributed + coarse Hypre + Armijo",
        "args": [
            "--distribution_strategy",
            "overlap_p2p",
            "--problem_build_mode",
            "root_bcast",
            "--mg_level_build_mode",
            "root_bcast",
            "--mg_transfer_build_mode",
            "owned_rows",
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
            "--line_search",
            "armijo",
        ],
    },
    "distributed_cached_hypre": {
        "label": "Distributed + cached transfers",
        "args": [
            "--distribution_strategy",
            "overlap_p2p",
            "--problem_build_mode",
            "root_bcast",
            "--mg_level_build_mode",
            "root_bcast",
            "--mg_transfer_build_mode",
            "owned_rows",
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
            "--line_search",
            "golden_fixed",
        ],
    },
    "distributed_cached_hypre_armijo": {
        "label": "Distributed + cached + Armijo",
        "args": [
            "--distribution_strategy",
            "overlap_p2p",
            "--problem_build_mode",
            "root_bcast",
            "--mg_level_build_mode",
            "root_bcast",
            "--mg_transfer_build_mode",
            "owned_rows",
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
            "--line_search",
            "armijo",
        ],
    },
}

COMMON_ARGS = [
    "--level",
    str(LEVEL),
    "--elem_degree",
    str(DEGREE),
    "--lambda-target",
    "1.0",
    "--profile",
    "performance",
    "--pc_type",
    "mg",
    "--mg_strategy",
    "same_mesh_p4_p2_p1_lminus1_p1",
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
]


def _ensure_assets() -> None:
    for level in (4, 5):
        for degree in (1, 2, 4):
            ensure_same_mesh_case_hdf5(level, degree)


def _sum(items: list[dict[str, object]], key: str) -> float:
    return float(sum(float(item.get(key, 0.0)) for item in items))


def _callback_value(callback: dict[str, object], phase: str, key: str) -> float:
    return float(dict(callback.get(str(phase), {})).get(str(key), 0.0))


def _aggregate(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload["result"])
    step = dict(result["steps"][0])
    history = list(step.get("history", []))
    linear = list(step.get("linear_timing", []))
    case = dict(payload["case"])
    timings = dict(payload["timings"])
    callback = dict(timings.get("callback_summary", {}))
    linear_summary = dict(step.get("linear_summary", {}))
    return {
        "solver_success": bool(result["solver_success"]),
        "status": str(result["status"]),
        "line_search": str(case.get("line_search", "")),
        "newton_iterations": int(step["nit"]),
        "linear_iterations": int(step["linear_iters"]),
        "line_search_evals": int(sum(int(item.get("ls_evals", 0)) for item in history)),
        "worst_true_relative_residual": float(
            linear_summary.get("worst_true_relative_residual", float("nan"))
        ),
        "one_time_setup_time_sec": float(timings.get("one_time_setup_time", 0.0)),
        "steady_state_setup_time_sec": float(timings.get("steady_state_setup_time", 0.0)),
        "solve_time_sec": float(timings.get("solve_time", 0.0)),
        "end_to_end_total_time_sec": float(timings.get("total_time", 0.0)),
        "steady_state_total_time_sec": float(timings.get("steady_state_total_time", 0.0)),
        "benchmark_total_time_sec": float(timings.get("benchmark_total_time", 0.0)),
        "problem_build_time_sec": float(timings.get("problem_build_time", 0.0)),
        "assembler_warmup_time_sec": float(
            dict(timings.get("assembler_setup_breakdown", {})).get("warmup", 0.0)
        ),
        "mg_hierarchy_build_time_sec": float(
            dict(timings.get("solver_bootstrap_breakdown", {})).get(
                "mg_hierarchy_build_time", 0.0
            )
        ),
        "mg_configure_time_sec": float(
            dict(timings.get("solver_bootstrap_breakdown", {})).get("mg_configure_time", 0.0)
        ),
        "energy_total_time_sec": _callback_value(callback, "energy", "total"),
        "gradient_total_time_sec": _callback_value(callback, "gradient", "total"),
        "hessian_total_time_sec": _callback_value(callback, "hessian", "total"),
        "line_search_time_sec": _sum(history, "t_ls"),
        "linear_pc_setup_time_sec": _sum(linear, "pc_setup_time"),
        "linear_ksp_solve_time_sec": _sum(linear, "solve_time"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=sorted(PMG_VARIANTS.keys()),
        required=True,
        help="Accepted L4 PMG variant to confirm on L5",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "raw_results" / "slope_stability_l5_p4_pmg_confirmation_lambda1",
    )
    parser.add_argument(
        "--skip-hypre-baseline",
        action="store_true",
        help="Run only the accepted PMG variant for the L5 confirmation pass",
    )
    args = parser.parse_args()

    _ensure_assets()
    output_root = Path(args.output_root)
    summary_path = output_root / "summary.json"
    rows: list[dict[str, object]] = []
    if summary_path.exists():
        rows = list(json.loads(summary_path.read_text(encoding="utf-8")))
    existing = {(str(row["variant"]), int(row["ranks"])) for row in rows}

    variants = [
        {
            "name": str(args.variant),
            "label": str(PMG_VARIANTS[str(args.variant)]["label"]),
            "kind": "pmg",
            "args": list(PMG_VARIANTS[str(args.variant)]["args"]),
        }
    ]
    if not bool(args.skip_hypre_baseline):
        variants.append(
            {
                "name": "tuned_hypre_nonmg",
                "label": "Tuned Hypre baseline",
                "kind": "hypre",
                "args": [
                    "--pc_type",
                    "hypre",
                    "--ksp_type",
                    "cg",
                    "--ksp_max_it",
                    "300",
                    "--distribution_strategy",
                    "overlap_p2p",
                    "--problem_build_mode",
                    "root_bcast",
                    "--line_search",
                    "golden_fixed",
                    "--hypre_nodal_coarsen",
                    "6",
                    "--hypre_vec_interp_variant",
                    "3",
                ],
            }
        )

    for variant in variants:
        for ranks in RANKS:
            key = (str(variant["name"]), int(ranks))
            if key in existing:
                continue
            case_dir = output_root / str(variant["name"]) / f"np{ranks}"
            case_dir.mkdir(parents=True, exist_ok=True)
            result_path = case_dir / "result.json"
            stdout_path = case_dir / "stdout.txt"
            stderr_path = case_dir / "stderr.txt"
            proc = subprocess.run(
                [
                    "mpiexec",
                    "-n",
                    str(ranks),
                    str(PYTHON),
                    "-u",
                    str(SOLVER),
                    *COMMON_ARGS,
                    *list(variant["args"]),
                    "--out",
                    str(result_path),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            stdout_path.write_text(proc.stdout, encoding="utf-8")
            stderr_path.write_text(proc.stderr, encoding="utf-8")
            if proc.returncode != 0:
                row = {
                    "variant": str(variant["name"]),
                    "variant_label": str(variant["label"]),
                    "variant_kind": str(variant["kind"]),
                    "ranks": int(ranks),
                    "solver_success": False,
                    "status": "subprocess_failed",
                    "message": proc.stderr.strip().splitlines()[-1]
                    if proc.stderr.strip()
                    else "subprocess failed",
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "result_json": str(result_path),
                }
            else:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                row = {
                    "variant": str(variant["name"]),
                    "variant_label": str(variant["label"]),
                    "variant_kind": str(variant["kind"]),
                    "ranks": int(ranks),
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "result_json": str(result_path),
                }
                row.update(_aggregate(payload))
            rows.append(row)
            rows.sort(key=lambda item: (str(item["variant"]), int(item["ranks"])))
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
