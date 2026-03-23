#!/usr/bin/env python3
"""Staged PMG smoother sweep for the assembled P4 solver on L4/L5."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

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
    / "slope_stability_p4_pmg_smoother_sweep_lambda1"
)
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

LEVELS = [4, 5]

COMMON_ARGS = [
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
    "--line_search",
    "golden_fixed",
]

STAGE_A_VARIANTS = [
    {
        "name": "tail_baseline",
        "label": "Tail baseline",
        "stage": "family_screen",
        "args": ["--mg_strategy", "same_mesh_p4_p2_p1_lminus1_p1"],
    },
    {
        "name": "tail_fine_richardson_jacobi",
        "label": "Tail fine Richardson/Jacobi",
        "stage": "family_screen",
        "args": [
            "--mg_strategy",
            "same_mesh_p4_p2_p1_lminus1_p1",
            "--mg_p4_smoother_pc_type",
            "jacobi",
        ],
    },
    {
        "name": "tail_fine_chebyshev_jacobi",
        "label": "Tail fine Chebyshev/Jacobi",
        "stage": "family_screen",
        "args": [
            "--mg_strategy",
            "same_mesh_p4_p2_p1_lminus1_p1",
            "--mg_p4_smoother_ksp_type",
            "chebyshev",
            "--mg_p4_smoother_pc_type",
            "jacobi",
        ],
    },
    {
        "name": "tail_fine_fgmres_none",
        "label": "Tail fine FGMRES/none",
        "stage": "family_screen",
        "args": [
            "--mg_strategy",
            "same_mesh_p4_p2_p1_lminus1_p1",
            "--mg_p4_smoother_ksp_type",
            "fgmres",
            "--mg_p4_smoother_pc_type",
            "none",
            "--mg_p4_smoother_steps",
            "4",
        ],
    },
    {
        "name": "tail_p2_jacobi",
        "label": "Tail P2 Richardson/Jacobi",
        "stage": "family_screen",
        "args": [
            "--mg_strategy",
            "same_mesh_p4_p2_p1_lminus1_p1",
            "--mg_p2_smoother_pc_type",
            "jacobi",
        ],
    },
    {
        "name": "tail_p1_jacobi",
        "label": "Tail P1 Richardson/Jacobi",
        "stage": "family_screen",
        "args": [
            "--mg_strategy",
            "same_mesh_p4_p2_p1_lminus1_p1",
            "--mg_p1_smoother_pc_type",
            "jacobi",
            "--mg_p1_smoother_steps",
            "2",
        ],
    },
    {
        "name": "no_tail_baseline",
        "label": "No-tail baseline",
        "stage": "family_screen",
        "args": ["--mg_strategy", "same_mesh_p4_p2_p1"],
    },
    {
        "name": "tail_replicated_assets",
        "label": "Tail baseline with replicated asset loads",
        "stage": "family_screen",
        "args": [
            "--mg_strategy",
            "same_mesh_p4_p2_p1_lminus1_p1",
            "--problem_build_mode",
            "replicated",
            "--mg_level_build_mode",
            "replicated",
        ],
    },
]

STEP_VARIANTS = [
    {
        "name": "tail_baseline_p4steps_2",
        "label": "Tail baseline P4 steps=2",
        "stage": "step_refine",
        "args": [
            "--mg_strategy",
            "same_mesh_p4_p2_p1_lminus1_p1",
            "--mg_p4_smoother_steps",
            "2",
        ],
    },
    {
        "name": "tail_baseline_p4steps_4",
        "label": "Tail baseline P4 steps=4",
        "stage": "step_refine",
        "args": [
            "--mg_strategy",
            "same_mesh_p4_p2_p1_lminus1_p1",
            "--mg_p4_smoother_steps",
            "4",
        ],
    },
    {
        "name": "tail_cheb_p4steps_2",
        "label": "Tail Chebyshev/Jacobi P4 steps=2",
        "stage": "step_refine",
        "args": [
            "--mg_strategy",
            "same_mesh_p4_p2_p1_lminus1_p1",
            "--mg_p4_smoother_ksp_type",
            "chebyshev",
            "--mg_p4_smoother_pc_type",
            "jacobi",
            "--mg_p4_smoother_steps",
            "2",
        ],
    },
    {
        "name": "tail_cheb_p4steps_4",
        "label": "Tail Chebyshev/Jacobi P4 steps=4",
        "stage": "step_refine",
        "args": [
            "--mg_strategy",
            "same_mesh_p4_p2_p1_lminus1_p1",
            "--mg_p4_smoother_ksp_type",
            "chebyshev",
            "--mg_p4_smoother_pc_type",
            "jacobi",
            "--mg_p4_smoother_steps",
            "4",
        ],
    },
    {
        "name": "tail_baseline_p2steps_2",
        "label": "Tail baseline P2 steps=2",
        "stage": "step_refine",
        "args": [
            "--mg_strategy",
            "same_mesh_p4_p2_p1_lminus1_p1",
            "--mg_p2_smoother_steps",
            "2",
        ],
    },
    {
        "name": "tail_baseline_p2steps_4",
        "label": "Tail baseline P2 steps=4",
        "stage": "step_refine",
        "args": [
            "--mg_strategy",
            "same_mesh_p4_p2_p1_lminus1_p1",
            "--mg_p2_smoother_steps",
            "4",
        ],
    },
    {
        "name": "tail_p2_jacobi_steps_2",
        "label": "Tail P2 Jacobi steps=2",
        "stage": "step_refine",
        "args": [
            "--mg_strategy",
            "same_mesh_p4_p2_p1_lminus1_p1",
            "--mg_p2_smoother_pc_type",
            "jacobi",
            "--mg_p2_smoother_steps",
            "2",
        ],
    },
    {
        "name": "tail_p2_jacobi_steps_4",
        "label": "Tail P2 Jacobi steps=4",
        "stage": "step_refine",
        "args": [
            "--mg_strategy",
            "same_mesh_p4_p2_p1_lminus1_p1",
            "--mg_p2_smoother_pc_type",
            "jacobi",
            "--mg_p2_smoother_steps",
            "4",
        ],
    },
    {
        "name": "tail_baseline_p1steps_1",
        "label": "Tail baseline P1 steps=1",
        "stage": "step_refine",
        "args": [
            "--mg_strategy",
            "same_mesh_p4_p2_p1_lminus1_p1",
            "--mg_p1_smoother_steps",
            "1",
        ],
    },
    {
        "name": "tail_baseline_p1steps_2",
        "label": "Tail baseline P1 steps=2",
        "stage": "step_refine",
        "args": [
            "--mg_strategy",
            "same_mesh_p4_p2_p1_lminus1_p1",
            "--mg_p1_smoother_steps",
            "2",
        ],
    },
    {
        "name": "tail_p1_jacobi_steps_1",
        "label": "Tail P1 Jacobi steps=1",
        "stage": "step_refine",
        "args": [
            "--mg_strategy",
            "same_mesh_p4_p2_p1_lminus1_p1",
            "--mg_p1_smoother_pc_type",
            "jacobi",
            "--mg_p1_smoother_steps",
            "1",
        ],
    },
    {
        "name": "tail_p1_jacobi_steps_3",
        "label": "Tail P1 Jacobi steps=3",
        "stage": "step_refine",
        "args": [
            "--mg_strategy",
            "same_mesh_p4_p2_p1_lminus1_p1",
            "--mg_p1_smoother_pc_type",
            "jacobi",
            "--mg_p1_smoother_steps",
            "3",
        ],
    },
]


def _ensure_assets() -> None:
    for level in LEVELS:
        for degree in (1, 2, 4):
            ensure_same_mesh_case_hdf5(level, degree)


def _load_rows() -> list[dict[str, object]]:
    if SUMMARY_PATH.exists():
        return list(json.loads(SUMMARY_PATH.read_text(encoding="utf-8")))
    return []


def _write_rows(rows: list[dict[str, object]]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def _sum(items: list[dict[str, object]], key: str) -> float:
    return float(sum(float(item.get(key, 0.0)) for item in items))


def _callback_value(callback: dict[str, object], phase: str, key: str) -> float:
    return float(dict(callback.get(str(phase), {})).get(str(key), 0.0))


def _collect_mg_family(linear_timing: list[dict[str, object]], family: str) -> dict[str, float]:
    observed_time = 0.0
    solve_invocations = 0
    total_iterations = 0
    for entry in linear_timing:
        for diag in list(entry.get("mg_runtime_diagnostics", [])):
            if str(diag.get("family")) != str(family):
                continue
            observed_time += float(diag.get("observed_time_sec", 0.0))
            solve_invocations += int(diag.get("solve_invocations", 0))
            total_iterations += int(diag.get("total_iterations", 0))
    return {
        "observed_time_sec": float(observed_time),
        "solve_invocations": int(solve_invocations),
        "total_iterations": int(total_iterations),
    }


def _aggregate(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload["result"])
    step = dict(result["steps"][0])
    history = list(step.get("history", []))
    linear = list(step.get("linear_timing", []))
    case = dict(payload["case"])
    linear_solver = dict(dict(payload["metadata"]).get("linear_solver", {}))
    timings = dict(payload["timings"])
    callback = dict(timings.get("callback_summary", {}))
    bootstrap = dict(timings.get("solver_bootstrap_breakdown", {}))
    linear_summary = dict(step.get("linear_summary", {}))
    fine = _collect_mg_family(linear, "fine")
    degree2 = _collect_mg_family(linear, "degree2")
    degree1 = _collect_mg_family(linear, "degree1")
    coarse = _collect_mg_family(linear, "coarse")
    return {
        "solver_success": bool(result["solver_success"]),
        "status": str(result["status"]),
        "message": str(step.get("message", "")),
        "newton_iterations": int(step["nit"]),
        "linear_iterations": int(step["linear_iters"]),
        "line_search_evals": int(sum(int(item.get("ls_evals", 0)) for item in history)),
        "worst_true_relative_residual": float(
            linear_summary.get("worst_true_relative_residual", float("nan"))
        ),
        "steady_state_total_time_sec": float(timings.get("steady_state_total_time", 0.0)),
        "end_to_end_total_time_sec": float(timings.get("total_time", 0.0)),
        "solve_time_sec": float(timings.get("solve_time", 0.0)),
        "one_time_setup_time_sec": float(timings.get("one_time_setup_time", 0.0)),
        "problem_build_time_sec": float(timings.get("problem_build_time", 0.0)),
        "assembler_setup_time_sec": float(timings.get("assembler_setup_time", 0.0)),
        "assembler_warmup_time_sec": float(
            dict(timings.get("assembler_setup_breakdown", {})).get("warmup", 0.0)
        ),
        "mg_hierarchy_build_time_sec": float(bootstrap.get("mg_hierarchy_build_time", 0.0)),
        "mg_transfer_build_time_sec": float(bootstrap.get("mg_transfer_build_time", 0.0)),
        "mg_configure_time_sec": float(bootstrap.get("mg_configure_time", 0.0)),
        "energy_total_time_sec": _callback_value(callback, "energy", "total"),
        "gradient_total_time_sec": _callback_value(callback, "gradient", "total"),
        "hessian_total_time_sec": _callback_value(callback, "hessian", "total"),
        "line_search_time_sec": float(sum(float(item.get("t_ls", 0.0)) for item in history)),
        "linear_pc_setup_time_sec": _sum(linear, "pc_setup_time"),
        "linear_ksp_solve_time_sec": _sum(linear, "solve_time"),
        "linear_assemble_time_sec": _sum(linear, "assemble_total_time"),
        "fine_observed_time_sec": float(fine["observed_time_sec"]),
        "p2_observed_time_sec": float(degree2["observed_time_sec"]),
        "p1_observed_time_sec": float(degree1["observed_time_sec"]),
        "coarse_observed_time_sec": float(coarse["observed_time_sec"]),
        "fine_solve_invocations": int(fine["solve_invocations"]),
        "p2_solve_invocations": int(degree2["solve_invocations"]),
        "p1_solve_invocations": int(degree1["solve_invocations"]),
        "coarse_solve_invocations": int(coarse["solve_invocations"]),
        "mg_strategy": str(linear_solver.get("mg_strategy", "")),
        "line_search": str(case.get("line_search", "")),
        "problem_build_mode": str(case.get("problem_build_mode", "")),
        "mg_level_build_mode": str(case.get("mg_level_build_mode", "")),
        "mg_transfer_build_mode": str(case.get("mg_transfer_build_mode", "")),
        "mg_legacy_level_smoothers": dict(linear_solver.get("mg_legacy_level_smoothers", {})),
    }


def _run_case(*, level: int, ranks: int, variant: dict[str, object]) -> dict[str, object]:
    case_dir = OUTPUT_ROOT / f"level{level}" / str(variant["name"]) / f"np{ranks}"
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    stdout_path = case_dir / "stdout.txt"
    stderr_path = case_dir / "stderr.txt"
    if not result_path.exists():
        command = [
            "mpiexec",
            "-n",
            str(ranks),
            str(PYTHON),
            "-u",
            str(SOLVER),
            "--level",
            str(level),
            *COMMON_ARGS,
            *list(variant["args"]),
            "--out",
            str(result_path),
        ]
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0 and not result_path.exists():
            return {
                "level": int(level),
                "ranks": int(ranks),
                "variant": str(variant["name"]),
                "label": str(variant["label"]),
                "stage": str(variant["stage"]),
                "solver_success": False,
                "status": "subprocess_failed",
                "message": proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "subprocess failed",
                "stdout_path": str(stdout_path.relative_to(REPO_ROOT)),
                "stderr_path": str(stderr_path.relative_to(REPO_ROOT)),
            }
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    row = {
        "level": int(level),
        "ranks": int(ranks),
        "variant": str(variant["name"]),
        "label": str(variant["label"]),
        "stage": str(variant["stage"]),
        "result_path": str(result_path.relative_to(REPO_ROOT)),
        "stdout_path": str(stdout_path.relative_to(REPO_ROOT)),
        "stderr_path": str(stderr_path.relative_to(REPO_ROOT)),
    }
    row.update(_aggregate(payload))
    return row


def _baseline_rows(rows: list[dict[str, object]], *, level: int) -> dict[int, dict[str, object]]:
    subset = [
        row
        for row in rows
        if int(row["level"]) == int(level)
        and str(row["variant"]) == "tail_baseline"
        and bool(row.get("solver_success"))
    ]
    return {int(row["ranks"]): row for row in subset}


def _passes_gate(row: dict[str, object], baseline: dict[str, object]) -> bool:
    if not bool(row.get("solver_success")):
        return False
    if not bool(baseline.get("solver_success")):
        return False
    return int(row["newton_iterations"]) <= int(np.ceil(1.1 * float(baseline["newton_iterations"])))


def _competitive_family(
    row: dict[str, object] | None,
    baseline: dict[str, object] | None,
    *,
    time_factor: float = 1.25,
) -> bool:
    if row is None or baseline is None:
        return False
    if not _passes_gate(row, baseline):
        return False
    return float(row["steady_state_total_time_sec"]) <= float(time_factor) * float(
        baseline["steady_state_total_time_sec"]
    )


def _run_stage(rows: list[dict[str, object]], *, variants: list[dict[str, object]], levels: list[int], ranks: list[int]) -> list[dict[str, object]]:
    existing = {
        (int(row["level"]), int(row["ranks"]), str(row["variant"]))
        for row in rows
    }
    for level in levels:
        for variant in variants:
            for ranks_count in ranks:
                key = (int(level), int(ranks_count), str(variant["name"]))
                if key in existing:
                    continue
                rows.append(_run_case(level=level, ranks=ranks_count, variant=variant))
                rows.sort(key=lambda item: (int(item["level"]), str(item["stage"]), str(item["variant"]), int(item["ranks"])))
                _write_rows(rows)
    return rows


def _row_for(rows: list[dict[str, object]], *, level: int, ranks: int, variant: str) -> dict[str, object] | None:
    return next(
        (
            row
            for row in rows
            if int(row["level"]) == int(level)
            and int(row["ranks"]) == int(ranks)
            and str(row["variant"]) == str(variant)
        ),
        None,
    )


def _pruned_rank8_variants(
    rows: list[dict[str, object]],
    *,
    level: int,
    variants: list[dict[str, object]],
    baseline_variant: str = "tail_baseline",
    always_keep: set[str] | None = None,
) -> list[dict[str, object]]:
    always_keep = set(always_keep or set())
    baseline_rank1 = _row_for(rows, level=level, ranks=1, variant=baseline_variant)
    selected: list[dict[str, object]] = []
    for variant in variants:
        name = str(variant["name"])
        rank1 = _row_for(rows, level=level, ranks=1, variant=name)
        if rank1 is None:
            continue
        if not bool(rank1.get("solver_success")):
            continue
        if name in always_keep:
            selected.append(variant)
            continue
        if baseline_rank1 is None or _passes_gate(rank1, baseline_rank1):
            selected.append(variant)
    return selected


def _select_step_variants(rows: list[dict[str, object]], *, level: int) -> list[dict[str, object]]:
    baseline_rank1 = _row_for(rows, level=level, ranks=1, variant="tail_baseline")
    family_rank1 = {
        "cheb": _row_for(rows, level=level, ranks=1, variant="tail_fine_chebyshev_jacobi"),
        "p2_jacobi": _row_for(rows, level=level, ranks=1, variant="tail_p2_jacobi"),
        "p1_jacobi": _row_for(rows, level=level, ranks=1, variant="tail_p1_jacobi"),
    }
    keep_names = {
        "tail_baseline_p4steps_2",
        "tail_baseline_p4steps_4",
        "tail_baseline_p2steps_2",
        "tail_baseline_p2steps_4",
        "tail_baseline_p1steps_1",
        "tail_baseline_p1steps_2",
    }
    if _competitive_family(family_rank1["cheb"], baseline_rank1):
        keep_names.update({"tail_cheb_p4steps_2", "tail_cheb_p4steps_4"})
    if _competitive_family(family_rank1["p2_jacobi"], baseline_rank1):
        keep_names.update({"tail_p2_jacobi_steps_2", "tail_p2_jacobi_steps_4"})
    if _competitive_family(family_rank1["p1_jacobi"], baseline_rank1):
        keep_names.update({"tail_p1_jacobi_steps_1", "tail_p1_jacobi_steps_3"})
    return [variant for variant in STEP_VARIANTS if str(variant["name"]) in keep_names]


def main() -> None:
    _ensure_assets()
    rows = _load_rows()

    rows = _run_stage(rows, variants=STAGE_A_VARIANTS, levels=[4], ranks=[1])
    rows = _run_stage(
        rows,
        variants=_pruned_rank8_variants(
            rows,
            level=4,
            variants=STAGE_A_VARIANTS,
            always_keep={"tail_baseline", "no_tail_baseline", "tail_replicated_assets"},
        ),
        levels=[4],
        ranks=[8],
    )
    selected_step_variants = _select_step_variants(rows, level=4)
    rows = _run_stage(rows, variants=selected_step_variants, levels=[4], ranks=[1])
    rows = _run_stage(
        rows,
        variants=_pruned_rank8_variants(
            rows,
            level=4,
            variants=selected_step_variants,
            always_keep=set(),
        ),
        levels=[4],
        ranks=[8],
    )

    l4_baseline = _baseline_rows(rows, level=4)
    l4_rank8 = [
        row
        for row in rows
        if int(row["level"]) == 4
        and int(row["ranks"]) == 8
        and _passes_gate(row, l4_baseline[8])
    ]
    l4_rank8.sort(key=lambda row: float(row["steady_state_total_time_sec"]))
    finalists = [row["variant"] for row in l4_rank8[:3]]
    finalist_variants = [
        variant
        for variant in (STAGE_A_VARIANTS + STEP_VARIANTS)
        if str(variant["name"]) in finalists or str(variant["name"]) == "tail_baseline"
    ]

    rows = _run_stage(rows, variants=finalist_variants, levels=[5], ranks=[1, 8])

    l5_baseline = _baseline_rows(rows, level=5)
    accepted_finalists: list[str] = []
    for name in finalists:
        level1 = next(
            (
                row for row in rows
                if int(row["level"]) == 5 and int(row["ranks"]) == 1 and str(row["variant"]) == str(name)
            ),
            None,
        )
        level8 = next(
            (
                row for row in rows
                if int(row["level"]) == 5 and int(row["ranks"]) == 8 and str(row["variant"]) == str(name)
            ),
            None,
        )
        if level1 is None or level8 is None:
            continue
        if _passes_gate(level1, l5_baseline[1]) and _passes_gate(level8, l5_baseline[8]):
            accepted_finalists.append(str(name))
    if not accepted_finalists:
        accepted_finalists = ["tail_baseline"]

    full_variants = [
        variant
        for variant in (STAGE_A_VARIANTS + STEP_VARIANTS)
        if str(variant["name"]) in set(accepted_finalists + ["tail_baseline"])
    ]
    rows = _run_stage(rows, variants=full_variants, levels=LEVELS, ranks=[1, 2, 4, 8])

    bottleneck_variant = {
        "name": "tail_replicated_assets_full",
        "label": "Tail baseline with replicated asset loads (full matrix)",
        "stage": "bottleneck_round",
        "args": [
            "--mg_strategy",
            "same_mesh_p4_p2_p1_lminus1_p1",
            "--problem_build_mode",
            "replicated",
            "--mg_level_build_mode",
            "replicated",
        ],
    }
    rows = _run_stage(rows, variants=[bottleneck_variant], levels=[4], ranks=[1, 2, 4, 8])

    armijo_variant = {
        "name": "tail_armijo",
        "label": "Tail baseline with Armijo",
        "stage": "line_search_round",
        "args": [
            "--mg_strategy",
            "same_mesh_p4_p2_p1_lminus1_p1",
            "--line_search",
            "armijo",
        ],
    }
    rows = _run_stage(rows, variants=[armijo_variant], levels=[4, 5], ranks=[1, 8])

    rows.sort(key=lambda item: (int(item["level"]), str(item["stage"]), str(item["variant"]), int(item["ranks"])))
    _write_rows(rows)


if __name__ == "__main__":
    main()
