#!/usr/bin/env python3
"""Run the staged L6 P4 deep-tail optimization benchmark at 8 MPI ranks."""

from __future__ import annotations

import json
import math
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
    / "slope_stability_l6_p4_deep_p1_tail_optimization_lambda1_np8_maxit20"
)
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

RANKS = 8
CUSTOM_HIERARCHY = "1:1,2:1,3:1,4:1,5:1,6:1,6:2,6:4"
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
    "--mg_strategy",
    "custom_mixed",
    "--mg_custom_hierarchy",
    CUSTOM_HIERARCHY,
    "--mg_variant",
    "legacy_pmg",
    "--ksp_type",
    "fgmres",
    "--ksp_rtol",
    "1e-2",
    "--ksp_max_it",
    "15",
    "--accept_ksp_maxit_direction",
    "--maxit",
    "20",
    "--benchmark_mode",
    "warmup_once_then_solve",
    "--save-history",
    "--save-linear-timing",
    "--quiet",
    "--no-use_trust_region",
    "--line_search",
    "armijo",
    "--distribution_strategy",
    "overlap_p2p",
    "--problem_build_mode",
    "root_bcast",
    "--mg_level_build_mode",
    "root_bcast",
    "--mg_transfer_build_mode",
    "owned_rows",
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
    "rank0_lu_broadcast",
]


def _ensure_assets() -> None:
    for level in range(1, 7):
        for degree in (1, 2, 4):
            ensure_same_mesh_case_hdf5(level, degree)


def _sum(items: list[dict[str, object]], key: str) -> float:
    return float(sum(float(item.get(key, 0.0)) for item in items))


def _make_variant(
    *,
    name: str,
    label: str,
    stage: str,
    args: list[str],
    smoother_desc: str,
    guard_enabled: bool,
    reuse_buffers: bool,
) -> dict[str, object]:
    return {
        "name": str(name),
        "label": str(label),
        "stage": str(stage),
        "args": list(args),
        "smoother_desc": str(smoother_desc),
        "guard_enabled": bool(guard_enabled),
        "reuse_buffers": bool(reuse_buffers),
    }


STEP1_VARIANTS = [
    _make_variant(
        name="step1_baseline_unguarded",
        label="Step 1 baseline: Armijo + capped solve (unguarded)",
        stage="step1",
        args=[
            "--no-guard_ksp_maxit_direction",
            "--ksp_maxit_direction_true_rel_cap",
            "6e-2",
            "--mg_p4_smoother_ksp_type",
            "richardson",
            "--mg_p4_smoother_pc_type",
            "sor",
            "--mg_p4_smoother_steps",
            "3",
            "--no-reuse_hessian_value_buffers",
        ],
        smoother_desc="P4 richardson+sor(3), P2/P1 richardson+sor(3)",
        guard_enabled=False,
        reuse_buffers=False,
    ),
    _make_variant(
        name="step1_guarded",
        label="Step 1 candidate: Armijo + capped solve (guarded)",
        stage="step1",
        args=[
            "--guard_ksp_maxit_direction",
            "--ksp_maxit_direction_true_rel_cap",
            "6e-2",
            "--mg_p4_smoother_ksp_type",
            "richardson",
            "--mg_p4_smoother_pc_type",
            "sor",
            "--mg_p4_smoother_steps",
            "3",
            "--no-reuse_hessian_value_buffers",
        ],
        smoother_desc="P4 richardson+sor(3), P2/P1 richardson+sor(3)",
        guard_enabled=True,
        reuse_buffers=False,
    ),
]


STEP2_CANDIDATE_SPECS = [
    ("richardson", "sor", 2),
    ("richardson", "sor", 3),
    ("richardson", "jacobi", 2),
    ("richardson", "jacobi", 3),
    ("richardson", "jacobi", 4),
    ("chebyshev", "jacobi", 2),
    ("chebyshev", "jacobi", 3),
    ("chebyshev", "jacobi", 4),
    ("richardson", "asm", 1),
    ("richardson", "asm", 2),
    ("fgmres", "none", 2),
    ("fgmres", "none", 3),
]


def _load_summary() -> dict[str, object]:
    if SUMMARY_PATH.exists():
        return dict(json.loads(SUMMARY_PATH.read_text(encoding="utf-8")))
    return {"rows": [], "selection": {}}


def _save_summary(rows: list[dict[str, object]], selection: dict[str, object]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rows": rows, "selection": selection}
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _command(extra_args: list[str], out: Path) -> list[str]:
    return [
        "mpiexec",
        "-n",
        str(RANKS),
        str(PYTHON),
        "-u",
        str(SOLVER),
        *COMMON_ARGS,
        *list(extra_args),
        "--out",
        str(out),
    ]


def _aggregate(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload.get("result", {}))
    steps = list(result.get("steps", []))
    last_step = dict(steps[-1]) if steps else {}
    history = list(last_step.get("history", []))
    linear = list(last_step.get("linear_timing", []))
    timings = dict(payload.get("timings", {}))
    linear_summary = dict(last_step.get("linear_summary", {}))
    linear_solver = dict(payload.get("metadata", {}).get("linear_solver", {}))

    return {
        "solver_success": bool(result.get("solver_success", False)),
        "status": str(result.get("status", "")),
        "message": str(last_step.get("message", "")),
        "newton_iterations": int(last_step.get("nit", len(steps))),
        "linear_iterations": int(last_step.get("linear_iters", 0)),
        "line_search_evals": int(sum(int(item.get("ls_evals", 0)) for item in history)),
        "line_search_time_sec": float(sum(float(item.get("t_ls", 0.0)) for item in history)),
        "gradient_stage_time_sec": float(sum(float(item.get("t_grad", 0.0)) for item in history)),
        "hessian_stage_time_sec": float(sum(float(item.get("t_hess", 0.0)) for item in history)),
        "iteration_time_sec": float(sum(float(item.get("t_iter", 0.0)) for item in history)),
        "worst_true_relative_residual": float(
            linear_summary.get("worst_true_relative_residual", float("nan"))
        ),
        "accepted_capped_step_count": int(
            last_step.get("accepted_capped_step_count", 0)
        ),
        "one_time_setup_time_sec": float(
            timings.get("one_time_setup_time", timings.get("setup_time", 0.0))
        ),
        "steady_state_setup_time_sec": float(timings.get("steady_state_setup_time", 0.0)),
        "solve_time_sec": float(timings.get("solve_time", 0.0)),
        "steady_state_total_time_sec": float(timings.get("steady_state_total_time", 0.0)),
        "end_to_end_total_time_sec": float(timings.get("total_time", 0.0)),
        "linear_pc_setup_time_sec": _sum(linear, "pc_setup_time"),
        "linear_ksp_solve_time_sec": _sum(linear, "solve_time"),
        "linear_assemble_time_sec": _sum(linear, "assemble_total_time"),
        "energy": float(last_step.get("energy", float("nan"))),
        "final_grad_norm": float(
            last_step.get("final_grad_norm", result.get("final_grad_norm", float("nan")))
        ),
        "omega": float(last_step.get("omega", float("nan"))),
        "u_max": float(last_step.get("u_max", float("nan"))),
        "guard_ksp_maxit_direction": bool(
            linear_solver.get("guard_ksp_maxit_direction", False)
        ),
        "ksp_maxit_direction_true_rel_cap": float(
            linear_solver.get("ksp_maxit_direction_true_rel_cap", float("nan"))
        ),
        "reuse_hessian_value_buffers": bool(
            linear_solver.get("reuse_hessian_value_buffers", True)
        ),
        "mg_p4_smoother_ksp_type": str(
            dict(linear_solver.get("mg_legacy_level_smoothers", {}))
            .get("fine", {})
            .get("ksp_type", "")
        ),
        "mg_p4_smoother_pc_type": str(
            dict(linear_solver.get("mg_legacy_level_smoothers", {}))
            .get("fine", {})
            .get("pc_type", "")
        ),
        "mg_p4_smoother_steps": int(
            dict(linear_solver.get("mg_legacy_level_smoothers", {}))
            .get("fine", {})
            .get("steps", 0)
        ),
    }


def _run_case(variant: dict[str, object]) -> dict[str, object]:
    case_dir = OUTPUT_ROOT / str(variant["name"])
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    stdout_path = case_dir / "stdout.txt"
    stderr_path = case_dir / "stderr.txt"

    if not result_path.exists():
        proc = subprocess.run(
            _command(list(variant["args"]), result_path),
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            return {
                "variant": str(variant["name"]),
                "label": str(variant["label"]),
                "stage": str(variant["stage"]),
                "solver_success": False,
                "status": "subprocess_failed",
                "message": proc.stderr.strip().splitlines()[-1]
                if proc.stderr.strip()
                else "subprocess failed",
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "result_json": str(result_path),
                "guard_enabled": bool(variant["guard_enabled"]),
                "reuse_buffers": bool(variant["reuse_buffers"]),
                "smoother_desc": str(variant["smoother_desc"]),
            }

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    aggregated = _aggregate(payload)
    aggregated.update(
        {
            "variant": str(variant["name"]),
            "label": str(variant["label"]),
            "stage": str(variant["stage"]),
            "guard_enabled": bool(variant["guard_enabled"]),
            "reuse_buffers": bool(variant["reuse_buffers"]),
            "smoother_desc": str(variant["smoother_desc"]),
            "result_json": str(result_path),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
    )
    return aggregated


def _within_factor(candidate: float, baseline: float, factor: float) -> bool:
    if not math.isfinite(candidate) or not math.isfinite(baseline):
        return False
    return candidate <= factor * baseline


def _meets_end_state_guards(
    candidate: dict[str, object], baseline: dict[str, object]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    required_keys = ("energy", "final_grad_norm", "linear_iterations", "steady_state_total_time_sec")
    for key in required_keys:
        if key not in candidate:
            reasons.append(f"missing {key}")
    if reasons:
        return False, reasons
    energy_delta = abs(float(candidate["energy"]) - float(baseline["energy"]))
    if not energy_delta <= 1.0e-6:
        reasons.append(f"energy delta {energy_delta:.3e} > 1e-6")
    if not _within_factor(
        float(candidate["final_grad_norm"]), float(baseline["final_grad_norm"]), 1.2
    ):
        reasons.append("final grad norm > 1.2x baseline")
    if not _within_factor(
        float(candidate["linear_iterations"]), float(baseline["linear_iterations"]), 1.1
    ):
        reasons.append("linear iterations > 1.1x baseline")
    return (len(reasons) == 0, reasons)


def _build_step2_variants(step1_winner: dict[str, object]) -> list[dict[str, object]]:
    variants: list[dict[str, object]] = []
    guard_flag = (
        ["--guard_ksp_maxit_direction"]
        if bool(step1_winner["guard_enabled"])
        else ["--no-guard_ksp_maxit_direction"]
    )
    for ksp_type, pc_type, steps in STEP2_CANDIDATE_SPECS:
        name = f"step2_p4_{ksp_type}_{pc_type}_{steps}"
        variants.append(
            _make_variant(
                name=name,
                label=f"Step 2 candidate: {ksp_type}+{pc_type} ({steps})",
                stage="step2",
                args=[
                    *guard_flag,
                    "--ksp_maxit_direction_true_rel_cap",
                    str(step1_winner["ksp_maxit_direction_true_rel_cap"]),
                    "--mg_p4_smoother_ksp_type",
                    str(ksp_type),
                    "--mg_p4_smoother_pc_type",
                    str(pc_type),
                    "--mg_p4_smoother_steps",
                    str(int(steps)),
                    "--no-reuse_hessian_value_buffers",
                ],
                smoother_desc=f"P4 {ksp_type}+{pc_type}({steps}), P2/P1 richardson+sor(3)",
                guard_enabled=bool(step1_winner["guard_enabled"]),
                reuse_buffers=False,
            )
        )
    return variants


def _variant_by_name(rows: list[dict[str, object]], name: str) -> dict[str, object]:
    for row in rows:
        if str(row.get("variant")) == str(name):
            return row
    raise KeyError(name)


def main() -> None:
    _ensure_assets()
    summary = _load_summary()
    rows = list(summary.get("rows", []))
    row_by_name = {str(row.get("variant")): row for row in rows}
    selection = dict(summary.get("selection", {}))

    for variant in STEP1_VARIANTS:
        if variant["name"] not in row_by_name:
            row = _run_case(variant)
            rows.append(row)
            row_by_name[str(variant["name"])] = row
            _save_summary(rows, selection)

    baseline = _variant_by_name(rows, "step1_baseline_unguarded")
    guarded = _variant_by_name(rows, "step1_guarded")
    step1_ok, step1_reasons = _meets_end_state_guards(guarded, baseline)
    if step1_ok and float(guarded["steady_state_total_time_sec"]) < float(
        baseline["steady_state_total_time_sec"]
    ):
        step1_winner_name = "step1_guarded"
    else:
        step1_winner_name = "step1_baseline_unguarded"
    selection["step1"] = {
        "baseline": "step1_baseline_unguarded",
        "candidate": "step1_guarded",
        "winner": step1_winner_name,
        "candidate_guard_ok": bool(step1_ok),
        "candidate_guard_reasons": list(step1_reasons),
    }
    _save_summary(rows, selection)

    step1_winner = _variant_by_name(rows, step1_winner_name)
    step2_variants = _build_step2_variants(step1_winner)
    for variant in step2_variants:
        if variant["name"] not in row_by_name:
            row = _run_case(variant)
            rows.append(row)
            row_by_name[str(variant["name"])] = row
            _save_summary(rows, selection)

    step2_baseline_name = step1_winner_name
    step2_winner_name = step2_baseline_name
    step2_evaluations: list[dict[str, object]] = []
    step2_baseline = _variant_by_name(rows, step2_baseline_name)
    for variant in step2_variants:
        row = _variant_by_name(rows, str(variant["name"]))
        ok, reasons = _meets_end_state_guards(row, step2_baseline)
        improves_time = float(row["steady_state_total_time_sec"]) < float(
            step2_baseline["steady_state_total_time_sec"]
        )
        step2_evaluations.append(
            {
                "variant": str(variant["name"]),
                "guard_ok": bool(ok),
                "guard_reasons": list(reasons),
                "improves_time": bool(improves_time),
            }
        )
        if ok and improves_time:
            if float(row["steady_state_total_time_sec"]) < float(
                _variant_by_name(rows, step2_winner_name)["steady_state_total_time_sec"]
            ):
                step2_winner_name = str(variant["name"])
    selection["step2"] = {
        "baseline": step2_baseline_name,
        "winner": step2_winner_name,
        "evaluations": step2_evaluations,
    }
    _save_summary(rows, selection)

    step2_winner = _variant_by_name(rows, step2_winner_name)
    step3_variant = _make_variant(
        name="step3_buffer_reuse",
        label="Step 3 candidate: persistent Hessian COO buffers",
        stage="step3",
        args=[
            (
                "--guard_ksp_maxit_direction"
                if bool(step2_winner["guard_enabled"])
                else "--no-guard_ksp_maxit_direction"
            ),
            "--ksp_maxit_direction_true_rel_cap",
            str(step2_winner["ksp_maxit_direction_true_rel_cap"]),
            "--mg_p4_smoother_ksp_type",
            str(step2_winner["mg_p4_smoother_ksp_type"]),
            "--mg_p4_smoother_pc_type",
            str(step2_winner["mg_p4_smoother_pc_type"]),
            "--mg_p4_smoother_steps",
            str(int(step2_winner["mg_p4_smoother_steps"])),
            "--reuse_hessian_value_buffers",
        ],
        smoother_desc=str(step2_winner["smoother_desc"]),
        guard_enabled=bool(step2_winner["guard_enabled"]),
        reuse_buffers=True,
    )
    if step3_variant["name"] not in row_by_name:
        row = _run_case(step3_variant)
        rows.append(row)
        row_by_name[str(step3_variant["name"])] = row
        _save_summary(rows, selection)

    step3_candidate = _variant_by_name(rows, "step3_buffer_reuse")
    step3_ok, step3_reasons = _meets_end_state_guards(step3_candidate, step2_winner)
    step3_improves_stage = (
        float(step3_candidate["hessian_stage_time_sec"])
        < float(step2_winner["hessian_stage_time_sec"])
        or float(step3_candidate["linear_assemble_time_sec"])
        < float(step2_winner["linear_assemble_time_sec"])
    )
    step3_improves_time = float(step3_candidate["steady_state_total_time_sec"]) < float(
        step2_winner["steady_state_total_time_sec"]
    )
    if step3_ok and step3_improves_stage and step3_improves_time:
        step3_winner_name = "step3_buffer_reuse"
    else:
        step3_winner_name = step2_winner_name
    selection["step3"] = {
        "baseline": step2_winner_name,
        "candidate": "step3_buffer_reuse",
        "winner": step3_winner_name,
        "candidate_guard_ok": bool(step3_ok),
        "candidate_guard_reasons": list(step3_reasons),
        "candidate_improves_stage": bool(step3_improves_stage),
        "candidate_improves_time": bool(step3_improves_time),
    }
    selection["final_stack"] = {
        "baseline": "step1_baseline_unguarded",
        "step1_winner": step1_winner_name,
        "step2_winner": step2_winner_name,
        "step3_winner": step3_winner_name,
        "final": step3_winner_name,
    }
    _save_summary(rows, selection)


if __name__ == "__main__":
    main()
