#!/usr/bin/env python3
"""Run the slope-stability JAX+PETSc minimizer sweep and select one winner."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from src.core.benchmark.replication import read_json, run_logged_command, write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
SOLVER = "src/problems/slope_stability/jax_petsc/solve_slope_stability_dof.py"
REP_CASES = ((3, 1), (3, 16))
VERIFICATION_CASE = (4, 32)


@dataclass(frozen=True)
class Candidate:
    name: str
    stage: str
    settings: dict[str, object]


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _bool_flag(name: str, value: bool) -> list[str]:
    return [f"--{name}" if value else f"--no-{name}"]


def _float_tag(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _parse_case_spec(spec: str) -> tuple[int, int]:
    parts = str(spec).split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected CASE spec 'level:nprocs', got {spec!r}")
    return int(parts[0]), int(parts[1])


def _candidate_command(
    candidate: Candidate,
    *,
    level: int,
    nprocs: int,
    lambda_target: float,
    out_json: Path,
) -> list[str]:
    settings = dict(candidate.settings)
    cmd = [
        "mpiexec",
        "-n",
        str(nprocs),
        str(PYTHON),
        "-u",
        SOLVER,
        "--level",
        str(level),
        "--lambda-target",
        str(lambda_target),
        "--profile",
        "performance",
        "--assembly_mode",
        "element",
        "--element_reorder_mode",
        str(settings["element_reorder_mode"]),
        "--local_hessian_mode",
        str(settings["local_hessian_mode"]),
        "--nproc",
        str(settings["nproc"]),
        "--linesearch_a",
        str(settings["linesearch_a"]),
        "--linesearch_b",
        str(settings["linesearch_b"]),
        "--linesearch_tol",
        str(settings["linesearch_tol"]),
        "--ksp_type",
        str(settings["ksp_type"]),
        "--pc_type",
        str(settings["pc_type"]),
        "--ksp_rtol",
        str(settings["ksp_rtol"]),
        "--ksp_max_it",
        str(settings["ksp_max_it"]),
        "--trust_radius_init",
        str(settings["trust_radius_init"]),
        "--trust_radius_min",
        str(settings["trust_radius_min"]),
        "--trust_radius_max",
        str(settings["trust_radius_max"]),
        "--trust_shrink",
        str(settings["trust_shrink"]),
        "--trust_expand",
        str(settings["trust_expand"]),
        "--trust_eta_shrink",
        str(settings["trust_eta_shrink"]),
        "--trust_eta_expand",
        str(settings["trust_eta_expand"]),
        "--trust_max_reject",
        str(settings["trust_max_reject"]),
        "--tolf",
        str(settings["tolf"]),
        "--tolg",
        str(settings["tolg"]),
        "--tolg_rel",
        str(settings["tolg_rel"]),
        "--tolx_rel",
        str(settings["tolx_rel"]),
        "--tolx_abs",
        str(settings["tolx_abs"]),
        "--maxit",
        str(settings["maxit"]),
        "--gamg_threshold",
        str(settings["gamg_threshold"]),
        "--gamg_agg_nsmooths",
        str(settings["gamg_agg_nsmooths"]),
        "--reg",
        str(settings["reg"]),
        "--save-history",
        "--save-linear-timing",
        "--quiet",
        "--out",
        str(out_json),
    ]
    cmd += _bool_flag("local_coloring", bool(settings["local_coloring"]))
    cmd += _bool_flag("use_near_nullspace", bool(settings["use_near_nullspace"]))
    cmd += _bool_flag("gamg_set_coordinates", bool(settings["gamg_set_coordinates"]))
    cmd += _bool_flag("retry_on_failure", bool(settings["retry_on_failure"]))
    cmd += _bool_flag("pc_setup_on_ksp_cap", bool(settings["pc_setup_on_ksp_cap"]))
    cmd += _bool_flag("use_trust_region", bool(settings["use_trust_region"]))
    cmd += _bool_flag(
        "trust_subproblem_line_search",
        bool(settings["trust_subproblem_line_search"]),
    )
    return cmd


def _summarize_case(candidate: Candidate, payload: dict, *, level: int, nprocs: int, json_path: Path) -> dict[str, object]:
    step = dict(payload["result"]["steps"][0])
    linear_timing = list(step.get("linear_timing", []))
    history = list(step.get("history", []))
    return {
        "candidate": candidate.name,
        "stage": candidate.stage,
        "level": int(level),
        "nprocs": int(nprocs),
        "status": str(payload["result"]["status"]),
        "solver_success": bool(payload["result"]["solver_success"]),
        "final_energy": float(step["energy"]),
        "omega": float(step["omega"]),
        "u_max": float(step["u_max"]),
        "newton_iters": int(step["nit"]),
        "linear_iters": int(step["linear_iters"]),
        "total_time_s": float(payload["timings"]["total_time"]),
        "assembly_time_s": float(sum(float(rec.get("assemble_total_time", 0.0)) for rec in linear_timing)),
        "pc_init_time_s": float(sum(float(rec.get("pc_setup_time", 0.0)) for rec in linear_timing)),
        "ksp_solve_time_s": float(sum(float(rec.get("solve_time", 0.0)) for rec in linear_timing)),
        "trust_rejects": int(sum(int(rec.get("trust_rejects", 0)) for rec in history)),
        "message": str(step["message"]),
        "json_path": _display_path(json_path),
    }


def _geo_mean(values: list[float]) -> float | None:
    if not values:
        return None
    if any(v <= 0.0 for v in values):
        return None
    return float(math.exp(sum(math.log(v) for v in values) / len(values)))


def _score_row(candidate: Candidate, rep_rows: list[dict[str, object]]) -> dict[str, object]:
    success_rows = [row for row in rep_rows if bool(row["solver_success"])]
    return {
        "candidate": candidate.name,
        "stage": candidate.stage,
        "settings": dict(candidate.settings),
        "representative_rows": rep_rows,
        "success_count": int(len(success_rows)),
        "geo_mean_time_s": _geo_mean([float(row["total_time_s"]) for row in success_rows]),
        "total_linear_iters": int(sum(int(row["linear_iters"]) for row in rep_rows)),
        "total_trust_rejects": int(sum(int(row["trust_rejects"]) for row in rep_rows)),
    }


def _score_key(row: dict[str, object]) -> tuple[float, float, int, int]:
    geo = row["geo_mean_time_s"]
    return (
        -float(row["success_count"]),
        float(geo) if geo is not None else float("inf"),
        int(row["total_linear_iters"]),
        int(row["total_trust_rejects"]),
    )


def _baseline_settings() -> dict[str, object]:
    return {
        "assembly_mode": "element",
        "local_hessian_mode": "element",
        "local_coloring": True,
        "element_reorder_mode": "block_xyz",
        "nproc": 1,
        "use_near_nullspace": True,
        "gamg_set_coordinates": True,
        "linesearch_a": -0.5,
        "linesearch_b": 2.0,
        "linesearch_tol": 1.0e-1,
        "retry_on_failure": False,
        "pc_setup_on_ksp_cap": False,
        "gamg_threshold": 0.05,
        "gamg_agg_nsmooths": 1,
        "trust_radius_init": 0.5,
        "trust_radius_min": 1.0e-8,
        "trust_radius_max": 1.0e6,
        "trust_shrink": 0.5,
        "trust_expand": 1.5,
        "trust_eta_shrink": 0.05,
        "trust_eta_expand": 0.75,
        "trust_max_reject": 6,
        "tolf": 1.0e-4,
        "tolg": 1.0e-3,
        "tolg_rel": 1.0e-3,
        "tolx_rel": 1.0e-3,
        "tolx_abs": 1.0e-10,
        "maxit": 100,
        "reg": 1.0e-12,
        "ksp_type": "stcg",
        "pc_type": "gamg",
        "ksp_rtol": 1.0e-1,
        "ksp_max_it": 30,
        "use_trust_region": True,
        "trust_subproblem_line_search": True,
    }


def _stage_a_candidates() -> list[Candidate]:
    baseline = _baseline_settings()
    variants = [
        ("cg_gamg_ls", {"ksp_type": "cg", "pc_type": "gamg", "use_trust_region": False, "trust_subproblem_line_search": False}),
        ("cg_gamg_tr", {"ksp_type": "cg", "pc_type": "gamg", "use_trust_region": True, "trust_subproblem_line_search": False}),
        ("cg_gamg_tr_ls", {"ksp_type": "cg", "pc_type": "gamg", "use_trust_region": True, "trust_subproblem_line_search": True}),
        ("stcg_gamg_tr", {"ksp_type": "stcg", "pc_type": "gamg", "use_trust_region": True, "trust_subproblem_line_search": False}),
        ("stcg_gamg_tr_ls", {"ksp_type": "stcg", "pc_type": "gamg", "use_trust_region": True, "trust_subproblem_line_search": True}),
    ]
    return [
        Candidate(name=name, stage="A", settings={**baseline, **overrides})
        for name, overrides in variants
    ]


def _stage_b_candidates(base: Candidate) -> list[Candidate]:
    candidates: list[Candidate] = []
    for pc_type in ("gamg", "hypre"):
        for ksp_rtol in (1.0e-1, 5.0e-2, 1.0e-2):
            for ksp_max_it in (30, 50):
                for pc_setup_on_ksp_cap in (False, True):
                    name = (
                        f"{base.name}__pc_{pc_type}_rtol_{_float_tag(ksp_rtol)}"
                        f"_kmax_{ksp_max_it}_pcsetup_{int(pc_setup_on_ksp_cap)}"
                    )
                    settings = {
                        **base.settings,
                        "pc_type": pc_type,
                        "ksp_rtol": ksp_rtol,
                        "ksp_max_it": ksp_max_it,
                        "pc_setup_on_ksp_cap": pc_setup_on_ksp_cap,
                    }
                    candidates.append(Candidate(name=name, stage="B", settings=settings))
    return candidates


def _stage_c_candidates(base: Candidate) -> list[Candidate]:
    return [
        Candidate(
            name=f"trust_radius_{_float_tag(trust_radius_init)}",
            stage="C",
            settings={**base.settings, "trust_radius_init": trust_radius_init},
        )
        for trust_radius_init in (0.25, 0.5, 1.0)
    ]


def _write_candidate_markdown(path: Path, scored: dict[str, object]) -> None:
    rep_rows = list(scored["representative_rows"])
    lines = [
        f"# {scored['candidate']}",
        "",
        f"- stage: `{scored['stage']}`",
        f"- success_count: `{scored['success_count']}`",
        f"- geo_mean_time_s: `{scored['geo_mean_time_s']}`",
        f"- total_linear_iters: `{scored['total_linear_iters']}`",
        f"- total_trust_rejects: `{scored['total_trust_rejects']}`",
        "",
        "## Settings",
        "",
        "```json",
        json.dumps(scored["settings"], indent=2, sort_keys=True),
        "```",
        "",
        "## Representative Cases",
        "",
        "| level | np | result | time [s] | Newton | Linear | omega | u_max | message |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rep_rows:
        lines.append(
            "| {level} | {nprocs} | {result} | {time:.4f} | {nit} | {lit} | {omega:.6f} | {u_max:.6f} | {msg} |".format(
                level=row["level"],
                nprocs=row["nprocs"],
                result=row["status"],
                time=float(row["total_time_s"]),
                nit=int(row["newton_iters"]),
                lit=int(row["linear_iters"]),
                omega=float(row["omega"]),
                u_max=float(row["u_max"]),
                msg=str(row["message"]).replace("|", "\\|"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_candidate_case(
    out_root: Path,
    candidate: Candidate,
    *,
    level: int,
    nprocs: int,
    lambda_target: float,
    resume: bool,
) -> dict[str, object]:
    leaf_dir = out_root / "candidates" / candidate.stage / candidate.name / f"l{level}_np{nprocs}"
    json_out = leaf_dir / "output.json"
    command = _candidate_command(
        candidate,
        level=level,
        nprocs=nprocs,
        lambda_target=lambda_target,
        out_json=json_out,
    )
    run_logged_command(
        command=command,
        cwd=REPO_ROOT,
        leaf_dir=leaf_dir,
        expected_outputs=[json_out],
        resume=resume,
        notes=f"Slope-stability PETSc minimizer sweep candidate {candidate.name} on l{level} np{nprocs}.",
    )
    payload = read_json(json_out)
    return _summarize_case(candidate, payload, level=level, nprocs=nprocs, json_path=json_out)


def _run_stage(
    out_root: Path,
    *,
    candidates: list[Candidate],
    rep_cases: list[tuple[int, int]],
    lambda_target: float,
    resume: bool,
) -> list[dict[str, object]]:
    scored_rows: list[dict[str, object]] = []
    for candidate in candidates:
        rep_rows = [
            _run_candidate_case(
                out_root,
                candidate,
                level=level,
                nprocs=nprocs,
                lambda_target=lambda_target,
                resume=resume,
            )
            for level, nprocs in rep_cases
        ]
        scored = _score_row(candidate, rep_rows)
        candidate_md = out_root / "candidates" / candidate.stage / candidate.name / "report.md"
        candidate_md.parent.mkdir(parents=True, exist_ok=True)
        _write_candidate_markdown(candidate_md, scored)
        scored["md_path"] = _display_path(candidate_md)
        scored_rows.append(scored)
    return sorted(scored_rows, key=_score_key)


def _run_verification(
    out_root: Path,
    *,
    candidate: Candidate,
    verification_case: tuple[int, int],
    lambda_target: float,
    resume: bool,
) -> dict[str, object]:
    level, nprocs = verification_case
    leaf_dir = out_root / "verification" / candidate.name / f"l{level}_np{nprocs}"
    json_out = leaf_dir / "output.json"
    command = _candidate_command(
        candidate,
        level=level,
        nprocs=nprocs,
        lambda_target=lambda_target,
        out_json=json_out,
    )
    run_logged_command(
        command=command,
        cwd=REPO_ROOT,
        leaf_dir=leaf_dir,
        expected_outputs=[json_out],
        resume=resume,
        notes=f"Slope-stability PETSc verification case for {candidate.name}.",
    )
    payload = read_json(json_out)
    return _summarize_case(candidate, payload, level=level, nprocs=nprocs, json_path=json_out)


def _select_verified_winner(
    out_root: Path,
    final_ranked: list[dict[str, object]],
    *,
    verification_case: tuple[int, int],
    lambda_target: float,
    verify_top_k: int | None,
    resume: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    verification_rows: list[dict[str, object]] = []
    winner = dict(final_ranked[0])
    ranked_rows = list(final_ranked)
    if verify_top_k is not None:
        ranked_rows = ranked_rows[: int(verify_top_k)]
    for ranked in ranked_rows:
        candidate = Candidate(
            name=str(ranked["candidate"]),
            stage=str(ranked["stage"]),
            settings=dict(ranked["settings"]),
        )
        verification = _run_verification(
            out_root,
            candidate=candidate,
            verification_case=verification_case,
            lambda_target=lambda_target,
            resume=resume,
        )
        verification["verification_passed"] = bool(verification["solver_success"])
        verification_rows.append(verification)
        if bool(verification["solver_success"]):
            winner = dict(ranked)
            winner["verification"] = verification
            return winner, verification_rows
    winner["verification"] = verification_rows[0] if verification_rows else {}
    return winner, verification_rows


def _write_summary_markdown(path: Path, summary: dict[str, object]) -> None:
    winner = dict(summary["winner"])
    lines = [
        "# Slope-Stability PETSc Minimizer Sweep",
        "",
        "## Winner",
        "",
        f"- candidate: `{winner['candidate']}`",
        f"- stage: `{winner['stage']}`",
        f"- success_count: `{winner['success_count']}`",
        f"- geo_mean_time_s: `{winner['geo_mean_time_s']}`",
        f"- verification_passed: `{winner['verification'].get('verification_passed', False)}`",
        "",
        "```json",
        json.dumps(winner["settings"], indent=2, sort_keys=True),
        "```",
        "",
        "## Ranked Final-Stage Candidates",
        "",
        "| candidate | stage | success | geo mean [s] | linear | trust rejects | md |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["final_ranked"]:
        lines.append(
            "| {candidate} | {stage} | {success} | {geo} | {linear} | {rejects} | `{md}` |".format(
                candidate=row["candidate"],
                stage=row["stage"],
                success=row["success_count"],
                geo=row["geo_mean_time_s"],
                linear=row["total_linear_iters"],
                rejects=row["total_trust_rejects"],
                md=row.get("md_path", "-"),
            )
        )
    lines.append("")
    lines.append("## Verification")
    lines.append("")
    lines.append("| candidate | result | time [s] | Newton | Linear | omega | u_max | message |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in summary["verification_rows"]:
        lines.append(
            "| {candidate} | {result} | {time:.4f} | {nit} | {lit} | {omega:.6f} | {u_max:.6f} | {msg} |".format(
                candidate=row["candidate"],
                result=row["status"],
                time=float(row["total_time_s"]),
                nit=int(row["newton_iters"]),
                lit=int(row["linear_iters"]),
                omega=float(row["omega"]),
                u_max=float(row["u_max"]),
                msg=str(row["message"]).replace("|", "\\|"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/raw_results/slope_stability_petsc_minimizer_sweep_lambda1p2",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--lambda-target", type=float, default=1.2)
    parser.add_argument(
        "--rep-case",
        action="append",
        default=[],
        help="Representative case spec as level:nprocs; repeatable",
    )
    parser.add_argument(
        "--verification-case",
        type=str,
        default="",
        help="Verification case spec as level:nprocs",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("A", "B", "C"),
        default=["A", "B", "C"],
        help="Sweep stages to execute; winner selection uses the last executed stage",
    )
    parser.add_argument(
        "--max-candidates-per-stage",
        type=int,
        default=None,
        help="Optional cap for generated candidates in each sweep stage",
    )
    parser.add_argument(
        "--verify-top-k",
        type=int,
        default=None,
        help="Optionally verify only the top-k ranked final-stage candidates",
    )
    parser.add_argument(
        "--fallback-stage-a-count",
        type=int,
        default=3,
        help="If Stage A has zero successful candidates, seed Stage B from this many top Stage A families",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    out_root = (REPO_ROOT / args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rep_cases = [_parse_case_spec(spec) for spec in args.rep_case] or list(REP_CASES)
    verification_case = (
        _parse_case_spec(args.verification_case)
        if args.verification_case
        else VERIFICATION_CASE
    )
    max_candidates = (
        None if args.max_candidates_per_stage is None else int(args.max_candidates_per_stage)
    )
    stages = [str(stage) for stage in args.stages]
    lambda_target = float(args.lambda_target)

    stage_a_ranked: list[dict[str, object]] = []
    stage_b_ranked: list[dict[str, object]] = []
    stage_c_ranked: list[dict[str, object]] = []
    stage_b_base_candidates: list[str] = []

    if "A" in stages:
        candidates = _stage_a_candidates()
        if max_candidates is not None:
            candidates = candidates[:max_candidates]
        stage_a_ranked = _run_stage(
            out_root,
            candidates=candidates,
            rep_cases=rep_cases,
            lambda_target=lambda_target,
            resume=bool(args.resume),
        )

    stage_a_successes = [row for row in stage_a_ranked if int(row["success_count"]) > 0]
    if stage_a_successes:
        base_rows_for_stage_b = [stage_a_successes[0]]
    else:
        base_rows_for_stage_b = stage_a_ranked[: max(1, int(args.fallback_stage_a_count))]
    stage_b_base_candidates = [str(row["candidate"]) for row in base_rows_for_stage_b]
    base_for_stage_b = base_rows_for_stage_b[0] if base_rows_for_stage_b else None
    if "B" in stages and base_for_stage_b is None:
        raise ValueError("Stage B requires Stage A results")
    if "B" in stages and base_for_stage_b is not None:
        candidates: list[Candidate] = []
        for base_row in base_rows_for_stage_b:
            candidates.extend(
                _stage_b_candidates(
                    Candidate(
                        name=str(base_row["candidate"]),
                        stage="B",
                        settings=dict(base_row["settings"]),
                    )
                )
            )
        if stage_a_successes:
            stage_b_base_candidates = [str(stage_a_successes[0]["candidate"])]
        if max_candidates is not None:
            candidates = candidates[:max_candidates]
        stage_b_ranked = _run_stage(
            out_root,
            candidates=candidates,
            rep_cases=rep_cases,
            lambda_target=lambda_target,
            resume=bool(args.resume),
        )

    base_for_stage_c = stage_b_ranked[0] if stage_b_ranked else None
    if "C" in stages and base_for_stage_c is None:
        raise ValueError("Stage C requires Stage B results")
    if "C" in stages and base_for_stage_c is not None:
        best_stage_b = Candidate(
            name=str(base_for_stage_b["candidate"]),
            stage="C",
            settings=dict(base_for_stage_b["settings"]),
        )
        if bool(best_stage_b.settings["use_trust_region"]):
            candidates = _stage_c_candidates(best_stage_b)
            if max_candidates is not None:
                candidates = candidates[:max_candidates]
            stage_c_ranked = _run_stage(
                out_root,
                candidates=candidates,
                rep_cases=rep_cases,
                lambda_target=lambda_target,
                resume=bool(args.resume),
            )

    if stage_c_ranked:
        final_ranked = stage_c_ranked
    elif stage_b_ranked:
        final_ranked = stage_b_ranked
    else:
        final_ranked = stage_a_ranked
    if not final_ranked:
        raise RuntimeError("No sweep stages produced any ranked candidates")

    winner, verification_rows = _select_verified_winner(
        out_root,
        final_ranked=final_ranked,
        verification_case=verification_case,
        lambda_target=lambda_target,
        verify_top_k=args.verify_top_k,
        resume=bool(args.resume),
    )
    winner_settings_path = out_root / "winner_settings.json"
    write_json(winner_settings_path, dict(winner["settings"]))

    summary = {
        "runner": "slope_stability_petsc_minimizer_sweep",
        "representative_cases": [
            {"level": int(level), "nprocs": int(nprocs)} for level, nprocs in rep_cases
        ],
        "verification_case": {
            "level": int(verification_case[0]),
            "nprocs": int(verification_case[1]),
        },
        "lambda_target": lambda_target,
        "stage_a_ranked": stage_a_ranked,
        "stage_b_base_candidates": stage_b_base_candidates,
        "stage_b_ranked": stage_b_ranked,
        "stage_c_ranked": stage_c_ranked,
        "final_ranked": final_ranked,
        "verification_rows": verification_rows,
        "winner": winner,
        "winner_settings_path": _display_path(winner_settings_path),
    }
    write_json(out_root / "summary.json", summary)
    _write_summary_markdown(out_root / "summary.md", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
