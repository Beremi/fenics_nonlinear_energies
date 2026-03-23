#!/usr/bin/env python3
"""Run the final slope-stability JAX+PETSc benchmark suite at lambda=1.2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.benchmark.replication import read_json, run_logged_command, write_json
from src.problems.slope_stability.support import DEFAULT_LEVEL, build_case_data


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
SOLVER = "src/problems/slope_stability/jax_petsc/solve_slope_stability_dof.py"


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _bool_flag(name: str, value: bool) -> list[str]:
    return [f"--{name}" if value else f"--no-{name}"]


def _default_settings() -> dict[str, object]:
    return {
        "profile": "performance",
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1.0e-2,
        "ksp_max_it": 50,
        "pc_setup_on_ksp_cap": False,
        "gamg_threshold": 0.05,
        "gamg_agg_nsmooths": 1,
        "gamg_set_coordinates": True,
        "use_near_nullspace": True,
        "assembly_mode": "element",
        "element_reorder_mode": "block_xyz",
        "local_hessian_mode": "element",
        "local_coloring": True,
        "use_trust_region": True,
        "trust_subproblem_line_search": True,
        "trust_radius_init": 0.5,
        "trust_radius_min": 1.0e-8,
        "trust_radius_max": 1.0e6,
        "trust_shrink": 0.5,
        "trust_expand": 1.5,
        "trust_eta_shrink": 0.05,
        "trust_eta_expand": 0.75,
        "trust_max_reject": 6,
        "linesearch_a": -0.5,
        "linesearch_b": 2.0,
        "linesearch_tol": 1.0e-1,
        "retry_on_failure": False,
        "tolf": 1.0e-4,
        "tolg": 1.0e-3,
        "tolg_rel": 1.0e-3,
        "tolx_rel": 1.0e-3,
        "tolx_abs": 1.0e-10,
        "maxit": 100,
        "step_time_limit_s": 60.0,
        "reg": 1.0e-12,
        "nproc": 1,
    }


def _settings_from_json(path: str | None) -> dict[str, object]:
    settings = _default_settings()
    if not path:
        return settings
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    settings.update(loaded)
    return settings


def _rank_allowed(level: int, nprocs: int, min_elements_per_rank: float) -> bool:
    case = build_case_data(f"ssr_homo_capture_p2_level{int(level)}")
    return float(case.n_elements) / float(nprocs) >= float(min_elements_per_rank)


def _case_command(
    *,
    level: int,
    nprocs: int,
    lambda_target: float,
    settings: dict[str, object],
    out_json: Path,
) -> list[str]:
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
        str(settings["profile"]),
        "--ksp_type",
        str(settings["ksp_type"]),
        "--pc_type",
        str(settings["pc_type"]),
        "--ksp_rtol",
        str(settings["ksp_rtol"]),
        "--ksp_max_it",
        str(settings["ksp_max_it"]),
        "--assembly_mode",
        str(settings["assembly_mode"]),
        "--element_reorder_mode",
        str(settings["element_reorder_mode"]),
        "--local_hessian_mode",
        str(settings["local_hessian_mode"]),
        "--nproc",
        str(settings["nproc"]),
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
        "--linesearch_a",
        str(settings["linesearch_a"]),
        "--linesearch_b",
        str(settings["linesearch_b"]),
        "--linesearch_tol",
        str(settings["linesearch_tol"]),
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
    if settings.get("step_time_limit_s") is not None:
        cmd += ["--step_time_limit_s", str(settings["step_time_limit_s"])]
    cmd += _bool_flag("local_coloring", bool(settings["local_coloring"]))
    cmd += _bool_flag("gamg_set_coordinates", bool(settings["gamg_set_coordinates"]))
    cmd += _bool_flag("use_near_nullspace", bool(settings["use_near_nullspace"]))
    cmd += _bool_flag("pc_setup_on_ksp_cap", bool(settings["pc_setup_on_ksp_cap"]))
    cmd += _bool_flag("use_trust_region", bool(settings["use_trust_region"]))
    cmd += _bool_flag("trust_subproblem_line_search", bool(settings["trust_subproblem_line_search"]))
    cmd += _bool_flag("retry_on_failure", bool(settings["retry_on_failure"]))
    return cmd


def _write_case_markdown(path: Path, payload: dict, row: dict[str, object], settings: dict[str, object]) -> None:
    lines = [
        f"# level {row['level']} np {row['nprocs']}",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| level | `{row['level']}` |",
        f"| h | `{row['h']}` |",
        f"| nprocs | `{row['nprocs']}` |",
        f"| nodes | `{row['nodes']}` |",
        f"| elements | `{row['elements']}` |",
        f"| free DOFs | `{row['free_dofs']}` |",
        f"| result | `{row['result']}` |",
        f"| final energy | `{row['final_energy']}` |",
        f"| omega | `{row['omega']}` |",
        f"| u_max | `{row['u_max']}` |",
        f"| Newton | `{row['newton_iters']}` |",
        f"| linear | `{row['linear_iters']}` |",
        f"| total time [s] | `{row['total_time_s']}` |",
        f"| assembly [s] | `{row['assembly_time_s']}` |",
        f"| PC init [s] | `{row['pc_init_time_s']}` |",
        f"| KSP solve [s] | `{row['ksp_solve_time_s']}` |",
        "",
        "## Solver Settings",
        "",
        "```json",
        json.dumps(settings, indent=2, sort_keys=True),
        "```",
        "",
        "## Raw Payload",
        "",
        "```json",
        json.dumps(payload, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summarize_case(payload: dict, *, json_path: Path, md_path: Path) -> dict[str, object]:
    step = dict(payload["result"]["steps"][0])
    linear_timing = list(step.get("linear_timing", []))
    history = list(step.get("history", []))
    return {
        "level": int(payload["mesh"]["level"]),
        "h": float(payload["mesh"]["h"]),
        "nprocs": int(payload["metadata"]["nprocs"]),
        "nodes": int(payload["mesh"]["nodes"]),
        "elements": int(payload["mesh"]["elements"]),
        "free_dofs": int(payload["mesh"]["free_dofs"]),
        "result": str(payload["result"]["status"]),
        "final_energy": float(step["energy"]),
        "omega": float(step["omega"]),
        "u_max": float(step["u_max"]),
        "newton_iters": int(step["nit"]),
        "linear_iters": int(step["linear_iters"]),
        "setup_time_s": float(payload["timings"]["setup_time"]),
        "solve_time_s": float(payload["timings"]["solve_time"]),
        "total_time_s": float(payload["timings"]["total_time"]),
        "assembly_time_s": float(sum(float(rec.get("assemble_total_time", 0.0)) for rec in linear_timing)),
        "pc_init_time_s": float(sum(float(rec.get("pc_setup_time", 0.0)) for rec in linear_timing)),
        "ksp_solve_time_s": float(sum(float(rec.get("solve_time", 0.0)) for rec in linear_timing)),
        "trust_rejects": int(sum(int(rec.get("trust_rejects", 0)) for rec in history)),
        "message": str(step["message"]),
        "json_path": _display_path(json_path),
        "md_path": _display_path(md_path),
    }


def _write_summary_markdown(path: Path, rows: list[dict[str, object]], settings: dict[str, object]) -> None:
    lines = [
        "# Slope-Stability PETSc Final Suite",
        "",
        "## Frozen Settings",
        "",
        "```json",
        json.dumps(settings, indent=2, sort_keys=True),
        "```",
        "",
        "## Cases",
        "",
        "| level | h | np | elements | free DOFs | result | energy | omega | u_max | total [s] | json | md |",
        "| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {level} | {h:.2f} | {nprocs} | {elements} | {free_dofs} | {result} | {energy:.6f} | {omega:.6f} | {u_max:.6f} | {time:.4f} | `{json_path}` | `{md_path}` |".format(
                level=row["level"],
                h=float(row["h"]),
                nprocs=row["nprocs"],
                elements=row["elements"],
                free_dofs=row["free_dofs"],
                result=row["result"],
                energy=float(row["final_energy"]),
                omega=float(row["omega"]),
                u_max=float(row["u_max"]),
                time=float(row["total_time_s"]),
                json_path=row["json_path"],
                md_path=row["md_path"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--nprocs", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--lambda-target", type=float, default=1.2)
    parser.add_argument("--min-elements-per-rank", type=float, default=40.0)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/raw_results/slope_stability_petsc_final_suite_lambda1p2",
    )
    parser.add_argument("--settings-json", type=str, default="")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    out_root = (REPO_ROOT / args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    settings = _settings_from_json(args.settings_json or None)
    lambda_target = float(args.lambda_target)
    min_elements_per_rank = float(args.min_elements_per_rank)

    rows: list[dict[str, object]] = []
    for level in args.levels:
        for nprocs in args.nprocs:
            if not _rank_allowed(level, nprocs, min_elements_per_rank):
                continue
            leaf_dir = out_root / f"l{level}_np{nprocs}"
            json_out = leaf_dir / "output.json"
            md_out = leaf_dir / "report.md"
            command = _case_command(
                level=level,
                nprocs=nprocs,
                lambda_target=lambda_target,
                settings=settings,
                out_json=json_out,
            )
            run_logged_command(
                command=command,
                cwd=REPO_ROOT,
                leaf_dir=leaf_dir,
                expected_outputs=[json_out],
                resume=bool(args.resume),
                notes=f"Slope-stability PETSc final-suite case level {level} np {nprocs}.",
            )
            payload = read_json(json_out)
            row = _summarize_case(payload, json_path=json_out, md_path=md_out)
            _write_case_markdown(md_out, payload, row, settings)
            rows.append(row)
            rows.sort(key=lambda item: (int(item["level"]), int(item["nprocs"])))
            write_json(
                out_root / "summary.json",
                {
                    "runner": "slope_stability_petsc_final_suite",
                    "lambda_target": lambda_target,
                    "min_elements_per_rank": min_elements_per_rank,
                    "settings": settings,
                    "rows": rows,
                },
            )
            _write_summary_markdown(out_root / "summary.md", rows, settings)
            print(json.dumps(row, indent=2))

    summary = {
        "runner": "slope_stability_petsc_final_suite",
        "lambda_target": lambda_target,
        "min_elements_per_rank": min_elements_per_rank,
        "settings": settings,
        "rows": rows,
    }
    write_json(out_root / "summary.json", summary)
    _write_summary_markdown(out_root / "summary.md", rows, settings)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
