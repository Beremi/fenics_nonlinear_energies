#!/usr/bin/env python3
"""Generate a markdown report for the slope-stability JAX+PETSc sweep and suite."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _fmt_float(value: object, digits: int = 6) -> str:
    if value is None:
        return "-"
    val = float(value)
    if not math.isfinite(val):
        return str(val)
    return f"{val:.{digits}f}"


def _solve_other_time(row: dict[str, object]) -> float:
    solve = float(row.get("solve_time_s", 0.0))
    assembly = float(row.get("assembly_time_s", 0.0))
    pc = float(row.get("pc_init_time_s", 0.0))
    ksp = float(row.get("ksp_solve_time_s", 0.0))
    return max(0.0, solve - assembly - pc - ksp)


def _solve_share(value: object, row: dict[str, object]) -> str:
    solve = float(row.get("solve_time_s", 0.0))
    if solve <= 0.0:
        return "-"
    return f"{100.0 * float(value) / solve:.1f}%"


def _write_mesh_progression_plot(rows: list[dict[str, object]], out_path: Path) -> list[dict[str, object]]:
    mesh_rows = [
        row for row in rows if int(row["nprocs"]) == 1 and str(row["result"]) == "completed"
    ]
    mesh_rows.sort(key=lambda row: int(row["level"]))
    if not mesh_rows:
        return []

    free_dofs = [int(row["free_dofs"]) for row in mesh_rows]
    total_time = [float(row["total_time_s"]) for row in mesh_rows]
    solve_time = [float(row["solve_time_s"]) for row in mesh_rows]

    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.plot(free_dofs, total_time, marker="o", linewidth=2.0, label="total time")
    ax.plot(free_dofs, solve_time, marker="s", linewidth=2.0, label="solve time")
    for row, x, y in zip(mesh_rows, free_dofs, total_time):
        ax.annotate(f"L{row['level']}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Free DOFs")
    ax.set_ylabel("Time [s]")
    ax.set_title("Mesh Progression at 1 MPI Rank")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return mesh_rows


def _write_strong_scaling_plots(rows: list[dict[str, object]], out_dir: Path) -> tuple[Path, Path, list[int]]:
    eligible_levels = sorted(
        {
            int(row["level"])
            for row in rows
            if str(row["result"]) == "completed"
            and int(row["nprocs"]) > 1
        }
    )
    total_plot = out_dir / "strong_scaling_total_time.png"
    speedup_plot = out_dir / "strong_scaling_speedup.png"
    if not eligible_levels:
        return total_plot, speedup_plot, []

    fig_total, ax_total = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    fig_speedup, ax_speedup = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)

    for level in eligible_levels:
        level_rows = [
            row
            for row in rows
            if int(row["level"]) == level and str(row["result"]) == "completed"
        ]
        level_rows.sort(key=lambda row: int(row["nprocs"]))
        if len(level_rows) < 2:
            continue
        nprocs = [int(row["nprocs"]) for row in level_rows]
        total_time = [float(row["total_time_s"]) for row in level_rows]
        baseline_time = total_time[0]
        speedup = [baseline_time / value if value > 0.0 else float("nan") for value in total_time]
        ax_total.plot(nprocs, total_time, marker="o", linewidth=2.0, label=f"L{level}")
        ax_speedup.plot(nprocs, speedup, marker="o", linewidth=2.0, label=f"L{level}")

    all_nprocs = sorted({int(row["nprocs"]) for row in rows if int(row["nprocs"]) > 0})
    if all_nprocs:
        ax_speedup.plot(all_nprocs, all_nprocs, linestyle="--", color="black", alpha=0.4, label="ideal")

    ax_total.set_xscale("log", base=2)
    ax_total.set_yscale("log")
    ax_total.set_xlabel("MPI ranks")
    ax_total.set_ylabel("Total time [s]")
    ax_total.set_title("Strong Scaling by Mesh Level")
    ax_total.grid(True, which="both", alpha=0.3)
    ax_total.legend()
    fig_total.savefig(total_plot, dpi=180)
    plt.close(fig_total)

    ax_speedup.set_xscale("log", base=2)
    ax_speedup.set_yscale("log", base=2)
    ax_speedup.set_xlabel("MPI ranks")
    ax_speedup.set_ylabel("Speedup vs first completed point")
    ax_speedup.set_title("Strong Scaling Speedup")
    ax_speedup.grid(True, which="both", alpha=0.3)
    ax_speedup.legend()
    fig_speedup.savefig(speedup_plot, dpi=180)
    plt.close(fig_speedup)
    return total_plot, speedup_plot, eligible_levels


def _write_report(
    out_path: Path,
    *,
    sweep: dict[str, object],
    suite: dict[str, object],
    sweep_summary_path: Path,
    suite_summary_path: Path,
    mesh_rows: list[dict[str, object]],
    mesh_plot: Path,
    strong_total_plot: Path,
    strong_speedup_plot: Path,
    eligible_levels: list[int],
) -> None:
    winner = dict(sweep["winner"])
    final_ranked = list(sweep["final_ranked"])
    verification_rows = list(sweep["verification_rows"])
    rows = list(suite["rows"])
    rows.sort(key=lambda row: (int(row["level"]), int(row["nprocs"])))

    lines = [
        "# Slope-Stability JAX+PETSc Benchmark Report",
        "",
        f"- lambda target: `{suite['lambda_target']}`",
        f"- sweep summary: `{_repo_rel(sweep_summary_path)}`",
        f"- suite summary: `{_repo_rel(suite_summary_path)}`",
        f"- winner settings: `{sweep.get('winner_settings_path', '-')}`",
    ]
    lines.extend(
        [
            "",
            "## Frozen Winner",
            "",
            f"- candidate: `{winner['candidate']}`",
            f"- stage: `{winner['stage']}`",
            f"- success_count: `{winner['success_count']}`",
            f"- geo_mean_time_s: `{winner['geo_mean_time_s']}`",
            f"- verification_passed: `{winner.get('verification', {}).get('verification_passed', False)}`",
            "",
            "```json",
            json.dumps(winner["settings"], indent=2, sort_keys=True),
            "```",
            "",
            "## Sweep Ranking",
            "",
            "| candidate | stage | success | geo mean [s] | linear iters | trust rejects | md |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in final_ranked:
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

    lines.extend(
        [
            "",
            "## Verification Case",
            "",
            "| candidate | result | time [s] | Newton | Linear | omega | u_max | message |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in verification_rows:
        lines.append(
            "| {candidate} | {result} | {time} | {nit} | {lit} | {omega} | {u_max} | {msg} |".format(
                candidate=row["candidate"],
                result=row["status"],
                time=_fmt_float(row["total_time_s"], 4),
                nit=row["newton_iters"],
                lit=row["linear_iters"],
                omega=_fmt_float(row["omega"], 6),
                u_max=_fmt_float(row["u_max"], 6),
                msg=str(row["message"]).replace("|", "\\|"),
            )
        )

    lines.extend(
        [
            "",
            "## Mesh Progression at 1 Rank",
            "",
            f"![Mesh progression]({mesh_plot.name})",
            "",
            "| level | h | elements | free DOFs | total [s] | solve [s] | energy | omega | u_max |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in mesh_rows:
        lines.append(
            "| {level} | {h} | {elements} | {free_dofs} | {total} | {solve} | {energy} | {omega} | {u_max} |".format(
                level=row["level"],
                h=_fmt_float(row["h"], 2),
                elements=row["elements"],
                free_dofs=row["free_dofs"],
                total=_fmt_float(row["total_time_s"], 4),
                solve=_fmt_float(row["solve_time_s"], 4),
                energy=_fmt_float(row["final_energy"], 6),
                omega=_fmt_float(row["omega"], 6),
                u_max=_fmt_float(row["u_max"], 6),
            )
        )

    lines.extend(
        [
            "",
            "## Strong Scaling",
            "",
            f"Eligible levels: `{eligible_levels}`",
            "",
            f"![Strong scaling total time]({strong_total_plot.name})",
            "",
            f"![Strong scaling speedup]({strong_speedup_plot.name})",
        ]
    )
    for level in eligible_levels:
        level_rows = [row for row in rows if int(row["level"]) == level]
        lines.extend(
            [
                "",
                f"### Level {level}",
                "",
                "| np | result | total [s] | assembly [s] | PC [s] | KSP [s] | energy | omega | u_max |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in level_rows:
            lines.append(
                "| {nprocs} | {result} | {total} | {assembly} | {pc} | {ksp} | {energy} | {omega} | {u_max} |".format(
                    nprocs=row["nprocs"],
                    result=row["result"],
                    total=_fmt_float(row["total_time_s"], 4),
                    assembly=_fmt_float(row["assembly_time_s"], 4),
                    pc=_fmt_float(row["pc_init_time_s"], 4),
                    ksp=_fmt_float(row["ksp_solve_time_s"], 4),
                    energy=_fmt_float(row["final_energy"], 6),
                    omega=_fmt_float(row["omega"], 6),
                    u_max=_fmt_float(row["u_max"], 6),
                )
            )

    lines.extend(
        [
            "",
            "## Iterations And Timing Breakdown",
            "",
            "| level | np | result | Newton | Linear | setup [s] | solve [s] | assembly [s] | assembly % | PC [s] | PC % | KSP [s] | KSP % | other solve [s] | other % | message |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        other = _solve_other_time(row)
        lines.append(
            "| {level} | {nprocs} | {result} | {nit} | {lit} | {setup} | {solve} | {assembly} | {assembly_share} | {pc} | {pc_share} | {ksp} | {ksp_share} | {other} | {other_share} | {message} |".format(
                level=row["level"],
                nprocs=row["nprocs"],
                result=row["result"],
                nit=row["newton_iters"],
                lit=row["linear_iters"],
                setup=_fmt_float(row["setup_time_s"], 4),
                solve=_fmt_float(row["solve_time_s"], 4),
                assembly=_fmt_float(row["assembly_time_s"], 4),
                assembly_share=_solve_share(row["assembly_time_s"], row),
                pc=_fmt_float(row["pc_init_time_s"], 4),
                pc_share=_solve_share(row["pc_init_time_s"], row),
                ksp=_fmt_float(row["ksp_solve_time_s"], 4),
                ksp_share=_solve_share(row["ksp_solve_time_s"], row),
                other=_fmt_float(other, 4),
                other_share=_solve_share(other, row),
                message=str(row["message"]).replace("|", "\\|"),
            )
        )

    lines.extend(
        [
            "",
            "## Final Response Table",
            "",
            "| level | np | elements | free DOFs | result | energy | omega | u_max | Newton | Linear | total [s] | assembly [s] | PC [s] | KSP [s] | trust rejects | json | md |",
            "| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {level} | {nprocs} | {elements} | {free_dofs} | {result} | {energy} | {omega} | {u_max} | {nit} | {lit} | {total} | {assembly} | {pc} | {ksp} | {rejects} | `{json_path}` | `{md_path}` |".format(
                level=row["level"],
                nprocs=row["nprocs"],
                elements=row["elements"],
                free_dofs=row["free_dofs"],
                result=row["result"],
                energy=_fmt_float(row["final_energy"], 6),
                omega=_fmt_float(row["omega"], 6),
                u_max=_fmt_float(row["u_max"], 6),
                nit=row["newton_iters"],
                lit=row["linear_iters"],
                total=_fmt_float(row["total_time_s"], 4),
                assembly=_fmt_float(row["assembly_time_s"], 4),
                pc=_fmt_float(row["pc_init_time_s"], 4),
                ksp=_fmt_float(row["ksp_solve_time_s"], 4),
                rejects=row["trust_rejects"],
                json_path=row["json_path"],
                md_path=row["md_path"],
            )
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-summary", required=True, type=str)
    parser.add_argument("--suite-summary", required=True, type=str)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/reports/slope_stability_petsc_benchmarks",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep = _load_json(args.sweep_summary)
    suite = _load_json(args.suite_summary)
    rows = list(suite["rows"])

    mesh_plot = out_dir / "mesh_progression_np1.png"
    strong_total_plot = out_dir / "strong_scaling_total_time.png"
    strong_speedup_plot = out_dir / "strong_scaling_speedup.png"
    mesh_rows = _write_mesh_progression_plot(rows, mesh_plot)
    strong_total_plot, strong_speedup_plot, eligible_levels = _write_strong_scaling_plots(
        rows, out_dir
    )

    report_path = out_dir / "report.md"
    _write_report(
        report_path,
        sweep=sweep,
        suite=suite,
        sweep_summary_path=Path(args.sweep_summary).resolve(),
        suite_summary_path=Path(args.suite_summary).resolve(),
        mesh_rows=mesh_rows,
        mesh_plot=mesh_plot,
        strong_total_plot=strong_total_plot,
        strong_speedup_plot=strong_speedup_plot,
        eligible_levels=eligible_levels,
    )
    print(json.dumps({"report": _repo_rel(report_path)}, indent=2))


if __name__ == "__main__":
    main()
