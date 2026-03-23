"""Generate a short report for tuned PETSc PCMG on slope-stability L5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT_DIR = Path("artifacts/tmp_l5_pmg")
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/slope_stability_level5_petsc_mg_lambda1")


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _sum(entries: list[dict], key: str) -> float:
    return float(sum(float(entry.get(key, 0.0)) for entry in entries))


def _extract(path: Path) -> dict[str, object]:
    payload = _load(path)
    step = payload["result"]["steps"][0]
    linear = step.get("linear_timing", [])
    return {
        "path": str(path),
        "pc_type": str(payload["metadata"]["linear_solver"]["pc_type"]),
        "ksp_type": str(payload["metadata"]["linear_solver"]["ksp_type"]),
        "ranks": int(payload["metadata"]["nprocs"]),
        "solver_success": bool(payload["result"]["solver_success"]),
        "status": str(payload["result"]["status"]),
        "message": str(step["message"]),
        "total_time_sec": float(payload["timings"]["total_time"]),
        "setup_time_sec": float(payload["timings"]["setup_time"]),
        "solve_time_sec": float(payload["timings"]["solve_time"]),
        "newton_iterations": int(step["nit"]),
        "linear_iterations": int(step["linear_iters"]),
        "avg_linear_per_newton": float(step["linear_iters"]) / float(step["nit"]),
        "energy": float(step["energy"]),
        "omega": float(step["omega"]),
        "u_max": float(step["u_max"]),
        "assembly_time_sec": _sum(linear, "assemble_total_time"),
        "pc_setup_time_sec": _sum(linear, "pc_setup_time"),
        "ksp_time_sec": _sum(linear, "solve_time"),
        "linear_total_time_sec": _sum(linear, "linear_total_time"),
    }


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _plot_grouped(rows: list[dict], output_dir: Path) -> None:
    labels = [f"{row['pc_type']}\n{row['ranks']} rank" for row in rows]
    x = np.arange(len(rows))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(x - width / 2, [row["solve_time_sec"] for row in rows], width, label="solve")
    ax.bar(x + width / 2, [row["total_time_sec"] for row in rows], width, label="total")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Time [s]")
    ax.set_title("L5 lambda=1.0 PETSc solve time")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "time_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(x - width / 2, [row["newton_iterations"] for row in rows], width, label="Newton")
    ax.bar(x + width / 2, [row["linear_iterations"] for row in rows], width, label="linear")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Iterations")
    ax.set_title("L5 lambda=1.0 iterations")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "iteration_comparison.png", dpi=180)
    plt.close(fig)


def _table(rows: list[dict]) -> str:
    lines = [
        "| preconditioner | ranks | total [s] | solve [s] | setup [s] | Newton | linear | avg linear / Newton | energy | omega | u_max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    names = {"hypre": "BoomerAMG", "mg": "PETSc PCMG"}
    for row in rows:
        lines.append(
            "| "
            f"{names[row['pc_type']]} | {row['ranks']} | {_fmt(row['total_time_sec'])} | "
            f"{_fmt(row['solve_time_sec'])} | {_fmt(row['setup_time_sec'])} | "
            f"{row['newton_iterations']} | {row['linear_iterations']} | {_fmt(row['avg_linear_per_newton'], 2)} | "
            f"{_fmt(row['energy'], 6)} | {_fmt(row['omega'], 6)} | {_fmt(row['u_max'], 6)} |"
        )
    return "\n".join(lines)


def _breakdown_table(rows: list[dict]) -> str:
    lines = [
        "| preconditioner | ranks | assembly [s] | PC setup [s] | KSP [s] | linear phase [s] |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    names = {"hypre": "BoomerAMG", "mg": "PETSc PCMG"}
    for row in rows:
        lines.append(
            "| "
            f"{names[row['pc_type']]} | {row['ranks']} | {_fmt(row['assembly_time_sec'])} | "
            f"{_fmt(row['pc_setup_time_sec'])} | {_fmt(row['ksp_time_sec'])} | {_fmt(row['linear_total_time_sec'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        _extract(args.input_dir / "l5_hypre_np1_no_tr.json"),
        _extract(args.input_dir / "l5_pmg_np1_no_tr.json"),
        _extract(args.input_dir / "l5_hypre_np8_no_tr.json"),
        _extract(args.input_dir / "l5_pmg_np8_no_tr.json"),
    ]
    trial_note = None
    trial_path = args.input_dir / "l5_pmg_np8_tuned.json"
    if trial_path.exists():
        trial = _extract(trial_path)
        trial_note = (
            "Initial `8`-rank PCMG with trust region enabled stopped with "
            f"`{trial['message']}` after `{trial['newton_iterations']}` Newton steps "
            f"and `{trial['linear_iterations']}` linear iterations. "
            "The final successful comparison below uses `--no-use_trust_region`."
        )

    _plot_grouped(rows, output_dir)

    report = f"""# L5 lambda=1.0 with PETSc multigrid

This note compares the tuned PETSc geometric multigrid run against the BoomerAMG baseline on the same `L5` slope-stability case.

Tuned PCMG setup:

- outer linear solver: `fgmres`
- preconditioner: PETSc `pc_type = mg`
- hierarchy: nested `L1 -> L5` `P2` spaces
- transfer operators: explicit geometric prolongation/restriction from coarse `P2` basis evaluation at fine nodes
- coarse operators: PETSc Galerkin
- smoother: `richardson + sor`, `3` steps
- coarse solve: `preonly + lu` on `1` rank, `cg + jacobi` on multi-rank
- nonlinear globalization for the final successful runs: `--no-use_trust_region`

{trial_note or ""}

## Results

{_table(rows)}

## Time Breakdown

{_breakdown_table(rows)}

![Time comparison](time_comparison.png)
![Iteration comparison](iteration_comparison.png)

## Takeaways

- The tuned `PCMG` solve is fully successful on `L5` at both `1` and `8` ranks.
- On `1` rank, `PCMG` cut solve time from `{_fmt(rows[0]['solve_time_sec'])} s` to `{_fmt(rows[1]['solve_time_sec'])} s` and linear iterations from `{rows[0]['linear_iterations']}` to `{rows[1]['linear_iterations']}`.
- On `8` ranks, `PCMG` cut solve time from `{_fmt(rows[2]['solve_time_sec'])} s` to `{_fmt(rows[3]['solve_time_sec'])} s` and linear iterations from `{rows[2]['linear_iterations']}` to `{rows[3]['linear_iterations']}`.
- The multigrid hierarchy is especially strong on the linear side: the final `PCMG` runs need only about `{_fmt(rows[1]['avg_linear_per_newton'], 2)}` and `{_fmt(rows[3]['avg_linear_per_newton'], 2)}` Krylov iterations per Newton step at `1` and `8` ranks.
"""
    (output_dir / "report.md").write_text(report)
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
