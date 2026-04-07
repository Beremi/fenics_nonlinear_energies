from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


BASELINE_ROOT = Path(
    os.environ.get(
        "P4_L1_2_BASELINE_ROOT",
        "artifacts/raw_results/scaling_probe/p4_l1_2_uniform_tail_maxit1_threads1",
    )
)
OPT_ROOT = Path(
    os.environ.get(
        "P4_L1_2_OPT_ROOT",
        "artifacts/raw_results/assembly_opt_ladder/coo_chunk4_sweep",
    )
)
OUTDIR = Path(
    os.environ.get(
        "P4_L1_2_OPT_COMPARE_OUTDIR",
        str(OPT_ROOT / "compare_assets"),
    )
)
REPORT = Path(
    os.environ.get(
        "P4_L1_2_OPT_COMPARE_REPORT",
        str(OPT_ROOT / "COMPARE.md"),
    )
)

RANKS = (1, 2, 4, 8, 16, 32)
CHUNK_NAMES = ("coo_chunk2", "coo_chunk4", "coo_chunk8", "coo_chunk16", "coo_chunk32", "coo_chunk64")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _first_linear(obj: dict) -> dict:
    return obj["parallel_diagnostics"][0]["linear_history"][0]


def _phase_values(obj: dict) -> dict[str, float]:
    pd = obj["parallel_diagnostics"]
    return {
        "total_time": float(obj["total_time"]),
        "solve_time": float(obj["solve_time"]),
        "problem_load": max(float(rank["stage_timings"].get("problem_load", 0.0)) for rank in pd),
        "assembler_create": max(float(rank["stage_timings"].get("assembler_create", 0.0)) for rank in pd),
        "mg_hierarchy_build": max(float(rank["stage_timings"].get("mg_hierarchy_build", 0.0)) for rank in pd),
        "initial_guess_total": max(float(rank["stage_timings"].get("initial_guess_total", 0.0)) for rank in pd),
        "hessian_hvp": max(float(rank["assembly_callbacks"]["hessian"]["hvp_compute"]) for rank in pd),
        "hessian_extraction": max(float(rank["assembly_callbacks"]["hessian"]["extraction"]) for rank in pd),
        "hessian_coo": max(float(rank["assembly_callbacks"]["hessian"]["coo_assembly"]) for rank in pd),
        "linear1_t_assemble": max(float(rank["linear_history"][0]["t_assemble"]) for rank in pd),
        "linear1_t_setup": max(float(rank["linear_history"][0]["t_setup"]) for rank in pd),
        "linear1_t_solve": max(float(rank["linear_history"][0]["t_solve"]) for rank in pd),
    }


def _collect_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for np_ranks in RANKS:
        obj = _load_json(root / f"np{np_ranks}" / "output.json")
        row = {"np": float(np_ranks), "path": str(root / f"np{np_ranks}" / "output.json")}
        row.update(_phase_values(obj))
        rows.append(row)
    base_total = rows[0]["total_time"]
    base_solve = rows[0]["solve_time"]
    for row in rows:
        row["speedup"] = base_total / row["total_time"]
        row["efficiency"] = row["speedup"] / row["np"]
        row["solve_speedup"] = base_solve / row["solve_time"]
        row["solve_efficiency"] = row["solve_speedup"] / row["np"]
    return rows


def _chunk_screen_rows() -> list[dict]:
    rows: list[dict] = []
    for name in CHUNK_NAMES:
        path = Path("artifacts/raw_results/assembly_opt_ladder") / name / "np32" / "output.json"
        if not path.exists():
            continue
        obj = _load_json(path)
        lin = _first_linear(obj)
        rows.append(
            {
                "name": name,
                "chunk": int(obj["p4_hessian_chunk_size"]),
                "total_time": float(obj["total_time"]),
                "solve_time": float(obj["solve_time"]),
                "linear1_t_assemble": float(lin["t_assemble"]),
                "linear1_t_setup": float(lin["t_setup"]),
                "linear1_t_solve": float(lin["t_solve"]),
                "hessian_hvp": float(obj["assembly_callbacks"]["hessian"]["hvp_compute"]),
                "hessian_extraction": float(obj["assembly_callbacks"]["hessian"]["extraction"]),
                "hessian_coo": float(obj["assembly_callbacks"]["hessian"]["coo_assembly"]),
            }
        )
    rows.sort(key=lambda row: row["chunk"])
    return rows


def _save(fig: plt.Figure, name: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_scaling_compare(baseline: list[dict], opt: list[dict]) -> None:
    ranks = np.array([row["np"] for row in baseline], dtype=float)
    labels = [str(int(v)) for v in ranks]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ax = axes[0]
    for rows, label, color in (
        (baseline, "baseline total", "#AA3377"),
        (opt, "optimized total", "#117733"),
    ):
        vals = np.array([row["total_time"] for row in rows], dtype=float)
        ax.loglog(ranks, vals, marker="o", linewidth=2.0, label=label, color=color)
    ax.loglog(ranks, np.array([baseline[0]["total_time"] / r for r in ranks]), linestyle="--", color="0.35", label="ideal 1/r")
    ax.set_title("Total Time Scaling")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Time [s]")
    ax.set_xticks(ranks, labels)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    for rows, label, color in (
        (baseline, "baseline solve", "#CC6677"),
        (opt, "optimized solve", "#44AA99"),
    ):
        vals = np.array([row["solve_time"] for row in rows], dtype=float)
        ax.loglog(ranks, vals, marker="s", linewidth=2.0, label=label, color=color)
    ax.loglog(ranks, np.array([baseline[0]["solve_time"] / r for r in ranks]), linestyle="--", color="0.35", label="ideal 1/r")
    ax.set_title("Solve Time Scaling")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Time [s]")
    ax.set_xticks(ranks, labels)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)

    _save(fig, "scaling_compare.png")


def _plot_improvement(baseline: list[dict], opt: list[dict]) -> None:
    ranks = np.array([row["np"] for row in baseline], dtype=float)
    labels = [str(int(v)) for v in ranks]
    total_gain = 100.0 * (
        np.array([b["total_time"] for b in baseline], dtype=float)
        - np.array([o["total_time"] for o in opt], dtype=float)
    ) / np.array([b["total_time"] for b in baseline], dtype=float)
    solve_gain = 100.0 * (
        np.array([b["solve_time"] for b in baseline], dtype=float)
        - np.array([o["solve_time"] for o in opt], dtype=float)
    ) / np.array([b["solve_time"] for b in baseline], dtype=float)
    assemble_gain = 100.0 * (
        np.array([b["linear1_t_assemble"] for b in baseline], dtype=float)
        - np.array([o["linear1_t_assemble"] for o in opt], dtype=float)
    ) / np.array([b["linear1_t_assemble"] for b in baseline], dtype=float)

    x = np.arange(len(ranks), dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    width = 0.24
    ax.bar(x - width, total_gain, width=width, label="total time")
    ax.bar(x, solve_gain, width=width, label="solve time")
    ax.bar(x + width, assemble_gain, width=width, label="first linear assemble")
    ax.axhline(0.0, color="0.2", linewidth=1.0)
    ax.set_xticks(x, labels)
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Improvement vs baseline [%]")
    ax.set_title("Optimized-vs-Baseline Gains")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, "improvement_bars.png")


def _plot_phase32_compare(baseline: list[dict], opt: list[dict]) -> None:
    b = baseline[-1]
    o = opt[-1]
    phases = [
        ("problem_load", "problem load"),
        ("assembler_create", "assembler create"),
        ("mg_hierarchy_build", "MG hierarchy"),
        ("hessian_hvp", "Hessian HVP"),
        ("hessian_extraction", "Hessian extraction"),
        ("linear1_t_assemble", "1st assemble"),
        ("linear1_t_setup", "1st KSP setup"),
        ("linear1_t_solve", "1st KSP solve"),
    ]
    labels = [label for _, label in phases]
    baseline_vals = np.array([b[key] for key, _ in phases], dtype=float)
    opt_vals = np.array([o[key] for key, _ in phases], dtype=float)
    y = np.arange(len(phases))
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.barh(y + 0.18, baseline_vals, height=0.34, label="baseline", color="#CC6677")
    ax.barh(y - 0.18, opt_vals, height=0.34, label="optimized", color="#44AA99")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Time [s]")
    ax.set_title("32-Rank Phase Comparison")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, "phase32_compare.png")


def _plot_chunk_screen(rows: list[dict]) -> None:
    if not rows:
        return
    chunks = np.array([row["chunk"] for row in rows], dtype=float)
    total = np.array([row["total_time"] for row in rows], dtype=float)
    assemble = np.array([row["linear1_t_assemble"] for row in rows], dtype=float)
    solve = np.array([row["solve_time"] for row in rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ax = axes[0]
    ax.plot(chunks, total, marker="o", linewidth=2.0, label="total")
    ax.plot(chunks, solve, marker="s", linewidth=2.0, label="solve")
    ax.set_xscale("log", base=2)
    ax.set_xticks(chunks, [str(int(c)) for c in chunks])
    ax.set_xlabel("P4 Hessian chunk size")
    ax.set_ylabel("Time [s]")
    ax.set_title("Chunk Screen at 32 Ranks")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(chunks, assemble, marker="d", linewidth=2.0, label="1st linear assemble")
    ax.set_xscale("log", base=2)
    ax.set_xticks(chunks, [str(int(c)) for c in chunks])
    ax.set_xlabel("P4 Hessian chunk size")
    ax.set_ylabel("Time [s]")
    ax.set_title("Chunk Effect on First Assembly")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, "chunk_screen.png")


def _report_text(baseline: list[dict], opt: list[dict], chunks: list[dict]) -> str:
    b32 = baseline[-1]
    o32 = opt[-1]
    rows = []
    for b, o in zip(baseline, opt, strict=True):
        rows.append(
            "| "
            + " | ".join(
                [
                    str(int(b["np"])),
                    f"{b['total_time']:.3f}",
                    f"{o['total_time']:.3f}",
                    f"{100.0 * (b['total_time'] - o['total_time']) / b['total_time']:.1f}",
                    f"{b['solve_time']:.3f}",
                    f"{o['solve_time']:.3f}",
                    f"{100.0 * (b['solve_time'] - o['solve_time']) / b['solve_time']:.1f}",
                ]
            )
            + " |"
        )

    chunk_lines = []
    for row in chunks:
        chunk_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["chunk"]),
                    f"{row['total_time']:.3f}",
                    f"{row['linear1_t_assemble']:.3f}",
                    f"{row['solve_time']:.3f}",
                    f"{row['hessian_hvp']:.3f}",
                    f"{row['hessian_extraction']:.3f}",
                ]
            )
            + " |"
        )

    return f"""# `P4(L1_2), lambda = 1.5` Assembly Optimization Comparison

This report compares the published refined-3D baseline against the promoted
optimized backend:

- baseline: `coo` assembly with the older maintained `P4` chunking on the
  refined `L1_2` scaling study
- optimized: `coo` assembly, `coo_vectorized` MG transfers, and tuned
  `P4` Hessian chunk size `4`

## Outcome

At `32` ranks the optimized stack improves the published fixed-work benchmark
from `{b32['total_time']:.3f} s` to `{o32['total_time']:.3f} s`
({100.0 * (b32['total_time'] - o32['total_time']) / b32['total_time']:.1f}% faster),
and improves the first assembled linearization from
`{b32['linear1_t_assemble']:.3f} s` to `{o32['linear1_t_assemble']:.3f} s`
({100.0 * (b32['linear1_t_assemble'] - o32['linear1_t_assemble']) / b32['linear1_t_assemble']:.1f}% faster).

![scaling compare](./compare_assets/scaling_compare.png)

![improvement bars](./compare_assets/improvement_bars.png)

## Per-Rank Comparison

| ranks | baseline total [s] | optimized total [s] | total gain [%] | baseline solve [s] | optimized solve [s] | solve gain [%] |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

## 32-Rank Phase View

![phase32](./compare_assets/phase32_compare.png)

The biggest structural win is the MG hierarchy build path, which dropped from
`{b32['mg_hierarchy_build']:.3f} s` to `{o32['mg_hierarchy_build']:.3f} s`.
The tuned `chunk4` setting then improves the repeated Hessian/linearization
path on top of that.

## Chunk Screen

![chunk screen](./compare_assets/chunk_screen.png)

| chunk | total [s] | first assemble [s] | solve [s] | Hessian HVP [s] | Hessian extraction [s] |
| ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(chunk_lines)}
"""


def main() -> None:
    baseline = _collect_rows(BASELINE_ROOT)
    opt = _collect_rows(OPT_ROOT)
    chunks = _chunk_screen_rows()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    _plot_scaling_compare(baseline, opt)
    _plot_improvement(baseline, opt)
    _plot_phase32_compare(baseline, opt)
    _plot_chunk_screen(chunks)
    REPORT.write_text(_report_text(baseline, opt, chunks), encoding="utf-8")


if __name__ == "__main__":
    main()
