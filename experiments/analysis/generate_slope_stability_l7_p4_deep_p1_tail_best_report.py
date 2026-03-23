#!/usr/bin/env python3
"""Generate the L7 P4 best-default benchmark report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT = Path(
    "artifacts/raw_results/slope_stability_l7_p4_deep_p1_tail_best_lambda1_np8_maxit20/summary.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/reports/slope_stability_l7_p4_deep_p1_tail_best_lambda1_np8_maxit20"
)


def _fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _load_result_payload(summary_row: dict[str, object]) -> dict[str, object]:
    result_json = summary_row.get("result_json")
    if not result_json:
        return {}
    path = Path(str(result_json))
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _peak_memory_record(summary_path: Path) -> dict[str, object] | None:
    watch_path = summary_path.parent / "memory_watch.jsonl"
    if not watch_path.exists():
        return None
    rows = [
        json.loads(line)
        for line in watch_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("rss_gb_total", 0.0)))


def _memory_rows(summary_path: Path) -> list[dict[str, object]]:
    watch_path = summary_path.parent / "memory_watch.jsonl"
    if not watch_path.exists():
        return []
    return [
        json.loads(line)
        for line in watch_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _extract_linear_rows(result_payload: dict[str, object]) -> list[dict[str, object]]:
    try:
        steps = result_payload["result"]["steps"]
    except Exception:
        return []
    if not steps:
        return []
    return list(steps[0].get("linear_timing", []))


def _save_memory_plot(rows: list[dict[str, object]], out_path: Path) -> bool:
    if not rows:
        return False
    t0 = rows[0].get("ts", "")
    if not t0:
        return False
    import datetime as dt

    ts0 = dt.datetime.fromisoformat(str(t0))
    xs = []
    ys = []
    for row in rows:
        ts = dt.datetime.fromisoformat(str(row.get("ts")))
        xs.append((ts - ts0).total_seconds() / 60.0)
        ys.append(float(row.get("rss_gb_total", 0.0)))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3.0)
    ax.set_xlabel("Elapsed time [min]")
    ax.set_ylabel("Watched RSS total [GiB]")
    ax.set_title("L7/P4 memory watch")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def _save_linear_breakdown_plot(rows: list[dict[str, object]], out_path: Path) -> bool:
    if not rows:
        return False
    xs = list(range(1, len(rows) + 1))
    assemble = [float(row.get("assemble_total_time", 0.0)) for row in rows]
    pc_setup = [float(row.get("pc_setup_time", 0.0)) for row in rows]
    ksp = [float(row.get("solve_time", 0.0)) for row in rows]
    total = [float(row.get("linear_total_time", 0.0)) for row in rows]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(xs, total, marker="o", linewidth=2.0, label="linear total")
    ax.plot(xs, ksp, marker="s", linewidth=1.6, label="KSP solve")
    ax.plot(xs, assemble, marker="^", linewidth=1.6, label="assemble")
    ax.plot(xs, pc_setup, marker="d", linewidth=1.6, label="PC setup")
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("Time [s]")
    ax.set_title("Per-Newton linear breakdown")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def _save_ksp_quality_plot(rows: list[dict[str, object]], out_path: Path) -> bool:
    if not rows:
        return False
    xs = list(range(1, len(rows) + 1))
    its = [float(row.get("ksp_its", 0.0)) for row in rows]
    rel = [
        max(float(row.get("true_relative_residual", 0.0)), 1e-16)
        for row in rows
    ]

    fig, ax1 = plt.subplots(figsize=(8.0, 4.8))
    ax1.plot(xs, its, color="tab:blue", marker="o", linewidth=1.8, label="KSP its")
    ax1.set_xlabel("Newton iteration")
    ax1.set_ylabel("KSP iterations", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        xs,
        rel,
        color="tab:red",
        marker="s",
        linewidth=1.6,
        label="true rel residual",
    )
    ax2.set_ylabel("True relative residual", color="tab:red")
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_title("Linear quality by Newton iteration")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def _save_callback_plot(result_payload: dict[str, object], out_path: Path) -> bool:
    callback_summary = (
        result_payload.get("timings", {}).get("callback_summary", {})
        if result_payload
        else {}
    )
    if not callback_summary:
        return False
    labels = ["energy", "gradient", "hessian"]
    values = [float(callback_summary.get(label, {}).get("total", 0.0)) for label in labels]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(labels, values, color=["tab:green", "tab:orange", "tab:purple"])
    ax.set_ylabel("Total time [s]")
    ax.set_title("Callback totals")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    row = dict(json.loads(args.input.read_text(encoding="utf-8")))
    result_payload = _load_result_payload(row)
    peak_mem = _peak_memory_record(args.input)
    memory_rows = _memory_rows(args.input)
    linear_rows = _extract_linear_rows(result_payload)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    memory_plot = "memory_watch.png"
    linear_plot = "newton_linear_breakdown.png"
    ksp_plot = "newton_ksp_quality.png"
    callback_plot = "callback_totals.png"
    have_memory_plot = _save_memory_plot(memory_rows, out_dir / memory_plot)
    have_linear_plot = _save_linear_breakdown_plot(linear_rows, out_dir / linear_plot)
    have_ksp_plot = _save_ksp_quality_plot(linear_rows, out_dir / ksp_plot)
    have_callback_plot = _save_callback_plot(result_payload, out_dir / callback_plot)

    peak_mem_block = ""
    if peak_mem is not None:
        peak_mem_block = f"""

## Memory Watch

| metric | value |
| --- | ---: |
| peak watched RSS total [GiB] | {_fmt(float(peak_mem.get('rss_gb_total', 0.0)), 3)} |
| peak sample timestamp | {peak_mem.get('ts', '')} |
| watched process count at peak | {int(peak_mem.get('proc_count', 0))} |
"""
        if have_memory_plot:
            peak_mem_block += f"""

![Memory watch]({memory_plot})
"""

    plot_block = ""
    if have_linear_plot or have_ksp_plot or have_callback_plot:
        plot_block = "\n## Plots\n"
        if have_linear_plot:
            plot_block += f"\n![Per-Newton linear breakdown]({linear_plot})\n"
        if have_ksp_plot:
            plot_block += f"\n![KSP quality by Newton iteration]({ksp_plot})\n"
        if have_callback_plot:
            plot_block += f"\n![Callback totals]({callback_plot})\n"

    report = f"""# `L7/P4` current best default on `8` ranks

Setting:

- problem: `L7`, `P4`, `lambda=1.0`
- hierarchy: `{row.get('mg_custom_hierarchy', '')}`
- ranks: `8`
- nonlinear: `armijo`, `maxit=20`, `--no-use_trust_region`
- linear: `fgmres`, `ksp_max_it=15`
- capped linear solve handling: guarded `accept_ksp_maxit_direction`, true-rel cap `6e-2`
- coarse solve: `rank0_lu_broadcast`
- smoothers: `P4/P2/P1 = richardson + sor`, `3` steps
- problem build mode: `rank_local`
- MG level build mode: `rank_local`
- transfer build mode: `owned_rows`
- hot-path distribution: `overlap_p2p`
- benchmark mode: `warmup_once_then_solve`
- Hessian buffer mode: `--no-reuse_hessian_value_buffers`

## Result

| metric | value |
| --- | ---: |
| status | {row.get('message', '')} |
| nodes | {int(row.get('nodes', 0))} |
| elements | {int(row.get('elements', 0))} |
| free dofs | {int(row.get('free_dofs', 0))} |
| Newton reached | {int(row.get('newton_iterations', 0))} |
| linear iterations | {int(row.get('linear_iterations', 0))} |
| accepted capped steps | {int(row.get('accepted_capped_step_count', 0))} |
| steady-state total [s] | {_fmt(float(row.get('steady_state_total_time_sec', 0.0)))} |
| solve [s] | {_fmt(float(row.get('solve_time_sec', 0.0)))} |
| one-time setup [s] | {_fmt(float(row.get('one_time_setup_time_sec', 0.0)))} |
| line-search time [s] | {_fmt(float(row.get('line_search_time_sec', 0.0)))} |
| line-search evals | {int(row.get('line_search_evals', 0))} |
| linear KSP solve [s] | {_fmt(float(row.get('linear_ksp_solve_time_sec', 0.0)))} |
| linear assemble [s] | {_fmt(float(row.get('linear_assemble_time_sec', 0.0)))} |
| linear PC setup [s] | {_fmt(float(row.get('linear_pc_setup_time_sec', 0.0)))} |
| energy | {_fmt(float(row.get('energy', 0.0)), 9)} |
| final grad norm | {_fmt(float(row.get('final_grad_norm', 0.0)), 6)} |
| worst true relative residual | {_fmt(float(row.get('worst_true_relative_residual', 0.0)), 6)} |
| omega | {_fmt(float(row.get('omega', 0.0)), 6)} |
| u_max | {_fmt(float(row.get('u_max', 0.0)), 6)} |
{peak_mem_block}
{plot_block}
"""

    (out_dir / "README.md").write_text(report, encoding="utf-8")
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
