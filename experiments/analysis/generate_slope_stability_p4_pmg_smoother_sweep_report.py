#!/usr/bin/env python3
"""Generate the staged PMG smoother sweep report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SUMMARY_PATH = Path(
    "artifacts/raw_results/slope_stability_p4_pmg_smoother_sweep_lambda1/summary.json"
)
OUTPUT_DIR = Path(
    "artifacts/reports/slope_stability_p4_pmg_smoother_sweep_lambda1"
)


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _rows_for(rows: list[dict[str, object]], *, level: int, variant: str) -> list[dict[str, object]]:
    return sorted(
        [
            row
            for row in rows
            if int(row["level"]) == int(level) and str(row["variant"]) == str(variant)
        ],
        key=lambda row: int(row["ranks"]),
    )


def _plot_available_scaling(
    rows: list[dict[str, object]],
    *,
    level: int,
    variants: list[str],
    output_path: Path,
) -> None:
    subset = [
        row
        for row in rows
        if int(row["level"]) == int(level)
        and bool(row.get("solver_success"))
        and str(row["variant"]) in set(variants)
        and int(row["ranks"]) in {1, 2, 4, 8}
    ]
    if not subset:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for variant in variants:
        vrows = sorted(
            [row for row in subset if str(row["variant"]) == str(variant)],
            key=lambda row: int(row["ranks"]),
        )
        if not vrows:
            continue
        x = np.array([int(row["ranks"]) for row in vrows], dtype=np.float64)
        y = np.array([float(row["steady_state_total_time_sec"]) for row in vrows], dtype=np.float64)
        ax.plot(x, y, marker="o", label=variant)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("ranks")
    ax.set_ylabel("steady-state total [s]")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    rows = list(json.loads(SUMMARY_PATH.read_text(encoding="utf-8")))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    l4_plot = OUTPUT_DIR / "l4_available_scaling.png"
    l5_plot = OUTPUT_DIR / "l5_available_scaling.png"
    finalists = ["tail_baseline", "tail_replicated_assets", "tail_baseline_p2steps_2"]
    _plot_available_scaling(rows, level=4, variants=finalists, output_path=l4_plot)
    _plot_available_scaling(rows, level=5, variants=finalists, output_path=l5_plot)

    l4_rank8_family = sorted(
        [
            row
            for row in rows
            if int(row["level"]) == 4
            and int(row["ranks"]) == 8
            and str(row["stage"]) == "family_screen"
            and bool(row.get("solver_success"))
        ],
        key=lambda row: float(row["steady_state_total_time_sec"]),
    )
    l4_rank8_step = sorted(
        [
            row
            for row in rows
            if int(row["level"]) == 4
            and int(row["ranks"]) == 8
            and str(row["stage"]) == "step_refine"
            and bool(row.get("solver_success"))
        ],
        key=lambda row: float(row["steady_state_total_time_sec"]),
    )
    baseline_l5_np2 = next(
        (
            row
            for row in rows
            if int(row["level"]) == 5
            and int(row["ranks"]) == 2
            and str(row["variant"]) == "tail_baseline"
        ),
        None,
    )

    lines: list[str] = []
    lines.append("# P4 PMG Smoother Sweep")
    lines.append("")
    lines.append(
        "This report summarizes the staged PMG sweep after adding per-level legacy smoother controls and MG runtime diagnostics. "
        "The campaign was intentionally pruned on failures and weak families, and it stopped once the strict L5 promotion gate was no longer being met."
    )
    lines.append("")

    lines.append("## L4 Family Screen")
    lines.append("")
    lines.append("| variant | ranks | success | Newton | linear | steady-state [s] | setup [s] | solve [s] |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(
        [
            row
            for row in rows
            if int(row["level"]) == 4
            and str(row["stage"]) == "family_screen"
            and int(row["ranks"]) in {1, 8}
        ],
        key=lambda item: (str(item["variant"]), int(item["ranks"])),
    ):
        lines.append(
            f"| {row['variant']} | {row['ranks']} | {'yes' if row.get('solver_success') else 'no'} | "
            f"{row.get('newton_iterations', '-')} | {row.get('linear_iterations', '-')} | "
            f"{_fmt(row.get('steady_state_total_time_sec'))} | {_fmt(row.get('one_time_setup_time_sec'))} | "
            f"{_fmt(row.get('solve_time_sec'))} |"
        )
    lines.append("")

    lines.append("## L4 Step Refinement")
    lines.append("")
    lines.append("| variant | ranks | success | Newton | linear | steady-state [s] | fine observed [s] | coarse observed [s] |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(
        [
            row
            for row in rows
            if int(row["level"]) == 4
            and str(row["stage"]) == "step_refine"
            and int(row["ranks"]) in {1, 8}
        ],
        key=lambda item: (str(item["variant"]), int(item["ranks"])),
    ):
        lines.append(
            f"| {row['variant']} | {row['ranks']} | {'yes' if row.get('solver_success') else 'no'} | "
            f"{row.get('newton_iterations', '-')} | {row.get('linear_iterations', '-')} | "
            f"{_fmt(row.get('steady_state_total_time_sec'))} | {_fmt(row.get('fine_observed_time_sec'))} | "
            f"{_fmt(row.get('coarse_observed_time_sec'))} |"
        )
    lines.append("")

    lines.append("## L5 Promotion Screen")
    lines.append("")
    lines.append("| variant | ranks | success | status | Newton | linear | steady-state [s] |")
    lines.append("| --- | ---: | --- | --- | ---: | ---: | ---: |")
    for row in sorted(
        [row for row in rows if int(row["level"]) == 5 and int(row["ranks"]) in {1, 2, 8}],
        key=lambda item: (str(item["variant"]), int(item["ranks"])),
    ):
        lines.append(
            f"| {row['variant']} | {row['ranks']} | {'yes' if row.get('solver_success') else 'no'} | "
            f"{row.get('status', '-')} | {row.get('newton_iterations', '-')} | "
            f"{row.get('linear_iterations', '-')} | {_fmt(row.get('steady_state_total_time_sec'))} |"
        )
    lines.append("")

    lines.append("## Available Full-Matrix Rows")
    lines.append("")
    lines.append("| level | variant | ranks | success | stage | steady-state [s] | energy [s] | gradient [s] | Hessian [s] | line search [s] |")
    lines.append("| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(
        [
            row
            for row in rows
            if str(row["variant"]) in set(finalists)
            and int(row["ranks"]) in {1, 2, 4, 8}
        ],
        key=lambda item: (int(item["level"]), str(item["variant"]), int(item["ranks"])),
    ):
        lines.append(
            f"| {row['level']} | {row['variant']} | {row['ranks']} | {'yes' if row.get('solver_success') else 'no'} | "
            f"{row['stage']} | {_fmt(row.get('steady_state_total_time_sec'))} | {_fmt(row.get('energy_total_time_sec'))} | "
            f"{_fmt(row.get('gradient_total_time_sec'))} | {_fmt(row.get('hessian_total_time_sec'))} | "
            f"{_fmt(row.get('line_search_time_sec'))} |"
        )
    lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    if l4_rank8_family:
        best_family = l4_rank8_family[0]
        lines.append(
            f"- On L4 rank 8, the best family-screen row is `{best_family['variant']}` at `{_fmt(best_family['steady_state_total_time_sec'])} s`."
        )
    if l4_rank8_step:
        best_step = l4_rank8_step[0]
        lines.append(
            f"- On L4 rank 8, the best completed step-refine row is `{best_step['variant']}` at `{_fmt(best_step['steady_state_total_time_sec'])} s`."
        )
    lines.append(
        "- No smoother-only change produced a convincing cross-level win over the tuned baseline. "
        "The strongest L4 gains were either negligible (`tail_replicated_assets` vs `tail_baseline`) or did not survive the L5 screen (`tail_baseline_p2steps_2`)."
    )
    lines.append(
        "- The P1-Jacobi branch helped at L4 rank 1, but it regressed materially at L4 rank 8, so it was not promoted."
    )
    if baseline_l5_np2 is not None:
        lines.append(
            f"- The strict full-matrix promotion stopped once the tuned baseline itself failed on L5 rank 2 "
            f"(`status={baseline_l5_np2['status']}`, `steady-state={_fmt(baseline_l5_np2['steady_state_total_time_sec'])} s`)."
        )
    lines.append(
        "- The main implemented outcomes of this campaign are therefore: coarse-Hypre observability, per-level smoother controls, a pruned staged sweep driver, and a clearer diagnosis that the remaining problem is not a missing smoother toggle but the larger L5 robustness/scaling path."
    )
    lines.append("")
    if l4_plot.exists():
        lines.append(f"![L4 available scaling]({l4_plot.name})")
        lines.append("")
    if l5_plot.exists():
        lines.append(f"![L5 available scaling]({l5_plot.name})")
        lines.append("")

    report_text = "\n".join(lines)
    (OUTPUT_DIR / "README.md").write_text(report_text + "\n", encoding="utf-8")
    (OUTPUT_DIR / "report.md").write_text(report_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
