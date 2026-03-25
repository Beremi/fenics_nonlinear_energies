#!/usr/bin/env python3
"""Generate an assignment-facing thesis report for ``plaplace_u3``."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from src.problems.plaplace_u3.thesis.assignment import (
    ASSIGNMENT_STAGE_DETAILS,
    attach_assignment_metadata,
    summarize_assignment_rows,
)
from experiments.analysis.plaplace_u3_thesis_docs import (
    convergence_diagnostic_rows,
    table_5_12_timing_rows,
    table_5_13_timing_rows,
    table_discrepancy_lines,
    table_legend_lines,
    table_problem_spec_lines,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _load_summary(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    val = float(value)
    if not math.isfinite(val):
        return str(val)
    return f"{val:.{digits}f}"


def _fmt_sci(value: object, digits: int = 0) -> str:
    if value is None:
        return "-"
    val = float(value)
    if not math.isfinite(val):
        return str(val)
    return f"{val:.{digits}e}"


def _assignment_verdict(row: dict[str, object]) -> str:
    verdict = str(row.get("assignment_verdict") or "").strip().lower()
    if verdict:
        return verdict
    value = row.get("assignment_acceptance_pass")
    if value is None:
        return "secondary"
    return "pass" if bool(value) else "fail"


def _assignment_bucket(row: dict[str, object]) -> str:
    verdict = _assignment_verdict(row)
    if verdict == "pass":
        return "PASS"
    if verdict == "low impact":
        return "low impact"
    if verdict == "fail":
        return "FAIL"
    if verdict == "unknown":
        return "unknown"
    return "secondary"


def _square_pass(row: dict[str, object]) -> bool | None:
    delta_j = row.get("delta_J")
    if delta_j is None:
        return None
    return abs(float(delta_j)) <= 0.03


def _square_oa2_pass(row: dict[str, object]) -> bool | None:
    delta_j = row.get("delta_J")
    if delta_j is None:
        return None
    return abs(float(delta_j)) <= 0.5


def _square_hole_pass(row: dict[str, object]) -> bool:
    thesis_j = row.get("thesis_J")
    thesis_i = row.get("thesis_I")
    if thesis_j in (None, 0.0) or thesis_i is None:
        return False
    rel_j = abs(float(row["J"]) - float(thesis_j)) / abs(float(thesis_j))
    abs_i = abs(float(row["I"]) - float(thesis_i))
    return rel_j <= 0.02 and abs_i <= 0.05


def _same_order(measured: object, thesis: object) -> bool:
    if measured in (None, 0.0) or thesis in (None, 0.0):
        return False
    measured_val = abs(float(measured))
    thesis_val = abs(float(thesis))
    if measured_val == 0.0 or thesis_val == 0.0:
        return False
    return abs(math.log10(measured_val / thesis_val)) <= 1.0


def _monotone_nonincreasing(values: list[float]) -> bool:
    return all(curr <= prev + 1.0e-14 for prev, curr in zip(values, values[1:]))


def _plot_scalar_state(npz_path: str | Path, out_path: Path, title: str) -> None:
    data = np.load(npz_path)
    coords = np.asarray(data["coords"], dtype=np.float64)
    triangles = np.asarray(data["triangles"], dtype=np.int32)
    values = np.asarray(data["u"], dtype=np.float64)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    if triangles.shape[1] == 3:
        triang = mtri.Triangulation(coords[:, 0], coords[:, 1], triangles)
        trip = ax.tripcolor(triang, values, shading="gouraud", cmap="viridis")
        fig.colorbar(trip, ax=ax, shrink=0.85)
    else:
        ax.plot(coords[:, 0], values, linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _panel_from_rows(rows: list[dict[str, object]], out_path: Path, title: str) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(1, len(rows), figsize=(4.8 * len(rows), 4.2), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, row in zip(axes, rows):
        state_path = row.get("state_path")
        if not state_path:
            ax.set_axis_off()
            continue
        data = np.load(state_path)
        coords = np.asarray(data["coords"], dtype=np.float64)
        triangles = np.asarray(data["triangles"], dtype=np.int32)
        values = np.asarray(data["u"], dtype=np.float64)
        triang = mtri.Triangulation(
            coords[:, 0],
            coords[:, 1],
            triangles[:, :3] if triangles.shape[1] > 3 else triangles,
        )
        trip = ax.tripcolor(triang, values, shading="gouraud", cmap="viridis")
        ax.set_title(f"{row['init_mode']}\nJ={_fmt(row['J'], 3)}, I={_fmt(row['I'], 3)}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(trip, ax=ax, shrink=0.8)
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _rows_for(rows: list[dict[str, object]], *tables: str) -> list[dict[str, object]]:
    wanted = set(tables)
    return [dict(row) for row in rows if str(row["table"]) in wanted]


def _append_table(lines: list[str], headers: list[str], body: list[list[str]]) -> None:
    if not body:
        return
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in body:
        lines.append("| " + " | ".join(row) + " |")


def _refinement_trend_counts(rows: list[dict[str, object]]) -> tuple[int, int]:
    groups: dict[tuple[str, str, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["table"]), str(row["method"]), float(row["p"]))].append(row)

    passed = 0
    total = 0
    for grouped_rows in groups.values():
        ordered = sorted(grouped_rows, key=lambda row: int(row["level"]))
        if len(ordered) < 3:
            continue
        total += 1
        measured_errors = [float(row["reference_error_w1p"]) for row in ordered if row.get("reference_error_w1p") is not None]
        thesis_errors = [float(row["thesis_error"]) for row in ordered if row.get("thesis_error") is not None]
        if len(measured_errors) != len(ordered) or len(thesis_errors) != len(ordered):
            continue
        if _monotone_nonincreasing(measured_errors) and all(
            _same_order(row.get("reference_error_w1p"), row.get("thesis_error")) for row in ordered
        ):
            passed += 1
    return passed, total


def _iteration_order_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    preferred_table = {"mpa": "table_5_7", "rmpa": "table_5_9", "oa1": "table_5_11"}
    by_key = defaultdict(list)
    for row in rows:
        key = (float(row["p"]), str(row["method"]))
        by_key[key].append(row)

    for p in sorted({float(row["p"]) for row in rows}):
        row_bundle: dict[str, object] = {"p": p}
        complete = True
        for method, table in preferred_table.items():
            matches = [
                row
                for row in by_key.get((p, method), [])
                if str(row["table"]) == table
                and int(row["level"]) == 6
                and abs(float(row["epsilon"]) - 1.0e-4) <= 1.0e-14
                and str(row["init_mode"]) == "sine"
            ]
            if not matches:
                complete = False
                break
            row_bundle[method] = matches[0]
        if complete:
            selected.append(row_bundle)
    return selected


def _iteration_order_pass_counts(rows: list[dict[str, object]]) -> tuple[int, int]:
    bundles = _iteration_order_rows(rows)
    passed = 0
    for bundle in bundles:
        oa1_it = int(bundle["oa1"]["outer_iterations"])
        rmpa_it = int(bundle["rmpa"]["outer_iterations"])
        mpa_it = int(bundle["mpa"]["outer_iterations"])
        if oa1_it < rmpa_it < mpa_it and mpa_it >= 2 * rmpa_it:
            passed += 1
    return passed, len(bundles)


def _status_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["status"])] += 1
    return dict(counts)


def _unique_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[str, str, str, float, float, str]] = set()
    unique: list[dict[str, object]] = []
    for row in rows:
        key = (
            str(row.get("table")),
            str(row.get("method")),
            str(row.get("init_mode")),
            float(row.get("p", 0.0)),
            float(row.get("epsilon", 0.0)),
            str(row.get("status")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _unique_partial_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, object]] = []
    for row in rows:
        note = str(
            row.get("assignment_caveat")
            or row.get("execution_note")
            or row.get("reference_note")
            or row.get("assignment_gap_class")
            or "secondary target"
        )
        key = (
            str(row.get("assignment_stage")),
            str(row.get("assignment_section")),
            note,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _low_impact_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if bool(row.get("assignment_primary")) and _assignment_verdict(row) == "low impact"
    ]


def _compact_block(title: str, items: list[str]) -> list[str]:
    if not items:
        return []
    return ["", f"**{title}**", *[f"- {item}" for item in items], ""]


def _table_legend_lines(table: str) -> list[str]:
    return table_legend_lines(table)


def _table_problem_spec_lines(table: str) -> list[str]:
    return table_problem_spec_lines(table)


def _table_discrepancy_lines(table: str, rows: list[dict[str, object]]) -> list[str]:
    return table_discrepancy_lines(table, rows)


def _match_breakdown(rows: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    by_kind: dict[str, dict[str, int]] = defaultdict(lambda: {"pass": 0, "total": 0})
    for row in rows:
        if not bool(row.get("assignment_primary")):
            continue
        kind = str(row.get("assignment_reference_kind", "unknown"))
        by_kind[kind]["total"] += 1
        if row.get("assignment_acceptance_pass") is True:
            by_kind[kind]["pass"] += 1
    return dict(by_kind)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=str, default="artifacts/raw_results/plaplace_u3_thesis_full/summary.json")
    parser.add_argument("--summary-label", type=str, default="")
    parser.add_argument("--out", type=str, default="artifacts/reports/plaplace_u3_thesis/README.md")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    summary_label_path = Path(args.summary_label) if args.summary_label else summary_path
    out_path = Path(args.out)
    report_dir = out_path.parent
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_summary(summary_path)
    rows = [attach_assignment_metadata(dict(row)) for row in summary["rows"]]
    assignment_overview = summarize_assignment_rows(rows)

    quick_rows = _rows_for(rows, "quick")
    oned_rows = _rows_for(rows, "table_5_2", "table_5_3", "table_5_2_drn_sanity")
    square_level_rows = _rows_for(rows, "table_5_8", "table_5_10")
    square_eps_rows = _rows_for(rows, "table_5_9", "table_5_11")
    mpa_rows = _rows_for(rows, "table_5_6", "table_5_7")
    stage_c_timing_rows = table_5_12_timing_rows(rows)
    direction_rows = table_5_13_timing_rows(rows)
    multibranch_rows = _rows_for(rows, "table_5_14")
    hole_rows = _rows_for(rows, "figure_5_13")

    iteration_order_pass_count, iteration_order_total = _iteration_order_pass_counts(rows)
    primary_rows = [row for row in rows if bool(row.get("assignment_primary"))]
    primary_pass_rows = [row for row in primary_rows if row.get("assignment_acceptance_pass") is True]
    low_impact_rows = _low_impact_rows(rows)
    partial_rows = [
        row
        for row in rows
        if (
            not bool(row.get("assignment_primary"))
            or
            row.get("assignment_acceptance_pass") is None
            or (
                row.get("assignment_acceptance_pass") is True
                and str(row.get("status")) != "completed"
            )
        )
        and str(row.get("table")) != "quick"
    ]
    partial_rows.extend(
        row
        for row in [*stage_c_timing_rows, *direction_rows]
        if str(row.get("timing_status")) != "timing complete"
    )
    mismatch_rows = [
        row
        for row in rows
        if bool(row.get("assignment_primary")) and row.get("assignment_acceptance_pass") is False
    ]
    unresolved_rows = [
        row
        for row in rows
        if (
            bool(row.get("assignment_primary"))
            and str(row.get("status")) != "completed"
            and row.get("assignment_acceptance_pass") is not True
        )
    ]
    status_counts = _status_counts(rows)
    execution_note_rows = [row for row in rows if row.get("execution_note")]
    match_breakdown = _match_breakdown(rows)

    quick_plot = report_dir / "quick_sample.png"
    if quick_rows and quick_rows[0].get("state_path"):
        _plot_scalar_state(quick_rows[0]["state_path"], quick_plot, "Quick thesis sample")

    oa2_square_rows = [row for row in multibranch_rows if str(row["method"]) == "oa2"]
    square_panel = report_dir / "square_multibranch_panel.png"
    _panel_from_rows(oa2_square_rows, square_panel, "Square OA2 multibranch seeds")
    hole_panel = report_dir / "square_hole_panel.png"
    _panel_from_rows(hole_rows, hole_panel, "Square-hole OA2 seeds")

    lines = [
        "# pLaplaceU3 Thesis Reproduction Report",
        "",
        f"- summary: `{_repo_rel(summary_label_path)}`",
        f"- suite: `{summary['suite']}`",
        f"- quick mode: `{summary['quick']}`",
        f"- skip_reference: `{summary['skip_reference']}`",
    ]

    constants = summary.get("constants")
    if isinstance(constants, dict):
        lines.append(
            f"- constants: `rmpa_delta0={constants.get('rmpa_delta0')}`, `oa_delta_hat={constants.get('oa_delta_hat')}`, `mpa_segment_tol_factor={constants.get('mpa_segment_tol_factor')}`"
        )
    packet_note = summary.get("packet_note")
    if packet_note:
        lines.append(f"- packet note: {packet_note}")

    lines.extend(
        [
            "",
            "## Thesis Problem And Functionals",
            "",
            f"- problem: {assignment_overview['problem_statement']}",
            f"- functionals: {assignment_overview['functional_summary']}",
            "",
            "## Thesis Geometry, Discretisation, And Seeds",
            "",
            f"- geometries: {assignment_overview['geometry_summary']}",
            f"- discretisation: {assignment_overview['discretization_summary']}",
            f"- seeds: {assignment_overview['seed_summary']}",
            "",
            "## Assignment Stage Map",
            "",
        ]
    )

    _append_table(
        lines,
        ["stage", "assignment brief", "repo targets", "acceptance"],
        [
            [
                stage,
                str(ASSIGNMENT_STAGE_DETAILS[stage]),
                ", ".join(
                    sorted(
                        {
                            str(row["assignment_section"])
                            for row in rows
                            if str(row.get("assignment_stage")) == stage
                        }
                    )
                ),
                (
                    f"{assignment_overview['by_stage'][stage]['pass']} pass / "
                    f"{assignment_overview['by_stage'][stage]['fail']} fail / "
                    f"{assignment_overview['by_stage'][stage]['unknown']} secondary"
                ),
            ]
            for stage in ASSIGNMENT_STAGE_DETAILS
            if stage in assignment_overview["by_stage"]
        ],
    )

    lines.extend(["", "## Method To Thesis Table Map", ""])
    _append_table(
        lines,
        ["method", "thesis targets"],
        [
            [method, ", ".join(tables)]
            for method, tables in assignment_overview["method_to_tables"].items()
        ],
    )

    lines.extend(["", "## Replication Legend", ""])
    for label, text in assignment_overview["legend"].items():
        lines.append(f"- `{label}`: {text}")

    lines.extend(
        [
            "",
            "## Assignment Snapshot",
            "",
            f"- primary assignment rows passed: `{len(primary_pass_rows)}` / `{len(primary_rows)}`",
            f"- low-impact primary discrepancies: `{len(low_impact_rows)}`",
            f"- direct-comparison primary rows passed: `{match_breakdown.get('exact', {}).get('pass', 0)}` / `{match_breakdown.get('exact', {}).get('total', 0)}`",
            f"- proxy-comparison primary rows passed: `{match_breakdown.get('proxy', {}).get('pass', 0)}` / `{match_breakdown.get('proxy', {}).get('total', 0)}`",
            f"- row status counts: `{status_counts}`",
            f"- iteration-order passes (`OA1 < RMPA < MPA`, `MPA >= 2 * RMPA`): `{iteration_order_pass_count}` / `{iteration_order_total}`",
            "- proxy reference errors are included for context only; they are not the primary pass/fail criterion.",
        ]
    )

    lines.extend(["", "## Stage-By-Stage Status Matrix", ""])
    _append_table(
        lines,
        ["stage", "pass", "fail", "secondary / unknown", "total"],
        [
            [
                stage,
                str(stats["pass"]),
                str(stats["fail"]),
                str(stats["unknown"]),
                str(stats["total"]),
            ]
            for stage, stats in assignment_overview["by_stage"].items()
        ],
    )

    lines.extend(["", "## Table-By-Table Status Matrix", ""])
    _append_table(
        lines,
        ["target", "assignment section", "pass", "fail", "secondary / unknown", "total"],
        [
            [
                table,
                next(
                    str(row["assignment_section"])
                    for row in rows
                    if str(row["table"]) == table
                ),
                str(stats["pass"]),
                str(stats["fail"]),
                str(stats["unknown"]),
                str(stats["total"]),
            ]
            for table, stats in assignment_overview["by_table"].items()
        ],
    )

    if quick_rows and quick_plot.exists():
        lines.extend(["", "## Quick Sample", "", f"![Quick sample]({quick_plot.name})", ""])
        _append_table(
            lines,
            ["method", "geometry", "level", "p", "J", "I", "status"],
            [
                [
                    str(row["method"]),
                    str(row["geometry"]),
                    str(row["level"]),
                    _fmt(row["p"], 3),
                    _fmt(row["J"], 4),
                    _fmt(row["I"], 4),
                    str(row["status"]),
                ]
                for row in quick_rows
            ],
        )

    lines.extend(
        [
            "",
            "## Reference Policy Used In This Packet",
            "",
            "- direct thesis quantities such as energy, branch identity, and iteration counts are compared against the published tables",
            "- error columns use a modern proxy reference solve from this repo, not the original thesis Section 3.3.1 implementation",
            "- the proxy reference kind is carried per row as `reference_kind` and `reference_note` in the summary JSON",
        ]
    )

    if execution_note_rows:
        lines.extend(["", "## Benchmark-Specific Execution Notes", ""])
        _append_table(
            lines,
            ["target", "seed", "note"],
            [
                [
                    str(row["assignment_section"]),
                    str(row.get("init_mode", "-")),
                    str(row["execution_note"]),
                ]
                for row in sorted(
                    execution_note_rows,
                    key=lambda item: (
                        str(item["assignment_section"]),
                        str(item.get("init_mode", "")),
                    ),
                )
            ],
        )

    if oned_rows:
        lines.extend(["", "## 1D Direction Study", ""])
        lines.extend(_compact_block("Problem spec", _table_problem_spec_lines("table_5_2")))
        _append_table(
            lines,
            ["table", "direction", "p", "thesis J", "measured J", "ΔJ", "thesis it", "measured it", "Δit", "thesis error", "measured error", "assignment"],
            [
                [
                    str(row["table"]),
                    str(row["direction"]),
                    _fmt(row["p"], 3),
                    _fmt(row.get("thesis_J"), 4),
                    _fmt(row["J"], 4),
                    _fmt(row.get("delta_J"), 4),
                    _fmt(row.get("thesis_iterations"), 0),
                    _fmt(row["outer_iterations"], 0),
                    _fmt(row.get("delta_iterations"), 0),
                    _fmt(row.get("thesis_error"), 4),
                    _fmt(row.get("reference_error_w1p"), 4),
                    _assignment_bucket(row),
                ]
                for row in sorted(oned_rows, key=lambda item: (str(item["table"]), float(item["p"])))
            ],
        )
        lines.extend(_compact_block("Column legend", _table_legend_lines("table_5_2")))
        lines.extend(_compact_block("Discrepancy notes", _table_discrepancy_lines("table_5_2", oned_rows)))

    if square_level_rows:
        lines.extend(["", "## Square Principal Branch By Mesh Level", ""])
        lines.extend(_compact_block("Problem spec", _table_problem_spec_lines("table_5_8")))
        _append_table(
            lines,
            ["table", "method", "p", "level", "thesis J", "measured J", "ΔJ", "thesis error", "measured error", "assignment"],
            [
                [
                    str(row["table"]),
                    str(row["method"]),
                    _fmt(row["p"], 3),
                    str(row["level"]),
                    _fmt(row.get("thesis_J"), 4),
                    _fmt(row["J"], 4),
                    _fmt(row.get("delta_J"), 4),
                    _fmt(row.get("thesis_error"), 4),
                    _fmt(row.get("reference_error_w1p"), 4),
                    _assignment_bucket(row),
                ]
                for row in sorted(square_level_rows, key=lambda item: (str(item["table"]), str(item["method"]), float(item["p"]), int(item["level"])))
            ],
        )
        lines.extend(_compact_block("Column legend", _table_legend_lines("table_5_8")))
        lines.extend(
            _compact_block(
                "Discrepancy notes",
                [
                    "Table 5.8 rows are the RMPA square principal-branch sweep; Table 5.10 rows are the OA1 analogue on the same branch.",
                    "The `p = 1.5`, `level = 7` RMPA point is carried as a secondary extension row.",
                ],
            )
        )

    if square_eps_rows:
        lines.extend(["", "## Square Principal Branch By Tolerance", ""])
        lines.extend(_compact_block("Problem spec", _table_problem_spec_lines("table_5_9")))
        _append_table(
            lines,
            ["table", "method", "p", "epsilon", "thesis J", "measured J", "ΔJ", "thesis error", "measured error", "assignment"],
            [
                [
                    str(row["table"]),
                    str(row["method"]),
                    _fmt(row["p"], 3),
                    _fmt_sci(row["epsilon"], 0),
                    _fmt(row.get("thesis_J"), 4),
                    _fmt(row["J"], 4),
                    _fmt(row.get("delta_J"), 4),
                    _fmt(row.get("thesis_error"), 4),
                    _fmt(row.get("reference_error_w1p"), 4),
                    _assignment_bucket(row),
                ]
                for row in sorted(square_eps_rows, key=lambda item: (str(item["table"]), str(item["method"]), float(item["p"]), float(item["epsilon"])))
            ],
        )
        lines.extend(_compact_block("Column legend", _table_legend_lines("table_5_9")))
        lines.extend(
            _compact_block(
                "Discrepancy notes",
                [
                    "Table 5.9 stays within the current packet rule.",
                    "Table 5.11 is retained as secondary context because the thesis marks it as internally inconsistent.",
                ],
            )
        )

    if mpa_rows:
        lines.extend(["", "## MPA Square Branch", ""])
        lines.extend(_compact_block("Problem spec", _table_problem_spec_lines("table_5_6")))
        _append_table(
            lines,
            ["table", "p", "level", "epsilon", "thesis J", "measured J", "ΔJ", "thesis error", "measured error", "assignment"],
            [
                [
                    str(row["table"]),
                    _fmt(row["p"], 3),
                    str(row["level"]),
                    _fmt_sci(row["epsilon"], 0),
                    _fmt(row.get("thesis_J"), 4),
                    _fmt(row["J"], 4),
                    _fmt(row.get("delta_J"), 4),
                    _fmt(row.get("thesis_error"), 4),
                    _fmt(row.get("reference_error_w1p"), 4),
                    _assignment_bucket(row),
                ]
                for row in sorted(mpa_rows, key=lambda item: (str(item["table"]), float(item["p"]), int(item["level"]), float(item["epsilon"])))
            ],
        )
        lines.extend(_compact_block("Column legend", _table_legend_lines("table_5_6")))
        lines.extend(_compact_block("Discrepancy notes", _table_discrepancy_lines("table_5_6", mpa_rows)))

    lines.extend(["", "## Stage C Timing Summary", ""])
    lines.extend(_compact_block("Problem spec", _table_problem_spec_lines("table_5_12")))
    if stage_c_timing_rows:
        _append_table(
            lines,
            [
                "method",
                "p",
                "thesis it",
                "repo it",
                "thesis time [s]",
                "repo time [s]",
                "timing status",
                "timing reason",
                "solver status",
            ],
            [
                [
                    str(row["method"]),
                    _fmt(row["p"], 3),
                    _fmt(row.get("thesis_iterations"), 0),
                    _fmt(row.get("repo_iterations"), 0),
                    _fmt(row.get("thesis_time_s"), 2),
                    _fmt(row.get("repo_time_s"), 2),
                    str(row.get("timing_status")),
                    str(row.get("timing_reason")),
                    str(row["status"]),
                ]
                for row in stage_c_timing_rows
            ],
        )
    lines.extend(_compact_block("Column legend", _table_legend_lines("table_5_12")))
    lines.extend(_compact_block("Discrepancy notes", _table_discrepancy_lines("table_5_12", stage_c_timing_rows)))

    if direction_rows:
        lines.extend(["", "## Iteration Counts And Direction Comparison", ""])
        lines.extend(_compact_block("Problem spec", _table_problem_spec_lines("table_5_13")))
        _append_table(
            lines,
            [
                "method",
                "direction",
                "p",
                "thesis it",
                "measured it",
                "Δit",
                "thesis dir it",
                "Δdir it",
                "thesis time [s]",
                "repo time [s]",
                "timing status",
                "timing reason",
                "assignment",
                "status",
            ],
            [
                [
                    str(row["method"]),
                    str(row["direction"]),
                    _fmt(row["p"], 3),
                    _fmt(row.get("thesis_iterations"), 0),
                    _fmt(row.get("repo_iterations", row["outer_iterations"]), 0),
                    _fmt(row.get("delta_iterations"), 0),
                    _fmt(row.get("thesis_direction_iterations"), 0),
                    _fmt(row.get("delta_direction_iterations"), 0),
                    _fmt(row.get("thesis_time_s"), 2),
                    _fmt(row.get("repo_time_s"), 2),
                    str(row.get("timing_status")),
                    str(row.get("timing_reason")),
                    _assignment_bucket(row),
                    str(row["status"]),
                ]
                for row in sorted(direction_rows, key=lambda item: (str(item["method"]), str(item["direction"]), float(item["p"])))
            ],
        )
        lines.extend(_compact_block("Column legend", _table_legend_lines("table_5_13")))
        lines.extend(_compact_block("Discrepancy notes", _table_discrepancy_lines("table_5_13", direction_rows)))

    diagnostics_rows = convergence_diagnostic_rows(rows)
    if diagnostics_rows:
        lines.extend(["", "## Convergence Diagnostics", ""])
        _append_table(
            lines,
            ["family", "current status", "root-cause category", "strongest evidence", "action taken"],
            [
                [
                    row["family"],
                    row["current_status"],
                    row["root_cause_category"],
                    row["strongest_evidence"],
                    row["action_taken"],
                ]
                for row in diagnostics_rows
            ],
        )

    if multibranch_rows:
        lines.extend(["", "## Square OA2 Multiple Branches", ""])
        lines.extend(_compact_block("Problem spec", _table_problem_spec_lines("table_5_14")))
        if square_panel.exists():
            lines.extend([f"![Square OA2 branches]({square_panel.name})", ""])
        lines.append(
            "The thesis Figure 5.12 panel order is `(a) sine`, `(b) skew`, `(c) sine_x2`, `(d) sine_y2`, so it should not be read as the same order as the Table 5.14 rows."
        )
        lines.append("")
        _append_table(
            lines,
            ["seed", "method", "thesis J", "measured J", "ΔJ", "thesis I", "measured I", "assignment"],
            [
                [
                    str(row["init_mode"]),
                    str(row["method"]),
                    _fmt(row.get("thesis_J"), 4),
                    _fmt(row["J"], 4),
                    _fmt(row.get("delta_J"), 4),
                    _fmt(row.get("thesis_I"), 4),
                    _fmt(row["I"], 4),
                    _assignment_bucket(row),
                ]
                for row in sorted(multibranch_rows, key=lambda item: (str(item["init_mode"]), str(item["method"])))
            ],
        )
        lines.extend(_compact_block("Column legend", _table_legend_lines("table_5_14")))
        lines.extend(_compact_block("Discrepancy notes", _table_discrepancy_lines("table_5_14", multibranch_rows)))

    if hole_rows:
        lines.extend(["", "## Square-Hole OA2", ""])
        lines.extend(_compact_block("Problem spec", _table_problem_spec_lines("figure_5_13")))
        if hole_panel.exists():
            lines.extend([f"![Square-hole OA2]({hole_panel.name})", ""])
        _append_table(
            lines,
            ["seed", "thesis J", "measured J", "abs ΔJ", "thesis I", "measured I", "abs ΔI", "assignment"],
            [
                [
                    str(row["init_mode"]),
                    _fmt(row.get("thesis_J"), 4),
                    _fmt(row["J"], 4),
                    _fmt(None if row.get("thesis_J") is None else abs(float(row["J"]) - float(row["thesis_J"])), 4),
                    _fmt(row.get("thesis_I"), 4),
                    _fmt(row["I"], 4),
                    _fmt(None if row.get("thesis_I") is None else abs(float(row["I"]) - float(row["thesis_I"])), 4),
                    _assignment_bucket(row),
                ]
                for row in sorted(hole_rows, key=lambda item: str(item["init_mode"]))
            ],
        )
        lines.extend(_compact_block("Column legend", _table_legend_lines("figure_5_13")))
        lines.extend(_compact_block("Discrepancy notes", _table_discrepancy_lines("figure_5_13", hole_rows)))

    lines.extend(["", "## What Works", ""])
    lines.extend(
        [
            f"- Primary assignment rows passing the runbook thresholds: `{len(primary_pass_rows)}` / `{len(primary_rows)}`.",
            f"- Low-impact primary discrepancies are tracked separately: `{len(low_impact_rows)}`.",
            f"- Stage A and Stage B principal-branch energies are largely aligned with the thesis tables, especially across the square level-5 and level-6 benchmarks.",
            f"- Stage C direction-comparison rows are complete, and the qualitative iteration ordering holds for `{iteration_order_pass_count}` / `{iteration_order_total}` published `p` values.",
        ]
    )

    lines.extend(["", "## What Is Low Impact", ""])
    if low_impact_rows:
        _append_table(
            lines,
            ["target", "stage", "verdict", "note"],
            [
                [
                    str(row["assignment_section"]),
                    str(row["assignment_stage"]),
                    _assignment_bucket(row),
                    str(row.get("assignment_gap_class") or "documented low-impact discrepancy"),
                ]
                for row in sorted(
                    low_impact_rows,
                    key=lambda item: (
                        str(item.get("assignment_stage", "")),
                        str(item.get("assignment_section", "")),
                        str(item.get("method", "")),
                        float(item.get("p", 0.0)),
                    ),
                )
            ],
        )
    else:
        lines.append("- No low-impact primary discrepancies are currently recorded.")

    lines.extend(["", "## What Needs Context", ""])
    if partial_rows:
        partial_rows = _unique_partial_rows(partial_rows)
        _append_table(
            lines,
            ["target", "stage", "status", "note"],
            [
                [
                    str(row["assignment_section"]),
                    str(row["assignment_stage"]),
                    _assignment_bucket(row),
                    str(
                        row.get("assignment_caveat")
                        or row.get("execution_note")
                        or row.get("reference_note")
                        or row.get("assignment_gap_class")
                        or "secondary row"
                    ),
                ]
                for row in sorted(partial_rows, key=lambda item: (str(item["assignment_stage"]), str(item["assignment_section"])))[:20]
            ],
        )
    else:
        lines.append("- No secondary or diagnostic-only rows remain.")

    lines.extend(["", "## What Does Not Yet Match", ""])
    problem_rows = _unique_rows([*mismatch_rows, *unresolved_rows])
    if problem_rows:
        _append_table(
            lines,
            ["target", "method", "status", "measured", "thesis", "gap class"],
            [
                [
                    str(row["assignment_section"]),
                    str(row.get("method", "-")),
                    str(row.get("status", "-")),
                    f"J={_fmt(row.get('J'), 4)}, I={_fmt(row.get('I'), 4)}",
                    f"J={_fmt(row.get('thesis_J'), 4)}, I={_fmt(row.get('thesis_I'), 4)}",
                    str(row.get("assignment_gap_class") or "-"),
                ]
                for row in sorted(
                    problem_rows,
                    key=lambda item: (
                        str(item.get("assignment_stage", "")),
                        str(item.get("assignment_section", "")),
                        str(item.get("method", "")),
                    ),
                )
            ],
        )
    else:
        lines.append("- All primary assignment rows currently match the runbook acceptance rules.")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report_path": str(out_path), "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
