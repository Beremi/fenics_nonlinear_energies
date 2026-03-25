#!/usr/bin/env python3
"""Generate the merged pLaplace_u3 thesis replication docs page."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path

from src.problems.plaplace_u3.thesis.assignment import (
    ASSIGNMENT_STAGE_DETAILS,
    METHOD_TO_TABLES,
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
DEFAULT_SUMMARY = REPO_ROOT / "artifacts" / "raw_results" / "plaplace_u3_thesis_full" / "summary.json"
DEFAULT_OUT = REPO_ROOT / "docs" / "problems" / "pLaplace_u3_thesis_replications.md"
DEFAULT_ASSET_DIR = REPO_ROOT / "docs" / "assets" / "plaplace_u3_thesis"
SOURCE_SAMPLE_PNG = DEFAULT_ASSET_DIR / "plaplace_u3_sample_state.png"
SOURCE_SAMPLE_PDF = DEFAULT_ASSET_DIR / "plaplace_u3_sample_state.pdf"
SOURCE_SQUARE_PANEL = DEFAULT_ASSET_DIR / "square_multibranch_panel.png"
SOURCE_HOLE_PANEL = DEFAULT_ASSET_DIR / "square_hole_panel.png"


def _load_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _fmt(value: object, digits: int = 4) -> str:
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


def _style_thesis(value: object, digits: int = 4) -> str:
    return f'<span style="color:#1d4ed8;"><em>{_fmt(value, digits)}</em></span>'


def _style_repo(value: object, digits: int = 4) -> str:
    return f'<span style="color:#b91c1c;"><strong>{_fmt(value, digits)}</strong></span>'


def _style_repo_sci(value: object, digits: int = 0) -> str:
    return f'<span style="color:#b91c1c;"><strong>{_fmt_sci(value, digits)}</strong></span>'


def _style_thesis_sci(value: object, digits: int = 0) -> str:
    return f'<span style="color:#1d4ed8;"><em>{_fmt_sci(value, digits)}</em></span>'


def _assignment_verdict(row: dict[str, object]) -> str:
    verdict = str(row.get("assignment_verdict") or "").strip().lower()
    if verdict:
        return verdict
    value = row.get("assignment_acceptance_pass")
    if value is None:
        return "secondary"
    return "pass" if bool(value) else "fail"


def _assignment_label(row: dict[str, object]) -> str:
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


def _append_table(lines: list[str], headers: list[str], rows: list[list[str]]) -> None:
    if not rows:
        return
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")


def _copy_assets(asset_dir: Path) -> dict[str, str]:
    asset_dir.mkdir(parents=True, exist_ok=True)
    copies = {
        "sample_png": asset_dir / "plaplace_u3_sample_state.png",
        "sample_pdf": asset_dir / "plaplace_u3_sample_state.pdf",
        "square_panel": asset_dir / "square_multibranch_panel.png",
        "hole_panel": asset_dir / "square_hole_panel.png",
    }
    for source, target in (
        (SOURCE_SAMPLE_PNG, copies["sample_png"]),
        (SOURCE_SAMPLE_PDF, copies["sample_pdf"]),
        (SOURCE_SQUARE_PANEL, copies["square_panel"]),
        (SOURCE_HOLE_PANEL, copies["hole_panel"]),
    ):
        if source.resolve() == target.resolve():
            continue
        shutil.copy2(source, target)
    return {key: path.name for key, path in copies.items()}


def _asset_link(out_path: Path, asset_dir: Path, filename: str) -> str:
    return str(Path(os.path.relpath(asset_dir / filename, start=out_path.parent))).replace("\\", "/")


def _repo_file_link(out_path: Path, repo_rel_path: str, *, label: str | None = None) -> str:
    target = REPO_ROOT / repo_rel_path
    rel = str(Path(os.path.relpath(target, start=out_path.parent))).replace("\\", "/")
    return f"[`{label or repo_rel_path}`]({rel})"


def _implementation_map_lines(out_path: Path) -> list[str]:
    return [
        "## Implementation Map",
        "",
        "### Core Library Code",
        "",
        f"- exact scalar P1 formulas for $A(u)$, $B(u)$, and $J(u)$: {_repo_file_link(out_path, 'src/problems/plaplace_u3/common.py')}",
        f"- reusable 2D structured meshes, seeds, and adjacency: {_repo_file_link(out_path, 'src/problems/plaplace_u3/support/mesh.py')}",
        f"- thesis 1D harness mesh support: {_repo_file_link(out_path, 'src/problems/plaplace_u3/thesis/mesh1d.py')}",
        f"- discrete thesis functionals, rescaling, and the standard Laplace helper matrix: {_repo_file_link(out_path, 'src/problems/plaplace_u3/thesis/functionals.py')}",
        f"- cached FE problem wrapper and common result payloads: {_repo_file_link(out_path, 'src/problems/plaplace_u3/thesis/solver_common.py')}",
        f"- descent directions and stopping criteria: {_repo_file_link(out_path, 'src/problems/plaplace_u3/thesis/directions.py')}",
        f"- thesis RMPA solver: {_repo_file_link(out_path, 'src/problems/plaplace_u3/thesis/solver_rmpa.py')}",
        f"- thesis OA1/OA2 solvers: {_repo_file_link(out_path, 'src/problems/plaplace_u3/thesis/solver_oa.py')}",
        f"- thesis MPA solver: {_repo_file_link(out_path, 'src/problems/plaplace_u3/thesis/solver_mpa.py')}",
        f"- thesis presets and published benchmark values: {_repo_file_link(out_path, 'src/problems/plaplace_u3/thesis/presets.py')} and {_repo_file_link(out_path, 'src/problems/plaplace_u3/thesis/tables.py')}",
        f"- proxy-reference policy and assignment/report labels: {_repo_file_link(out_path, 'src/problems/plaplace_u3/thesis/reference_policy.py')} and {_repo_file_link(out_path, 'src/problems/plaplace_u3/thesis/assignment.py')}",
        "",
        "### Scripts And Publication Helpers",
        "",
        f"- single-case thesis CLI and argument parsing: {_repo_file_link(out_path, 'src/problems/plaplace_u3/thesis/scripts/solve_case.py')}",
        f"- thesis-suite orchestration: {_repo_file_link(out_path, 'experiments/runners/run_plaplace_u3_thesis_suite.py')}",
        f"- docs page generator: {_repo_file_link(out_path, 'experiments/analysis/generate_plaplace_u3_thesis_problem_page.py')}",
        f"- report generator: {_repo_file_link(out_path, 'experiments/analysis/generate_plaplace_u3_thesis_report.py')}",
    ]


def _sorted_rows(rows: list[dict[str, object]], *tables: str) -> list[dict[str, object]]:
    wanted = set(tables)
    return [dict(row) for row in rows if str(row["table"]) in wanted]


def _status_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["status"])] += 1
    return dict(counts)


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


def _section_command(out_dir: str, *tables: str) -> str:
    args = " ".join(f"--only-table {table}" for table in tables)
    return (
        "```bash\n"
        f"./.venv/bin/python -u experiments/analysis/materialize_plaplace_u3_thesis_section.py \\\n"
        f"  --summary artifacts/raw_results/plaplace_u3_thesis_full/summary.json \\\n"
        f"  {args} \\\n"
        f"  --out-dir {out_dir}\n"
        "```"
    )


def _rebuild_packet_command() -> str:
    return (
        "```bash\n"
        "./.venv/bin/python -u experiments/analysis/generate_plaplace_u3_thesis_report.py \\\n"
        "  --summary artifacts/raw_results/plaplace_u3_thesis_full/summary.json \\\n"
        "  --summary-label artifacts/raw_results/plaplace_u3_thesis_full/summary.json \\\n"
        "  --out artifacts/reports/plaplace_u3_thesis/README.md\n"
        "```\n\n"
        "```bash\n"
        "./.venv/bin/python -u experiments/analysis/generate_plaplace_u3_thesis_problem_page.py \\\n"
        "  --summary artifacts/raw_results/plaplace_u3_thesis_full/summary.json \\\n"
        "  --out docs/problems/pLaplace_u3_thesis_replications.md \\\n"
        "  --asset-dir docs/assets/plaplace_u3_thesis\n"
        "```"
    )


def _stage_matrix_rows(overview: dict[str, object]) -> list[list[str]]:
    rows = []
    by_stage = dict(overview["by_stage"])
    for stage in ASSIGNMENT_STAGE_DETAILS:
        stats = by_stage.get(stage)
        if stats is None:
            continue
        rows.append(
            [
                stage,
                ASSIGNMENT_STAGE_DETAILS[stage],
                str(stats["pass"]),
                str(stats["fail"]),
                str(stats["unknown"]),
                str(stats["total"]),
            ]
        )
    return rows


def _table_matrix_rows(overview: dict[str, object], rows: list[dict[str, object]]) -> list[list[str]]:
    by_table = dict(overview["by_table"])
    out = []
    for table, stats in by_table.items():
        section = next(str(row["assignment_section"]) for row in rows if str(row["table"]) == table)
        out.append(
            [
                str(table),
                section,
                str(stats["pass"]),
                str(stats["fail"]),
                str(stats["unknown"]),
                str(stats["total"]),
            ]
        )
    return out


def _problem_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if bool(row.get("assignment_primary"))
        and (
            row.get("assignment_acceptance_pass") is False
            or (row.get("assignment_acceptance_pass") is not True and str(row.get("status")) != "completed")
        )
    ]


def _low_impact_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if bool(row.get("assignment_primary")) and _assignment_verdict(row) == "low impact"
    ]


def _unique_partial_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, object]] = []
    for row in rows:
        note = str(
            row.get("assignment_caveat")
            or row.get("execution_note")
            or row.get("reference_note")
            or row.get("assignment_gap_class")
            or "secondary target"
        )
        key = (str(row.get("assignment_section")), note)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=str, default=str(DEFAULT_SUMMARY))
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--asset-dir", type=str, default=str(DEFAULT_ASSET_DIR))
    args = parser.parse_args()

    summary_path = Path(args.summary)
    out_path = Path(args.out)
    asset_dir = Path(args.asset_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = _load_summary(summary_path)
    rows = [attach_assignment_metadata(dict(row)) for row in summary["rows"]]
    overview = summarize_assignment_rows(rows)
    status_counts = _status_counts(rows)
    primary_rows = [row for row in rows if bool(row.get("assignment_primary"))]
    primary_pass = [row for row in primary_rows if row.get("assignment_acceptance_pass") is True]
    low_impact_rows = _low_impact_rows(rows)
    stage_c_rows = table_5_12_timing_rows(rows)
    direction_rows = table_5_13_timing_rows(rows)
    match_breakdown = _match_breakdown(rows)
    partial_rows = [
        row
        for row in rows
        if (
            not bool(row.get("assignment_primary"))
            or row.get("assignment_acceptance_pass") is None
            or (row.get("assignment_acceptance_pass") is True and str(row.get("status")) != "completed")
        )
    ]
    partial_rows.extend(
        row
        for row in [*stage_c_rows, *direction_rows]
        if str(row.get("timing_status")) != "timing complete"
    )
    mismatch_rows = [
        row
        for row in rows
        if bool(row.get("assignment_primary")) and row.get("assignment_acceptance_pass") is False
    ]
    asset_names = _copy_assets(asset_dir)
    sample_png_link = _asset_link(out_path, asset_dir, asset_names["sample_png"])
    sample_pdf_link = _asset_link(out_path, asset_dir, asset_names["sample_pdf"])
    square_panel_link = _asset_link(out_path, asset_dir, asset_names["square_panel"])
    hole_panel_link = _asset_link(out_path, asset_dir, asset_names["hole_panel"])

    oned_rows = _sorted_rows(rows, "table_5_2", "table_5_3", "table_5_2_drn_sanity")
    rmpa_rows = _sorted_rows(rows, "table_5_8", "table_5_9")
    oa1_rows = _sorted_rows(rows, "table_5_10", "table_5_11")
    mpa_rows = _sorted_rows(rows, "table_5_6", "table_5_7")
    square_multibranch_rows = _sorted_rows(rows, "table_5_14")
    hole_rows = _sorted_rows(rows, "figure_5_13")

    lines = [
        "# pLaplace_u3 Thesis Replications",
        "",
        "Source of the original algorithms and published benchmark values:",
        "",
        "- Michaela Bailová, *Variational methods for solving engineering problems*, PhD Thesis, Ostrava, 2023",
        f"- local source PDF: `{summary.get('source_pdf', 'BAI0012_FEI_P1807_1103V036_2023.pdf')}`",
        "- this merged page combines the thesis/runbook description and the current canonical replication packet",
        "",
        "## Thesis Problem Statement And Functionals",
        "",
        "The thesis studies the nonlinear Dirichlet $p$-Laplacian problem",
        "",
        "$$",
        "-\\Delta_p u = u^3 \\quad \\text{in } \\Omega, \\qquad u = 0 \\quad \\text{on } \\partial \\Omega,",
        "$$",
        "",
        "with",
        "",
        "$$",
        "\\Delta_p u := \\operatorname{div}\\!\\left(\\lvert \\nabla u \\rvert^{p-2}\\nabla u\\right), \\qquad p \\in \\left(\\frac{4}{3}, 4\\right).",
        "$$",
        "",
        "The weak form is:",
        "",
        "$$",
        "\\int_\\Omega \\lvert \\nabla u \\rvert^{p-2}\\nabla u \\cdot \\nabla v\\,dx = \\int_\\Omega u^3 v\\,dx",
        "\\qquad \\forall v \\in W_0^{1,p}(\\Omega).",
        "$$",
        "",
        "The energy functional is",
        "",
        "$$",
        "J(u) = \\frac{1}{p}\\int_\\Omega \\lvert \\nabla u \\rvert^p\\,dx - \\frac{1}{4}\\int_\\Omega u^4\\,dx.",
        "$$",
        "",
        "The thesis also uses the scale-invariant quotient",
        "",
        "$$",
        "I(u) = \\frac{\\|u\\|_{1,p,0}}{\\|u\\|_{L^4(\\Omega)}}.",
        "$$",
        "",
        "For the ray methods, the positive ray maximiser is",
        "",
        "$$",
        "t^*(w) = \\left(\\frac{A(w)}{B(w)}\\right)^{\\frac{1}{4-p}},",
        "\\qquad",
        "A(w) = \\int_\\Omega \\lvert \\nabla w \\rvert^p\\,dx,",
        "\\qquad",
        "B(w) = \\int_\\Omega w^4\\,dx.",
        "$$",
        "",
        "## Thesis Geometries, Discretisation, And Seeds",
        "",
        "- primary geometry: $\\Omega = [0,\\pi] \\times [0,\\pi]$",
        "- secondary geometry: $\\Omega = ([0,\\pi] \\times [0,\\pi]) \\setminus ((\\pi/4,3\\pi/4) \\times (\\pi/4,3\\pi/4))$",
        "- boundary condition: homogeneous Dirichlet on the full boundary, including the inner hole boundary",
        "- discretisation: structured continuous $P_1$ finite elements on uniform right-triangle meshes with $h = \\pi / 2^L$",
        "- principal square seed: $\\sin(x)\\sin(y)$",
        "- square OA2 seeds: $\\sin(x)\\sin(y)$, $10\\sin(2x)\\sin(y)$, $10\\sin(x)\\sin(2y)$, $4(x-y)\\sin(x)\\sin(y)$",
        "- square-hole OA2 seeds: $\\sin(x)\\sin(y)$, $4|\\sin(x)\\sin(2y)|$, $4(x-y)\\sin(x)\\sin(y)$, $|4\\sin(3x)\\sin(3y)|$",
        "",
        "## Thesis Algorithms And Current Repo Implementation",
        "",
        "- `MPA`: classical polygonal-chain mountain-pass method",
        "- `RMPA`: ray mountain-pass method using the analytic ray projection $t^*(w)$",
        "- `OA1`: first-order descent on $I(u)$ with halving acceptance",
        "- `OA2`: first-order descent on $I(u)$ with a 1D minimisation step on $[0, \\delta]$",
        "",
    ]
    lines.extend(_implementation_map_lines(out_path))
    lines.extend(
        [
            "",
            "The section commands below rematerialize the current canonical thesis packet into dedicated experiment folders quickly. For a raw solver recomputation of the same families, use `experiments/runners/run_plaplace_u3_thesis_suite.py --only-table ...` with the table keys shown in each section.",
            "",
            f"![Current square sample state]({sample_png_link})",
            "",
            f"PDF version: [sample state]({sample_pdf_link})",
            "",
            "## Validation Metric And Replication Status",
            "",
            "The thesis validates computed solutions against a separate finite-element reference solution using the discrete $|u-\\bar u|_{1,p,0}$ seminorm. In this repository packet, the direct thesis quantities such as $J$, $I$, and iteration counts are compared against the published tables, while the error columns use the repo's proxy reference policy documented in the canonical thesis report.",
            "",
            f"- canonical summary: `{_repo_rel(summary_path)}`",
            f"- canonical thesis report: `artifacts/reports/plaplace_u3_thesis/README.md`",
            f"- packet note: {summary.get('packet_note', '-')}",
            f"- primary assignment rows passing: `{len(primary_pass)}` / `{len(primary_rows)}`",
            f"- low-impact primary discrepancies: `{len(low_impact_rows)}`",
            f"- direct-comparison primary rows passing: `{match_breakdown.get('exact', {}).get('pass', 0)}` / `{match_breakdown.get('exact', {}).get('total', 0)}`",
            f"- proxy-comparison primary rows passing: `{match_breakdown.get('proxy', {}).get('pass', 0)}` / `{match_breakdown.get('proxy', {}).get('total', 0)}`",
            f"- status counts: `{status_counts}`",
            "",
            "### Stage Map",
            "",
        ]
    )

    _append_table(
        lines,
        ["stage", "brief", "pass", "fail", "secondary", "total"],
        _stage_matrix_rows(overview),
    )

    lines.extend(["", "### Table Map", ""])
    _append_table(
        lines,
        ["method", "thesis targets"],
        [[method, ", ".join(tables)] for method, tables in METHOD_TO_TABLES.items()],
    )

    lines.extend(["", "### Current Table Coverage", ""])
    _append_table(
        lines,
        ["target", "assignment section", "pass", "fail", "secondary", "total"],
        _table_matrix_rows(overview, rows),
    )

    lines.extend(
        [
            "",
            "## 1D Direction Study",
            "",
            "This section merges the thesis 1D harness description with the current repo results for the same nonlinearity on $(0,\\pi)$.",
        ]
    )
    lines.extend(_compact_block("Problem spec", _table_problem_spec_lines("table_5_2")))
    lines.extend(
        [
            _section_command(
                "artifacts/raw_results/plaplace_u3_thesis_sections/one_dimensional",
                "table_5_2",
                "table_5_3",
                "table_5_2_drn_sanity",
            ),
            "",
        ]
    )
    _append_table(
        lines,
        ["table", "direction", "p", "thesis $J$", "repo $J$", "thesis error", "repo error", "repo iters", "status"],
        [
            [
                str(row["table"]),
                str(row["direction"]),
                _fmt(row["p"], 3),
                _style_thesis(row.get("thesis_J")),
                _style_repo(row.get("J")),
                _style_thesis(row.get("thesis_error")),
                _style_repo(row.get("reference_error_w1p")),
                _style_repo(row.get("outer_iterations"), 0),
                _assignment_label(row),
            ]
            for row in sorted(oned_rows, key=lambda item: (str(item["table"]), float(item["p"])))
        ],
    )
    lines.extend(_compact_block("Column legend", _table_legend_lines("table_5_2")))
    lines.extend(_compact_block("Discrepancy notes", _table_discrepancy_lines("table_5_2", oned_rows)))

    lines.extend(
        [
            "",
            "## RMPA Square Principal-Branch Replication",
            "",
            "The thesis uses the square benchmark as the main validation target for RMPA. The tables below merge the mesh-refinement and fixed-mesh tolerance studies.",
        ]
    )
    lines.extend(_compact_block("Problem spec", _table_problem_spec_lines("table_5_8")))
    lines.extend(
        [
            _section_command(
                "artifacts/raw_results/plaplace_u3_thesis_sections/rmpa_square",
                "table_5_8",
                "table_5_9",
            ),
            "",
            "### Table 5.8 — refinement study",
            "",
        ]
    )
    _append_table(
        lines,
        ["$p$", "level", "thesis $J$", "repo $J$", "thesis error", "repo error", "status"],
        [
            [
                _fmt(row["p"], 3),
                str(row["level"]),
                _style_thesis(row.get("thesis_J")),
                _style_repo(row.get("J")),
                _style_thesis(row.get("thesis_error")),
                _style_repo(row.get("reference_error_w1p")),
                _assignment_label(row),
            ]
            for row in sorted(
                [row for row in rmpa_rows if str(row["table"]) == "table_5_8"],
                key=lambda item: (float(item["p"]), int(item["level"])),
            )
        ],
    )
    lines.extend(_compact_block("Column legend", _table_legend_lines("table_5_8")))
    lines.extend(_compact_block("Discrepancy notes", _table_discrepancy_lines("table_5_8", rmpa_rows)))

    lines.extend(["", "### Table 5.9 — tolerance study", ""])
    _append_table(
        lines,
        ["$p$", "$\\varepsilon$", "thesis $J$", "repo $J$", "thesis error", "repo error", "status"],
        [
            [
                _fmt(row["p"], 3),
                _style_repo_sci(row["epsilon"]),
                _style_thesis(row.get("thesis_J")),
                _style_repo(row.get("J")),
                _style_thesis(row.get("thesis_error")),
                _style_repo(row.get("reference_error_w1p")),
                _assignment_label(row),
            ]
            for row in sorted(
                [row for row in rmpa_rows if str(row["table"]) == "table_5_9"],
                key=lambda item: (float(item["p"]), float(item["epsilon"])),
            )
        ],
    )
    lines.extend(_compact_block("Column legend", _table_legend_lines("table_5_9")))
    lines.extend(_compact_block("Discrepancy notes", _table_discrepancy_lines("table_5_9", rmpa_rows)))

    lines.extend(
        [
            "",
            "## OA1 Square Principal-Branch Replication",
            "",
            "OA1 uses the scale-invariant functional $I(u)$. The thesis notes that Table 5.11 should be treated cautiously, so Table 5.10 remains the primary OA1 benchmark.",
        ]
    )
    lines.extend(_compact_block("Problem spec", _table_problem_spec_lines("table_5_10")))
    lines.extend(
        [
            _section_command(
                "artifacts/raw_results/plaplace_u3_thesis_sections/oa1_square",
                "table_5_10",
                "table_5_11",
            ),
            "",
            "### Table 5.10 — refinement study",
            "",
        ]
    )
    _append_table(
        lines,
        ["$p$", "level", "thesis $J$", "repo $J$", "thesis error", "repo error", "status"],
        [
            [
                _fmt(row["p"], 3),
                str(row["level"]),
                _style_thesis(row.get("thesis_J")),
                _style_repo(row.get("J")),
                _style_thesis(row.get("thesis_error")),
                _style_repo(row.get("reference_error_w1p")),
                _assignment_label(row),
            ]
            for row in sorted(
                [row for row in oa1_rows if str(row["table"]) == "table_5_10"],
                key=lambda item: (float(item["p"]), int(item["level"])),
            )
        ],
    )
    lines.extend(_compact_block("Column legend", _table_legend_lines("table_5_10")))
    lines.extend(_compact_block("Discrepancy notes", _table_discrepancy_lines("table_5_10", oa1_rows)))

    lines.extend(["", "### Table 5.11 — tolerance study (secondary / inconsistent in thesis)", ""])
    _append_table(
        lines,
        ["$p$", "$\\varepsilon$", "thesis $J$", "repo $J$", "thesis error", "repo error", "status"],
        [
            [
                _fmt(row["p"], 3),
                _style_repo_sci(row["epsilon"]),
                _style_thesis(row.get("thesis_J")),
                _style_repo(row.get("J")),
                _style_thesis(row.get("thesis_error")),
                _style_repo(row.get("reference_error_w1p")),
                _assignment_label(row),
            ]
            for row in sorted(
                [row for row in oa1_rows if str(row["table"]) == "table_5_11"],
                key=lambda item: (float(item["p"]), float(item["epsilon"])),
            )
        ],
    )
    lines.extend(_compact_block("Column legend", _table_legend_lines("table_5_11")))
    lines.extend(_compact_block("Discrepancy notes", _table_discrepancy_lines("table_5_11", oa1_rows)))

    lines.extend(
        [
            "",
            "## Stage C Timing Summary",
            "",
            "Table 5.12 is the thesis wall-time comparison for the square principal-branch sweep at fixed mesh and tolerance. The packet surfaces the thesis timings directly and pairs them with the fresh local serial-python rerun.",
        ]
    )
    lines.extend(_compact_block("Problem spec", _table_problem_spec_lines("table_5_12")))
    lines.extend(
        [
            _section_command(
                "artifacts/raw_results/plaplace_u3_thesis_sections/stage_c_timing",
                "table_5_7",
                "table_5_9",
                "table_5_11",
            ),
            "",
        ]
    )
    _append_table(
        lines,
        [
            "method",
            "$p$",
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
                _style_thesis(row.get("thesis_iterations"), 0),
                _style_repo(row.get("repo_iterations"), 0),
                _style_thesis(row.get("thesis_time_s"), 2),
                _style_repo(row.get("repo_time_s"), 2),
                str(row.get("timing_status")),
                str(row.get("timing_reason")),
                str(row.get("status")),
            ]
            for row in stage_c_rows
        ],
    )
    lines.extend(_compact_block("Column legend", _table_legend_lines("table_5_12")))
    lines.extend(_compact_block("Discrepancy notes", _table_discrepancy_lines("table_5_12", stage_c_rows)))

    lines.extend(
        [
            "",
            "## Cross-Method Comparison: MPA, Iteration Counts, And Descent Directions",
            "",
            "This section combines the MPA square tables with the cross-method and descent-direction comparison rows.",
        ]
    )
    lines.extend(_compact_block("Problem spec", _table_problem_spec_lines("table_5_13")))
    lines.extend(
        [
            _section_command(
                "artifacts/raw_results/plaplace_u3_thesis_sections/method_comparison",
                "table_5_6",
                "table_5_7",
                "table_5_9",
                "table_5_11",
                "table_5_13",
            ),
            "",
            "### MPA square branch",
            "",
        ]
    )
    _append_table(
        lines,
        ["table", "$p$", "level", "$\\varepsilon$", "thesis $J$", "repo $J$", "thesis error", "repo error", "status"],
        [
            [
                str(row["table"]),
                _fmt(row["p"], 3),
                str(row["level"]),
                _style_repo_sci(row["epsilon"]),
                _style_thesis(row.get("thesis_J")),
                _style_repo(row.get("J")),
                _style_thesis(row.get("thesis_error")),
                _style_repo(row.get("reference_error_w1p")),
                _assignment_label(row),
            ]
            for row in sorted(mpa_rows, key=lambda item: (str(item["table"]), float(item["p"]), int(item["level"]), float(item["epsilon"])))
        ],
    )
    lines.extend(_compact_block("Column legend", _table_legend_lines("table_5_6")))
    lines.extend(_compact_block("Discrepancy notes", _table_discrepancy_lines("table_5_6", mpa_rows)))

    timing_rows_513 = direction_rows
    lines.extend(["", "### Table 5.13 — direction comparison", ""])
    _append_table(
        lines,
        [
            "method",
            "direction",
            "$p$",
            "thesis iters",
            "repo iters",
            "thesis direction iters",
            "thesis time [s]",
            "repo time [s]",
            "timing status",
            "timing reason",
            "status",
        ],
        [
            [
                str(row["method"]),
                str(row["direction"]),
                _fmt(row["p"], 3),
                _style_thesis(row.get("thesis_iterations"), 0),
                _style_repo(row.get("repo_iterations", row.get("outer_iterations")), 0),
                _style_thesis(row.get("thesis_direction_iterations"), 0),
                _style_thesis(row.get("thesis_time_s"), 2),
                _style_repo(row.get("repo_time_s"), 2),
                str(row.get("timing_status", "timing complete")),
                str(row.get("timing_reason")),
                _assignment_label(row),
            ]
            for row in timing_rows_513
        ],
    )
    lines.extend(_compact_block("Column legend", _table_legend_lines("table_5_13")))
    lines.extend(_compact_block("Discrepancy notes", _table_discrepancy_lines("table_5_13", timing_rows_513)))

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

    lines.extend(
        [
            "",
            "## Square Multiple-Solution Study (Table 5.14)",
            "",
            "OA1 stays on the principal branch for the square seeds, while OA2 can recover distinct higher branches depending on the initialisation.",
            "The thesis Figure 5.12 panel order is `(a) sine`, `(b) skew`, `(c) sine_x2`, `(d) sine_y2`, so it should not be read as the same order as the Table 5.14 rows.",
        ]
    )
    lines.extend(_compact_block("Problem spec", _table_problem_spec_lines("table_5_14")))
    lines.extend(
        [
            _section_command(
                "artifacts/raw_results/plaplace_u3_thesis_sections/square_multibranch",
                "table_5_14",
            ),
            "",
            f"![Square OA1/OA2 multibranch panel]({square_panel_link})",
            "",
        ]
    )
    _append_table(
        lines,
        ["seed", "method", "thesis $J$", "repo $J$", "thesis $I$", "repo $I$", "status"],
        [
            [
                str(row["init_mode"]),
                str(row["method"]),
                _style_thesis(row.get("thesis_J")),
                _style_repo(row.get("J")),
                _style_thesis(row.get("thesis_I")),
                _style_repo(row.get("I")),
                _assignment_label(row),
            ]
            for row in sorted(square_multibranch_rows, key=lambda item: (str(item["init_mode"]), str(item["method"])))
        ],
    )
    lines.extend(_compact_block("Column legend", _table_legend_lines("table_5_14")))
    lines.extend(_compact_block("Discrepancy notes", _table_discrepancy_lines("table_5_14", square_multibranch_rows)))

    lines.extend(
        [
            "",
            "## Square-With-Hole OA2 Study (Figure 5.13)",
            "",
            "This nonconvex domain is the sharpest multi-solution benchmark in the thesis packet and is the main extension case beyond the square.",
        ]
    )
    lines.extend(_compact_block("Problem spec", _table_problem_spec_lines("figure_5_13")))
    lines.extend(
        [
            _section_command(
                "artifacts/raw_results/plaplace_u3_thesis_sections/square_hole",
                "figure_5_13",
            ),
            "",
            f"![Square-hole OA2 panel]({hole_panel_link})",
            "",
        ]
    )
    _append_table(
        lines,
        ["seed", "thesis $J$", "repo $J$", "thesis $I$", "repo $I$", "status"],
        [
            [
                str(row["init_mode"]),
                _style_thesis(row.get("thesis_J")),
                _style_repo(row.get("J")),
                _style_thesis(row.get("thesis_I")),
                _style_repo(row.get("I")),
                _assignment_label(row),
            ]
            for row in sorted(hole_rows, key=lambda item: str(item["init_mode"]))
        ],
    )
    lines.extend(_compact_block("Column legend", _table_legend_lines("figure_5_13")))
    lines.extend(_compact_block("Discrepancy notes", _table_discrepancy_lines("figure_5_13", hole_rows)))

    lines.extend(
        [
            "",
            "## Rebuild The Canonical Thesis Packet And This Page",
            "",
            "Use the canonical summary to rebuild the assignment-facing report and this merged docs page without rerunning the full raw suite.",
            "",
            _rebuild_packet_command(),
            "",
            "## What Matches, What Needs Context, And What Does Not Match",
            "",
            f"- primary assignment rows passing the current thresholds: `{len(primary_pass)}` / `{len(primary_rows)}`",
            f"- low-impact primary discrepancies: `{len(low_impact_rows)}`",
            f"- secondary / diagnostic rows: `{len(partial_rows)}`",
            f"- unresolved rows: `{len(_problem_rows(rows))}`",
            "",
            "### What works",
            "",
            "- Stage A and Stage B square principal-branch energies largely track the published thesis values.",
            "- Stage E square-with-hole OA2 values currently match all four published seeds in the canonical packet.",
            "- The merged page uses docs-owned assets and only repo-relative links.",
            "",
            "### What is low impact",
            "",
        ]
    )
    _append_table(
        lines,
        ["target", "verdict", "note"],
        [
            [
                str(row["assignment_section"]),
                _assignment_label(row),
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

    lines.extend(
        [
            "",
            "### What needs context",
            "",
        ]
    )
    _append_table(
        lines,
        ["target", "note"],
        [
            [
                str(row["assignment_section"]),
                str(
                    row.get("assignment_caveat")
                    or row.get("execution_note")
                    or row.get("reference_note")
                    or row.get("assignment_gap_class")
                    or "secondary row"
                ),
            ]
            for row in sorted(
                _unique_partial_rows(partial_rows),
                key=lambda item: (str(item["assignment_stage"]), str(item["assignment_section"])),
            )[:20]
        ],
    )

    lines.extend(["", "### What does not match", ""])
    _append_table(
        lines,
        ["target", "method", "status", "gap class"],
        [
            [
                str(row["assignment_section"]),
                str(row.get("method", "-")),
                str(row.get("status", "-")),
                str(row.get("assignment_gap_class") or "-"),
            ]
            for row in sorted(
                _problem_rows(rows),
                key=lambda item: (
                    str(item.get("assignment_stage", "")),
                    str(item.get("assignment_section", "")),
                    str(item.get("method", "")),
                    float(item.get("p", 0.0)),
                ),
            )
        ],
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "page_path": str(out_path),
                "asset_dir": str(asset_dir),
                "summary_path": str(summary_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
