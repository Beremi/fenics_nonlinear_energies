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

from src.problems.plaplace_u3.thesis.assignment import ASSIGNMENT_STAGE_DETAILS, METHOD_TO_TABLES


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


def _pass_label(value: object) -> str:
    if value is None:
        return "secondary"
    return "PASS" if bool(value) else "FAIL"


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


def _sorted_rows(rows: list[dict[str, object]], *tables: str) -> list[dict[str, object]]:
    wanted = set(tables)
    return [dict(row) for row in rows if str(row["table"]) in wanted]


def _status_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["status"])] += 1
    return dict(counts)


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
        if row.get("assignment_acceptance_pass") is False
        or (row.get("assignment_acceptance_pass") is not True and str(row.get("status")) != "completed")
    ]


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
    rows = [dict(row) for row in summary["rows"]]
    overview = dict(summary["assignment_overview"])
    status_counts = _status_counts(rows)
    primary_rows = [row for row in rows if bool(row.get("assignment_primary"))]
    primary_pass = [row for row in primary_rows if row.get("assignment_acceptance_pass") is True]
    partial_rows = [
        row
        for row in rows
        if row.get("assignment_acceptance_pass") is None
        or (row.get("assignment_acceptance_pass") is True and str(row.get("status")) != "completed")
    ]
    mismatch_rows = [row for row in rows if row.get("assignment_acceptance_pass") is False]
    asset_names = _copy_assets(asset_dir)
    sample_png_link = _asset_link(out_path, asset_dir, asset_names["sample_png"])
    sample_pdf_link = _asset_link(out_path, asset_dir, asset_names["sample_pdf"])
    square_panel_link = _asset_link(out_path, asset_dir, asset_names["square_panel"])
    hole_panel_link = _asset_link(out_path, asset_dir, asset_names["hole_panel"])

    oned_rows = _sorted_rows(rows, "table_5_2", "table_5_3", "table_5_2_drn_sanity")
    rmpa_rows = _sorted_rows(rows, "table_5_8", "table_5_9")
    oa1_rows = _sorted_rows(rows, "table_5_10", "table_5_11")
    mpa_rows = _sorted_rows(rows, "table_5_6", "table_5_7")
    direction_rows = _sorted_rows(rows, "table_5_13")
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
        "The repository implementation for this thesis packet lives under `src/problems/plaplace_u3/thesis/` and is backed by the same structured thesis geometries, exact $P_1$ quartic integration, and the current canonical summary at `artifacts/raw_results/plaplace_u3_thesis_full/summary.json`.",
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
        f"- status counts: `{status_counts}`",
        "",
        "### Stage Map",
        "",
    ]

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
            "",
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
                _pass_label(row.get("assignment_acceptance_pass")),
            ]
            for row in sorted(oned_rows, key=lambda item: (str(item["table"]), float(item["p"])))
        ],
    )

    lines.extend(
        [
            "",
            "## RMPA Square Principal-Branch Replication",
            "",
            "The thesis uses the square benchmark as the main validation target for RMPA. The tables below merge the mesh-refinement and fixed-mesh tolerance studies.",
            "",
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
                _pass_label(row.get("assignment_acceptance_pass")),
            ]
            for row in sorted(
                [row for row in rmpa_rows if str(row["table"]) == "table_5_8"],
                key=lambda item: (float(item["p"]), int(item["level"])),
            )
        ],
    )

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
                _pass_label(row.get("assignment_acceptance_pass")),
            ]
            for row in sorted(
                [row for row in rmpa_rows if str(row["table"]) == "table_5_9"],
                key=lambda item: (float(item["p"]), float(item["epsilon"])),
            )
        ],
    )

    lines.extend(
        [
            "",
            "## OA1 Square Principal-Branch Replication",
            "",
            "OA1 uses the scale-invariant functional $I(u)$. The thesis notes that Table 5.11 should be treated cautiously, so Table 5.10 remains the primary OA1 benchmark.",
            "",
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
                _pass_label(row.get("assignment_acceptance_pass")),
            ]
            for row in sorted(
                [row for row in oa1_rows if str(row["table"]) == "table_5_10"],
                key=lambda item: (float(item["p"]), int(item["level"])),
            )
        ],
    )

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
                _pass_label(row.get("assignment_acceptance_pass")),
            ]
            for row in sorted(
                [row for row in oa1_rows if str(row["table"]) == "table_5_11"],
                key=lambda item: (float(item["p"]), float(item["epsilon"])),
            )
        ],
    )

    lines.extend(
        [
            "",
            "## Cross-Method Comparison: MPA, Iteration Counts, And Descent Directions",
            "",
            "This section combines the MPA square tables with the cross-method and descent-direction comparison rows.",
            "",
            _section_command(
                "artifacts/raw_results/plaplace_u3_thesis_sections/method_comparison",
                "table_5_6",
                "table_5_7",
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
                _pass_label(row.get("assignment_acceptance_pass")),
            ]
            for row in sorted(mpa_rows, key=lambda item: (str(item["table"]), float(item["p"]), int(item["level"]), float(item["epsilon"])))
        ],
    )

    lines.extend(["", "### Table 5.13 — direction comparison", ""])
    _append_table(
        lines,
        ["method", "direction", "$p$", "thesis iters", "repo iters", "thesis direction iters", "status"],
        [
            [
                str(row["method"]),
                str(row["direction"]),
                _fmt(row["p"], 3),
                _style_thesis(row.get("thesis_iterations"), 0),
                _style_repo(row.get("outer_iterations"), 0),
                _style_thesis(row.get("thesis_direction_iterations"), 0),
                _pass_label(row.get("assignment_acceptance_pass")),
            ]
            for row in sorted(direction_rows, key=lambda item: (str(item["method"]), str(item["direction"]), float(item["p"])))
        ],
    )

    lines.extend(
        [
            "",
            "## Square Multiple-Solution Study (Table 5.14)",
            "",
            "OA1 stays on the principal branch for the square seeds, while OA2 can recover distinct higher branches depending on the initialisation.",
            "",
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
                _pass_label(row.get("assignment_acceptance_pass")),
            ]
            for row in sorted(square_multibranch_rows, key=lambda item: (str(item["init_mode"]), str(item["method"])))
        ],
    )

    lines.extend(
        [
            "",
            "## Square-With-Hole OA2 Study (Figure 5.13)",
            "",
            "This nonconvex domain is the sharpest multi-solution benchmark in the thesis packet and is the main extension case beyond the square.",
            "",
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
                _pass_label(row.get("assignment_acceptance_pass")),
            ]
            for row in sorted(hole_rows, key=lambda item: str(item["init_mode"]))
        ],
    )

    lines.extend(
        [
            "",
            "## Rebuild The Canonical Thesis Packet And This Page",
            "",
            "Use the canonical summary to rebuild the assignment-facing report and this merged docs page without rerunning the full raw suite.",
            "",
            _rebuild_packet_command(),
            "",
            "## What Matches, What Is Partial, And What Does Not Match",
            "",
            f"- primary assignment rows passing the current thresholds: `{len(primary_pass)}` / `{len(primary_rows)}`",
            f"- secondary / partial rows: `{len(partial_rows)}`",
            f"- mismatch rows: `{len(mismatch_rows)}`",
            "",
            "### What works",
            "",
            "- Stage A and Stage B square principal-branch energies largely track the published thesis values.",
            "- Stage E square-with-hole OA2 values currently match all four published seeds in the canonical packet.",
            "- The merged page uses docs-owned assets and only repo-relative links.",
            "",
            "### What is partial",
            "",
        ]
    )
    _append_table(
        lines,
        ["target", "note"],
        [
            [
                str(row["assignment_section"]),
                str(row.get("assignment_caveat") or row.get("execution_note") or "secondary target"),
            ]
            for row in sorted(partial_rows, key=lambda item: (str(item["assignment_stage"]), str(item["assignment_section"])))[:20]
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
