#!/usr/bin/env python3
"""Materialize a section-level thesis packet from the canonical pLaplace_u3 summary."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = REPO_ROOT / "artifacts" / "raw_results" / "plaplace_u3_thesis_full" / "summary.json"


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _load_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_overview(rows: list[dict[str, object]]) -> dict[str, object]:
    by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    by_table: dict[str, Counter[str]] = defaultdict(Counter)
    status_counts: Counter[str] = Counter()
    for row in rows:
        stage = str(row.get("assignment_stage", "unknown"))
        table = str(row.get("table", "unknown"))
        verdict = row.get("assignment_acceptance_pass")
        bucket = "pass" if verdict is True else "fail" if verdict is False else "unknown"
        by_stage[stage][bucket] += 1
        by_stage[stage]["total"] += 1
        by_table[table][bucket] += 1
        by_table[table]["total"] += 1
        status_counts[str(row.get("status", "unknown"))] += 1
    return {
        "by_stage": {key: dict(value) for key, value in sorted(by_stage.items())},
        "by_table": {key: dict(value) for key, value in sorted(by_table.items())},
        "status_counts": dict(status_counts),
    }


def _write_readme(
    out_path: Path,
    summary_path: Path,
    only_tables: list[str],
    rows: list[dict[str, object]],
    overview: dict[str, object],
) -> None:
    lines = [
        "# pLaplace_u3 Thesis Section Packet",
        "",
        f"- source summary: `{_repo_rel(summary_path)}`",
        f"- selected tables: `{', '.join(only_tables) if only_tables else 'all'}`",
        f"- rows: `{len(rows)}`",
        f"- status counts: `{overview['status_counts']}`",
        "",
        "This packet rematerializes the current canonical thesis results for the selected experiment family.",
        "For raw solver recomputation, use `experiments/runners/run_plaplace_u3_thesis_suite.py --only-table ...`.",
        "",
        "## Rows",
        "",
        "| table | method | geometry | level | p | init | status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("table", "-")),
                    str(row.get("method", "-")),
                    str(row.get("geometry", "-")),
                    str(row.get("level", "-")),
                    str(row.get("p", "-")),
                    str(row.get("init_mode", "-")),
                    str(row.get("status", "-")),
                ]
            )
            + " |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=str, default=str(DEFAULT_SUMMARY))
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--only-table", action="append", default=[])
    args = parser.parse_args()

    summary_path = Path(args.summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_summary(summary_path)
    rows = [dict(row) for row in summary["rows"]]
    wanted = {str(table) for table in args.only_table}
    if wanted:
        rows = [row for row in rows if str(row.get("table")) in wanted]
    if not rows:
        raise SystemExit(f"No rows found for tables {sorted(wanted)} in {summary_path}")

    overview = _build_overview(rows)
    filtered = dict(summary)
    filtered["source_summary"] = _repo_rel(summary_path)
    filtered["selected_tables"] = sorted(wanted)
    filtered["rows"] = rows
    filtered["num_rows"] = len(rows)
    filtered["assignment_overview"] = overview

    summary_out = out_dir / "summary.json"
    readme_out = out_dir / "README.md"
    summary_out.write_text(json.dumps(filtered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_readme(readme_out, summary_path, sorted(wanted), rows, overview)
    print(
        json.dumps(
            {
                "summary_path": _repo_rel(summary_out),
                "readme_path": _repo_rel(readme_out),
                "num_rows": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
