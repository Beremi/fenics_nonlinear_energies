#!/usr/bin/env python3
"""Merge chunked thesis-suite summaries into one canonical summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.runners.run_plaplace_u3_thesis_suite import (
    _attach_thesis_reference,
    _case_name,
    _iter_cases,
)
from src.problems.plaplace_u3.thesis.assignment import summarize_assignment_rows

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _refresh_derived_fields(row: dict[str, object]) -> dict[str, object]:
    """Recompute thesis and assignment fields from the current source tables."""
    base = {
        key: value
        for key, value in row.items()
        if not key.startswith(("assignment_", "delta_", "thesis_"))
        and key not in {"launcher", "process_count", "runtime_context", "timing_table"}
    }
    return _attach_thesis_reference(base)


def _expected_maxit_for_method(method: object) -> int:
    return 1000 if str(method).lower() == "mpa" else 500


def _row_budget_matches(row: dict[str, object]) -> bool:
    configured = row.get("configured_maxit")
    if configured is None:
        return False
    return int(configured) == int(_expected_maxit_for_method(row.get("method")))


def _row_result_in_repo(row: dict[str, object]) -> bool:
    result_path = row.get("result_path")
    if not result_path:
        return False
    try:
        Path(str(result_path)).resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return False
    return True


def _source_summary_rank(source_path: Path) -> int:
    try:
        rel = source_path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return 0
    parts = rel.parts
    if len(parts) >= 4 and parts[:3] == ("artifacts", "raw_results", "plaplace_u3_thesis_chunks"):
        return 0
    if len(parts) >= 3 and parts[:2] == ("artifacts", "raw_results"):
        return 2
    return 1


def _row_requires_publishable_timing(row: dict[str, object]) -> bool:
    table = str(row.get("table", ""))
    if table == "table_5_13":
        return True
    return str(row.get("timing_table", "")) == "table_5_12"


def _row_positive_solve_time(row: dict[str, object]) -> bool:
    value = row.get("solve_time_s")
    if value is None:
        return False
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _row_has_publishable_timing(row: dict[str, object]) -> bool:
    if not _row_requires_publishable_timing(row):
        return False
    return str(row.get("status", "")) == "completed" and _row_positive_solve_time(row)


def _row_timestamp(row: dict[str, object], *, source_path: Path) -> int:
    result_path = row.get("result_path")
    if result_path:
        candidate = Path(str(result_path))
        if candidate.exists():
            return int(candidate.stat().st_mtime_ns)
    return int(source_path.stat().st_mtime_ns)


def _row_score(row: dict[str, object], *, source_path: Path, source_priority: int) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(_row_budget_matches(row)),
        int(_row_has_publishable_timing(row)),
        int(_row_positive_solve_time(row)),
        int(_row_result_in_repo(row)),
        _source_summary_rank(source_path),
        int(source_priority),
        _row_timestamp(row, source_path=source_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-dir", type=str, default="artifacts/raw_results/plaplace_u3_thesis_chunks")
    parser.add_argument("--out", type=str, default="artifacts/raw_results/plaplace_u3_thesis_full/summary.json")
    parser.add_argument(
        "--extra-summary",
        action="append",
        default=[],
        help="Optional additional summary.json overlays to consider when selecting the best row per exact case",
    )
    parser.add_argument("--quick", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir)
    summaries = sorted(chunks_dir.glob("chunk*/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No chunk summaries found under {chunks_dir}")
    overlay_summaries = [Path(path) for path in args.extra_summary]
    for path in overlay_summaries:
        if not path.exists():
            raise FileNotFoundError(path)

    row_by_case: dict[str, dict[str, object]] = {}
    score_by_case: dict[str, tuple[int, int, int, int, int, int, int]] = {}
    first_summary = _load_json(summaries[0])
    for source_priority, source_path in [
        *((0, path) for path in summaries),
        *((1, path) for path in overlay_summaries),
    ]:
        payload = _load_json(source_path)
        for row in payload["rows"]:
            key = _case_name(
                type("CaseLike", (), {
                    "table": row["table"],
                    "method": row["method"],
                    "direction": row["direction"],
                    "dimension": row["dimension"],
                    "geometry": row["geometry"],
                    "level": row["level"],
                    "p": row["p"],
                    "epsilon": row["epsilon"],
                    "init_mode": row["init_mode"],
                })()
            )
            refreshed = _refresh_derived_fields(dict(row))
            score = _row_score(refreshed, source_path=source_path, source_priority=source_priority)
            if key not in row_by_case or score > score_by_case[key]:
                row_by_case[key] = refreshed
                score_by_case[key] = score

    cases = _iter_cases(full=not bool(args.quick))
    ordered_rows = []
    missing = []
    for case in cases:
        key = _case_name(case)
        row = row_by_case.get(key)
        if row is None:
            missing.append(key)
        else:
            ordered_rows.append(row)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} cases; first missing: {missing[:5]}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "suite": "plaplace_u3_thesis_quick" if bool(args.quick) else "plaplace_u3_thesis_full",
        "quick": bool(args.quick),
        "skip_reference": bool(first_summary["skip_reference"]),
        "chunked": True,
        "chunk_count": len(summaries),
        "constants": dict(first_summary["constants"]),
        "assignment_overview": summarize_assignment_rows(ordered_rows),
        "rows": ordered_rows,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(out_path), "num_rows": len(ordered_rows)}, indent=2))


if __name__ == "__main__":
    main()
