#!/usr/bin/env python3
"""Merge chunked thesis-suite summaries into one canonical summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.runners.run_plaplace_u3_thesis_suite import _case_name, _iter_cases
from src.problems.plaplace_u3.thesis.assignment import (
    attach_assignment_metadata,
    summarize_assignment_rows,
)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-dir", type=str, default="artifacts/raw_results/plaplace_u3_thesis_chunks")
    parser.add_argument("--out", type=str, default="artifacts/raw_results/plaplace_u3_thesis_full/summary.json")
    parser.add_argument("--quick", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir)
    summaries = sorted(chunks_dir.glob("chunk*/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No chunk summaries found under {chunks_dir}")

    row_by_case: dict[str, dict[str, object]] = {}
    first_summary = _load_json(summaries[0])
    for path in summaries:
        payload = _load_json(path)
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
            row_by_case[key] = attach_assignment_metadata(dict(row))

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
