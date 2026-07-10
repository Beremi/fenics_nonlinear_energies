#!/usr/bin/env python3
"""Offline integrity preflight for a prepared Karolina campaign archive.

This command only reads local files.  It never invokes or queries Slurm and is
safe to run both before handoff and after a campaign archive is copied back.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.runners.paper_revision_karolina.prepare_campaign import (
    DEFAULT_MATRIX,
    offline_preflight,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON receipt. The default prints the result only.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        result = offline_preflight(args.campaign_root, matrix=args.matrix)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
