#!/usr/bin/env python3
"""Check that hard `[H]` float placements are intentionally allowlisted."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from common import PAPER_ROOT


FLOAT_RE = re.compile(
    r"\\begin\{(?P<env>figure|table|algorithm)\}\[H\](?P<body>.*?)\\end\{(?P=env)\}",
    re.DOTALL,
)
LABEL_RE = re.compile(r"\\label\{(?P<label>[^{}]+)\}")

ALLOWED_H_FLOATS: dict[tuple[str, str], str] = {
    ("algorithm", "alg:hybrid-newton"): "method pseudocode kept adjacent to solver-policy definition",
    ("algorithm", "alg:armijo-linesearch"): "method pseudocode kept adjacent to globalization definition",
    ("algorithm", "alg:colored-hessian-recovery"): "method pseudocode kept adjacent to sparse-recovery definition",
    ("algorithm", "alg:constitutive-ad-assembly"): "method pseudocode kept adjacent to constitutive-AD definition",
    ("figure", "fig:plasticity3d-validation"): "paired endpoint-surrogate validation panels must precede the summary table",
    ("table", "tab:plasticity3d-validation"): "validation summary must stay attached to the paired validation panels",
    ("figure", "fig:gl-results"): "single scaling figure interpreted immediately after the float",
    ("figure", "fig:hyperelasticity-results"): "single scaling figure interpreted immediately after the float",
    ("table", "tab:hyperelasticity-pmg-sensitivity"): "fixed-work PMG table interpreted immediately after the float",
    ("figure", "fig:hyperelasticity-karolina-pmg-scaling"): "paired fixed-work scaling figure kept with its timing table",
    ("table", "tab:hyperelasticity-karolina-pmg-scaling"): "paired fixed-work scaling table kept with its figure",
    ("table", "tab:topology-rank-consistency"): "rank-consistency table kept with the controlled-test interpretation",
    ("table", "tab:plasticity2d-reference-continuation"): "appendix support table kept with its brief interpretation",
}


@dataclass(frozen=True)
class HardFloat:
    path: Path
    line_number: int
    env: str
    label: str | None


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def find_hard_floats(paths: Iterable[Path]) -> list[HardFloat]:
    floats: list[HardFloat] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for match in FLOAT_RE.finditer(text):
            label_match = LABEL_RE.search(match.group("body"))
            label = label_match.group("label") if label_match is not None else None
            floats.append(
                HardFloat(
                    path=path,
                    line_number=_line_number(text, match.start()),
                    env=match.group("env"),
                    label=label,
                )
            )
    return floats


def unapproved_hard_floats(
    floats: Iterable[HardFloat],
    *,
    allowed: dict[tuple[str, str], str] = ALLOWED_H_FLOATS,
) -> list[HardFloat]:
    findings: list[HardFloat] = []
    for float_ in floats:
        if float_.label is None or (float_.env, float_.label) not in allowed:
            findings.append(float_)
    return findings


def default_tex_paths() -> list[Path]:
    return [PAPER_ROOT / "main.tex", *sorted((PAPER_ROOT / "sections").glob("*.tex"))]


def _format_float(float_: HardFloat) -> str:
    label = float_.label if float_.label is not None else "<missing label>"
    try:
        rel_path = float_.path.relative_to(PAPER_ROOT.parent)
    except ValueError:
        rel_path = float_.path
    return f"{rel_path}:{float_.line_number}: {float_.env}[H] label={label}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="TeX files to scan; defaults to paper/main.tex and sections")
    args = parser.parse_args(argv)
    paths = args.paths or default_tex_paths()
    floats = find_hard_floats(paths)
    findings = unapproved_hard_floats(floats)
    if findings:
        print("Unapproved hard `[H]` floats found:", file=sys.stderr)
        for finding in findings:
            print(f"  - {_format_float(finding)}", file=sys.stderr)
        print("Add a reasoned allowlist entry or relax the float placement.", file=sys.stderr)
        return 1
    print(f"Hard float placement OK: {len(floats)} allowlisted `[H]` floats.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
