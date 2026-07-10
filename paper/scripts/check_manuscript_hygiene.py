#!/usr/bin/env python3
"""Check visible manuscript text for process-local wording leaks."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HygienePattern:
    name: str
    regex: re.Pattern[str]


DEFAULT_PATTERNS = (
    HygienePattern(
        "local filesystem or raw-result path",
        re.compile(r"(?i)(/home/|/workdir/|tmp/|tmp_work|local_env|\.venv|raw_results|source_compare)"),
    ),
    HygienePattern(
        "draft or review-process marker",
        re.compile(r"(?i)\b(TODO|FIXME|placeholder|locked|promoted|campaign|reviewer)\b"),
    ),
    HygienePattern(
        "internal implementation label",
        re.compile(r"(?i)\b(mainline|sourcefixed|source[- ]operator|source[- ]formula|source[- ]submission)\b"),
    ),
    HygienePattern(
        "process-local or defensive comparison framing",
        re.compile(
            r"(?i)\b("
            r"codebase|repository[- ]local|run[- ]tag|"
            r"broad\s+software(?:\s+ranking|\s+comparison)?|software\s+ranking|framework\s+ranking"
            r")\b"
        ),
    ),
    HygienePattern(
        "local platform name in manuscript body",
        re.compile(r"\b(Karolina|Barbora)\b"),
    ),
    HygienePattern(
        "overused weak construction",
        re.compile(r"(?i)\bwe\s+can\b"),
    ),
    HygienePattern(
        "implementation mesh alias",
        re.compile(r"\b(?:P[124]\(L1[_-]\d?\)|L1_2)\b"),
    ),
)


def pdf_text(pdf_path: Path) -> str:
    try:
        result = subprocess.run(
            ["pdftotext", str(pdf_path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit("pdftotext is required for manuscript hygiene checks") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        detail = f": {stderr}" if stderr else ""
        raise SystemExit(f"pdftotext failed for {pdf_path}{detail}") from exc
    return result.stdout


def manuscript_body(text: str, *, include_references: bool = False) -> str:
    if include_references:
        return text
    # ``pdftotext`` commonly emits a form-feed immediately before a heading
    # that starts a new page.  Match horizontal whitespace and that page-break
    # marker without allowing ``\s*`` to consume preceding manuscript lines.
    match = re.search(r"(?m)^[ \t\f]*References[ \t\r]*$", text)
    if match is None:
        return text
    return text[: match.start()]


def find_hygiene_findings(text: str, patterns: tuple[HygienePattern, ...] = DEFAULT_PATTERNS) -> list[str]:
    findings: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = " ".join(line.split())
        if not stripped:
            continue
        for pattern in patterns:
            match = pattern.regex.search(stripped)
            if match is None:
                continue
            findings.append(
                f"line {line_number}: {pattern.name}: {match.group(0)!r} in {stripped!r}"
            )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check visible manuscript text for local/process wording that should not appear in a submitted paper."
    )
    parser.add_argument("pdf", nargs="?", type=Path, default=Path("build/main.pdf"))
    parser.add_argument(
        "--include-references",
        action="store_true",
        help="scan the bibliography as well as the manuscript body",
    )
    args = parser.parse_args(argv)
    if not args.pdf.is_file():
        print(f"missing PDF: {args.pdf}", file=sys.stderr)
        return 2
    text = manuscript_body(pdf_text(args.pdf), include_references=args.include_references)
    findings = find_hygiene_findings(text)
    if findings:
        print("Manuscript hygiene check failed:", file=sys.stderr)
        for finding in findings:
            print(f"  - {finding}", file=sys.stderr)
        return 1
    print(f"Manuscript hygiene OK: {args.pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
