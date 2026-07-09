"""Check that figure and table numbers appear monotonically in the LaTeX aux file."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ENTRY_RE = re.compile(
    r"\\@writefile\{(?P<list>lof|lot)\}"
    r"\{\\contentsline \{(?P<kind>figure|table)\}"
    r"\{\\numberline \{(?P<number>[^}]*)\}"
)
TOKEN_RE = re.compile(r"[A-Za-z]+|\d+")


def _letter_value(token: str) -> int:
    value = 0
    for char in token.upper():
        value = value * 26 + (ord(char) - ord("A") + 1)
    return 1000 + value


def _number_key(label: str) -> tuple[int, ...] | None:
    tokens = TOKEN_RE.findall(label)
    if not tokens:
        return None
    values: list[int] = []
    for token in tokens:
        if token.isdigit():
            values.append(int(token))
        else:
            values.append(_letter_value(token))
    return tuple(values)


def check_aux_order(aux_path: Path) -> list[str]:
    previous: dict[str, tuple[tuple[int, ...], str, int]] = {}
    failures: list[str] = []
    for line_number, line in enumerate(aux_path.read_text(encoding="utf-8").splitlines(), start=1):
        match = ENTRY_RE.search(line)
        if not match:
            continue
        kind = match.group("kind")
        label = match.group("number").strip()
        key = _number_key(label)
        if key is None:
            continue
        old = previous.get(kind)
        if old is not None:
            old_key, old_label, old_line = old
            if key <= old_key:
                failures.append(
                    f"{kind} number {label!r} at line {line_number} appears after "
                    f"{old_label!r} at line {old_line}"
                )
        previous[kind] = (key, label, line_number)
    return failures


def main(argv: list[str]) -> int:
    aux_path = Path(argv[1]) if len(argv) > 1 else Path("build/main.aux")
    if not aux_path.is_file():
        print(f"missing aux file: {aux_path}", file=sys.stderr)
        return 2
    failures = check_aux_order(aux_path)
    if failures:
        print("LaTeX aux ordering check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"Figure/table aux ordering OK: {aux_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
