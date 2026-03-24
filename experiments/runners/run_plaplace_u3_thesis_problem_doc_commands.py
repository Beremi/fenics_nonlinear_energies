#!/usr/bin/env python3
"""Execute every bash command block from the merged pLaplace_u3 thesis docs page."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOC = REPO_ROOT / "docs" / "problems" / "pLaplace_u3_thesis_replications.md"


def _extract_bash_blocks(text: str) -> list[str]:
    return [match.group(1).strip() for match in re.finditer(r"```bash\n(.*?)```", text, flags=re.DOTALL)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", type=str, default=str(DEFAULT_DOC))
    args = parser.parse_args()

    doc_path = Path(args.doc)
    commands = _extract_bash_blocks(doc_path.read_text(encoding="utf-8"))
    if not commands:
        raise RuntimeError(f"No bash blocks found in {doc_path}")

    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {command.splitlines()[0]}")
        subprocess.run(command, cwd=REPO_ROOT, shell=True, check=True, text=True)


if __name__ == "__main__":
    main()
