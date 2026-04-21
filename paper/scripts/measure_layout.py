#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

from common import BUILD_ROOT, PAPER_ROOT, ensure_paper_dirs, pt_to_in, write_json


TEX_TEMPLATE = r"""
\documentclass[10pt,a4paper]{article}
\usepackage[margin=1.5cm]{geometry}
\makeatletter
\newwrite\layoutfile
\begin{document}
\immediate\openout\layoutfile=%(outfile)s
\immediate\write\layoutfile{columnwidth_pt=\strip@pt\columnwidth}
\immediate\write\layoutfile{textwidth_pt=\strip@pt\textwidth}
\immediate\closeout\layoutfile
\end{document}
"""


def parse_measurements(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([a-z_]+)=([0-9.]+)$", line.strip())
        if not match:
            continue
        values[match.group(1)] = float(match.group(2))
    if "columnwidth_pt" not in values or "textwidth_pt" not in values:
        raise RuntimeError(f"Failed to parse layout measurements from {path}")
    return {
        "columnwidth_pt": values["columnwidth_pt"],
        "textwidth_pt": values["textwidth_pt"],
        "columnwidth_in": pt_to_in(values["columnwidth_pt"]),
        "textwidth_in": pt_to_in(values["textwidth_pt"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure LaTeX article layout widths for paper figures.")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=BUILD_ROOT / "layout.json",
        help="Where to write the measured layout JSON.",
    )
    args = parser.parse_args()

    ensure_paper_dirs()
    with tempfile.TemporaryDirectory(prefix="paper_layout_", dir=BUILD_ROOT) as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        measurement_txt = tmp_dir / "layout_measurements.txt"
        tex_path = tmp_dir / "measure_layout.tex"
        tex_path.write_text(TEX_TEMPLATE % {"outfile": measurement_txt.name}, encoding="utf-8")
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tmp_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        payload = parse_measurements(measurement_txt)
    payload["class"] = "article"
    payload["options"] = ["10pt", "a4paper"]
    payload["paper_root"] = str(PAPER_ROOT)
    write_json(args.out_json, payload)
    print(f"Wrote layout measurements to {args.out_json}")


if __name__ == "__main__":
    main()
