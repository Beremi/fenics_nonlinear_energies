#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

from common import BUILD_ROOT, PAPER_ROOT, ensure_paper_dirs, pt_to_in, write_json


EXPECTED_LAYOUT_CONTRACT = {
    "class": "article",
    "options": ["10pt", "a4paper"],
    "geometry_options": ["margin=1.5cm"],
}

TEX_TEMPLATE = r"""
\documentclass[%(options)s]{%(document_class)s}
\usepackage[%(geometry_options)s]{geometry}
\makeatletter
\newwrite\layoutfile
\begin{document}
\immediate\openout\layoutfile=%(outfile)s
\immediate\write\layoutfile{columnwidth_pt=\strip@pt\columnwidth}
\immediate\write\layoutfile{textwidth_pt=\strip@pt\textwidth}
\immediate\closeout\layoutfile
\end{document}
"""


def split_latex_options(options: str | None) -> list[str]:
    if not options:
        return []
    return [part.strip() for part in options.split(",") if part.strip()]


def read_main_layout_contract(main_tex: Path = PAPER_ROOT / "main.tex") -> dict[str, object]:
    text = main_tex.read_text(encoding="utf-8")
    documentclass = re.search(
        r"^\\documentclass(?:\[(?P<options>[^\]]*)\])?\{(?P<class>[^}]*)\}",
        text,
        flags=re.MULTILINE,
    )
    if documentclass is None:
        raise RuntimeError(f"Could not find documentclass in {main_tex}")

    geometry_matches = list(
        re.finditer(
            r"^\\usepackage(?:\[(?P<options>[^\]]*)\])?\{geometry\}",
            text,
            flags=re.MULTILINE,
        )
    )
    if len(geometry_matches) != 1:
        raise RuntimeError(f"Expected exactly one geometry package declaration in {main_tex}")

    return {
        "class": documentclass.group("class"),
        "options": split_latex_options(documentclass.group("options")),
        "geometry_options": split_latex_options(geometry_matches[0].group("options")),
    }


def validate_layout_contract(contract: dict[str, object]) -> None:
    mismatches: list[str] = []
    for key, expected in EXPECTED_LAYOUT_CONTRACT.items():
        observed = contract.get(key)
        if observed != expected:
            mismatches.append(f"{key}: expected {expected!r}, observed {observed!r}")
    if mismatches:
        details = "; ".join(mismatches)
        raise RuntimeError(
            "The measured figure layout is tied to the current A4 article "
            f"contract. Update measure_layout.py and the figure sizing policy before "
            f"regenerating target-template figures ({details})."
        )


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
    layout_contract = read_main_layout_contract()
    validate_layout_contract(layout_contract)
    with tempfile.TemporaryDirectory(prefix="paper_layout_", dir=BUILD_ROOT) as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        measurement_txt = tmp_dir / "layout_measurements.txt"
        tex_path = tmp_dir / "measure_layout.tex"
        tex_path.write_text(
            TEX_TEMPLATE
            % {
                "document_class": layout_contract["class"],
                "options": ",".join(layout_contract["options"]),
                "geometry_options": ",".join(layout_contract["geometry_options"]),
                "outfile": measurement_txt.name,
            },
            encoding="utf-8",
        )
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tmp_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        payload = parse_measurements(measurement_txt)
    payload["class"] = layout_contract["class"]
    payload["options"] = layout_contract["options"]
    payload["geometry_options"] = layout_contract["geometry_options"]
    payload["paper_root"] = str(PAPER_ROOT)
    write_json(args.out_json, payload)
    print(f"Wrote layout measurements to {args.out_json}")


if __name__ == "__main__":
    main()
