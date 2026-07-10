from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = REPO_ROOT / "paper"
BUILD_ROOT = PAPER_ROOT / "build"
FIGURES_ROOT = PAPER_ROOT / "figures" / "generated"
TABLES_ROOT = PAPER_ROOT / "tables" / "generated"
SCRIPTS_ROOT = PAPER_ROOT / "scripts"
LITERATURE_ROOT = PAPER_ROOT / "literature"
FULLTEXT_ROOT = LITERATURE_ROOT / "fulltext"
LAYOUT_JSON = BUILD_ROOT / "layout.json"
PAPER_BUNDLE_ROOT_ENV = "FNE_PAPER_BUNDLE_ROOT"
PAPER_BUNDLE_ROOT_OPTIONS = ("--paper-bundle-root", "--bundle-root")
HISTORICAL_PAPER_BUNDLE_RELATIVE = Path(
    "artifacts/reproduction/paper_submission_2026_07_08"
)
_DETERMINISTIC_DATE = datetime(2000, 1, 1, tzinfo=timezone.utc)
PDF_METADATA = {
    "Creator": "Matplotlib",
    "CreationDate": _DETERMINISTIC_DATE,
    "ModDate": _DETERMINISTIC_DATE,
}
PNG_METADATA = {"Software": "Matplotlib"}


def _paper_bundle_cli_override(argv: Sequence[str]) -> str | None:
    """Return the last explicit paper-bundle root in ``argv``, if present."""

    override: str | None = None
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--":
            break
        for option in PAPER_BUNDLE_ROOT_OPTIONS:
            if argument == option:
                if index + 1 >= len(argv):
                    raise ValueError(f"{option} requires a nonempty path")
                override = argv[index + 1]
                index += 1
                break
            prefix = f"{option}="
            if argument.startswith(prefix):
                override = argument[len(prefix) :]
                break
        index += 1
    return override


def resolve_paper_bundle_root(
    override: str | Path | None = None,
    *,
    repo_root: Path = REPO_ROOT,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve and validate the shared paper evidence-bundle root.

    The explicit argument takes precedence over ``FNE_PAPER_BUNDLE_ROOT``.
    Relative paths are repository-relative.  Paper bundles must remain inside
    ``artifacts/reproduction`` so generated manifests can use safe,
    repository-relative paths and archive-neutral validation remains valid.
    """

    environment = os.environ if environ is None else environ
    raw_value: str | Path
    if override is not None:
        raw_value = override
    elif PAPER_BUNDLE_ROOT_ENV in environment:
        raw_value = environment[PAPER_BUNDLE_ROOT_ENV]
    else:
        raw_value = HISTORICAL_PAPER_BUNDLE_RELATIVE

    raw_text = str(raw_value).strip()
    if not raw_text:
        raise ValueError(
            f"Paper bundle root is empty; unset {PAPER_BUNDLE_ROOT_ENV} or provide a nonempty path"
        )

    resolved_repo = repo_root.expanduser().resolve()
    candidate = Path(raw_text).expanduser()
    if not candidate.is_absolute():
        candidate = resolved_repo / candidate
    candidate = candidate.resolve()
    reproduction_root = (resolved_repo / "artifacts" / "reproduction").resolve()
    try:
        relative = candidate.relative_to(reproduction_root)
    except ValueError as exc:
        raise ValueError(
            "Paper bundle root must be inside the repository's "
            f"artifacts/reproduction directory: {candidate}"
        ) from exc
    if not relative.parts:
        raise ValueError(
            "Paper bundle root must name one campaign below artifacts/reproduction, "
            "not the reproduction directory itself"
        )
    if candidate.exists() and not candidate.is_dir():
        raise ValueError(f"Paper bundle root exists but is not a directory: {candidate}")
    return candidate


def configured_paper_bundle_root(
    argv: Sequence[str] | None = None,
    *,
    repo_root: Path = REPO_ROOT,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the CLI/environment paper-bundle configuration for a process."""

    arguments = sys.argv[1:] if argv is None else argv
    return resolve_paper_bundle_root(
        _paper_bundle_cli_override(arguments),
        repo_root=repo_root,
        environ=environ,
    )


PAPER_BUNDLE_ROOT = configured_paper_bundle_root()
PAPER_BUNDLE_INPUT_ROOT = PAPER_BUNDLE_ROOT / "inputs"
PAPER_BUNDLE_MANIFEST = PAPER_BUNDLE_ROOT / "manifest.json"


def add_paper_bundle_root_argument(parser: argparse.ArgumentParser) -> None:
    """Add the common paper-bundle override to a paper-script CLI."""

    parser.add_argument(
        *PAPER_BUNDLE_ROOT_OPTIONS,
        dest="paper_bundle_root",
        type=resolve_paper_bundle_root,
        default=PAPER_BUNDLE_ROOT,
        metavar="PATH",
        help=(
            "paper evidence-bundle campaign root below artifacts/reproduction; "
            f"defaults to ${PAPER_BUNDLE_ROOT_ENV} or "
            f"{HISTORICAL_PAPER_BUNDLE_RELATIVE.as_posix()}"
        ),
    )


def ensure_paper_dirs() -> None:
    for path in (BUILD_ROOT, FIGURES_ROOT, TABLES_ROOT, SCRIPTS_ROOT, LITERATURE_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_layout() -> dict[str, float]:
    if not LAYOUT_JSON.exists():
        raise FileNotFoundError(f"Layout JSON missing: {LAYOUT_JSON}")
    payload = read_json(LAYOUT_JSON)
    return {
        "columnwidth_pt": float(payload["columnwidth_pt"]),
        "textwidth_pt": float(payload["textwidth_pt"]),
        "columnwidth_in": float(payload["columnwidth_in"]),
        "textwidth_in": float(payload["textwidth_in"]),
    }


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd or REPO_ROOT, check=True)


def pt_to_in(value_pt: float) -> float:
    return float(value_pt) / 72.27


def column_figure_size(
    layout: dict[str, float],
    *,
    width_scale: float = 1.0,
    height_ratio: float | None = None,
    height_in: float | None = None,
) -> tuple[float, float]:
    width = layout["columnwidth_in"] * width_scale
    if height_in is not None:
        return width, height_in
    if height_ratio is None:
        height_ratio = 0.62
    return width, width * height_ratio


def text_figure_size(
    layout: dict[str, float],
    *,
    width_scale: float = 1.0,
    height_ratio: float | None = None,
    height_in: float | None = None,
) -> tuple[float, float]:
    width = layout["textwidth_in"] * width_scale
    if height_in is not None:
        return width, height_in
    if height_ratio is None:
        height_ratio = 0.38
    return width, width * height_ratio


def paper_width_in(layout: dict[str, float], preset: str = "full") -> float:
    preset = str(preset)
    scales = {
        "subfigure": 0.46,
        "full": 1.0,
        "medium": 0.84,
        "narrow": 0.72,
    }
    if preset not in scales:
        raise ValueError(f"Unsupported figure preset {preset!r}")
    return layout["textwidth_in"] * scales[preset]


def paper_figure_size(
    layout: dict[str, float],
    *,
    preset: str = "full",
    height_ratio: float | None = None,
    height_in: float | None = None,
) -> tuple[float, float]:
    width = paper_width_in(layout, preset)
    if height_in is not None:
        return width, height_in
    if height_ratio is None:
        height_ratio = 0.40
    return width, width * height_ratio


def configure_paper_matplotlib(font_size: float = 10.0):
    from experiments.analysis.docs_assets.common import configure_matplotlib

    plt = configure_matplotlib()
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "legend.fontsize": font_size,
            "xtick.labelsize": font_size - 0.5,
            "ytick.labelsize": font_size - 0.5,
            "axes.titlepad": 2.5,
        }
    )
    return plt


def save_pdf_and_png(fig, pdf_path: Path, *, png_dpi: int = 240) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format="pdf", dpi=600, metadata=PDF_METADATA)
    fig.savefig(pdf_path.with_suffix(".png"), format="png", dpi=png_dpi, metadata=PNG_METADATA)


def copy_asset(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = str(text)
    for key, value in replacements.items():
        out = out.replace(key, value)
    return out


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
