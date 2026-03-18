"""Shared helpers for rebuilding curated documentation assets."""

from __future__ import annotations

import csv
import json
import math
import shlex
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_ASSETS_ROOT = REPO_ROOT / "docs" / "assets"
BUILD_ROOT = REPO_ROOT / "experiments" / "analysis" / "docs_assets"
DATA_ROOT = BUILD_ROOT / "data"
PDF_ROOT = DOCS_ASSETS_ROOT
PNG_ROOT = DOCS_ASSETS_ROOT
GIF_ROOT = DOCS_ASSETS_ROOT
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

FIGURE_WIDTH_CM = 11.0
OVERVIEW_FONT_PT = 12.0

FAMILY_TITLES = {
    "plaplace": "pLaplace",
    "ginzburg_landau": "GinzburgLandau",
    "hyperelasticity": "HyperElasticity",
    "topology": "Topology",
}

IMPLEMENTATION_STYLES = {
    "fenics_custom": {"label": "FEniCS custom", "color": "#111111", "marker": "o", "linestyle": "-"},
    "fenics_snes": {"label": "FEniCS SNES", "color": "#4d4d4d", "marker": "s", "linestyle": "--"},
    "jax_serial": {"label": "pure JAX serial", "color": "#6b6b6b", "marker": "^", "linestyle": "-."},
    "jax_petsc_element": {"label": "JAX+PETSc element", "color": "#2f2f2f", "marker": "D", "linestyle": ":"},
    "jax_petsc_local_sfd": {"label": "JAX+PETSc local-SFD", "color": "#8a8a8a", "marker": "v", "linestyle": (0, (5, 2))},
    "jax_parallel": {"label": "parallel JAX+PETSc", "color": "#2f2f2f", "marker": "D", "linestyle": ":"},
}

THREAD_ENV = {
    "JAX_PLATFORMS": "cpu",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
}


def ensure_docs_asset_dirs() -> None:
    for path in (DOCS_ASSETS_ROOT, BUILD_ROOT, DATA_ROOT, PDF_ROOT, PNG_ROOT, GIF_ROOT):
        path.mkdir(parents=True, exist_ok=True)
    for family in FAMILY_TITLES:
        (DATA_ROOT / family).mkdir(parents=True, exist_ok=True)
        (PDF_ROOT / family).mkdir(parents=True, exist_ok=True)
        (PNG_ROOT / family).mkdir(parents=True, exist_ok=True)
        (GIF_ROOT / family).mkdir(parents=True, exist_ok=True)


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def shell_join(argv: Sequence[str]) -> str:
    return shlex.join([str(part) for part in argv])


def normalize_command(command: str) -> str:
    if not command:
        return ""
    replacements = [
        (str(PYTHON), "./.venv/bin/python"),
        (str(REPO_ROOT) + "/", ""),
        (str(REPO_ROOT), "."),
    ]
    normalized = command
    for source, target in replacements:
        normalized = normalized.replace(source, target)
    return normalized


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def format_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "nan"
    return f"{number:.{digits}f}"


def format_int(value: Any) -> str:
    if value is None:
        return "-"
    return str(int(value))


def record_provenance(
    path: Path,
    *,
    script_name: str,
    argv: Sequence[str] | None = None,
    inputs: Sequence[str] | None = None,
    outputs: Sequence[str] | None = None,
    notes: str = "",
) -> None:
    command_argv = list(argv or ["./.venv/bin/python", *sys.argv])
    payload = {
        "script": script_name,
        "command": shell_join(command_argv),
        "cwd": repo_rel(REPO_ROOT),
        "inputs": list(inputs or []),
        "outputs": list(outputs or []),
        "notes": notes,
    }
    write_json(path, payload)


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": OVERVIEW_FONT_PT,
            "axes.titlesize": OVERVIEW_FONT_PT,
            "axes.labelsize": OVERVIEW_FONT_PT,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.4,
            "grid.linewidth": 0.6,
            "grid.linestyle": ":",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 0.8,
            "axes.titlepad": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def figure_width_in() -> float:
    return FIGURE_WIDTH_CM / 2.54


def springer_figure_size(height_ratio: float | None = None, *, height_in: float | None = None) -> tuple[float, float]:
    width = figure_width_in()
    if height_in is not None:
        return width, height_in
    assert height_ratio is not None
    return width, width * height_ratio


def ideal_strong_scaling(ranks: Sequence[float], times: Sequence[float]) -> np.ndarray:
    x = np.asarray(ranks, dtype=np.float64)
    y = np.asarray(times, dtype=np.float64)
    return y[0] * x[0] / x


def ideal_mesh_scaling(problem_sizes: Sequence[float], times: Sequence[float]) -> np.ndarray:
    x = np.asarray(problem_sizes, dtype=np.float64)
    y = np.asarray(times, dtype=np.float64)
    return y[0] * x / x[0]


def save_pdf(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", dpi=600)
    preview_path = PNG_ROOT / path.relative_to(PDF_ROOT)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview_path.with_suffix(".png"), format="png", dpi=320)
    fig.clf()


def family_pdf(name: str) -> Path:
    family, filename = name.split("/", 1)
    return PDF_ROOT / family / filename


def family_png(name: str) -> Path:
    family, filename = name.split("/", 1)
    return PNG_ROOT / family / Path(filename).with_suffix(".png")


def family_gif(name: str) -> Path:
    family, filename = name.split("/", 1)
    return GIF_ROOT / family / Path(filename).with_suffix(".gif")


def family_data(name: str) -> Path:
    family, filename = name.split("/", 1)
    return DATA_ROOT / family / filename


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def build_boundary_faces(tetrahedra: np.ndarray) -> np.ndarray:
    faces = np.asarray(
        [
            tetrahedra[:, [0, 1, 2]],
            tetrahedra[:, [0, 1, 3]],
            tetrahedra[:, [0, 2, 3]],
            tetrahedra[:, [1, 2, 3]],
        ],
        dtype=np.int32,
    ).reshape((-1, 3))
    counts: Counter[tuple[int, int, int]] = Counter(tuple(sorted(face.tolist())) for face in faces)
    boundary = [face for face in faces if counts[tuple(sorted(face.tolist()))] == 1]
    return np.asarray(boundary, dtype=np.int32)


def parse_summary_status(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("result", "unknown")) for row in rows)
    return {key: int(value) for key, value in sorted(counts.items())}


def implementation_style(name: str) -> dict[str, str]:
    return IMPLEMENTATION_STYLES.get(name, {"label": name, "color": "#4c4c4c", "marker": "o"})
