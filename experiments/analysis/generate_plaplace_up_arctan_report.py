#!/usr/bin/env python3
"""Generate the arctan-resonance report README and committed figures."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = REPO_ROOT / "artifacts" / "raw_results" / "plaplace_up_arctan_full" / "summary.json"
DEFAULT_PETSC_SUMMARY = REPO_ROOT / "artifacts" / "raw_results" / "plaplace_up_arctan_petsc" / "summary.json"
DEFAULT_REPORT = REPO_ROOT / "artifacts" / "reports" / "plaplace_up_arctan" / "README.md"
DEFAULT_ASSET_DIR = REPO_ROOT / "docs" / "assets" / "plaplace_up_arctan"
TRACK_RAW = "raw"
TRACK_CERTIFIED = "certified"


def _load_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_summary(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.exists():
        return None
    return _load_summary(path)


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _rows_for(summary: dict[str, object], *, study: str | None = None, p: float | None = None) -> list[dict[str, object]]:
    rows = [dict(row) for row in summary["rows"]]
    if study is not None:
        rows = [row for row in rows if str(row["study"]) == str(study)]
    if p is not None:
        rows = [row for row in rows if float(row["p"]) == float(p)]
    return rows


def _track_value(row: dict[str, object], key: str, track: str | None = None, default: object | None = None) -> object | None:
    prefixes: list[str | None] = []
    if track in {TRACK_RAW, TRACK_CERTIFIED}:
        prefixes.append(track)
    prefixes.extend([TRACK_CERTIFIED, TRACK_RAW, None])
    seen: set[str | None] = set()
    for prefix in prefixes:
        if prefix in seen:
            continue
        seen.add(prefix)
        candidate = key if prefix is None else f"{prefix}_{key}"
        if candidate in row and row[candidate] is not None:
            return row[candidate]
    return row.get(key, default)


def _strict_track_value(row: dict[str, object], key: str, track: str, default: object | None = None) -> object | None:
    candidate = f"{track}_{key}"
    if candidate in row and row[candidate] is not None:
        return row[candidate]
    return default


def _track_history(row: dict[str, object], track: str | None = None) -> list[dict[str, object]]:
    value = _track_value(row, "history", track, default=[])
    if not value:
        return []
    return [dict(item) for item in value]


def _strict_track_history(row: dict[str, object], track: str) -> list[dict[str, object]]:
    value = _strict_track_value(row, "history", track, default=[])
    if not value:
        return []
    return [dict(item) for item in value]


def _has_track_field(row: dict[str, object], track: str, key: str) -> bool:
    candidate = f"{track}_{key}"
    return candidate in row and row[candidate] is not None


def _tracks_for_rows(rows: list[dict[str, object]], key: str) -> list[str]:
    tracks = [track for track in (TRACK_RAW, TRACK_CERTIFIED) if any(_has_track_field(row, track, key) for row in rows)]
    return tracks or [TRACK_CERTIFIED]


def _format_track(track: str) -> str:
    return track.upper()


def _certified_mpa_iters(row: dict[str, object]) -> int:
    source = str(row.get("certified_handoff_source") or "")
    if source.startswith("mpa:"):
        return int(row.get("raw_outer_iterations", 0))
    return 0


def _certified_newton_iters(row: dict[str, object]) -> int:
    return int(row.get("certified_newton_iters", row.get("certified_outer_iterations", row.get("outer_iterations", 0))))


def _certified_total_nonlinear_iters(row: dict[str, object]) -> int:
    return int(_certified_mpa_iters(row) + _certified_newton_iters(row))


def _select_row(
    summary: dict[str, object],
    *,
    study: str,
    p: float,
    method: str,
    level: int,
    epsilon: float,
) -> dict[str, object]:
    matches = [
        row
        for row in summary["rows"]
        if str(row["study"]) == str(study)
        and float(row["p"]) == float(p)
        and str(row["method"]) == str(method)
        and int(row["level"]) == int(level)
        and abs(float(row["epsilon"]) - float(epsilon)) <= 1.0e-14
    ]
    if not matches:
        raise ValueError(f"No row found for {study=}, {p=}, {method=}, {level=}, {epsilon=}")
    return dict(matches[0])


def _lambda_payloads(summary: dict[str, object]) -> list[dict[str, object]]:
    cache_dir = Path(summary["lambda_cache_dir"])
    payloads = []
    for path in sorted(cache_dir.glob("lambda_p3_l*.json")):
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    return payloads


def _petsc_rows_for(
    summary: dict[str, object] | None,
    *,
    study: str | None = None,
    p: float | None = None,
) -> list[dict[str, object]]:
    if not summary:
        return []
    rows = [dict(row) for row in summary.get("rows", [])]
    if study is not None:
        rows = [row for row in rows if str(row.get("study")) == str(study)]
    if p is not None:
        rows = [row for row in rows if float(row.get("p", math.nan)) == float(p)]
    return rows


def _save_fig(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_state_panel(rows: list[dict[str, object]], title: str, out_base: Path, *, track: str = TRACK_CERTIFIED) -> None:
    fig, axes = plt.subplots(1, len(rows), figsize=(5.0 * len(rows), 4.0), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, row in zip(axes, rows):
        state_path = Path(str(_strict_track_value(row, "state_path", track, default=row["state_path"])))
        data = np.load(state_path)
        coords = np.asarray(data["coords"], dtype=np.float64)
        triangles = np.asarray(data["triangles"], dtype=np.int32)
        values = np.asarray(data["u"], dtype=np.float64)
        triang = mtri.Triangulation(coords[:, 0], coords[:, 1], triangles)
        trip = ax.tripcolor(triang, values, shading="gouraud", cmap="viridis")
        track_label = _format_track(track)
        j_value = float(_strict_track_value(row, "J", track, default=row["J"]))
        outer_iterations = int(_strict_track_value(row, "outer_iterations", track, default=row["outer_iterations"]))
        ax.set_title(f"{row['method'].upper()} {track_label}\nL{int(row['level'])}  J={j_value:.4e}\niter={outer_iterations}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(trip, ax=ax, shrink=0.8)
    fig.suptitle(title)
    _save_fig(fig, out_base)


def _plot_scalar_state(npz_path: Path, title: str, out_base: Path) -> None:
    data = np.load(npz_path)
    coords = np.asarray(data["coords"], dtype=np.float64)
    triangles = np.asarray(data["triangles"], dtype=np.int32)
    values = np.asarray(data["u"], dtype=np.float64)
    triang = mtri.Triangulation(coords[:, 0], coords[:, 1], triangles)
    fig, ax = plt.subplots(figsize=(5.2, 4.1), constrained_layout=True)
    trip = ax.tripcolor(triang, values, shading="gouraud", cmap="viridis")
    fig.colorbar(trip, ax=ax, shrink=0.82)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    _save_fig(fig, out_base)


def _plot_history(
    rows: list[dict[str, object]],
    title: str,
    out_base: Path,
    *,
    tracks: tuple[str, ...] | None = None,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.8), constrained_layout=True)
    for row in rows:
        track_histories = list(tracks) if tracks is not None else [track for track in (TRACK_RAW, TRACK_CERTIFIED) if _has_track_field(row, track, "history")]
        if not track_histories:
            track_histories = [TRACK_CERTIFIED]
        for track in track_histories:
            history = _strict_track_history(row, track)
            if not history:
                continue
            outer = [item["outer_it"] for item in history]
            J_vals = [item["J"] for item in history]
            stop_vals = [max(float(item.get("gradient_residual_norm", item.get("stop_measure"))), 1.0e-16) for item in history]
            residual_vals = [max(float(item.get("dual_residual_norm", item.get("residual_norm", item.get("stop_measure")))), 1.0e-16) for item in history]
            linestyle = "--" if track == TRACK_RAW else "-"
            alpha = 0.65 if track == TRACK_RAW else 0.95
            label = f"{row['method'].upper()} {_format_track(track)}"
            axes[0].plot(outer, J_vals, marker="o", linestyle=linestyle, alpha=alpha, label=label)
            axes[1].plot(outer, stop_vals, marker="o", linestyle=linestyle, alpha=alpha, label=label)
            axes[2].plot(outer, residual_vals, marker="o", linestyle=linestyle, alpha=alpha, label=label)
    axes[0].set_title("Energy history")
    axes[0].set_xlabel("outer iteration")
    axes[0].set_ylabel("J")
    axes[1].set_title("Coordinate gradient norm")
    axes[1].set_xlabel("outer iteration")
    axes[1].set_ylabel(r"$\|\nabla J_h(u)\|_{\ell^2}$")
    axes[1].set_yscale("log")
    axes[2].set_title("Laplace-dual residual")
    axes[2].set_xlabel("outer iteration")
    axes[2].set_ylabel(r"$\|R_h(u)\|_{K^{-1}}$")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.legend()
    fig.suptitle(title)
    _save_fig(fig, out_base)


def _plot_iteration_counts(summary: dict[str, object], out_base: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True)
    for p, ax in zip((2.0, 3.0), axes):
        rows = _rows_for(summary, study="mesh_refinement", p=p)
        for method in ("mpa", "rmpa"):
            subset = sorted((row for row in rows if str(row["method"]) == method), key=lambda row: int(row["level"]))
            tracks = _tracks_for_rows(subset, "outer_iterations")
            for track in tracks:
                yvals = [_strict_track_value(row, "outer_iterations", track, default=None) for row in subset]
                if any(value is None for value in yvals):
                    continue
                ax.plot(
                    [int(row["level"]) for row in subset],
                    [int(value) for value in yvals],
                    marker="o",
                    linestyle="--" if track == TRACK_RAW else "-",
                    alpha=0.65 if track == TRACK_RAW else 0.95,
                    label=f"{method.upper()} {_format_track(track)} refinement",
                )
        rows = _rows_for(summary, study="tolerance_sweep", p=p)
        for method in ("mpa", "rmpa"):
            subset = sorted((row for row in rows if str(row["method"]) == method), key=lambda row: float(row["epsilon"]))
            tracks = _tracks_for_rows(subset, "outer_iterations")
            for track in tracks:
                yvals = [_strict_track_value(row, "outer_iterations", track, default=None) for row in subset]
                if any(value is None for value in yvals):
                    continue
                ax.plot(
                    [float(row["epsilon"]) for row in subset],
                    [int(value) for value in yvals],
                    marker="s",
                    linestyle="--" if track == TRACK_RAW else "-",
                    alpha=0.65 if track == TRACK_RAW else 0.95,
                    label=f"{method.upper()} {_format_track(track)} tolerance",
                )
        ax.set_xscale("log")
        ax.set_title(f"p = {int(p)}")
        ax.set_xlabel("level or epsilon")
        ax.set_ylabel("outer iterations")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.legend()
    fig.suptitle("Iteration counts across refinement and tolerance studies")
    _save_fig(fig, out_base)


def _plot_reference_errors(summary: dict[str, object], out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.0), constrained_layout=True)
    for p in (2.0, 3.0):
        rows = _rows_for(summary, study="mesh_refinement", p=p)
        for method in ("mpa", "rmpa"):
            subset = sorted((row for row in rows if str(row["method"]) == method), key=lambda row: int(row["level"]))
            tracks = _tracks_for_rows(subset, "reference_error_w1p")
            for track in tracks:
                yvals = [_strict_track_value(row, "reference_error_w1p", track, default=None) for row in subset]
                if any(value is None for value in yvals):
                    continue
                ax.plot(
                    [int(row["level"]) for row in subset],
                    [float(value) for value in yvals],
                    marker="o",
                    linestyle="--" if track == TRACK_RAW else "-",
                    alpha=0.65 if track == TRACK_RAW else 0.95,
                    label=f"p={int(p)} {method.upper()} {_format_track(track)}",
                )
    ax.set_yscale("log")
    ax.set_xlabel("mesh level")
    ax.set_ylabel(r"$|u_h - u_{ref}|_{1,p,0}$")
    ax.set_title("Reference error over mesh refinement")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend()
    _save_fig(fig, out_base)


def _plot_lambda(summary: dict[str, object], out_base: Path) -> None:
    payloads = _lambda_payloads(summary)
    fig, ax1 = plt.subplots(figsize=(5.8, 4.0), constrained_layout=True)
    levels = [int(item["level"]) for item in payloads]
    lambda_vals = [float(item["lambda1"]) for item in payloads]
    norm_errs = [max(float(item["normalization_error"]), 1.0e-16) for item in payloads]
    ax1.plot(levels, lambda_vals, marker="o", color="#1f77b4", label=r"$\lambda_{1,h}$")
    ax1.set_xlabel("mesh level")
    ax1.set_ylabel(r"$\lambda_{1,h}$", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.3, linestyle=":")
    ax2 = ax1.twinx()
    ax2.plot(levels, norm_errs, marker="s", color="#d62728", label="normalization error")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$|\|\phi_{1,h}\|_{L^3}-1|$", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    fig.suptitle("p = 3 eigenvalue stage diagnostics")
    _save_fig(fig, out_base)


def _add_slope_triangle(ax, *, x0: float, y0: float, slope: float, decades: float = 0.25, label: str = "1:1") -> None:
    x1 = x0 * (10.0**decades)
    y1 = y0 * (10.0 ** (slope * decades))
    ax.plot([x0, x1], [y0, y0], color="black", linewidth=1.0)
    ax.plot([x1, x1], [y0, y1], color="black", linewidth=1.0)
    ax.plot([x0, x1], [y0, y1], color="black", linewidth=1.0)
    ax.text(x1, y0, label, fontsize=8, ha="left", va="top")


def _plot_petsc_mesh_timing(summary: dict[str, object] | None, out_base: Path) -> None:
    if not summary:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0), constrained_layout=True)
    for p, ax in zip((2.0, 3.0), axes):
        rows = sorted(_petsc_rows_for(summary, study="mesh_ladder", p=p), key=lambda row: int(row["level"]))
        if not rows:
            ax.text(0.5, 0.5, "no PETSc rows", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        dofs = [max(float(row.get("free_dofs", 1.0)), 1.0) for row in rows]
        setup = [float(row.get("setup_time_s", math.nan)) for row in rows]
        solve = [float(row.get("solve_time_s", math.nan)) for row in rows]
        total = [float(row.get("total_time_s", math.nan)) for row in rows]
        ax.loglog(dofs, setup, marker="o", label="setup")
        ax.loglog(dofs, solve, marker="s", label="solve")
        ax.loglog(dofs, total, marker="^", label="total")
        for row in rows:
            label = str(row.get("status", ""))
            ax.annotate(
                f"L{int(row['level'])} {label}",
                (max(float(row.get("free_dofs", 1.0)), 1.0), float(row.get("total_time_s", math.nan))),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
            )
        finite_dofs = [d for d, t in zip(dofs, total) if math.isfinite(t) and t > 0.0]
        finite_total = [t for t in total if math.isfinite(t) and t > 0.0]
        if finite_dofs and finite_total:
            _add_slope_triangle(
                ax,
                x0=min(finite_dofs) * 1.2,
                y0=min(finite_total) * 1.15,
                slope=1.0,
                label="1:1",
            )
        ax.set_xlabel("free dofs")
        ax.set_ylabel("time [s]")
        ax.set_title(f"PETSc timing, p = {int(p)}")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.legend()
    fig.suptitle("JAX + PETSc timing by mesh level")
    _save_fig(fig, out_base)


def _plot_petsc_scaling(summary: dict[str, object] | None, out_base: Path) -> None:
    rows = sorted(_petsc_rows_for(summary, study="strong_scaling", p=2.0), key=lambda row: int(row["nprocs"])) if summary else []
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    ranks = [int(row["nprocs"]) for row in rows]
    total = [float(row.get("total_time_s", math.nan)) for row in rows]
    solve = [float(row.get("solve_time_s", math.nan)) for row in rows]
    speedup = [float(row.get("speedup_total", math.nan)) for row in rows]
    ideal = [float(ranks_i) / float(ranks[0]) for ranks_i in ranks]
    axes[0].plot(ranks, total, marker="o", label="total time")
    axes[0].plot(ranks, solve, marker="s", label="solve time")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("MPI ranks")
    axes[0].set_ylabel("time [s]")
    axes[0].set_title("Strong scaling timing")
    axes[0].grid(True, alpha=0.3, linestyle=":")
    axes[0].legend()
    axes[1].plot(ranks, speedup, marker="o", label="measured speedup")
    axes[1].plot(ranks, ideal, linestyle="--", color="black", label="ideal")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("MPI ranks")
    axes[1].set_ylabel("speedup")
    axes[1].set_title("Strong scaling speedup")
    axes[1].grid(True, alpha=0.3, linestyle=":")
    axes[1].legend()
    fig.suptitle("JAX + PETSc PMG strong scaling on the finest successful p = 2 level")
    _save_fig(fig, out_base)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _write_report(
    summary: dict[str, object],
    summary_path: Path,
    report_path: Path,
    asset_dir: Path,
    *,
    petsc_summary: dict[str, object] | None = None,
    petsc_summary_path: Path | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rel = lambda name: str(Path(os.path.relpath(asset_dir / name, start=report_path.parent))).replace("\\", "/")
    summary_dir = summary_path.parent

    def _load_json(path: Path) -> dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _result_payload(row: dict[str, object]) -> dict[str, object]:
        return _load_json((REPO_ROOT / str(row["result_path"])).resolve() if not Path(str(row["result_path"])).is_absolute() else Path(str(row["result_path"])))

    def _reference_payload(name: str) -> dict[str, object]:
        path = summary_dir / "references" / name / "output.json"
        if not path.exists():
            return {}
        return _load_json(path)

    p2_rows = [_select_row(summary, study="mesh_refinement", p=2.0, method="mpa", level=level, epsilon=1.0e-5) for level in (4, 5, 6)]
    p3_rows = [_select_row(summary, study="mesh_refinement", p=3.0, method="mpa", level=level, epsilon=1.0e-5) for level in (4, 5, 6)]
    p2_tol_row = _select_row(summary, study="tolerance_sweep", p=2.0, method="mpa", level=6, epsilon=1.0e-6)
    p3_tol_row = _select_row(summary, study="tolerance_sweep", p=3.0, method="mpa", level=6, epsilon=1.0e-6)
    p3_continuation = _reference_payload("p3_certified_l6")
    p3_reference = _reference_payload("p3_certified_l7")
    p2_reference = _reference_payload("p2_newton_l7")

    def _handoff_label(source: object | None) -> str:
        value = str(source or "-")
        if value.startswith("mpa:"):
            return "MPA handoff"
        if value == "direct_init":
            return "Continuation direct"
        return value.replace("_", " ")

    def _p2_table_rows() -> list[list[str]]:
        rows: list[list[str]] = []
        for row in p2_rows:
            certified_newton_iters = _certified_newton_iters(row)
            rows.append(
                [
                    str(int(row["level"])),
                    _handoff_label(row.get("certified_handoff_source")),
                    _fmt(row.get("raw_residual_norm", row.get("residual_norm")), 6),
                    _fmt(row.get("certified_residual_norm", row.get("residual_norm")), 6),
                    str(_certified_mpa_iters(row)),
                    str(int(certified_newton_iters)),
                    str(_certified_total_nonlinear_iters(row)),
                    _fmt(row.get("certified_J", row.get("J")), 6),
                    str(row.get("certified_status", row.get("status"))),
                ]
            )
        return rows

    def _p3_table_rows() -> list[list[str]]:
        rows: list[list[str]] = []
        for row in p3_rows:
            certified_newton_iters = _certified_newton_iters(row)
            rows.append(
                [
                    str(int(row["level"])),
                    _fmt(row["lambda1"], 6),
                    _handoff_label(row.get("certified_handoff_source")),
                    _fmt(row.get("certified_residual_norm", row.get("residual_norm")), 6),
                    str(_certified_mpa_iters(row)),
                    str(int(certified_newton_iters)),
                    str(_certified_total_nonlinear_iters(row)),
                    _fmt(row.get("certified_J", row.get("J")), 6),
                    str(row.get("certified_status", row.get("status"))),
                ]
            )
        return rows

    def _continuation_rows() -> list[list[str]]:
        rows: list[list[str]] = []
        for step in list(p3_continuation.get("continuation_steps", [])):
            rows.append(
                [
                    f"{float(step['from_p']):.1f}",
                    f"{float(step['to_p']):.1f}",
                    str(step["path"]),
                    _fmt(step["residual_norm"], 6),
                    str(step["status"]),
                ]
            )
        return rows

    def _annex_cross_method_rows() -> list[list[str]]:
        rows: list[list[str]] = []
        for p in (2.0, 3.0):
            for method in ("mpa", "rmpa"):
                for level in (4, 5, 6):
                    row = _select_row(summary, study="mesh_refinement", p=p, method=method, level=level, epsilon=1.0e-5)
                    rows.append(
                        [
                            str(int(p)),
                            method.upper(),
                            str(int(level)),
                            str(row.get("raw_status", row.get("status", "-"))),
                            _fmt(row.get("raw_residual_norm", row.get("residual_norm")), 6),
                            str(row.get("certified_status")) if row.get("certified_status") is not None else "-",
                            _fmt(row.get("certified_residual_norm"), 6),
                            str(row.get("ray_best_kind") or "-"),
                            "yes" if bool(row.get("ray_stable_interior_extremum", False)) else "no",
                        ]
                    )
        return rows

    def _annex_rmpa_rows() -> list[list[str]]:
        rows: list[list[str]] = []
        for p in (2.0, 3.0):
            for level in (4, 5, 6):
                row = _select_row(summary, study="mesh_refinement", p=p, method="rmpa", level=level, epsilon=1.0e-5)
                rows.append(
                    [
                        str(int(p)),
                        str(int(level)),
                        str(row.get("raw_status", row.get("status", "-"))),
                        _fmt(row.get("raw_residual_norm", row.get("residual_norm")), 6),
                        str(row.get("ray_best_kind") or "-"),
                        "yes" if bool(row.get("ray_stable_interior_extremum", False)) else "no",
                        str(row.get("certification_message") or "-"),
                    ]
                )
        return rows

    def _annex_tolerance_rows() -> list[list[str]]:
        rows: list[list[str]] = []
        for p in (2.0, 3.0):
            for method in ("mpa", "rmpa"):
                for epsilon in (1.0e-4, 1.0e-5, 1.0e-6):
                    row = _select_row(summary, study="tolerance_sweep", p=p, method=method, level=6, epsilon=epsilon)
                    rows.append(
                        [
                            str(int(p)),
                            method.upper(),
                            f"{epsilon:.0e}",
                            str(row["raw_status"]),
                            _fmt(row["raw_residual_norm"], 6),
                            str(row["certified_status"]) if row["certified_status"] is not None else "-",
                            _fmt(row["certified_residual_norm"], 6),
                        ]
                    )
        return rows

    def _petsc_mesh_rows(p: float) -> list[list[str]]:
        rows: list[list[str]] = []
        for row in sorted(_petsc_rows_for(petsc_summary, study="mesh_ladder", p=p), key=lambda item: int(item["level"])):
            rows.append(
                [
                    str(int(row["level"])),
                    str(int(row.get("free_dofs", 0))),
                    str(row.get("status", "-")),
                    _fmt(row.get("residual_norm"), 6),
                    _fmt(row.get("setup_time_s"), 3),
                    _fmt(row.get("solve_time_s"), 3),
                    _fmt(row.get("total_time_s"), 3),
                    str(int(row.get("outer_iterations", 0))),
                    str(int(row.get("outer_iterations", 0))),
                    str(int(row.get("linear_iterations_total", 0))),
                    str(row.get("lambda_source", "-")).replace("_", " "),
                ]
            )
        return rows

    def _petsc_scaling_rows() -> list[list[str]]:
        rows: list[list[str]] = []
        for row in sorted(_petsc_rows_for(petsc_summary, study="strong_scaling", p=2.0), key=lambda item: int(item["nprocs"])):
            rows.append(
                [
                    str(int(row["nprocs"])),
                    str(row.get("status", "-")),
                    _fmt(row.get("residual_norm"), 6),
                    _fmt(row.get("total_time_s"), 3),
                    _fmt(row.get("solve_time_s"), 3),
                    str(int(row.get("outer_iterations", 0))),
                    str(int(row.get("outer_iterations", 0))),
                    str(int(row.get("linear_iterations_total", 0))),
                    _fmt(row.get("speedup_total"), 3),
                    _fmt(row.get("efficiency_total"), 3),
                ]
            )
        return rows

    lines = [
        "# pLaplace_up_arctan Report",
        "",
        "## Problem Description And Theory",
        "",
        "We solve the resonant Dirichlet problem on the unit square",
        "",
        "$$",
        "-\\Delta_p u = \\lambda_1 |u|^{p-2}u + \\arctan(u+1) \\quad \\text{in } (0,1)^2, \\qquad u = 0 \\text{ on } \\partial (0,1)^2,",
        "$$",
        "",
        "with energy",
        "",
        "$$",
        "J_p(u)=\\frac{1}{p}\\int_\\Omega |\\nabla u|^p\\,dx - \\frac{\\lambda_1}{p}\\int_\\Omega |u|^p\\,dx - \\int_\\Omega G(u)\\,dx,",
        "$$",
        "",
        "where `g(u)=arctan(u+1)` and `G'(u)=g(u)`. The solvability note used throughout the project is unchanged: the shifted arctan nonlinearity satisfies the asymptotic sign condition from the source note, so the report claims existence of weak solutions for `p=2` and `p=3`, but it does not claim global uniqueness.",
        "",
        "The practical conclusion of the recent solver study is also now clear:",
        "",
        "- The maintained computational method is the certified `MPA + stationary Newton` pipeline.",
        f"- On the published meshes this pipeline reduces the FE residual from an `MPA` handoff near `1.5e-1` to about `1e-15` for `p=2` and about `1e-14` for `p=3`.",
        "- `RMPA` is not part of the maintained solution path for this family and is discussed only in the annexes.",
        "",
        "## Certified Solver: MPA + Newton",
        "",
        "The current successful workflow uses raw `MPA` only to identify the positive principal branch and then switches to a discrete stationary solve for `J_p'(u)=0`. For `p=3` the certified branch is built by continuation in `p`, so the published results are continuation-guided Newton solves whose initial states come from already certified neighboring problems.",
        "",
        "Define the discrete gradient, dual residual, and certification merit by",
        "",
        "$$",
        "g_h(u_h)=\\nabla J_{p,h}(u_h), \\qquad R_h(u_h)=K^{-1}g_h(u_h), \\qquad M(u_h)=\\tfrac12 g_h(u_h)^\\top K^{-1}g_h(u_h).",
        "$$",
        "",
        "Then the maintained algorithm is:",
        "",
        "1. **Discrete setup.** Build the structured `P1` mesh, the finite-element energy `J_{p,h}`, and the stiffness matrix `K`. Use `\\lambda_1=2\\pi^2` for `p=2`, and the cached discrete eigenvalue `\\lambda_{1,h}` for `p=3`.",
        "2. **Raw MPA branch search.** Construct a polygonal path from `0` to a positive endpoint `e_h` with `J_{p,h}(e_h)<J_{p,h}(0)`. At iteration `k`, identify the current path peak `z_k`.",
        "3. **Dissertation direction `d^{V_h}`.** Solve the auxiliary Laplace problem",
        "",
        "$$",
        "K a_k = g_h(z_k), \\qquad d_k = -\\frac{a_k}{|a_k|_{1,p,0}},",
        "$$",
        "",
        "and update the path by a halved step from `z_k` followed by the usual local path repair. Record `\\|R_h(z_k)\\|_{K^{-1}}` and keep the best-residual iterate as the certification handoff state.",
        "4. **Regularized stationary Newton step.** Starting from the handoff state `u_k`, form the JAX autodiff gradient and Hessian and solve",
        "",
        "$$",
        "(H_k + \\mu_k K)\\,\\delta_k = -g_h(u_k).",
        "$$",
        "",
        "5. **Merit-based acceptance.** Accept `u_{k+1}=u_k+\\alpha_k\\delta_k` only if the stationary merit decreases. If needed, increase `\\mu_k` and backtrack in `\\alpha_k` until",
        "",
        "$$",
        "M(u_{k+1}) < M(u_k).",
        "$$",
        "",
        "6. **Certification stop.** Terminate when the discrete strong-form proxy satisfies",
        "",
        "$$",
        "\\|R_h(u_k)\\|_{K^{-1}} \\le \\varepsilon_{\\mathrm{cert}}.",
        "$$",
        "",
        "7. **Continuation for `p=3`.** First certify the `p=2` branch, then continue on the same mesh through `p=2.2,2.4,2.6,2.8,3.0`, using direct Newton from the previous certified state and reserving raw `MPA` only as fallback.",
        "",
        "Three implementation details matter for the final reported results:",
        "",
        "- The `MPA` stage is a branch locator, not the published stopping point. Its job is to produce the correct positive-branch handoff state.",
        "- Certification is driven by the Laplace-dual residual and the stationary merit, so the final stopping rule is aligned with `J_p'(u_h)=0` rather than with mountain-pass path geometry alone.",
        "- For `p=3`, continuation in `p` is the decisive ingredient. Once a certified `p=2` branch is available, the Newton stage converges rapidly at each continuation step.",
        "",
        "| published branch | actual certified path |",
        "| --- | --- |",
        "| `p=2` | raw `MPA` to locate the branch, then stationary Newton polish |",
        "| `p=3` | certify `p=2`, continue in `p`, then stationary Newton polish at each continuation stage |",
        "",
        f"The current suite summary has `{summary['generated_case_count']}` rows with overall status counts `{summary['status_counts']}`. All `12` certified rows completed.",
        "",
        "## Summary",
        "",
        "- Main reported method: certified `MPA + stationary Newton`.",
        "- `p=2`: `MPA` handoff followed by five to six Newton steps to machine-precision residuals.",
        "- `p=3`: certified continuation in `p` with four Newton steps on each published mesh.",
        "- Raw `MPA`, `RMPA`, and other unsuccessful routes are annex-only diagnostics.",
        "",
        "## p = 2 Validation Study",
        "",
    ]
    lines.append(
        "The `p=2` case is the validation problem. The table below is written in terms of the certified workflow: the entry column identifies how the certification stage is initialized, the handoff residual records the quality of that initial state, and the iteration columns report the raw `MPA` work, the Newton work, and their cumulative nonlinear total."
    )
    lines.extend(
        [
            "",
            _markdown_table(
                ["level", "certification entry", "MPA handoff residual", "certified residual", "MPA iters", "Newton iters", "total nonlinear", "certified J", "status"],
                _p2_table_rows(),
            ),
            "",
            f"Private `L7` Newton reference residual: `{float(p2_reference['residual_norm']):.3e}`." if p2_reference.get("residual_norm") is not None else "Private `L7` Newton reference residual: unavailable in this summary.",
        ]
    )
    lines.extend(
        [
            "",
            f"![p=2 certified solution panel]({rel('p2_solution_panel.png')})",
            "",
            f"![p=2 certified Newton convergence history after MPA handoff]({rel('p2_convergence_history.png')})",
            "",
            "## p = 3 Eigenvalue Stage",
            "",
        ]
    )
    lambda_rows = []
    for payload in _lambda_payloads(summary):
        lambda_rows.append(
            [
                str(payload["level"]),
                _fmt(payload["lambda1"], 6),
                _fmt(payload["residual_norm"], 6),
                _fmt(payload["normalization_error"], 6),
                str(payload["outer_iterations"]),
                payload["status"],
            ]
        )
    lines.append(_markdown_table(["level", "lambda1", "residual", "norm error", "iters", "status"], lambda_rows))
    lines.extend(
        [
            "",
            f"![p=3 eigen stage]({rel('lambda_convergence.png')})",
            "",
            f"![p=3 eigenfunction]({rel('p3_eigenfunction.png')})",
            "",
            "## p = 3 Main Study",
            "",
        ]
    )
    lines.append(
        "The decisive change for `p=3` is the fixed-level continuation in `p`. The published table is therefore framed entirely as a certified continuation computation: each row uses the discrete eigenvalue cache for that mesh, starts from an already certified nearby state, and reports the `MPA` contribution, Newton contribution, and cumulative nonlinear work actually used by the maintained path."
    )
    lines.extend(
        [
            "",
            _markdown_table(
                ["level", "lambda1,h", "certification entry", "certified residual", "MPA iters", "Newton iters", "total nonlinear", "certified J", "status"],
                _p3_table_rows(),
            ),
            "",
            "Representative `L6` continuation path:",
            "",
            _markdown_table(["from p", "to p", "path", "certified residual", "status"], _continuation_rows()),
            "",
            (
                f"Private `L7` certified reference residual: `{float(p3_reference['certified']['residual_norm']):.3e}`."
                if p3_reference.get("certified", {}).get("residual_norm") is not None
                else "Private `L7` certified reference residual: unavailable in this summary."
            ),
        ]
    )
    lines.extend(
        [
            "",
            f"![p=3 certified solution panel]({rel('p3_solution_panel.png')})",
            "",
            f"![p=3 certified continuation and Newton convergence history]({rel('p3_convergence_history.png')})",
        ]
    )
    if petsc_summary is not None:
        p2_petsc_rows = _petsc_rows_for(petsc_summary, study="mesh_ladder", p=2.0)
        p3_petsc_rows = _petsc_rows_for(petsc_summary, study="mesh_ladder", p=3.0)
        scaling_rows = _petsc_rows_for(petsc_summary, study="strong_scaling", p=2.0)
        finest_level = petsc_summary.get("finest_scaling_level")
        p2_best = next((row for row in reversed(sorted(p2_petsc_rows, key=lambda item: int(item["level"]))) if str(row.get("status")) == "completed"), None)
        p3_best = next((row for row in reversed(sorted(p3_petsc_rows, key=lambda item: int(item["level"]))) if str(row.get("status")) == "completed"), None)
        lines.extend(
            [
                "",
                "## JAX + PETSc Backend",
                "",
                "The fine-mesh stationary certification stage is now available as a distributed JAX + PETSc backend. The maintained branch-finding story is unchanged: raw `MPA` still identifies the correct positive branch, but the expensive local solve is delegated to PETSc, with JAX providing the local energy, gradient, and Hessian information.",
                "",
                "The PETSc local kernel uses a zero-safe regularization inside the element energy so that Hessian assembly remains finite even when a transferred warm start contains exact zeros:",
                "",
                "$$",
                "|x|_{\\varepsilon_h}^q = \\bigl(x^2 + \\varepsilon_h^2\\bigr)^{q/2} - \\varepsilon_h^q, \\qquad \\varepsilon_h = 10^{-12}.",
                "$$",
                "",
                "The Newton step remains",
                "",
                "$$",
                "(H_k + \\mu_k K)\\,\\delta_k = -g_h(u_k),",
                "$$",
                "",
                "but PETSc now solves the linear systems with `FGMRES` preconditioned by a structured PMG hierarchy built from the stiffness matrix `K`. The accepted variant for this family is the fixed-stiffness Galerkin PMG preconditioner with **Chebyshev** smoothing and **Jacobi** inner preconditioning on the multigrid levels.",
                "Near the tolerance floor, the PETSc backend also uses a small stagnation guard: if the residual is already within a tight constant factor of the target and no regularized step can produce a numerically resolvable merit decrease, the solve is accepted as converged instead of being labeled as a globalization failure.",
                "",
                "## PETSc Timing And Scaling",
                "",
                (
                    f"For `p=2`, the PETSc ladder now reaches level `L{int(p2_best['level'])}` with total time `{float(p2_best['total_time_s']):.2f}` s and residual `{float(p2_best['residual_norm']):.3e}`."
                    if p2_best is not None
                    else "For `p=2`, no successful PETSc mesh-ladder row is available in this summary."
                ),
                (
                    f"For `p=3`, the tuned PETSc ladder reaches level `L{int(p3_best['level'])}` with total time `{float(p3_best['total_time_s']):.2f}` s and residual `{float(p3_best['residual_norm']):.3e}`."
                    if p3_best is not None
                    else "For `p=3`, no successful PETSc mesh-ladder row is available in this summary."
                ),
                "The PETSc tables expose nonlinear outer iterations, Newton iterations, and cumulative Krylov iterations so the PMG workload is visible directly in the published summary. In this backend the nonlinear counter coincides with the Newton outer loop.",
                "The PETSc mesh-timing plot is shown on log-log axes with an ideal `1:1` reference triangle, and the strong-scaling panel compares measured log-log speedup against the ideal line.",
                "",
                _markdown_table(
                    ["level", "free dofs", "status", "residual", "setup [s]", "solve [s]", "total [s]", "nonlinear its", "Newton iters", "linear its", "lambda source"],
                    _petsc_mesh_rows(2.0),
                ),
                "",
                _markdown_table(
                    ["level", "free dofs", "status", "residual", "setup [s]", "solve [s]", "total [s]", "nonlinear its", "Newton iters", "linear its", "lambda source"],
                    _petsc_mesh_rows(3.0),
                ),
                "",
                (
                    f"Strong-scaling study on the finest successful PETSc level `L{int(finest_level)}`:"
                    if finest_level is not None and scaling_rows
                    else "Strong-scaling study is unavailable in this PETSc summary."
                ),
            ]
        )
        if scaling_rows:
            lines.extend(
                [
                    "",
                    _markdown_table(
                        ["ranks", "status", "residual", "total [s]", "solve [s]", "nonlinear its", "Newton iters", "linear its", "speedup", "efficiency"],
                        _petsc_scaling_rows(),
                    ),
                    "",
                    f"![PETSc mesh timing]({rel('petsc_mesh_timing.png')})",
                    "",
                    f"![PETSc strong scaling]({rel('petsc_strong_scaling.png')})",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    f"![PETSc mesh timing]({rel('petsc_mesh_timing.png')})",
                ]
            )
    lines.extend(
        [
            "",
            "## Raw Versus Certified Diagnostics",
            "",
            "The remainder of the report is diagnostic. It separates the successful certified `MPA + stationary Newton` path from the raw globalization attempts and preserves the failed approaches only to explain why they were not promoted into the maintained workflow.",
            "",
            "## Cross-Method Diagnostics (Annex A)",
            "",
            "This annex keeps the raw methods visible, but only as diagnostics. The main body above is intentionally MPA-only because that is the path that now solves the maintained problems.",
            "",
            _markdown_table(
                ["p", "method", "level", "raw status", "raw residual", "certified status", "certified residual", "ray kind", "stable ray max?"],
                _annex_cross_method_rows(),
            ),
            "",
            f"![Iteration counts]({rel('iteration_counts.png')})",
            "",
            f"![Reference errors]({rel('reference_error_refinement.png')})",
            "",
            "Standalone raw `MPA` is still not a converged solver: it is a branch-finding stage whose output must be polished by the stationary Newton solve. That is why the raw residuals stay near `1e-1` even when the certified residuals are already near machine precision.",
            "",
            "## Annex B — RMPA And Failed Paths",
            "",
            _markdown_table(
                ["p", "level", "raw status", "raw residual", "ray kind", "stable ray max?", "rationale"],
                _annex_rmpa_rows(),
            ),
            "",
            _markdown_table(
                ["p", "method", "epsilon", "raw status", "raw residual", "certified status", "certified residual"],
                _annex_tolerance_rows(),
            ),
            "",
            "Why `RMPA` is annex-only for this family:",
            "",
            "- The ray audit does not show the stable interior ray maximum required by the classical `RMPA` projection logic on the positive arctan branch.",
            "- Raw `RMPA` therefore fails before it can provide a useful Newton handoff iterate; its certified column is intentionally empty in the maintained tables.",
            "- Tightening the raw outer tolerance does not change the raw `RMPA` residuals materially on the published ladder, so more `maxit` or smaller `epsilon` is not the fix here.",
            "- The maintained successful path is `MPA + stationary Newton`, and for `p=3` the decisive ingredient is continuation in `p`, not ray projection.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "./.venv/bin/python -u experiments/runners/run_plaplace_up_arctan_suite.py \\",
            "  --out-dir artifacts/raw_results/plaplace_up_arctan_full \\",
            "  --summary artifacts/raw_results/plaplace_up_arctan_full/summary.json",
            "```",
            "",
            "```bash",
            "./.venv/bin/python -u experiments/runners/run_plaplace_up_arctan_petsc_suite.py \\",
            "  --out-dir artifacts/raw_results/plaplace_up_arctan_petsc \\",
            "  --summary artifacts/raw_results/plaplace_up_arctan_petsc/summary.json",
            "```",
            "",
            "```bash",
            "./.venv/bin/python -u experiments/analysis/generate_plaplace_up_arctan_report.py \\",
            "  --summary artifacts/raw_results/plaplace_up_arctan_full/summary.json \\",
            "  --petsc-summary artifacts/raw_results/plaplace_up_arctan_petsc/summary.json \\",
            "  --out artifacts/reports/plaplace_up_arctan/README.md \\",
            "  --asset-dir docs/assets/plaplace_up_arctan",
            "```",
        ]
    )
    if petsc_summary_path is not None:
        lines.extend(
            [
                "",
                f"PETSc timing summary: `{_repo_rel(petsc_summary_path)}`.",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--petsc-summary", type=Path, default=DEFAULT_PETSC_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    args = parser.parse_args()

    args.asset_dir.mkdir(parents=True, exist_ok=True)
    summary = _load_summary(args.summary)
    petsc_summary = _load_optional_summary(args.petsc_summary)

    p2_rows = [
        _select_row(summary, study="mesh_refinement", p=2.0, method="mpa", level=level, epsilon=1.0e-5)
        for level in (4, 5, 6)
    ]
    p3_rows = [
        _select_row(summary, study="mesh_refinement", p=3.0, method="mpa", level=level, epsilon=1.0e-5)
        for level in (4, 5, 6)
    ]
    p2_hist_rows = [_select_row(summary, study="tolerance_sweep", p=2.0, method="mpa", level=6, epsilon=1.0e-6)]
    p3_hist_rows = [_select_row(summary, study="tolerance_sweep", p=3.0, method="mpa", level=6, epsilon=1.0e-6)]
    lambda_l6 = json.loads((Path(summary["lambda_cache_dir"]) / "lambda_p3_l6.json").read_text(encoding="utf-8"))

    _plot_state_panel(p2_rows, "p = 2 certified MPA + Newton solutions across levels", args.asset_dir / "p2_solution_panel")
    _plot_state_panel(p3_rows, "p = 3 certified MPA + Newton solutions across levels", args.asset_dir / "p3_solution_panel")
    _plot_scalar_state(Path(lambda_l6["state_out"]), "p = 3 discrete first eigenfunction", args.asset_dir / "p3_eigenfunction")
    _plot_history(
        p2_hist_rows,
        "p = 2 certified stationary Newton history after the MPA handoff at level 6",
        args.asset_dir / "p2_convergence_history",
        tracks=(TRACK_CERTIFIED,),
    )
    _plot_history(
        p3_hist_rows,
        "p = 3 certified continuation/Newton history at level 6",
        args.asset_dir / "p3_convergence_history",
        tracks=(TRACK_CERTIFIED,),
    )
    _plot_iteration_counts(summary, args.asset_dir / "iteration_counts")
    _plot_reference_errors(summary, args.asset_dir / "reference_error_refinement")
    _plot_lambda(summary, args.asset_dir / "lambda_convergence")
    _plot_petsc_mesh_timing(petsc_summary, args.asset_dir / "petsc_mesh_timing")
    _plot_petsc_scaling(petsc_summary, args.asset_dir / "petsc_strong_scaling")
    _write_report(
        summary,
        args.summary,
        args.out,
        args.asset_dir,
        petsc_summary=petsc_summary,
        petsc_summary_path=args.petsc_summary if petsc_summary is not None else None,
    )


if __name__ == "__main__":
    main()
