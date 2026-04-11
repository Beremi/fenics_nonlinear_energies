#!/usr/bin/env python3
"""Generate assets for the Plasticity3D implementation scaling comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_COLOR_BY_IMPLEMENTATION = {
    "maintained_local_best": "#1f77b4",
    "source_petsc4py": "#d62728",
    "maintained_local_pmg": "#2ca02c",
    "source_petsc4py_pmg": "#ff7f0e",
}
DEFAULT_COLOR_CYCLE = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
)


def _plot_ideal_reference(
    ax,
    ranks: np.ndarray,
    values: np.ndarray,
    *,
    color: str,
) -> None:
    finite = np.isfinite(ranks) & np.isfinite(values) & (ranks > 0) & (values > 0.0)
    if not np.any(finite):
        return
    base_rank = float(ranks[finite][0])
    base_value = float(values[finite][0])
    ideal = base_value * (base_rank / ranks.astype(np.float64))
    ax.plot(
        ranks,
        ideal,
        linewidth=1.2,
        linestyle=(0, (4, 3)),
        color=color,
        alpha=0.45,
        zorder=1,
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _implementation_order(summary: dict[str, object]) -> list[str]:
    implementations = list(summary.get("implementations", []))
    ordered = [
        str(item.get("implementation", "")).strip()
        for item in implementations
        if isinstance(item, dict) and str(item.get("implementation", "")).strip()
    ]
    if ordered:
        return ordered
    seen: list[str] = []
    for row in list(summary.get("rows", [])):
        if not isinstance(row, dict):
            continue
        impl = str(row.get("implementation", "")).strip()
        if impl and impl not in seen:
            seen.append(impl)
    return seen


def _implementation_labels(summary: dict[str, object], order: list[str]) -> dict[str, str]:
    labels = {impl: impl for impl in order}
    for item in list(summary.get("implementations", [])):
        if not isinstance(item, dict):
            continue
        impl = str(item.get("implementation", "")).strip()
        if not impl:
            continue
        label = str(item.get("display_label", "")).strip()
        if label:
            labels[impl] = label
    for row in list(summary.get("rows", [])):
        if not isinstance(row, dict):
            continue
        impl = str(row.get("implementation", "")).strip()
        if not impl:
            continue
        label = str(row.get("display_label", "")).strip()
        if label:
            labels[impl] = label
    return labels


def _implementation_families(summary: dict[str, object], order: list[str]) -> dict[str, str]:
    families = {impl: "" for impl in order}
    for item in list(summary.get("implementations", [])):
        if not isinstance(item, dict):
            continue
        impl = str(item.get("implementation", "")).strip()
        if not impl:
            continue
        families[impl] = str(item.get("family", "")).strip()
    for row in list(summary.get("rows", [])):
        if not isinstance(row, dict):
            continue
        impl = str(row.get("implementation", "")).strip()
        if not impl or families.get(impl):
            continue
        families[impl] = str(row.get("family", "")).strip()
    return families


def _implementation_colors(order: list[str]) -> dict[str, str]:
    colors: dict[str, str] = {}
    next_idx = 0
    for impl in order:
        if impl in DEFAULT_COLOR_BY_IMPLEMENTATION:
            colors[impl] = DEFAULT_COLOR_BY_IMPLEMENTATION[impl]
            continue
        colors[impl] = DEFAULT_COLOR_CYCLE[next_idx % len(DEFAULT_COLOR_CYCLE)]
        next_idx += 1
    return colors


def _component_plot_name(implementation: str) -> str:
    safe = str(implementation).strip().replace("/", "_").replace(" ", "_")
    return f"{safe}_component_scaling.png"


def _stage_map(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    stages: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        stage = str(payload.get("stage", "")).strip()
        if stage:
            stages[stage] = payload
    return stages


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _payload_stage_map(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    stage_timings = dict(payload.get("stage_timings", {}))
    if not stage_timings:
        return {}
    cumulative = 0.0
    stages: dict[str, dict[str, object]] = {}
    for key in ("problem_load", "assembler_create", "mg_hierarchy_build", "initial_guess_total"):
        if key not in stage_timings:
            continue
        value = _safe_float(stage_timings.get(key))
        if not np.isfinite(value):
            continue
        cumulative += value
        if key == "initial_guess_total":
            stages["local_initial_guess_done"] = {"elapsed_s": cumulative}
        else:
            stages[key] = {"elapsed_s": cumulative}
    if cumulative > 0.0:
        stages["backend_ready"] = {
            "elapsed_s": cumulative - _safe_float(stage_timings.get("initial_guess_total", 0.0))
        }
    return stages


def _load_source_timing_bundle(
    row: dict[str, object],
    builder_timings: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    run_info_data: dict[str, object] = {}
    timing_bundle: dict[str, object] = dict(builder_timings)
    native_run_info = str(row.get("native_run_info", "") or "").strip()
    if native_run_info:
        run_info_path = _repo_path(native_run_info)
        if run_info_path.exists():
            run_info_data = _read_json(run_info_path)
            timings = dict(run_info_data.get("timings", {}))
            constitutive = dict(timings.get("constitutive", {}))
            linear = dict(timings.get("linear", {}))
            if constitutive:
                timing_bundle = {**timing_bundle, **constitutive}
            if linear:
                timing_bundle = {**timing_bundle, **linear}
    return timing_bundle, run_info_data


def _sum_history(history: list[dict[str, object]], key: str) -> float:
    total = 0.0
    for row in history:
        if not isinstance(row, dict):
            continue
        value = row.get(key)
        if value is None:
            continue
        try:
            total += float(value)
        except Exception:
            continue
    return float(total)


def _sum_linear_history(history: list[dict[str, object]], key: str) -> float:
    total = 0.0
    for row in history:
        if not isinstance(row, dict):
            continue
        value = row.get(key)
        if value is None:
            continue
        try:
            total += float(value)
        except Exception:
            continue
    return float(total)


def _extract_local_components(payload: dict[str, object], stages: dict[str, dict[str, object]]) -> dict[str, float]:
    history = list(payload.get("history", []))
    linear_history = list(payload.get("linear_history", []))
    assembly_callbacks = dict(payload.get("assembly_callbacks", {}))
    hessian_cb = dict(assembly_callbacks.get("hessian", {}))

    if not stages:
        stages = _payload_stage_map(payload)
    stage_timings = dict(payload.get("stage_timings", {}))
    backend_build_s = float(stages.get("backend_ready", {}).get("elapsed_s", math.nan))
    if not np.isfinite(backend_build_s):
        backend_build_s = float(
            sum(
                _safe_float(stage_timings.get(key))
                for key in ("problem_load", "assembler_create", "mg_hierarchy_build")
                if key in stage_timings
            )
        )
    initial_guess_s = float(dict(payload.get("initial_guess", {})).get("solve_time", math.nan))
    if not np.isfinite(initial_guess_s):
        initial_guess_s = _safe_float(stage_timings.get("initial_guess_total"))
    solve_total_s = float(payload.get("solve_time", math.nan))
    wall_total_s = float(payload.get("total_time", math.nan))

    gradient_top_s = _sum_history(history, "t_grad")
    line_search_s = _sum_history(history, "t_ls")
    update_s = _sum_history(history, "t_update")
    linear_assemble_s = _sum_linear_history(linear_history, "t_assemble")
    linear_setup_s = _sum_linear_history(linear_history, "t_setup")
    linear_solve_s = _sum_linear_history(linear_history, "t_solve")
    known_newton = (
        gradient_top_s
        + line_search_s
        + update_s
        + linear_assemble_s
        + linear_setup_s
        + linear_solve_s
    )
    other_newton_s = float(max(0.0, solve_total_s - known_newton)) if np.isfinite(solve_total_s) else math.nan

    return {
        "backend_build_s": backend_build_s,
        "initial_guess_s": initial_guess_s,
        "solve_total_s": solve_total_s,
        "wall_total_s": wall_total_s,
        "newton_gradient_top_s": gradient_top_s,
        "newton_line_search_s": line_search_s,
        "newton_update_s": update_s,
        "newton_linear_assemble_s": linear_assemble_s,
        "newton_linear_setup_s": linear_setup_s,
        "newton_linear_solve_s": linear_solve_s,
        "newton_other_s": other_newton_s,
        "callback_energy_s": float(dict(assembly_callbacks.get("energy", {})).get("total", math.nan)),
        "callback_gradient_s": float(dict(assembly_callbacks.get("gradient", {})).get("total", math.nan)),
        "callback_hessian_s": float(hessian_cb.get("total", math.nan)),
        "callback_hessian_kernel_s": float(hessian_cb.get("hvp_compute", math.nan)),
        "callback_hessian_extraction_s": float(hessian_cb.get("extraction", math.nan)),
        "callback_hessian_accumulate_s": float(hessian_cb.get("accumulate", math.nan)),
        "callback_hessian_coo_s": float(
            hessian_cb.get("coo_assembly", hessian_cb.get("coo_insert", math.nan))
        ),
    }


def _extract_source_components(
    payload: dict[str, object],
    stages: dict[str, dict[str, object]],
    timing_bundle: dict[str, object],
) -> dict[str, float]:
    history = list(payload.get("history", []))
    backend_build_s = float(stages.get("backend_ready", {}).get("elapsed_s", math.nan))
    initial_guess_s = float(dict(payload.get("initial_guess", {})).get("solve_time", math.nan))
    solve_total_s = float(payload.get("solve_time", math.nan))
    wall_total_s = float(payload.get("total_time", math.nan))

    linear_solve_s = _sum_history(history, "linear_solve_time")
    linear_preconditioner_s = _sum_history(history, "linear_preconditioner_time")
    linear_orthogonalization_s = _sum_history(history, "linear_orthogonalization_time")
    iteration_wall_s = _sum_history(history, "iteration_wall_time")
    known_newton = linear_solve_s + linear_preconditioner_s + linear_orthogonalization_s
    other_newton_s = float(max(0.0, iteration_wall_s - known_newton)) if np.isfinite(iteration_wall_s) else math.nan

    builder: dict[str, float] = {}
    for k, v in dict(timing_bundle).items():
        try:
            builder[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return {
        "backend_build_s": backend_build_s,
        "initial_guess_s": initial_guess_s,
        "solve_total_s": solve_total_s,
        "wall_total_s": wall_total_s,
        "iteration_wall_s": iteration_wall_s,
        "newton_linear_solve_s": linear_solve_s,
        "newton_linear_preconditioner_s": linear_preconditioner_s,
        "newton_linear_orthogonalization_s": linear_orthogonalization_s,
        "newton_other_s": other_newton_s,
        "builder_local_strain_s": float(builder.get("local_strain", math.nan)),
        "builder_local_constitutive_s": float(builder.get("local_constitutive", math.nan)),
        "builder_local_constitutive_comm_s": float(builder.get("local_constitutive_comm", math.nan)),
        "builder_build_tangent_local_s": float(builder.get("build_tangent_local", math.nan)),
        "builder_local_force_assembly_s": float(builder.get("local_force_assembly", math.nan)),
        "builder_local_force_gather_s": float(builder.get("local_force_gather", math.nan)),
        "builder_build_F_s": float(builder.get("build_F", math.nan)),
        "pmg_setup_time_s": float(builder.get("pmg_setup_time_s", math.nan)),
        "preconditioner_setup_time_total_s": float(
            builder.get("preconditioner_setup_time_total", math.nan)
        ),
        "preconditioner_apply_time_total_s": float(
            builder.get("preconditioner_apply_time_total", math.nan)
        ),
    }


def _enrich_row(row: dict[str, object]) -> dict[str, object]:
    result_path = _repo_path(str(row["result_json"]))
    payload = _read_json(result_path)
    stage_path_str = str(row.get("stage_jsonl", "") or "").strip()
    stages = _stage_map(_repo_path(stage_path_str)) if stage_path_str else {}
    builder_timings: dict[str, object] = {}
    builder_path_str = str(row.get("source_builder_timings_json", "") or "").strip()
    if builder_path_str:
        builder_path = _repo_path(builder_path_str)
        if builder_path.exists():
            builder_timings = _read_json(builder_path)
    timing_bundle, _run_info_data = _load_source_timing_bundle(row, builder_timings)

    enriched = dict(row)
    if str(row.get("family", "")).strip() == "source":
        enriched["components"] = _extract_source_components(payload, stages, timing_bundle)
    else:
        enriched["components"] = _extract_local_components(payload, stages)
    return enriched


def _rows_by_impl(
    rows: list[dict[str, object]],
    implementation_order: list[str],
) -> dict[str, list[dict[str, object]]]:
    grouped = {name: [] for name in implementation_order}
    for row in rows:
        impl = str(row.get("implementation", ""))
        if impl in grouped:
            grouped[impl].append(row)
    for impl in grouped:
        grouped[impl].sort(key=lambda row: int(row["ranks"]))
    return grouped


def _speedup_series(rows: list[dict[str, object]], key: str) -> list[tuple[int, float, float]]:
    if not rows:
        return []
    baseline = float(rows[0]["components"].get(key, math.nan))
    baseline_ranks = float(rows[0]["ranks"])
    out: list[tuple[int, float, float]] = []
    for row in rows:
        value = float(row["components"].get(key, math.nan))
        speedup = baseline / value if np.isfinite(baseline) and np.isfinite(value) and value > 0.0 else math.nan
        rank_ratio = float(row["ranks"]) / baseline_ranks if baseline_ranks > 0.0 else math.nan
        efficiency = speedup / rank_ratio if np.isfinite(speedup) and np.isfinite(rank_ratio) and rank_ratio > 0.0 else math.nan
        out.append((int(row["ranks"]), float(speedup), float(efficiency)))
    return out


def _speedup_map(rows: list[dict[str, object]], key: str) -> dict[int, tuple[float, float]]:
    return {ranks: (speedup, efficiency) for ranks, speedup, efficiency in _speedup_series(rows, key)}


def _plot_overall_scaling(
    rows_by_impl: dict[str, list[dict[str, object]]],
    *,
    implementation_order: list[str],
    labels: dict[str, str],
    colors: dict[str, str],
    out_path: Path,
    show_ideal: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for impl in implementation_order:
        rows = rows_by_impl.get(impl, [])
        if not rows:
            continue
        ranks = np.asarray([int(row["ranks"]) for row in rows], dtype=np.int64)
        wall = np.asarray([float(row["wall_time_s"]) for row in rows], dtype=np.float64)
        solve = np.asarray([float(row["solve_time_s"]) for row in rows], dtype=np.float64)
        color = colors[impl]
        label = labels[impl]
        axes[0].plot(ranks, wall, marker="o", linewidth=2.0, color=color, label=label)
        axes[1].plot(ranks, solve, marker="o", linewidth=2.0, color=color, label=label)
        if show_ideal:
            _plot_ideal_reference(axes[0], ranks, wall, color=color)
            _plot_ideal_reference(axes[1], ranks, solve, color=color)
    for ax, title, ylabel in (
        (axes[0], "Wall Time Scaling", "Wall time [s]"),
        (axes[1], "Solve Time Scaling", "Solve time [s]"),
    ):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(True, which="both", alpha=0.25)
        ax.set_title(title)
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel(ylabel)
    if show_ideal:
        axes[1].plot([], [], linewidth=1.2, linestyle=(0, (4, 3)), color="#555555", alpha=0.7, label="Ideal 1/r")
    axes[1].legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_common_components(
    rows_by_impl: dict[str, list[dict[str, object]]],
    *,
    implementation_order: list[str],
    labels: dict[str, str],
    colors: dict[str, str],
    out_path: Path,
    show_ideal: bool = False,
) -> None:
    keys = ("backend_build_s", "initial_guess_s", "solve_total_s")
    component_labels = {
        "backend_build_s": "Backend build",
        "initial_guess_s": "Elastic initial guess",
        "solve_total_s": "Newton solve",
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for impl in implementation_order:
        rows = rows_by_impl.get(impl, [])
        if not rows:
            continue
        ranks = np.asarray([int(row["ranks"]) for row in rows], dtype=np.int64)
        color = colors[impl]
        for idx, key in enumerate(keys):
            values = np.asarray([float(row["components"].get(key, math.nan)) for row in rows], dtype=np.float64)
            linestyle = ("-", "--", ":")[idx]
            ax.plot(
                ranks,
                values,
                marker="o",
                linewidth=1.8,
                linestyle=linestyle,
                color=color,
                label=f"{labels[impl]} / {component_labels[key]}",
            )
            if show_ideal:
                _plot_ideal_reference(ax, ranks, values, color=color)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title("Common Component Scaling")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Time [s]")
    if show_ideal:
        ax.plot([], [], linewidth=1.2, linestyle=(0, (4, 3)), color="#555555", alpha=0.7, label="Ideal 1/r")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_impl_breakdown(
    rows: list[dict[str, object]],
    *,
    title: str,
    component_keys: list[tuple[str, str]],
    color: str,
    out_path: Path,
    show_ideal: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ranks = np.asarray([int(row["ranks"]) for row in rows], dtype=np.int64)
    for idx, (key, label) in enumerate(component_keys):
        values = np.asarray([float(row["components"].get(key, math.nan)) for row in rows], dtype=np.float64)
        linestyle = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (1, 1))][idx % 6]
        ax.plot(
            ranks,
            values,
            marker="o",
            linewidth=1.8,
            linestyle=linestyle,
            color=color,
            label=label,
        )
        if show_ideal:
            _plot_ideal_reference(ax, ranks, values, color=color)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Time [s]")
    if show_ideal:
        ax.plot([], [], linewidth=1.2, linestyle=(0, (4, 3)), color="#555555", alpha=0.7, label="Ideal 1/r")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _write_report(
    summary: dict[str, object],
    rows_by_impl: dict[str, list[dict[str, object]]],
    all_rows: list[dict[str, object]],
    implementation_order: list[str],
    implementation_labels: dict[str, str],
    implementation_families: dict[str, str],
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Plasticity3D Implementation Scaling Comparison")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    stop_metric_name = str(summary.get("stop_metric_name", "relative_correction"))
    stop_metric_tol = (
        summary.get("grad_stop_tol")
        if stop_metric_name == "grad_norm"
        else summary.get("stop_tol")
    )
    lines.append(f"- Ranks: `{', '.join(str(v) for v in summary.get('ranks', []))}`")
    lines.append(f"- Stop metric: `{stop_metric_name} < {stop_metric_tol}`")
    lines.append(f"- Max Newton iterations: `{summary.get('maxit')}`")
    lines.append("- Compared implementations:")
    for impl in implementation_order:
        lines.append(f"  - `{implementation_labels.get(impl, impl)}`")
    lines.append("")

    lines.append("## Overall Results")
    lines.append("")
    lines.append("| implementation | ranks | status | wall [s] | solve [s] | Newton its | linear its | final metric |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    impl_order = {name: idx for idx, name in enumerate(implementation_order)}
    for row in sorted(
        all_rows,
        key=lambda item: (
            int(item.get("ranks", 10**6)),
            impl_order.get(str(item.get("implementation", "")), 10**6),
        ),
    ):
        impl = str(row.get("implementation", ""))
        label = implementation_labels.get(impl, impl)
        status = str(row.get("status", ""))
        wall = float(row.get("wall_time_s", math.nan))
        solve = float(row.get("solve_time_s", math.nan))
        nit = int(row.get("nit", 0))
        linear_its = int(row.get("linear_iterations_total", 0))
        final_metric = float(row.get("final_metric", math.nan))
        lines.append(
            f"| {label} | {int(row['ranks'])} | {status} | "
            f"{wall:.3f} | {solve:.3f} | {nit} | {linear_its} | {final_metric:.6e} |"
        )
    lines.append("")

    failed_rows = [
        row
        for row in all_rows
        if str(row.get("status", "")).strip().lower() != "completed"
    ]
    if failed_rows:
        lines.append("## Failed Or Incomplete Rows")
        lines.append("")
        for row in sorted(
            failed_rows,
            key=lambda item: (
                int(item.get("ranks", 10**6)),
                impl_order.get(str(item.get("implementation", "")), 10**6),
            ),
        ):
            impl = str(row.get("implementation", ""))
            label = implementation_labels.get(impl, impl)
            lines.append(
                f"- `{label}` at `{int(row['ranks'])}` ranks: {str(row.get('message', '')).strip() or str(row.get('status', 'failed'))}"
            )
        lines.append("")

    lines.append("## Common Scaling Buckets")
    lines.append("")
    lines.append("These are the cleanest like-for-like timings across both implementations:")
    lines.append("")
    lines.append("- backend build to `backend_ready`")
    lines.append("- elastic initial guess solve")
    lines.append("- Newton solve total")
    lines.append("")

    for impl in implementation_order:
        rows = rows_by_impl.get(impl, [])
        if not rows:
            continue
        baseline_ranks = int(rows[0]["ranks"])
        lines.append(f"### {implementation_labels[impl]}")
        lines.append("")
        lines.append(f"| ranks | backend build [s] | initial guess [s] | Newton solve [s] | wall [s] | speedup vs {baseline_ranks} rank | efficiency vs {baseline_ranks} |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        speedups = _speedup_series(rows, "wall_total_s")
        speed_map = {r: (s, e) for r, s, e in speedups}
        for row in rows:
            ranks = int(row["ranks"])
            comp = dict(row["components"])
            speedup, efficiency = speed_map.get(ranks, (math.nan, math.nan))
            lines.append(
                f"| {ranks} | {float(comp.get('backend_build_s', math.nan)):.3f} | "
                f"{float(comp.get('initial_guess_s', math.nan)):.3f} | "
                f"{float(comp.get('solve_total_s', math.nan)):.3f} | "
                f"{float(comp.get('wall_total_s', math.nan)):.3f} | "
                f"{float(speedup):.3f} | {float(efficiency):.3f} |"
            )
        lines.append("")

    for family in ("local", "source"):
        family_impls = [
            impl
            for impl in implementation_order
            if implementation_families.get(impl, "") == family and rows_by_impl.get(impl)
        ]
        if not family_impls:
            continue
        lines.append(f"## {family.title()} Breakdown")
        lines.append("")
        for impl in family_impls:
            rows = rows_by_impl.get(impl, [])
            if not rows:
                continue
            lines.append(f"### {implementation_labels[impl]}")
            lines.append("")
            if family == "local":
                lines.append("| ranks | linear assemble [s] | linear setup [s] | linear solve [s] | top grad [s] | line search [s] | Hessian callback [s] |")
                lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
                for row in rows:
                    comp = dict(row["components"])
                    lines.append(
                        f"| {int(row['ranks'])} | "
                        f"{float(comp.get('newton_linear_assemble_s', math.nan)):.3f} | "
                        f"{float(comp.get('newton_linear_setup_s', math.nan)):.3f} | "
                        f"{float(comp.get('newton_linear_solve_s', math.nan)):.3f} | "
                        f"{float(comp.get('newton_gradient_top_s', math.nan)):.3f} | "
                        f"{float(comp.get('newton_line_search_s', math.nan)):.3f} | "
                        f"{float(comp.get('callback_hessian_s', math.nan)):.3f} |"
                    )
                lines.append("")
                lines.append("| ranks | Newton solve speedup | linear assemble speedup | linear solve speedup | Hessian callback speedup |")
                lines.append("| ---: | ---: | ---: | ---: | ---: |")
                solve_map = _speedup_map(rows, "solve_total_s")
                assemble_map = _speedup_map(rows, "newton_linear_assemble_s")
                linear_map = _speedup_map(rows, "newton_linear_solve_s")
                hessian_map = _speedup_map(rows, "callback_hessian_s")
                for row in rows:
                    ranks = int(row["ranks"])
                    lines.append(
                        f"| {ranks} | "
                        f"{float(solve_map.get(ranks, (math.nan, math.nan))[0]):.3f} | "
                        f"{float(assemble_map.get(ranks, (math.nan, math.nan))[0]):.3f} | "
                        f"{float(linear_map.get(ranks, (math.nan, math.nan))[0]):.3f} | "
                        f"{float(hessian_map.get(ranks, (math.nan, math.nan))[0]):.3f} |"
                    )
                lines.append("")
            else:
                lines.append("| ranks | linear solve [s] | preconditioner [s] | orthogonalization [s] | local strain [s] | local constitutive [s] | build tangent [s] |")
                lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
                for row in rows:
                    comp = dict(row["components"])
                    lines.append(
                        f"| {int(row['ranks'])} | "
                        f"{float(comp.get('newton_linear_solve_s', math.nan)):.3f} | "
                        f"{float(comp.get('newton_linear_preconditioner_s', math.nan)):.3f} | "
                        f"{float(comp.get('newton_linear_orthogonalization_s', math.nan)):.3f} | "
                        f"{float(comp.get('builder_local_strain_s', math.nan)):.3f} | "
                        f"{float(comp.get('builder_local_constitutive_s', math.nan)):.3f} | "
                        f"{float(comp.get('builder_build_tangent_local_s', math.nan)):.3f} |"
                    )
                lines.append("")
                lines.append("| ranks | Newton solve speedup | linear solve speedup | preconditioner speedup | build tangent speedup |")
                lines.append("| ---: | ---: | ---: | ---: | ---: |")
                solve_map = _speedup_map(rows, "solve_total_s")
                linear_map = _speedup_map(rows, "newton_linear_solve_s")
                prec_map = _speedup_map(rows, "newton_linear_preconditioner_s")
                tangent_map = _speedup_map(rows, "builder_build_tangent_local_s")
                for row in rows:
                    ranks = int(row["ranks"])
                    lines.append(
                        f"| {ranks} | "
                        f"{float(solve_map.get(ranks, (math.nan, math.nan))[0]):.3f} | "
                        f"{float(linear_map.get(ranks, (math.nan, math.nan))[0]):.3f} | "
                        f"{float(prec_map.get(ranks, (math.nan, math.nan))[0]):.3f} | "
                        f"{float(tangent_map.get(ranks, (math.nan, math.nan))[0]):.3f} |"
                    )
                lines.append("")

    lines.append("## Plots")
    lines.append("")
    lines.append("### Overall Scaling")
    lines.append("")
    lines.append("![Overall scaling](overall_scaling.png)")
    lines.append("")
    lines.append("### Common Component Scaling")
    lines.append("")
    lines.append("![Common component scaling](common_component_scaling.png)")
    lines.append("")
    for impl in implementation_order:
        if rows_by_impl.get(impl):
            lines.append(f"### {implementation_labels.get(impl, impl)}")
            lines.append("")
            lines.append(
                f"![{implementation_labels.get(impl, impl)} component scaling]({_component_plot_name(impl)})"
            )
            lines.append("")

    lines.append("## Assets")
    lines.append("")
    lines.append("- `overall_scaling.png`: overall wall and solve scaling")
    lines.append("- `common_component_scaling.png`: backend build, initial guess, Newton solve")
    for impl in implementation_order:
        if rows_by_impl.get(impl):
            lines.append(
                f"- `{_component_plot_name(impl)}`: component scaling for `{implementation_labels.get(impl, impl)}`"
            )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate assets for the Plasticity3D implementation scaling comparison."
    )
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = _read_json(Path(args.summary_json))
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    enriched_rows = []
    for row in list(summary.get("rows", [])):
        if not isinstance(row, dict):
            continue
        result_path = _repo_path(str(row.get("result_json", "") or ""))
        if not result_path.exists():
            continue
        enriched_rows.append(_enrich_row(dict(row)))
    all_rows = [
        dict(row)
        for row in list(summary.get("rows", []))
        if isinstance(row, dict)
    ]
    implementation_order = _implementation_order(summary)
    implementation_labels = _implementation_labels(summary, implementation_order)
    implementation_families = _implementation_families(summary, implementation_order)
    implementation_colors = _implementation_colors(implementation_order)
    rows_by_impl = _rows_by_impl(enriched_rows, implementation_order)

    breakdown_payload = {
        "summary_json": str(Path(args.summary_json).resolve()),
        "rows": enriched_rows,
    }
    (out_dir / "scaling_breakdown.json").write_text(
        json.dumps(breakdown_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    _plot_overall_scaling(
        rows_by_impl,
        implementation_order=implementation_order,
        labels=implementation_labels,
        colors=implementation_colors,
        out_path=out_dir / "overall_scaling.png",
    )
    _plot_common_components(
        rows_by_impl,
        implementation_order=implementation_order,
        labels=implementation_labels,
        colors=implementation_colors,
        out_path=out_dir / "common_component_scaling.png",
    )
    for impl in implementation_order:
        rows = rows_by_impl.get(impl, [])
        if not rows:
            continue
        family = implementation_families.get(impl, "")
        if family == "source":
            component_keys = [
                ("newton_linear_solve_s", "Linear solve"),
                ("newton_linear_preconditioner_s", "Preconditioner"),
                ("newton_linear_orthogonalization_s", "Orthogonalization"),
                ("builder_local_strain_s", "Local strain"),
                ("builder_local_constitutive_s", "Local constitutive"),
                ("builder_build_tangent_local_s", "Build tangent"),
            ]
        else:
            component_keys = [
                ("newton_linear_assemble_s", "Linear assemble"),
                ("newton_linear_setup_s", "Linear setup"),
                ("newton_linear_solve_s", "Linear solve"),
                ("newton_line_search_s", "Line search"),
                ("callback_hessian_s", "Hessian callback"),
                ("callback_hessian_kernel_s", "Hessian kernel"),
            ]
        _plot_impl_breakdown(
            rows,
            title=f"{implementation_labels.get(impl, impl)} Component Scaling",
            component_keys=component_keys,
            color=implementation_colors[impl],
            out_path=out_dir / _component_plot_name(impl),
        )

    _write_report(
        summary,
        rows_by_impl,
        all_rows,
        implementation_order,
        implementation_labels,
        implementation_families,
        out_dir / "REPORT.md",
    )


if __name__ == "__main__":
    main()
