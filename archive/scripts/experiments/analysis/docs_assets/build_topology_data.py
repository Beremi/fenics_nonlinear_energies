#!/usr/bin/env python3
"""Build curated Topology overview data files."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from common import (
    DATA_ROOT,
    DIRECT_SPEED_PATHS,
    REPLICATION_ROOT,
    RUNS_ROOT,
    family_gif,
    SUITE_SCALING_PATHS,
    ensure_overview_dirs,
    load_direct_speed_rows,
    read_json,
    record_provenance,
    repo_rel,
    write_csv,
    write_json,
)


FAMILY = "topology"
FAMILY_DIR = DATA_ROOT / FAMILY
SERIAL_ROOT = REPLICATION_ROOT / "runs" / FAMILY / "serial_reference"
PARALLEL_FINAL_ROOT = REPLICATION_ROOT / "runs" / FAMILY / "parallel_final"
PARALLEL_SCALING_ROOT = REPLICATION_ROOT / "runs" / FAMILY / "parallel_scaling"
OVERVIEW_MESH_SCALING_ROOT = RUNS_ROOT / FAMILY / "mesh_scaling"


def _read_scaling_summary() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with SUITE_SCALING_PATHS[FAMILY].open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "ranks": int(row["ranks"]),
                    "result": row["result"],
                    "outer_iterations": int(row["outer_iterations"]),
                    "final_p_penal": float(row["final_p"]),
                    "final_volume_fraction": float(row["final_volume"]),
                    "final_compliance": float(row["final_compliance"]),
                    "wall_time_s": float(row["wall_time"]),
                    "solve_time_s": float(row["solve_time"]),
                    "speedup_vs_1": 0.0,
                }
            )
    if rows:
        baseline = rows[0]["wall_time_s"]
        for row in rows:
            row["speedup_vs_1"] = float(baseline) / max(float(row["wall_time_s"]), 1e-12)
    return rows


def _strong_scaling_rows() -> list[dict[str, object]]:
    parallel = read_json(PARALLEL_FINAL_ROOT / "parallel_full_run.json")
    mesh = parallel["mesh"]
    problem_size = int(mesh["displacement_free_dofs"] + mesh["design_free_dofs"])
    rows = []
    for row in _read_scaling_summary():
        rows.append(
            {
                "solver": "jax_parallel",
                "ranks": int(row["ranks"]),
                "problem_size": problem_size,
                "wall_time_s": float(row["wall_time_s"]),
                "solve_time_s": float(row["solve_time_s"]),
                "final_p_penal": float(row["final_p_penal"]),
                "final_compliance": float(row["final_compliance"]),
                "final_volume_fraction": float(row["final_volume_fraction"]),
                "outer_iterations": int(row["outer_iterations"]),
                "speedup_vs_1": float(row["speedup_vs_1"]),
                "result": row["result"],
            }
        )
    return rows


def _mesh_timing_rows() -> list[dict[str, object]]:
    rows = []
    for path in sorted(OVERVIEW_MESH_SCALING_ROOT.glob("nx*_ny*_np8/output.json")):
        payload = read_json(path)
        mesh = payload["mesh"]
        final = payload["final_metrics"]
        rows.append(
            {
                "solver": "jax_parallel",
                "mesh_label": f"{mesh['nx']}x{mesh['ny']}",
                "nx": int(mesh["nx"]),
                "ny": int(mesh["ny"]),
                "nprocs": int(payload["nprocs"]),
                "problem_size": int(mesh["displacement_free_dofs"] + mesh["design_free_dofs"]),
                "wall_time_s": float(payload["time"]),
                "solve_time_s": float(payload["time"] - payload["setup_time"]),
                "final_compliance": float(final["final_compliance"]),
                "final_volume_fraction": float(final["final_volume_fraction"]),
                "outer_iterations": int(final["outer_iterations"]),
                "result": payload["result"],
                "json_path": repo_rel(path),
            }
        )
    return sorted(rows, key=lambda row: row["problem_size"])


def _objective_history_rows() -> list[dict[str, object]]:
    history_path = PARALLEL_FINAL_ROOT / "parallel_full_outer_history.csv"
    rows = []
    with history_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "outer_iteration": int(row["outer_iter"]),
                    "p_penal": float(row["p_penal"]),
                    "compliance": float(row["compliance"]),
                    "volume_fraction": float(row["volume_fraction"]),
                    "state_change": float(row["theta_state_change"]),
                    "design_change": float(row["design_change"]),
                }
            )
    return rows


def _resolution_rows() -> list[dict[str, object]]:
    serial = read_json(SERIAL_ROOT / "report_run.json")
    parallel = read_json(PARALLEL_FINAL_ROOT / "parallel_full_run.json")
    rows = [
        {
            "label": "serial_reference",
            "mesh": "192x96",
            "ranks": 1,
            "result": serial["result"],
            "outer_iterations": int(serial["final_metrics"]["outer_iterations"]),
            "final_compliance": float(serial["final_metrics"]["final_compliance"]),
            "final_volume_fraction": float(serial["final_metrics"]["final_volume_fraction"]),
            "wall_time_s": float(serial["time"]),
        },
        {
            "label": "parallel_final",
            "mesh": "768x384",
            "ranks": int(parallel["nprocs"]),
            "result": parallel["result"],
            "outer_iterations": int(parallel["final_metrics"]["outer_iterations"]),
            "final_compliance": float(parallel["final_metrics"]["final_compliance"]),
            "final_volume_fraction": float(parallel["final_metrics"]["final_volume_fraction"]),
            "wall_time_s": float(parallel["time"]),
        },
    ]
    for row in _read_scaling_summary():
        rows.append(
            {
                "label": f"parallel_scaling_r{row['ranks']}",
                "mesh": "768x384",
                "ranks": int(row["ranks"]),
                "result": row["result"],
                "outer_iterations": int(row["outer_iterations"]),
                "final_compliance": float(row["final_compliance"]),
                "final_volume_fraction": float(row["final_volume_fraction"]),
                "wall_time_s": float(row["wall_time_s"]),
            }
        )
    return rows


def _direct_comparison_rows() -> list[dict[str, object]]:
    rows = load_direct_speed_rows(FAMILY)
    return [
        {
            "case_id": row["case_id"],
            "implementation": row["implementation"],
            "mpi_ranks": int(row["mpi_ranks"]),
            "median_wall_time_s": float(row["median_wall_time_s"]),
            "median_final_compliance": float(row["median_final_compliance"]),
            "median_final_volume_fraction": float(row["median_final_volume_fraction"]),
            "status": row["status"],
        }
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    del args

    ensure_overview_dirs()
    FAMILY_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(
        FAMILY_DIR / "objective_history.csv",
        _objective_history_rows(),
        ["outer_iteration", "p_penal", "compliance", "volume_fraction", "state_change", "design_change"],
    )
    write_csv(
        FAMILY_DIR / "resolution_objectives.csv",
        _resolution_rows(),
        ["label", "mesh", "ranks", "result", "outer_iterations", "final_compliance", "final_volume_fraction", "wall_time_s"],
    )
    write_csv(
        FAMILY_DIR / "direct_comparison.csv",
        _direct_comparison_rows(),
        ["case_id", "implementation", "mpi_ranks", "median_wall_time_s", "median_final_compliance", "median_final_volume_fraction", "status"],
    )
    write_csv(
        FAMILY_DIR / "parallel_scaling.csv",
        _read_scaling_summary(),
        ["ranks", "result", "outer_iterations", "final_p_penal", "final_volume_fraction", "final_compliance", "wall_time_s", "solve_time_s", "speedup_vs_1"],
    )
    write_csv(
        FAMILY_DIR / "strong_scaling.csv",
        _strong_scaling_rows(),
        ["solver", "ranks", "problem_size", "wall_time_s", "solve_time_s", "final_p_penal", "final_compliance", "final_volume_fraction", "outer_iterations", "speedup_vs_1", "result"],
    )
    write_csv(
        FAMILY_DIR / "mesh_timing.csv",
        _mesh_timing_rows(),
        ["solver", "mesh_label", "nx", "ny", "nprocs", "problem_size", "wall_time_s", "solve_time_s", "final_compliance", "final_volume_fraction", "outer_iterations", "result", "json_path"],
    )

    shutil.copyfile(SERIAL_ROOT / "report_state.npz", FAMILY_DIR / "serial_state.npz")
    shutil.copyfile(PARALLEL_FINAL_ROOT / "parallel_full_state.npz", FAMILY_DIR / "parallel_final_state.npz")
    topology_gif = family_gif(f"{FAMILY}/topology_parallel_final_evolution.gif")
    shutil.copyfile(PARALLEL_FINAL_ROOT / "density_evolution.gif", topology_gif)

    sources = {
        "direct_speed_csv": repo_rel(DIRECT_SPEED_PATHS[FAMILY]),
        "parallel_scaling_csv": repo_rel(SUITE_SCALING_PATHS[FAMILY]),
        "serial_report_json": repo_rel(SERIAL_ROOT / "report_run.json"),
        "parallel_report_json": repo_rel(PARALLEL_FINAL_ROOT / "parallel_full_run.json"),
        "serial_command": (SERIAL_ROOT / "command.txt").read_text(encoding="utf-8").strip(),
        "parallel_final_command": (PARALLEL_FINAL_ROOT / "command.txt").read_text(encoding="utf-8").strip(),
        "parallel_scaling_command": (PARALLEL_SCALING_ROOT / "command.txt").read_text(encoding="utf-8").strip(),
        "mesh_scaling_commands": {
            path.parent.name: (path.parent / "command.txt").read_text(encoding="utf-8").strip()
            for path in sorted(OVERVIEW_MESH_SCALING_ROOT.glob("*/output.json"))
        },
    }
    write_json(FAMILY_DIR / "sources.json", sources)
    record_provenance(
        FAMILY_DIR / "build_topology_data.provenance.json",
        script_name="overview/img/scripts/build_topology_data.py",
        inputs=[
            repo_rel(DIRECT_SPEED_PATHS[FAMILY]),
            repo_rel(SUITE_SCALING_PATHS[FAMILY]),
            repo_rel(SERIAL_ROOT / "report_run.json"),
            repo_rel(PARALLEL_FINAL_ROOT / "parallel_full_outer_history.csv"),
            repo_rel(PARALLEL_FINAL_ROOT / "parallel_full_run.json"),
        ],
        outputs=[
            repo_rel(FAMILY_DIR / "objective_history.csv"),
            repo_rel(FAMILY_DIR / "resolution_objectives.csv"),
            repo_rel(FAMILY_DIR / "direct_comparison.csv"),
            repo_rel(FAMILY_DIR / "parallel_scaling.csv"),
            repo_rel(FAMILY_DIR / "strong_scaling.csv"),
            repo_rel(FAMILY_DIR / "mesh_timing.csv"),
            repo_rel(FAMILY_DIR / "serial_state.npz"),
            repo_rel(FAMILY_DIR / "parallel_final_state.npz"),
            repo_rel(topology_gif),
            repo_rel(FAMILY_DIR / "sources.json"),
        ],
        notes="Curated topology overview data extracted from the maintained replication campaign.",
    )


if __name__ == "__main__":
    main()
