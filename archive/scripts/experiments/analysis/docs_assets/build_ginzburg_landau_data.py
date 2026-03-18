#!/usr/bin/env python3
"""Build curated GinzburgLandau overview data files."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import (
    DATA_ROOT,
    DIRECT_SPEED_PATHS,
    RUNS_ROOT,
    SUITE_SUMMARY_PATHS,
    ensure_overview_dirs,
    load_direct_speed_rows,
    load_suite_rows,
    publication_command,
    publication_run_dir,
    read_json,
    record_provenance,
    repo_rel,
    write_csv,
    write_json,
)


FAMILY = "ginzburg_landau"
FAMILY_DIR = DATA_ROOT / FAMILY


def _scalar_row(path):
    payload = read_json(path)
    if "results" in payload:
        return dict(payload["results"][0])
    step = payload["steps"][0]
    return {
        "energy": step["energy"],
        "iters": step["nit"],
        "total_ksp_its": step["linear_iters"],
        "solve_time": payload["solve_time_total"],
    }


def _energy_levels_rows():
    rows = load_suite_rows(FAMILY)
    solvers = ["fenics_custom", "jax_petsc_element", "jax_petsc_local_sfd"]
    table = []
    for level in range(5, 10):
        row = {"level": level}
        for solver in solvers:
            match = next(
                (
                    item for item in rows
                    if item["solver"] == solver and int(item["level"]) == level and int(item["nprocs"]) == 1
                ),
                None,
            )
            row[solver] = None if match is None else float(match["final_energy"])
        table.append(row)
    return table


def _parity_rows():
    base = publication_run_dir(FAMILY, "showcase")
    outputs = {
        "fenics_custom": _scalar_row(base / "fenics_custom" / "output.json"),
        "fenics_snes": _scalar_row(base / "fenics_snes" / "output.json"),
        "jax_petsc_element": _scalar_row(base / "jax_petsc_element" / "output.json"),
        "jax_petsc_local_sfd": _scalar_row(base / "jax_petsc_local_sfd" / "output.json"),
    }
    ref_energy = float(outputs["fenics_custom"]["energy"])
    rows = []
    for implementation, payload in outputs.items():
        energy = float(payload["energy"])
        rows.append(
            {
                "implementation": implementation,
                "result": "completed",
                "final_energy": energy,
                "energy_delta_vs_ref": energy - ref_energy,
                "rel_energy_delta_vs_ref": abs(energy - ref_energy) / max(abs(ref_energy), 1.0),
                "newton_iters": int(payload.get("iters", 0)),
                "linear_iters": int(payload.get("total_ksp_its", 0)),
                "wall_time_s": float(payload.get("solve_time", payload.get("time", 0.0))),
                "command": publication_command(base / implementation),
                "json_path": repo_rel(base / implementation / "output.json"),
            }
        )
    return rows


def _suite_scaling_rows():
    rows = load_suite_rows(FAMILY)
    selected = []
    for item in rows:
        if int(item["level"]) != 9:
            continue
        if item["solver"] not in {"fenics_custom", "jax_petsc_element", "jax_petsc_local_sfd"}:
            continue
        selected.append(
            {
                "solver": item["solver"],
                "nprocs": int(item["nprocs"]),
                "total_time_s": float(item["total_time_s"]),
                "final_energy": float(item["final_energy"]),
                "total_newton_iters": int(item["total_newton_iters"]),
                "total_linear_iters": int(item["total_linear_iters"]),
                "result": item["result"],
            }
        )
    return selected


def _problem_size_from_row(row: dict[str, object]) -> int:
    payload = read_json(Path(str(row["json_path"])))
    result = payload["result"]
    return int(result.get("free_dofs", result.get("total_dofs", 0)))


def _strong_scaling_rows():
    return _suite_scaling_rows()


def _mesh_timing_rows():
    rows = load_suite_rows(FAMILY)
    selected = []
    for item in rows:
        if int(item["nprocs"]) != 32:
            continue
        if item["solver"] not in {"fenics_custom", "jax_petsc_element", "jax_petsc_local_sfd"}:
            continue
        selected.append(
            {
                "solver": item["solver"],
                "level": int(item["level"]),
                "problem_size": _problem_size_from_row(item),
                "nprocs": int(item["nprocs"]),
                "total_time_s": float(item["total_time_s"]),
                "final_energy": float(item["final_energy"]),
                "total_newton_iters": int(item["total_newton_iters"]),
                "total_linear_iters": int(item["total_linear_iters"]),
                "result": item["result"],
            }
        )
    return sorted(selected, key=lambda row: (row["solver"], row["level"]))


def _showcase_speed_rows():
    rows = load_direct_speed_rows(FAMILY)
    return [
        {
            "case_id": row["case_id"],
            "implementation": row["implementation"],
            "mpi_ranks": int(row["mpi_ranks"]),
            "median_wall_time_s": float(row["median_wall_time_s"]),
            "median_newton_iters": int(row["median_newton_iters"]),
            "median_linear_iters": int(row["median_linear_iters"]),
            "median_final_energy": float(row["median_final_energy"]),
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

    energy_levels = _energy_levels_rows()
    parity_rows = _parity_rows()
    suite_scaling = _suite_scaling_rows()
    strong_scaling = _strong_scaling_rows()
    mesh_timing = _mesh_timing_rows()
    showcase_speed = _showcase_speed_rows()

    write_csv(
        FAMILY_DIR / "energy_levels.csv",
        energy_levels,
        ["level", "fenics_custom", "jax_petsc_element", "jax_petsc_local_sfd"],
    )
    write_csv(
        FAMILY_DIR / "parity_showcase.csv",
        parity_rows,
        [
            "implementation",
            "result",
            "final_energy",
            "energy_delta_vs_ref",
            "rel_energy_delta_vs_ref",
            "newton_iters",
            "linear_iters",
            "wall_time_s",
            "command",
            "json_path",
        ],
    )
    write_csv(
        FAMILY_DIR / "suite_scaling_level9.csv",
        suite_scaling,
        ["solver", "nprocs", "total_time_s", "final_energy", "total_newton_iters", "total_linear_iters", "result"],
    )
    write_csv(
        FAMILY_DIR / "strong_scaling.csv",
        strong_scaling,
        ["solver", "nprocs", "total_time_s", "final_energy", "total_newton_iters", "total_linear_iters", "result"],
    )
    write_csv(
        FAMILY_DIR / "mesh_timing.csv",
        mesh_timing,
        ["solver", "level", "problem_size", "nprocs", "total_time_s", "final_energy", "total_newton_iters", "total_linear_iters", "result"],
    )
    write_csv(
        FAMILY_DIR / "showcase_speed.csv",
        showcase_speed,
        [
            "case_id",
            "implementation",
            "mpi_ranks",
            "median_wall_time_s",
            "median_newton_iters",
            "median_linear_iters",
            "median_final_energy",
            "status",
        ],
    )

    sample_state_src = RUNS_ROOT / FAMILY / "showcase" / "fenics_custom" / "state.npz"
    sample_state_dst = FAMILY_DIR / "sample_state.npz"
    shutil.copyfile(sample_state_src, sample_state_dst)

    sources = {
        "suite_summary": repo_rel(SUITE_SUMMARY_PATHS[FAMILY]),
        "direct_speed_csv": repo_rel(DIRECT_SPEED_PATHS[FAMILY]),
        "publication_manifest": repo_rel(RUNS_ROOT / "manifest.json"),
        "publication_commands": {
            key: publication_command(publication_run_dir(FAMILY, "showcase") / key)
            for key in (
                "fenics_custom",
                "fenics_snes",
                "jax_petsc_element",
                "jax_petsc_local_sfd",
            )
        },
    }
    write_json(FAMILY_DIR / "sources.json", sources)
    record_provenance(
        FAMILY_DIR / "build_ginzburg_landau_data.provenance.json",
        script_name="overview/img/scripts/build_ginzburg_landau_data.py",
        inputs=[
            repo_rel(SUITE_SUMMARY_PATHS[FAMILY]),
            repo_rel(DIRECT_SPEED_PATHS[FAMILY]),
            repo_rel(RUNS_ROOT / "manifest.json"),
        ],
        outputs=[
            repo_rel(FAMILY_DIR / "energy_levels.csv"),
            repo_rel(FAMILY_DIR / "parity_showcase.csv"),
            repo_rel(FAMILY_DIR / "suite_scaling_level9.csv"),
            repo_rel(FAMILY_DIR / "strong_scaling.csv"),
            repo_rel(FAMILY_DIR / "mesh_timing.csv"),
            repo_rel(FAMILY_DIR / "showcase_speed.csv"),
            repo_rel(sample_state_dst),
            repo_rel(FAMILY_DIR / "sources.json"),
        ],
        notes="Curated overview data extracted from the maintained replication campaign and publication reruns.",
    )


if __name__ == "__main__":
    main()
