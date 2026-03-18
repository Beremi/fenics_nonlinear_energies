#!/usr/bin/env python3
"""Build curated HyperElasticity overview data files."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import (
    DATA_ROOT,
    DIRECT_SPEED_PATHS,
    REPLICATION_ROOT,
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


FAMILY = "hyperelasticity"
FAMILY_DIR = DATA_ROOT / FAMILY
PURE_JAX_ROOT = REPLICATION_ROOT / "runs" / FAMILY / "pure_jax_suite_best"


def _load_publication_row(path: Path) -> dict:
    payload = read_json(path)
    steps = payload.get("steps", [])
    final_step = steps[-1] if steps else {}
    return {
        "final_energy": float(final_step.get("energy", float("nan"))),
        "completed_steps": int(len(steps)),
        "newton_iters": int(sum(int(step.get("iters", step.get("nit", 0))) for step in steps)),
        "linear_iters": int(sum(int(step.get("linear_iters", 0)) for step in steps)),
        "wall_time_s": float(payload.get("solve_time_total", payload.get("time", 0.0))),
        "result": str(payload.get("result", "completed")),
    }


def _load_pure_jax_level(level: int, total_steps: int = 24) -> dict:
    path = PURE_JAX_ROOT / f"pure_jax_steps{total_steps}_l{level}.json"
    payload = read_json(path)
    final_step = payload["steps"][-1]
    return {
        "level": level,
        "total_steps": total_steps,
        "final_energy": float(final_step["energy"]),
        "total_time_s": float(payload["time"]),
        "total_newton_iters": int(payload["total_newton_iters"]),
        "total_linear_iters": int(payload["total_linear_iters"]),
        "result": str(payload["result"]),
        "json_path": repo_rel(path),
    }


def _energy_levels_rows() -> list[dict[str, object]]:
    suite_rows = load_suite_rows(FAMILY)
    pure_rows = {level: _load_pure_jax_level(level, 24) for level in (1, 2, 3)}
    table = []
    for level in (1, 2, 3, 4):
        row = {"level": level}
        for solver in ("fenics_custom", "jax_petsc_element"):
            match = next(
                (
                    item for item in suite_rows
                    if item["solver"] == solver
                    and int(item["level"]) == level
                    and int(item["nprocs"]) == 1
                    and int(item["total_steps"]) == 24
                ),
                None,
            )
            row[solver] = None if match is None else float(match["final_energy"])
        row["jax_serial"] = None if level not in pure_rows else float(pure_rows[level]["final_energy"])
        table.append(row)
    return table


def _parity_rows() -> list[dict[str, object]]:
    base = publication_run_dir(FAMILY, "showcase")
    outputs = {
        "fenics_custom": _load_publication_row(base / "fenics_custom" / "output.json"),
        "jax_petsc_element": _load_publication_row(base / "jax_petsc_element" / "output.json"),
        "jax_serial": _load_publication_row(base / "jax_serial" / "output.json"),
    }
    ref_energy = outputs["fenics_custom"]["final_energy"]
    rows = []
    for implementation, payload in outputs.items():
        energy = float(payload["final_energy"])
        rows.append(
            {
                "implementation": implementation,
                "result": payload["result"],
                "completed_steps": int(payload["completed_steps"]),
                "final_energy": energy,
                "energy_delta_vs_ref": energy - ref_energy,
                "rel_energy_delta_vs_ref": abs(energy - ref_energy) / max(abs(ref_energy), 1.0),
                "total_newton_iters": int(payload["newton_iters"]),
                "total_linear_iters": int(payload["linear_iters"]),
                "wall_time_s": float(payload["wall_time_s"]),
                "command": publication_command(base / implementation),
                "json_path": repo_rel(base / implementation / "output.json"),
            }
        )
    return rows


def _showcase_speed_rows() -> list[dict[str, object]]:
    rows = load_direct_speed_rows(FAMILY)
    return [
        {
            "case_id": row["case_id"],
            "implementation": row["implementation"],
            "mpi_ranks": int(row["mpi_ranks"]),
            "median_wall_time_s": float(row["median_wall_time_s"]),
            "median_newton_iters": int(row["median_newton_iters"]),
            "median_linear_iters": int(row["median_linear_iters"]),
            "median_final_energy": float(row["median_final_energy"]) if row["median_final_energy"] != "nan" else None,
            "status": row["status"],
        }
        for row in rows
    ]


def _problem_size_from_suite_row(row: dict[str, object]) -> int:
    payload = read_json(Path(str(row["json_path"])))
    result = payload["result"]
    return int(result.get("total_dofs", 0))


def _suite_scaling_rows() -> list[dict[str, object]]:
    suite_rows = load_suite_rows(FAMILY)
    selected = []
    for item in suite_rows:
        if int(item["level"]) != 4 or int(item["total_steps"]) != 24:
            continue
        if item["solver"] not in {"fenics_custom", "jax_petsc_element"}:
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


def _mesh_timing_rows() -> list[dict[str, object]]:
    suite_rows = load_suite_rows(FAMILY)
    selected = []
    for item in suite_rows:
        if int(item["nprocs"]) != 32 or int(item["total_steps"]) != 24:
            continue
        if item["solver"] not in {"fenics_custom", "jax_petsc_element"}:
            continue
        selected.append(
            {
                "solver": item["solver"],
                "level": int(item["level"]),
                "problem_size": _problem_size_from_suite_row(item),
                "nprocs": int(item["nprocs"]),
                "total_time_s": float(item["total_time_s"]),
                "final_energy": float(item["final_energy"]),
                "total_newton_iters": int(item["total_newton_iters"]),
                "total_linear_iters": int(item["total_linear_iters"]),
                "result": item["result"],
            }
        )
    return sorted(selected, key=lambda row: (row["solver"], row["level"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    del args

    ensure_overview_dirs()
    FAMILY_DIR.mkdir(parents=True, exist_ok=True)

    energy_levels = _energy_levels_rows()
    parity_rows = _parity_rows()
    showcase_speed = _showcase_speed_rows()
    suite_scaling = _suite_scaling_rows()
    mesh_timing = _mesh_timing_rows()

    write_csv(
        FAMILY_DIR / "energy_levels.csv",
        energy_levels,
        ["level", "fenics_custom", "jax_petsc_element", "jax_serial"],
    )
    write_csv(
        FAMILY_DIR / "parity_showcase.csv",
        parity_rows,
        [
            "implementation",
            "result",
            "completed_steps",
            "final_energy",
            "energy_delta_vs_ref",
            "rel_energy_delta_vs_ref",
            "total_newton_iters",
            "total_linear_iters",
            "wall_time_s",
            "command",
            "json_path",
        ],
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
    write_csv(
        FAMILY_DIR / "suite_scaling_level4_steps24.csv",
        suite_scaling,
        ["solver", "nprocs", "total_time_s", "final_energy", "total_newton_iters", "total_linear_iters", "result"],
    )
    write_csv(
        FAMILY_DIR / "strong_scaling.csv",
        suite_scaling,
        ["solver", "nprocs", "total_time_s", "final_energy", "total_newton_iters", "total_linear_iters", "result"],
    )
    write_csv(
        FAMILY_DIR / "mesh_timing.csv",
        mesh_timing,
        ["solver", "level", "problem_size", "nprocs", "total_time_s", "final_energy", "total_newton_iters", "total_linear_iters", "result"],
    )

    sample_state_src = RUNS_ROOT / FAMILY / "sample_render" / "jax_petsc_element_l4_np32" / "state.npz"
    sample_state_dst = FAMILY_DIR / "sample_state.npz"
    shutil.copyfile(sample_state_src, sample_state_dst)

    sources = {
        "suite_summary": repo_rel(SUITE_SUMMARY_PATHS[FAMILY]),
        "pure_jax_summary": repo_rel(SUITE_SUMMARY_PATHS["hyperelasticity_pure_jax"]),
        "direct_speed_csv": repo_rel(DIRECT_SPEED_PATHS[FAMILY]),
        "publication_manifest": repo_rel(RUNS_ROOT / "manifest.json"),
        "publication_commands": {
            key: publication_command(publication_run_dir(FAMILY, "showcase") / key)
            for key in ("fenics_custom", "jax_petsc_element", "jax_serial")
        },
        "sample_render_command": publication_command(
            publication_run_dir(FAMILY, "sample_render") / "jax_petsc_element_l4_np32"
        ),
    }
    write_json(FAMILY_DIR / "sources.json", sources)
    record_provenance(
        FAMILY_DIR / "build_hyperelasticity_data.provenance.json",
        script_name="overview/img/scripts/build_hyperelasticity_data.py",
        inputs=[
            repo_rel(SUITE_SUMMARY_PATHS[FAMILY]),
            repo_rel(SUITE_SUMMARY_PATHS["hyperelasticity_pure_jax"]),
            repo_rel(DIRECT_SPEED_PATHS[FAMILY]),
            repo_rel(RUNS_ROOT / "manifest.json"),
        ],
        outputs=[
            repo_rel(FAMILY_DIR / "energy_levels.csv"),
            repo_rel(FAMILY_DIR / "parity_showcase.csv"),
            repo_rel(FAMILY_DIR / "showcase_speed.csv"),
            repo_rel(FAMILY_DIR / "suite_scaling_level4_steps24.csv"),
            repo_rel(FAMILY_DIR / "strong_scaling.csv"),
            repo_rel(FAMILY_DIR / "mesh_timing.csv"),
            repo_rel(sample_state_dst),
            repo_rel(FAMILY_DIR / "sources.json"),
        ],
        notes="Curated overview data extracted from the maintained replication campaign, pure-JAX suite, and publication reruns.",
    )


if __name__ == "__main__":
    main()
