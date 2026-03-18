#!/usr/bin/env python3
"""Sync tracked docs-asset data files from a fresh local reproduction campaign."""

from __future__ import annotations

import argparse
import csv
import shutil
import math
from pathlib import Path
from typing import Any

from common import (
    DOCS_ASSETS_ROOT,
    family_data,
    family_gif,
    normalize_command,
    read_csv_rows,
    read_json,
    record_provenance,
    repo_rel,
    write_csv,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _smoke_rows(smoke_summary: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(smoke_summary)
    return {str(row["id"]): dict(row) for row in payload.get("rows", [])}


def _suite_rows(summary_path: Path) -> list[dict[str, Any]]:
    payload = read_json(summary_path)
    return list(payload.get("rows", []))


def _scalar_result(json_path: Path) -> dict[str, Any]:
    payload = read_json(json_path)
    return dict(payload["results"][0])


def _gl_payload(json_path: Path) -> dict[str, Any]:
    return read_json(json_path)


def _he_payload(json_path: Path) -> dict[str, Any]:
    return read_json(json_path)


def _pure_jax_final_energy(campaign_root: Path, level: int, total_steps: int = 24) -> float:
    payload = read_json(
        campaign_root
        / "runs"
        / "hyperelasticity"
        / "pure_jax_suite_best"
        / f"pure_jax_steps{total_steps}_l{level}.json"
    )
    steps = list(payload.get("steps", []))
    if not steps:
        return math.nan
    return float(steps[-1].get("energy", math.nan))


def _pure_jax_case_payload(campaign_root: Path, level: int, total_steps: int = 24) -> dict[str, Any]:
    return read_json(
        campaign_root
        / "runs"
        / "hyperelasticity"
        / "pure_jax_suite_best"
        / f"pure_jax_steps{total_steps}_l{level}.json"
    )


def _problem_size_from_suite_json(json_path: Path) -> int:
    payload = read_json(json_path)
    if "results" in payload:
        result = payload["results"][0]
        return int(result.get("dofs", result.get("free_dofs", result.get("total_dofs", 0))))
    if "free_dofs" in payload:
        return int(payload.get("free_dofs", payload.get("total_dofs", 0)))
    if "result" in payload:
        result = payload["result"]
        if isinstance(result, dict):
            return int(result.get("free_dofs", result.get("total_dofs", 0)))
    return int(payload.get("total_dofs", 0))


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _maybe_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _load_topology_scaling_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
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
        baseline = float(rows[0]["wall_time_s"])
        for row in rows:
            row["speedup_vs_1"] = baseline / max(float(row["wall_time_s"]), 1e-12)
    return rows


def _sync_plaplace(campaign_root: Path, smoke: dict[str, dict[str, Any]]) -> list[str]:
    family = "plaplace"
    suite_rows = _suite_rows(campaign_root / "runs" / family / "final_suite" / "summary.json")
    family_dir = family_data(f"{family}/.")
    outputs: list[str] = []

    energy_levels = []
    for level in range(5, 10):
        row: dict[str, Any] = {"level": level}
        for solver in ("fenics_custom", "jax_petsc_element", "jax_petsc_local_sfd"):
            match = next(
                (
                    item
                    for item in suite_rows
                    if item["solver"] == solver and int(item["level"]) == level and int(item["nprocs"]) == 1
                ),
                None,
            )
            row[solver] = None if match is None else float(match["final_energy"])
        energy_levels.append(row)
    path = family_dir / "energy_levels.csv"
    write_csv(path, energy_levels, ["level", "fenics_custom", "jax_petsc_element", "jax_petsc_local_sfd"])
    outputs.append(repo_rel(path))

    parity_rows = []
    fenics_custom = smoke["plaplace_fenics_custom_l5"]
    fenics_snes = smoke["plaplace_fenics_snes_l5"]
    jax_serial = smoke["plaplace_jax_serial_l5"]
    jax_petsc = smoke["plaplace_jax_petsc_element_l5"]
    for row in (fenics_custom, fenics_snes, jax_serial, jax_petsc):
        pass
    ref_energy = float(fenics_custom["final_energy"])
    parity_rows.extend(
        [
            {
                "implementation": "fenics_custom",
                "result": fenics_custom["result"],
                "final_energy": float(fenics_custom["final_energy"]),
                "energy_delta_vs_ref": 0.0,
                "rel_energy_delta_vs_ref": 0.0,
                "newton_iters": int(fenics_custom["newton_iters"]),
                "linear_iters": int(fenics_custom["linear_iters"]),
                "wall_time_s": float(fenics_custom["solver_wall_time_s"]),
            },
            {
                "implementation": "fenics_snes",
                "result": fenics_snes["result"],
                "final_energy": float(fenics_snes["final_energy"]),
                "energy_delta_vs_ref": float(fenics_snes["final_energy"]) - ref_energy,
                "rel_energy_delta_vs_ref": abs(float(fenics_snes["final_energy"]) - ref_energy) / max(abs(ref_energy), 1.0),
                "newton_iters": int(fenics_snes["newton_iters"]),
                "linear_iters": int(fenics_snes["linear_iters"]),
                "wall_time_s": float(fenics_snes["solver_wall_time_s"]),
            },
            {
                "implementation": "jax_serial",
                "result": jax_serial["result"],
                "final_energy": float(jax_serial["final_energy"]),
                "energy_delta_vs_ref": float(jax_serial["final_energy"]) - ref_energy,
                "rel_energy_delta_vs_ref": abs(float(jax_serial["final_energy"]) - ref_energy) / max(abs(ref_energy), 1.0),
                "newton_iters": int(jax_serial["newton_iters"]),
                "linear_iters": int(jax_serial["linear_iters"]),
                "wall_time_s": float(jax_serial["solver_wall_time_s"]),
            },
            {
                "implementation": "jax_petsc_element",
                "result": jax_petsc["result"],
                "final_energy": float(jax_petsc["final_energy"]),
                "energy_delta_vs_ref": float(jax_petsc["final_energy"]) - ref_energy,
                "rel_energy_delta_vs_ref": abs(float(jax_petsc["final_energy"]) - ref_energy) / max(abs(ref_energy), 1.0),
                "newton_iters": int(jax_petsc["newton_iters"]),
                "linear_iters": int(jax_petsc["linear_iters"]),
                "wall_time_s": float(jax_petsc["solver_wall_time_s"]),
            },
        ]
    )
    local_sfd = next(
        item
        for item in suite_rows
        if item["solver"] == "jax_petsc_local_sfd" and int(item["level"]) == 5 and int(item["nprocs"]) == 1
    )
    parity_rows.append(
        {
            "implementation": "jax_petsc_local_sfd",
            "result": local_sfd["result"],
            "final_energy": float(local_sfd["final_energy"]),
            "energy_delta_vs_ref": float(local_sfd["final_energy"]) - ref_energy,
            "rel_energy_delta_vs_ref": abs(float(local_sfd["final_energy"]) - ref_energy) / max(abs(ref_energy), 1.0),
            "newton_iters": int(local_sfd["total_newton_iters"]),
            "linear_iters": int(local_sfd["total_linear_iters"]),
            "wall_time_s": float(local_sfd["total_time_s"]),
        }
    )
    path = family_dir / "parity_showcase.csv"
    write_csv(
        path,
        parity_rows,
        ["implementation", "result", "final_energy", "energy_delta_vs_ref", "rel_energy_delta_vs_ref", "newton_iters", "linear_iters", "wall_time_s"],
    )
    outputs.append(repo_rel(path))

    strong_scaling = [
        {
            "solver": row["solver"],
            "nprocs": int(row["nprocs"]),
            "total_time_s": float(row["total_time_s"]),
            "final_energy": float(row["final_energy"]),
            "total_newton_iters": int(row["total_newton_iters"]),
            "total_linear_iters": int(row["total_linear_iters"]),
            "result": row["result"],
        }
        for row in suite_rows
        if int(row["level"]) == 9 and row["solver"] in {"fenics_custom", "jax_petsc_element", "jax_petsc_local_sfd"}
    ]
    strong_scaling.sort(key=lambda row: (row["solver"], row["nprocs"]))
    for name in ("suite_scaling_level9.csv", "strong_scaling.csv"):
        path = family_dir / name
        write_csv(path, strong_scaling, ["solver", "nprocs", "total_time_s", "final_energy", "total_newton_iters", "total_linear_iters", "result"])
        outputs.append(repo_rel(path))

    mesh_rows = []
    for row in suite_rows:
        if int(row["nprocs"]) != 32:
            continue
        if row["solver"] not in {"fenics_custom", "jax_petsc_element", "jax_petsc_local_sfd"}:
            continue
        json_path = Path(str(row["json_path"]))
        mesh_rows.append(
            {
                "solver": row["solver"],
                "level": int(row["level"]),
                "problem_size": _problem_size_from_suite_json(json_path),
                "nprocs": int(row["nprocs"]),
                "total_time_s": float(row["total_time_s"]),
                "final_energy": float(row["final_energy"]),
                "total_newton_iters": int(row["total_newton_iters"]),
                "total_linear_iters": int(row["total_linear_iters"]),
                "result": row["result"],
            }
        )
    mesh_rows.sort(key=lambda row: (row["solver"], row["level"]))
    path = family_dir / "mesh_timing.csv"
    write_csv(
        path,
        mesh_rows,
        ["solver", "level", "problem_size", "nprocs", "total_time_s", "final_energy", "total_newton_iters", "total_linear_iters", "result"],
    )
    outputs.append(repo_rel(path))

    _copy(Path(smoke["plaplace_fenics_custom_l5"]["state_path"]), family_dir / "sample_state.npz")
    outputs.append(repo_rel(family_dir / "sample_state.npz"))

    write_json(
        family_dir / "sources.json",
        {
            "campaign_root": repo_rel(campaign_root),
            "smoke_summary": repo_rel(campaign_root / "runs" / "readme_docs_smoke" / "summary.json"),
            "suite_summary": repo_rel(campaign_root / "runs" / family / "final_suite" / "summary.json"),
            "sample_state_case": smoke["plaplace_fenics_custom_l5"]["command"],
        },
    )
    outputs.append(repo_rel(family_dir / "sources.json"))

    record_provenance(
        family_dir / "sync_tracked_docs_data.provenance.json",
        script_name="experiments/analysis/docs_assets/sync_tracked_docs_data.py",
        inputs=[
            repo_rel(campaign_root / "runs" / "readme_docs_smoke" / "summary.json"),
            repo_rel(campaign_root / "runs" / family / "final_suite" / "summary.json"),
        ],
        outputs=outputs,
        notes="Tracked pLaplace docs data regenerated from the fresh local canonical campaign.",
    )
    outputs.append(repo_rel(family_dir / "sync_tracked_docs_data.provenance.json"))
    return outputs


def _sync_ginzburg_landau(campaign_root: Path, smoke: dict[str, dict[str, Any]]) -> list[str]:
    family = "ginzburg_landau"
    suite_rows = _suite_rows(campaign_root / "runs" / family / "final_suite" / "summary.json")
    family_dir = family_data(f"{family}/.")
    outputs: list[str] = []

    energy_levels = []
    for level in range(5, 10):
        row: dict[str, Any] = {"level": level}
        for solver in ("fenics_custom", "jax_petsc_element", "jax_petsc_local_sfd"):
            match = next(
                (
                    item
                    for item in suite_rows
                    if item["solver"] == solver and int(item["level"]) == level and int(item["nprocs"]) == 1
                ),
                None,
            )
            row[solver] = None if match is None else float(match["final_energy"])
        energy_levels.append(row)
    path = family_dir / "energy_levels.csv"
    write_csv(path, energy_levels, ["level", "fenics_custom", "jax_petsc_element", "jax_petsc_local_sfd"])
    outputs.append(repo_rel(path))

    fenics_custom = smoke["gl_fenics_custom_l5"]
    fenics_snes = smoke["gl_fenics_snes_l5"]
    jax_petsc = smoke["gl_jax_petsc_element_l5"]
    ref_energy = float(fenics_custom["final_energy"])
    parity_rows = [
        {
            "implementation": "fenics_custom",
            "result": fenics_custom["result"],
            "final_energy": float(fenics_custom["final_energy"]),
            "energy_delta_vs_ref": 0.0,
            "rel_energy_delta_vs_ref": 0.0,
            "newton_iters": int(fenics_custom["newton_iters"]),
            "linear_iters": int(fenics_custom["linear_iters"]),
            "wall_time_s": float(fenics_custom["solver_wall_time_s"]),
        },
        {
            "implementation": "fenics_snes",
            "result": fenics_snes["result"],
            "final_energy": float(fenics_snes["final_energy"]),
            "energy_delta_vs_ref": float(fenics_snes["final_energy"]) - ref_energy,
            "rel_energy_delta_vs_ref": abs(float(fenics_snes["final_energy"]) - ref_energy) / max(abs(ref_energy), 1.0),
            "newton_iters": int(fenics_snes["newton_iters"]),
            "linear_iters": int(fenics_snes["linear_iters"]),
            "wall_time_s": float(fenics_snes["solver_wall_time_s"]),
        },
        {
            "implementation": "jax_petsc_element",
            "result": jax_petsc["result"],
            "final_energy": float(jax_petsc["final_energy"]),
            "energy_delta_vs_ref": float(jax_petsc["final_energy"]) - ref_energy,
            "rel_energy_delta_vs_ref": abs(float(jax_petsc["final_energy"]) - ref_energy) / max(abs(ref_energy), 1.0),
            "newton_iters": int(jax_petsc["newton_iters"]),
            "linear_iters": int(jax_petsc["linear_iters"]),
            "wall_time_s": float(jax_petsc["solver_wall_time_s"]),
        },
    ]
    local_sfd = next(
        item
        for item in suite_rows
        if item["solver"] == "jax_petsc_local_sfd" and int(item["level"]) == 5 and int(item["nprocs"]) == 1
    )
    parity_rows.append(
        {
            "implementation": "jax_petsc_local_sfd",
            "result": local_sfd["result"],
            "final_energy": float(local_sfd["final_energy"]),
            "energy_delta_vs_ref": float(local_sfd["final_energy"]) - ref_energy,
            "rel_energy_delta_vs_ref": abs(float(local_sfd["final_energy"]) - ref_energy) / max(abs(ref_energy), 1.0),
            "newton_iters": int(local_sfd["total_newton_iters"]),
            "linear_iters": int(local_sfd["total_linear_iters"]),
            "wall_time_s": float(local_sfd["total_time_s"]),
        }
    )
    path = family_dir / "parity_showcase.csv"
    write_csv(
        path,
        parity_rows,
        ["implementation", "result", "final_energy", "energy_delta_vs_ref", "rel_energy_delta_vs_ref", "newton_iters", "linear_iters", "wall_time_s"],
    )
    outputs.append(repo_rel(path))

    strong_scaling = [
        {
            "solver": row["solver"],
            "nprocs": int(row["nprocs"]),
            "total_time_s": float(row["total_time_s"]),
            "final_energy": float(row["final_energy"]),
            "total_newton_iters": int(row["total_newton_iters"]),
            "total_linear_iters": int(row["total_linear_iters"]),
            "result": row["result"],
        }
        for row in suite_rows
        if int(row["level"]) == 9 and row["solver"] in {"fenics_custom", "jax_petsc_element", "jax_petsc_local_sfd"}
    ]
    strong_scaling.sort(key=lambda row: (row["solver"], row["nprocs"]))
    for name in ("suite_scaling_level9.csv", "strong_scaling.csv"):
        path = family_dir / name
        write_csv(path, strong_scaling, ["solver", "nprocs", "total_time_s", "final_energy", "total_newton_iters", "total_linear_iters", "result"])
        outputs.append(repo_rel(path))

    mesh_rows = []
    for row in suite_rows:
        if int(row["nprocs"]) != 32:
            continue
        if row["solver"] not in {"fenics_custom", "jax_petsc_element", "jax_petsc_local_sfd"}:
            continue
        json_path = Path(str(row["json_path"]))
        mesh_rows.append(
            {
                "solver": row["solver"],
                "level": int(row["level"]),
                "problem_size": _problem_size_from_suite_json(json_path),
                "nprocs": int(row["nprocs"]),
                "total_time_s": float(row["total_time_s"]),
                "final_energy": float(row["final_energy"]),
                "total_newton_iters": int(row["total_newton_iters"]),
                "total_linear_iters": int(row["total_linear_iters"]),
                "result": row["result"],
            }
        )
    mesh_rows.sort(key=lambda row: (row["solver"], row["level"]))
    path = family_dir / "mesh_timing.csv"
    write_csv(
        path,
        mesh_rows,
        ["solver", "level", "problem_size", "nprocs", "total_time_s", "final_energy", "total_newton_iters", "total_linear_iters", "result"],
    )
    outputs.append(repo_rel(path))

    _copy(Path(smoke["gl_fenics_custom_l5"]["state_path"]), family_dir / "sample_state.npz")
    outputs.append(repo_rel(family_dir / "sample_state.npz"))

    write_json(
        family_dir / "sources.json",
        {
            "campaign_root": repo_rel(campaign_root),
            "smoke_summary": repo_rel(campaign_root / "runs" / "readme_docs_smoke" / "summary.json"),
            "suite_summary": repo_rel(campaign_root / "runs" / family / "final_suite" / "summary.json"),
            "sample_state_case": smoke["gl_fenics_custom_l5"]["command"],
        },
    )
    outputs.append(repo_rel(family_dir / "sources.json"))
    record_provenance(
        family_dir / "sync_tracked_docs_data.provenance.json",
        script_name="experiments/analysis/docs_assets/sync_tracked_docs_data.py",
        inputs=[
            repo_rel(campaign_root / "runs" / "readme_docs_smoke" / "summary.json"),
            repo_rel(campaign_root / "runs" / family / "final_suite" / "summary.json"),
        ],
        outputs=outputs,
        notes="Tracked GinzburgLandau docs data regenerated from the fresh local canonical campaign.",
    )
    outputs.append(repo_rel(family_dir / "sync_tracked_docs_data.provenance.json"))
    return outputs


def _sync_hyperelasticity(campaign_root: Path, smoke: dict[str, dict[str, Any]]) -> list[str]:
    family = "hyperelasticity"
    suite_rows = _suite_rows(campaign_root / "runs" / family / "final_suite_best" / "summary.json")
    pure_rows = _suite_rows(campaign_root / "runs" / family / "pure_jax_suite_best" / "summary.json")
    family_dir = family_data(f"{family}/.")
    outputs: list[str] = []

    pure_lookup = {
        (int(row["level"]), int(row["total_steps"])): row
        for row in pure_rows
    }
    energy_levels = []
    for level in (1, 2, 3, 4):
        row: dict[str, Any] = {"level": level}
        for solver in ("fenics_custom", "jax_petsc_element"):
            match = next(
                (
                    item
                    for item in suite_rows
                    if item["solver"] == solver and int(item["level"]) == level and int(item["nprocs"]) == 1 and int(item["total_steps"]) == 24
                ),
                None,
            )
            row[solver] = None if match is None else float(match["final_energy"])
        pure_match = pure_lookup.get((level, 24))
        row["jax_serial"] = None if pure_match is None else _pure_jax_final_energy(campaign_root, level, 24)
        energy_levels.append(row)
    path = family_dir / "energy_levels.csv"
    write_csv(path, energy_levels, ["level", "fenics_custom", "jax_petsc_element", "jax_serial"])
    outputs.append(repo_rel(path))

    fenics_custom = next(
        item
        for item in suite_rows
        if item["solver"] == "fenics_custom" and int(item["level"]) == 1 and int(item["nprocs"]) == 1 and int(item["total_steps"]) == 24
    )
    jax_petsc = next(
        item
        for item in suite_rows
        if item["solver"] == "jax_petsc_element" and int(item["level"]) == 1 and int(item["nprocs"]) == 1 and int(item["total_steps"]) == 24
    )
    pure_summary = pure_lookup[(1, 24)]
    pure_payload = _pure_jax_case_payload(campaign_root, 1, 24)
    pure_steps = list(pure_payload.get("steps", []))
    jax_serial = {
        "implementation": "jax_serial",
        "result": str(pure_summary["result"]),
        "completed_steps": len(pure_steps),
        "final_energy": float(pure_steps[-1].get("energy", math.nan)) if pure_steps else math.nan,
        "total_newton_iters": int(pure_summary["total_newton_iters"]),
        "total_linear_iters": int(pure_summary["total_linear_iters"]),
        "wall_time_s": float(pure_summary["time"]),
    }
    ref_energy = float(fenics_custom["final_energy"])
    parity_rows = []
    for row in (fenics_custom, jax_petsc):
        parity_rows.append(
            {
                "implementation": str(row["solver"]),
                "result": row["result"],
                "completed_steps": int(row["completed_steps"]),
                "final_energy": float(row["final_energy"]),
                "energy_delta_vs_ref": float(row["final_energy"]) - ref_energy,
                "rel_energy_delta_vs_ref": abs(float(row["final_energy"]) - ref_energy) / max(abs(ref_energy), 1.0),
                "total_newton_iters": int(row["total_newton_iters"]),
                "total_linear_iters": int(row["total_linear_iters"]),
                "wall_time_s": float(row["total_time_s"]),
            }
        )
    parity_rows.append(
        {
            "implementation": jax_serial["implementation"],
            "result": jax_serial["result"],
            "completed_steps": int(jax_serial["completed_steps"]),
            "final_energy": float(jax_serial["final_energy"]),
            "energy_delta_vs_ref": float(jax_serial["final_energy"]) - ref_energy,
            "rel_energy_delta_vs_ref": abs(float(jax_serial["final_energy"]) - ref_energy) / max(abs(ref_energy), 1.0),
            "total_newton_iters": int(jax_serial["total_newton_iters"]),
            "total_linear_iters": int(jax_serial["total_linear_iters"]),
            "wall_time_s": float(jax_serial["wall_time_s"]),
        }
    )
    path = family_dir / "parity_showcase.csv"
    write_csv(
        path,
        parity_rows,
        ["implementation", "result", "completed_steps", "final_energy", "energy_delta_vs_ref", "rel_energy_delta_vs_ref", "total_newton_iters", "total_linear_iters", "wall_time_s"],
    )
    outputs.append(repo_rel(path))

    strong_scaling = [
        {
            "solver": row["solver"],
            "nprocs": int(row["nprocs"]),
            "total_time_s": float(row["total_time_s"]),
            "final_energy": float(row["final_energy"]),
            "total_newton_iters": int(row["total_newton_iters"]),
            "total_linear_iters": int(row["total_linear_iters"]),
            "result": row["result"],
        }
        for row in suite_rows
        if int(row["total_steps"]) == 24 and int(row["level"]) == 4 and row["solver"] in {"fenics_custom", "jax_petsc_element"}
    ]
    strong_scaling.sort(key=lambda row: (row["solver"], row["nprocs"]))
    for name in ("suite_scaling_level4_steps24.csv", "strong_scaling.csv"):
        path = family_dir / name
        write_csv(path, strong_scaling, ["solver", "nprocs", "total_time_s", "final_energy", "total_newton_iters", "total_linear_iters", "result"])
        outputs.append(repo_rel(path))

    mesh_rows = []
    for row in suite_rows:
        if int(row["total_steps"]) != 24 or int(row["nprocs"]) != 32:
            continue
        if row["solver"] not in {"fenics_custom", "jax_petsc_element"}:
            continue
        json_path = Path(str(row["json_path"]))
        mesh_rows.append(
            {
                "solver": row["solver"],
                "level": int(row["level"]),
                "problem_size": _problem_size_from_suite_json(json_path),
                "nprocs": int(row["nprocs"]),
                "total_time_s": float(row["total_time_s"]),
                "final_energy": float(row["final_energy"]),
                "total_newton_iters": int(row["total_newton_iters"]),
                "total_linear_iters": int(row["total_linear_iters"]),
                "result": row["result"],
            }
        )
    mesh_rows.sort(key=lambda row: (row["solver"], row["level"]))
    path = family_dir / "mesh_timing.csv"
    write_csv(
        path,
        mesh_rows,
        ["solver", "level", "problem_size", "nprocs", "total_time_s", "final_energy", "total_newton_iters", "total_linear_iters", "result"],
    )
    outputs.append(repo_rel(path))

    _copy(Path(smoke["he_jax_petsc_element_l4_np32_showcase"]["state_path"]), family_dir / "sample_state.npz")
    outputs.append(repo_rel(family_dir / "sample_state.npz"))

    write_json(
        family_dir / "sources.json",
        {
            "campaign_root": repo_rel(campaign_root),
            "smoke_summary": repo_rel(campaign_root / "runs" / "readme_docs_smoke" / "summary.json"),
            "suite_summary": repo_rel(campaign_root / "runs" / family / "final_suite_best" / "summary.json"),
            "pure_jax_summary": repo_rel(campaign_root / "runs" / family / "pure_jax_suite_best" / "summary.json"),
            "showcase_command": smoke["he_jax_petsc_element_l4_np32_showcase"]["command"],
        },
    )
    outputs.append(repo_rel(family_dir / "sources.json"))
    record_provenance(
        family_dir / "sync_tracked_docs_data.provenance.json",
        script_name="experiments/analysis/docs_assets/sync_tracked_docs_data.py",
        inputs=[
            repo_rel(campaign_root / "runs" / "readme_docs_smoke" / "summary.json"),
            repo_rel(campaign_root / "runs" / family / "final_suite_best" / "summary.json"),
            repo_rel(campaign_root / "runs" / family / "pure_jax_suite_best" / "summary.json"),
        ],
        outputs=outputs,
        notes="Tracked HyperElasticity docs data regenerated from the fresh local canonical campaign.",
    )
    outputs.append(repo_rel(family_dir / "sync_tracked_docs_data.provenance.json"))
    return outputs


def _sync_topology(campaign_root: Path) -> list[str]:
    family = "topology"
    family_dir = family_data(f"{family}/.")
    topo_root = campaign_root / "runs" / family
    outputs: list[str] = []

    parallel_final_json = topo_root / "parallel_final" / "parallel_full_run.json"
    serial_json = topo_root / "serial_reference" / "report_run.json"
    scaling_rows = _load_topology_scaling_rows(topo_root / "parallel_scaling" / "scaling_summary.csv")
    direct_rows = read_csv_rows(topo_root / "direct_comparison" / "direct_comparison.csv")
    parallel_final = read_json(parallel_final_json)
    serial = read_json(serial_json)
    direct_lookup = {
        (row["implementation"], int(row["mpi_ranks"])): row
        for row in direct_rows
    }

    serial_direct = direct_lookup.get(("jax_serial", 1))
    serial_result = serial["result"] if serial_direct is None else serial_direct["status"]
    serial_outer = int(serial["final_metrics"]["outer_iterations"])
    serial_compliance = float(serial["final_metrics"]["final_compliance"])
    serial_volume = float(serial["final_metrics"]["final_volume_fraction"])
    serial_wall = float(serial["time"])
    if serial_direct is not None:
        serial_compliance = float(serial_direct["median_final_compliance"])
        serial_volume = float(serial_direct["median_final_volume_fraction"])
        serial_wall = float(serial_direct["median_wall_time_s"])

    validated_scaling_rows: list[dict[str, Any]] = []
    for row in scaling_rows:
        updated = dict(row)
        if int(updated["ranks"]) == int(parallel_final["nprocs"]):
            updated["result"] = str(parallel_final["result"])
            updated["outer_iterations"] = int(parallel_final["final_metrics"]["outer_iterations"])
            updated["final_p_penal"] = float(parallel_final["final_metrics"]["final_p_penal"])
            updated["final_volume_fraction"] = float(parallel_final["final_metrics"]["final_volume_fraction"])
            updated["final_compliance"] = float(parallel_final["final_metrics"]["final_compliance"])
            updated["wall_time_s"] = float(parallel_final["time"])
            updated["solve_time_s"] = float(parallel_final["time"] - parallel_final.get("setup_time", 0.0))
        validated_scaling_rows.append(updated)
    if validated_scaling_rows:
        baseline = float(validated_scaling_rows[0]["wall_time_s"])
        for row in validated_scaling_rows:
            row["speedup_vs_1"] = baseline / max(float(row["wall_time_s"]), 1e-12)

    _copy(topo_root / "parallel_final" / "parallel_full_outer_history.csv", family_dir / "objective_history.csv")
    outputs.append(repo_rel(family_dir / "objective_history.csv"))

    resolution_rows = [
        {
            "label": "serial_reference",
            "mesh": "192x96",
            "ranks": 1,
            "result": serial_result,
            "outer_iterations": serial_outer,
            "final_compliance": serial_compliance,
            "final_volume_fraction": serial_volume,
            "wall_time_s": serial_wall,
        },
        {
            "label": "parallel_final",
            "mesh": "768x384",
            "ranks": int(parallel_final["nprocs"]),
            "result": parallel_final["result"],
            "outer_iterations": int(parallel_final["final_metrics"]["outer_iterations"]),
            "final_compliance": float(parallel_final["final_metrics"]["final_compliance"]),
            "final_volume_fraction": float(parallel_final["final_metrics"]["final_volume_fraction"]),
            "wall_time_s": float(parallel_final["time"]),
        },
    ]
    for row in validated_scaling_rows:
        resolution_rows.append(
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
    path = family_dir / "resolution_objectives.csv"
    write_csv(path, resolution_rows, ["label", "mesh", "ranks", "result", "outer_iterations", "final_compliance", "final_volume_fraction", "wall_time_s"])
    outputs.append(repo_rel(path))

    converted_direct = [
        {
            "case_id": row["case_id"],
            "implementation": row["implementation"],
            "mpi_ranks": int(row["mpi_ranks"]),
            "median_wall_time_s": float(row["median_wall_time_s"]),
            "median_final_compliance": float(row["median_final_compliance"]),
            "median_final_volume_fraction": float(row["median_final_volume_fraction"]),
            "status": row["status"],
        }
        for row in direct_rows
    ]
    path = family_dir / "direct_comparison.csv"
    write_csv(path, converted_direct, ["case_id", "implementation", "mpi_ranks", "median_wall_time_s", "median_final_compliance", "median_final_volume_fraction", "status"])
    outputs.append(repo_rel(path))

    path = family_dir / "parallel_scaling.csv"
    write_csv(path, validated_scaling_rows, ["ranks", "result", "outer_iterations", "final_p_penal", "final_volume_fraction", "final_compliance", "wall_time_s", "solve_time_s", "speedup_vs_1"])
    outputs.append(repo_rel(path))

    mesh = parallel_final["mesh"]
    problem_size = int(mesh["displacement_free_dofs"] + mesh["design_free_dofs"])
    strong_scaling = [
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
        for row in validated_scaling_rows
    ]
    path = family_dir / "strong_scaling.csv"
    write_csv(path, strong_scaling, ["solver", "ranks", "problem_size", "wall_time_s", "solve_time_s", "final_p_penal", "final_compliance", "final_volume_fraction", "outer_iterations", "speedup_vs_1", "result"])
    outputs.append(repo_rel(path))

    mesh_rows = [
        {
            "solver": row["solver"],
            "mesh_label": row["mesh_label"],
            "nx": int(row["nx"]),
            "ny": int(row["ny"]),
            "nprocs": int(row["nprocs"]),
            "problem_size": int(row["problem_size"]),
            "wall_time_s": float(row["wall_time_s"]),
            "solve_time_s": float(row["solve_time_s"]),
            "final_compliance": float(row["final_compliance"]),
            "final_volume_fraction": float(row["final_volume_fraction"]),
            "outer_iterations": int(row["outer_iterations"]),
            "result": row["result"],
            "json_path": row["json_path"],
        }
        for row in read_csv_rows(topo_root / "mesh_timing" / "mesh_timing_summary.csv")
    ]
    path = family_dir / "mesh_timing.csv"
    write_csv(path, mesh_rows, ["solver", "mesh_label", "nx", "ny", "nprocs", "problem_size", "wall_time_s", "solve_time_s", "final_compliance", "final_volume_fraction", "outer_iterations", "result", "json_path"])
    outputs.append(repo_rel(path))

    _copy(topo_root / "serial_reference" / "report_state.npz", family_dir / "serial_state.npz")
    _copy(topo_root / "parallel_final" / "parallel_full_state.npz", family_dir / "parallel_final_state.npz")
    outputs.append(repo_rel(family_dir / "serial_state.npz"))
    outputs.append(repo_rel(family_dir / "parallel_final_state.npz"))

    topology_gif = family_gif(f"{family}/topology_parallel_final_evolution.gif")
    _copy(topo_root / "parallel_final" / "density_evolution.gif", topology_gif)
    outputs.append(repo_rel(topology_gif))

    write_json(
        family_dir / "sources.json",
        {
            "campaign_root": repo_rel(campaign_root),
            "serial_report_json": repo_rel(serial_json),
            "parallel_final_json": repo_rel(parallel_final_json),
            "parallel_scaling_csv": repo_rel(topo_root / "parallel_scaling" / "scaling_summary.csv"),
            "direct_comparison_csv": repo_rel(topo_root / "direct_comparison" / "direct_comparison.csv"),
            "mesh_timing_csv": repo_rel(topo_root / "mesh_timing" / "mesh_timing_summary.csv"),
            "serial_reference_wall_time_source": "direct_comparison median (jax_serial, np=1)",
            "rank32_scaling_source": "parallel_final validated benchmark rerun",
            "serial_command": normalize_command((topo_root / "serial_reference" / "command.txt").read_text(encoding="utf-8").strip()),
            "parallel_final_command": normalize_command((topo_root / "parallel_final" / "command.txt").read_text(encoding="utf-8").strip()),
            "parallel_scaling_command": normalize_command((topo_root / "parallel_scaling" / "command.txt").read_text(encoding="utf-8").strip()),
        },
    )
    outputs.append(repo_rel(family_dir / "sources.json"))
    record_provenance(
        family_dir / "sync_tracked_docs_data.provenance.json",
        script_name="experiments/analysis/docs_assets/sync_tracked_docs_data.py",
        inputs=[
            repo_rel(serial_json),
            repo_rel(parallel_final_json),
            repo_rel(topo_root / "parallel_scaling" / "scaling_summary.csv"),
            repo_rel(topo_root / "direct_comparison" / "direct_comparison.csv"),
            repo_rel(topo_root / "mesh_timing" / "mesh_timing_summary.csv"),
        ],
        outputs=outputs,
        notes="Tracked topology docs data regenerated from the fresh local canonical campaign.",
    )
    outputs.append(repo_rel(family_dir / "sync_tracked_docs_data.provenance.json"))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()

    campaign_root = args.campaign_root.resolve()
    smoke = _smoke_rows(campaign_root / "runs" / "readme_docs_smoke" / "summary.json")
    outputs = {
        "plaplace": _sync_plaplace(campaign_root, smoke),
        "ginzburg_landau": _sync_ginzburg_landau(campaign_root, smoke),
        "hyperelasticity": _sync_hyperelasticity(campaign_root, smoke),
        "topology": _sync_topology(campaign_root),
    }
    write_json(campaign_root / "runs" / "docs_assets_sync_summary.json", outputs)
    print(outputs)


if __name__ == "__main__":
    main()
