#!/usr/bin/env python3
"""Execute one validated row of the prepared Karolina revision matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
P3D_CASE_RUNNER = REPO_ROOT / "experiments/runners/run_plasticity3d_backend_mix_case.py"
P3D_FIXED_RUNNER = REPO_ROOT / "experiments/runners/run_plasticity3d_fixed_state_route_screen.py"
P3D_QUADRATURE_RUNNER = REPO_ROOT / "experiments/runners/run_plasticity3d_fixed_state_quadrature.py"
HE_RUNNER = REPO_ROOT / "experiments/runners/run_trust_region_case.py"
ROUTE_FACTOR_RUNNER = REPO_ROOT / "experiments/runners/run_route_factor_microbenchmarks.py"


def load_row(matrix: Path, case_id: str) -> dict[str, str]:
    with matrix.open(newline="", encoding="utf-8") as handle:
        matches = [row for row in csv.DictReader(handle) if row["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one matrix row for {case_id!r}, found {len(matches)}")
    return matches[0]


def cpu_map(count: int) -> str:
    if not 1 <= int(count) <= 128:
        raise ValueError("Karolina ranks_per_node must be in 1..128")
    return ",".join(str(index) for index in range(int(count)))


def srun_prefix(row: dict[str, str], *, tasks: int | None = None) -> list[str]:
    ranks = int(row["total_ranks"] if tasks is None else tasks)
    if tasks is None:
        nodes = int(row["nodes"])
        ranks_per_node = int(row["ranks_per_node"])
    else:
        nodes = 1
        ranks_per_node = int(tasks)
    return [
        "srun",
        "--kill-on-bad-exit=1",
        f"--nodes={nodes}",
        f"--ntasks={ranks}",
        f"--ntasks-per-node={ranks_per_node}",
        "--cpus-per-task=1",
        "--distribution=block:block",
        f"--cpu-bind=map_cpu:{cpu_map(ranks_per_node)}",
        "--mem-bind=local",
    ]


def _route_backend(route: str) -> str:
    return {
        "element_ad": "local",
        "colored_sfd": "local_sfd",
        "constitutive_ad": "local_constitutiveAD",
    }[route]


def p3d_fixed_command(
    row: dict[str, str],
    *,
    python: str,
    run_dir: Path,
    route: str | None = None,
    route_order_position: int = 0,
    use_srun: bool = True,
) -> list[str]:
    selected_route = str(row["route"] if route is None else route)
    return [
        *(srun_prefix(row) if use_srun else []),
        python,
        "-u",
        str(P3D_FIXED_RUNNER),
        "--route",
        selected_route,
        "--tier",
        row["tier"],
        "--mesh-name",
        row["mesh_name"],
        "--element-degree",
        row["element_degree"],
        "--quadrature-rule",
        row["quadrature_rule"],
        "--constraint-variant",
        "glued_bottom",
        "--lambda-target",
        "1.55",
        "--state-label",
        row["state_label"],
        "--state-amplitude",
        row["state_amplitude"],
        "--warmup-repetitions",
        row["warmups"],
        "--measured-repetitions",
        row["repetitions"],
        "--probe-count",
        row.get("probe_count", "1"),
        "--comparison-id",
        row.get("comparison_id", "legacy_unpaired"),
        "--block-repetition",
        row.get("block_repetition", "0"),
        "--route-order-position",
        str(route_order_position),
        "--route-order-policy",
        row.get("route_order_policy", "legacy_unpaired"),
        "--output",
        str(run_dir / "output.json"),
        "--action-out",
        str(run_dir / "tangent_action.npz"),
    ]


def p3d_fixed_block_commands(
    row: dict[str, str], *, python: str, run_dir: Path, use_srun: bool = True
) -> list[tuple[str, list[str]]]:
    route_order = [value.strip() for value in row["route_order"].split("|") if value.strip()]
    if len(route_order) not in {2, 3} or len(set(route_order)) != len(route_order):
        raise ValueError(f"invalid paired route_order {row['route_order']!r}")
    commands: list[tuple[str, list[str]]] = []
    for position, route in enumerate(route_order):
        route_dir = run_dir / route
        command = p3d_fixed_command(
            row,
            python=python,
            run_dir=route_dir,
            route=route,
            route_order_position=position,
            use_srun=use_srun,
        )
        if int(row["total_ranks"]) == 1 and int(row["element_degree"]) == 1:
            command.append("--save-direct-matrix")
        commands.append((route, command))
    return commands


def p3d_solve_command(
    row: dict[str, str], *, python: str, run_dir: Path, route: str | None = None
) -> list[str]:
    selected_route = str(row["route"] if route is None else route)
    command = [
        *srun_prefix(row),
        python,
        "-u",
        str(P3D_CASE_RUNNER),
        "--assembly-backend",
        _route_backend(selected_route),
        "--solver-backend",
        row["solver_backend"],
        "--out-dir",
        str(run_dir),
        "--output-json",
        str(run_dir / "output.json"),
        "--mesh-name",
        row["mesh_name"],
        "--elem-degree",
        row["element_degree"],
        "--quadrature-rule",
        row["quadrature_rule"],
        "--constraint-variant",
        "glued_bottom",
        "--lambda-target",
        "1.55",
        "--pmg-strategy",
        row["pmg_strategy"],
        "--ksp-rtol",
        row["ksp_rtol"],
        "--ksp-max-it",
        row["ksp_max_it"],
        "--convergence-mode",
        "all",
        "--convergence-metric",
        "reference_elastic_energy",
        "--riesz-ksp-type",
        "gmres",
        "--riesz-pc-type",
        "hypre",
        "--riesz-ksp-rtol",
        "1e-10",
        "--riesz-ksp-atol",
        "1e-14",
        "--riesz-ksp-max-it",
        "1000",
        "--riesz-true-residual-rtol",
        "1e-8",
        "--riesz-spd-factor-solver-type",
        "mumps",
        "--riesz-symmetry-tol",
        "1e-12",
        "--stop-tol",
        row["stop_tol"],
        "--grad-stop-tol",
        row["grad_stop_tol"],
        "--maxit",
        row["maxit"],
        "--line-search",
        "armijo",
        "--linesearch-tol",
        "1e-3",
        "--use-trust-region",
        "--trust-subproblem-line-search",
    ]
    if row["tier"] != "smoke" or row["experiment_id"] == "EXP-DISC-001":
        command.extend(["--state-out", str(run_dir / "state.npz")])
    return command


def p3d_solve_block_commands(
    row: dict[str, str], *, python: str, run_dir: Path
) -> list[tuple[str, list[str]]]:
    route_order = [value.strip() for value in row["route_order"].split("|") if value.strip()]
    if set(route_order) != {"element_ad", "constitutive_ad"} or len(route_order) != 2:
        raise ValueError("full-solve paired block requires element/constitutive routes")
    return [
        (
            route,
            p3d_solve_command(
                row, python=python, run_dir=run_dir / route, route=route
            ),
        )
        for route in route_order
    ]


def he_command(row: dict[str, str], *, python: str, run_dir: Path) -> list[str]:
    return [
        *srun_prefix(row),
        python,
        "-u",
        str(HE_RUNNER),
        "--problem",
        "he",
        "--backend",
        "element",
        "--level",
        "5",
        "--out",
        str(run_dir / "output.json"),
        "--state-out",
        str(run_dir / "state.npz"),
        "--steps",
        "1",
        "--start-step",
        "1",
        "--total-steps",
        "24",
        "--profile",
        "performance",
        "--ksp-type",
        "stcg",
        "--pc-type",
        "mg",
        "--ksp-rtol",
        row["ksp_rtol"],
        "--ksp-max-it",
        row["ksp_max_it"],
        "--no-pc-setup-on-ksp-cap",
        "--he-pmg-coarsest-level",
        "3",
        "--he-pmg-smoother-ksp-type",
        "chebyshev",
        "--he-pmg-smoother-pc-type",
        "jacobi",
        "--he-pmg-smoother-steps",
        "2",
        "--he-pmg-coarse-pc-type",
        "hypre",
        "--he-pmg-coarse-hypre-nodal-coarsen",
        "6",
        "--he-pmg-coarse-hypre-vec-interp-variant",
        "3",
        "--he-pmg-coarse-hypre-max-iter",
        "2",
        "--he-pmg-coarse-hypre-tol",
        "0.0",
        "--he-pmg-galerkin",
        "both",
        "--tolf",
        row["stop_tol"],
        "--tolg",
        row["grad_stop_tol"],
        "--tolg-rel",
        "1e-3",
        "--tolx-rel",
        "1e-4",
        "--tolx-abs",
        "1e-10",
        "--maxit",
        row["maxit"],
        "--step-time-limit-s",
        "840",
        "--line-search",
        "armijo",
        "--linesearch-tol",
        "1e-1",
        "--use-trust-region",
        "--trust-subproblem-line-search",
        "--save-history",
        "--save-linear-timing",
        "--quiet",
        "--nproc-threads",
        "1",
        "--element-reorder-mode",
        "block_xyz",
        "--local-hessian-mode",
        "element",
        "--problem-build-mode",
        "rank_local",
        "--he-mesh-source",
        "procedural",
        "--distribution-strategy",
        "overlap_p2p",
        "--assembly-backend",
        "coo_local",
        "--local-coloring",
    ]


def route_factor_command(
    row: dict[str, str], *, python: str, run_dir: Path
) -> list[str]:
    return [
        *srun_prefix(row),
        python,
        "-u",
        str(ROUTE_FACTOR_RUNNER),
        "--output",
        str(run_dir / "output.json"),
        "--repetitions",
        row["repetitions"],
        "--block-repetition",
        row["block_repetition"],
    ]


def build_command(
    row: dict[str, str], *, python: str, run_dir: Path
) -> list[str]:
    if row["runner"] == "p3d_fixed_state":
        return p3d_fixed_command(row, python=python, run_dir=run_dir)
    if row["runner"] == "p3d_fixed_state_block":
        # Preview the first command; execute() runs the complete frozen block.
        return p3d_fixed_block_commands(row, python=python, run_dir=run_dir)[0][1]
    if row["runner"] == "p3d_solve":
        return p3d_solve_command(row, python=python, run_dir=run_dir)
    if row["runner"] == "p3d_solve_block":
        return p3d_solve_block_commands(row, python=python, run_dir=run_dir)[0][1]
    if row["runner"] == "he_first_step":
        return he_command(row, python=python, run_dir=run_dir)
    if row["runner"] == "route_factor_microbench":
        return route_factor_command(row, python=python, run_dir=run_dir)
    raise ValueError(f"unsupported runner {row['runner']!r}")


def quadrature_command(
    row: dict[str, str], *, python: str, run_dir: Path
) -> list[str]:
    return [
        *srun_prefix(row, tasks=1),
        python,
        "-u",
        str(P3D_QUADRATURE_RUNNER),
        "--state",
        str(run_dir / "state.npz"),
        "--output",
        str(run_dir / "quadrature_reference.json"),
        "--quadrature-rules",
        "tetra_24point,tetra_duffy_125point",
        "--action-output-dir",
        str(run_dir / "quadrature_vectors"),
    ]


def _run(command: list[str], *, stdout: Path, stderr: Path, timeout_s: float | None) -> dict[str, object]:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with stdout.open("w", encoding="utf-8") as out_handle, stderr.open(
        "w", encoding="utf-8"
    ) as err_handle:
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=False,
                stdout=out_handle,
                stderr=err_handle,
                text=True,
                timeout=timeout_s,
            )
            returncode = int(completed.returncode)
            timed_out = False
        except subprocess.TimeoutExpired:
            returncode = 124
            timed_out = True
    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_time_s": float(time.perf_counter() - started),
    }


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _relative_vector_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    if reference.shape != candidate.shape:
        return float("inf")
    denominator = max(float(np.linalg.norm(reference)), np.finfo(np.float64).tiny)
    return float(np.linalg.norm(candidate - reference) / denominator)


def _load_fixed_route_output(route_dir: Path) -> dict[str, object]:
    with (route_dir / "output.json").open(encoding="utf-8") as handle:
        payload = json.load(handle)
    with np.load(route_dir / "tangent_action.npz", allow_pickle=False) as archive:
        state = np.asarray(archive["state"], dtype=np.float64)
        actions = np.asarray(
            archive["tangent_actions"] if "tangent_actions" in archive else archive["tangent_action"][None, :],
            dtype=np.float64,
        )
        gradient = np.asarray(archive["gradient"], dtype=np.float64)
    direct = None
    matrix_path = route_dir / "tangent_matrix_csr.npz"
    if matrix_path.is_file():
        with np.load(matrix_path, allow_pickle=False) as archive:
            direct = {
                "indptr": np.asarray(archive["indptr"], dtype=np.int64),
                "indices": np.asarray(archive["indices"], dtype=np.int64),
                "values": np.asarray(archive["values"], dtype=np.float64),
                "shape": np.asarray(archive["shape"], dtype=np.int64),
            }
    return {
        "payload": payload,
        "state": state,
        "actions": actions,
        "gradient": gradient,
        "direct_matrix": direct,
    }


def validate_fixed_state_block(
    row: dict[str, str], route_dirs: dict[str, Path]
) -> dict[str, object]:
    route_order = [value.strip() for value in row["route_order"].split("|") if value.strip()]
    loaded = {route: _load_fixed_route_output(route_dirs[route]) for route in route_order}
    reference_route = "element_ad"
    if reference_route not in loaded:
        raise ValueError("paired fixed-state block lacks element_ad reference")
    reference = loaded[reference_route]
    probe_count = int(row["probe_count"])
    expected_ranks = int(row["total_ranks"])
    direct_required = expected_ranks == 1 and int(row["element_degree"]) == 1
    if reference["actions"].shape[0] != probe_count:
        raise ValueError("saved fixed-state action count differs from matrix")
    comparisons: dict[str, object] = {}
    for route, record in loaded.items():
        payload = dict(record["payload"])
        design = dict(payload.get("comparison_design") or {})
        exact_payload = {
            "experiment_id": row["experiment_id"],
            "tier": row["tier"],
            "mesh_name": row["mesh_name"],
            "element_degree": int(row["element_degree"]),
            "quadrature_rule_id": row["quadrature_rule"],
            "state_label": row["state_label"],
            "state_amplitude": float(row["state_amplitude"]),
            "mpi_ranks": int(row["total_ranks"]),
            "route": route,
            "constraint_variant": "glued_bottom",
            "lambda_target": 1.55,
            "warmup_repetitions": int(row["warmups"]),
            "measured_repetitions": int(row["repetitions"]),
        }
        for key, expected in exact_payload.items():
            if payload.get(key) != expected:
                raise ValueError(f"{route} payload {key} differs from the matrix row")
        if payload.get("wall_time_reduction") != "mpi_collective_max":
            raise ValueError(f"{route} did not report collective-max timing")
        if design.get("comparison_id") != row["comparison_id"]:
            raise ValueError(f"{route} comparison_id mismatch")
        if int(design.get("block_repetition", -1)) != int(row["block_repetition"]):
            raise ValueError(f"{route} block repetition mismatch")
        if design.get("route_order_policy") != row["route_order_policy"]:
            raise ValueError(f"{route} route-order policy mismatch")
        if int(design.get("route_order_position", -1)) != route_order.index(route):
            raise ValueError(f"{route} route-order position mismatch")
        if int(payload.get("probe_count", -1)) != probe_count:
            raise ValueError(f"{route} probe count mismatch")
        state_error = _relative_vector_error(reference["state"], record["state"])
        gradient_error = _relative_vector_error(reference["gradient"], record["gradient"])
        action_errors = [
            _relative_vector_error(reference["actions"][index], record["actions"][index])
            for index in range(probe_count)
        ]
        if not np.array_equal(reference["state"], record["state"]):
            raise ValueError(f"{route} state array is not exactly paired")
        if gradient_error > 1.0e-12:
            raise ValueError(f"{route} gradient/residual mismatch: {gradient_error:.3e}")
        if max(action_errors) > 1.0e-8:
            raise ValueError(f"{route} multi-probe tangent mismatch")
        if payload.get("branch_diagnostics", {}).get("counts") != reference[
            "payload"
        ].get("branch_diagnostics", {}).get("counts"):
            raise ValueError(f"{route} active-branch counts differ")
        direct_error = None
        reference_direct = reference["direct_matrix"]
        candidate_direct = record["direct_matrix"]
        if direct_required and (reference_direct is None or candidate_direct is None):
            raise ValueError("rank-one degree-one blocks require a direct CSR matrix for every route")
        if reference_direct is not None or candidate_direct is not None:
            if reference_direct is None or candidate_direct is None:
                raise ValueError("direct matrix is missing for one route")
            for key in ("indptr", "indices", "shape"):
                if not np.array_equal(reference_direct[key], candidate_direct[key]):
                    raise ValueError(f"{route} direct-matrix {key} mismatch")
            direct_error = _relative_vector_error(
                reference_direct["values"], candidate_direct["values"]
            )
            if direct_error > 1.0e-8:
                raise ValueError(f"{route} direct matrix values mismatch")
        timings = np.asarray(payload.get("wall_times_s", []), dtype=np.float64)
        if timings.size < 5 or not np.all(np.isfinite(timings)) or np.any(timings <= 0.0):
            raise ValueError(f"{route} collective timing samples are invalid")
        raw_rank_timings = payload.get("wall_times_by_rank_s")
        if not isinstance(raw_rank_timings, list) or len(raw_rank_timings) != timings.size:
            raise ValueError(f"{route} lacks one raw rank-timing row per sample")
        rank_timings = np.asarray(raw_rank_timings, dtype=np.float64)
        if rank_timings.shape != (timings.size, expected_ranks):
            raise ValueError(f"{route} raw rank-timing shape disagrees with the matrix")
        if not np.all(np.isfinite(rank_timings)) or np.any(rank_timings <= 0.0):
            raise ValueError(f"{route} raw rank timings are invalid")
        if not np.allclose(
            timings,
            np.max(rank_timings, axis=1),
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            raise ValueError(f"{route} MPI_MAX samples disagree with raw rank timings")
        comparisons[route] = {
            "state_exact": True,
            "state_relative_error": state_error,
            "gradient_residual_relative_error": gradient_error,
            "action_relative_errors": action_errors,
            "direct_matrix_relative_error": direct_error,
            "collective_max_wall_time_s": float(np.median(timings)),
            "collective_max_wall_times_s": timings.tolist(),
            "wall_times_by_rank_s": rank_timings.tolist(),
            "timing_rank_count": expected_ranks,
            "timing_provenance": "rank_allgather_then_MPI_MAX",
        }
    return {
        "status": "admitted_correctness_block",
        "comparison_id": row["comparison_id"],
        "block_repetition": int(row["block_repetition"]),
        "route_order": route_order,
        "route_order_policy": row["route_order_policy"],
        "timing_reduction": "mpi_collective_max",
        "probe_count": probe_count,
        "routes": comparisons,
        "timing_claim_released": False,
    }


def execute_fixed_state_block(
    row: dict[str, str], *, out_root: Path, python: str
) -> list[dict[str, object]]:
    case_root = out_root / "cases" / row["case_id"] / f"job_{os.environ.get('SLURM_JOB_ID', 'local')}"
    run_dir = case_root / "measure_01"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(case_root / "matrix_row.json", row)
    records: list[dict[str, object]] = []
    route_dirs: dict[str, Path] = {}
    for route, command in p3d_fixed_block_commands(row, python=python, run_dir=run_dir):
        route_dir = run_dir / route
        route_dir.mkdir(parents=True, exist_ok=True)
        route_dirs[route] = route_dir
        (route_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
        result = _run(
            command,
            stdout=route_dir / "stdout.txt",
            stderr=route_dir / "stderr.txt",
            timeout_s=None,
        )
        records.append({"route": route, "command": shlex.join(command), **result})
        _write_json(case_root / "run_records.json", records)
    if all(int(record["returncode"]) == 0 for record in records):
        try:
            block_result = validate_fixed_state_block(row, route_dirs)
            block_result["job_metadata"] = {
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
                "slurm_cluster_name": os.environ.get("SLURM_CLUSTER_NAME", ""),
            }
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            block_result = {
                "status": "invalid",
                "comparison_id": row["comparison_id"],
                "block_repetition": int(row["block_repetition"]),
                "route_order": row["route_order"].split("|"),
                "timing_reduction": "mpi_collective_max",
                "reason": str(exc),
                "timing_claim_released": False,
            }
            records.append({"route": "block_validation", "returncode": 86, "reason": str(exc)})
        _write_json(run_dir / "block_result.json", block_result)
        _write_json(case_root / "run_records.json", records)
    else:
        _write_json(
            run_dir / "block_result.json",
            {
                "status": "censored",
                "comparison_id": row["comparison_id"],
                "block_repetition": int(row["block_repetition"]),
                "route_order": row["route_order"].split("|"),
                "timing_reduction": "mpi_collective_max",
                "reason": "one_or_more_route_processes_failed",
                "timing_claim_released": False,
            },
        )
    return records


def execute_p3d_solve_block(
    row: dict[str, str], *, out_root: Path, python: str
) -> list[dict[str, object]]:
    case_root = out_root / "cases" / row["case_id"] / f"job_{os.environ.get('SLURM_JOB_ID', 'local')}"
    run_dir = case_root / "measure_01"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(case_root / "matrix_row.json", row)
    records: list[dict[str, object]] = []
    route_payloads: dict[str, dict[str, object]] = {}
    for route, command in p3d_solve_block_commands(row, python=python, run_dir=run_dir):
        route_dir = run_dir / route
        route_dir.mkdir(parents=True, exist_ok=True)
        (route_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
        result = _run(
            command,
            stdout=route_dir / "stdout.txt",
            stderr=route_dir / "stderr.txt",
            timeout_s=None,
        )
        record: dict[str, object] = {"route": route, "command": shlex.join(command), **result}
        if int(result["returncode"]) == 0:
            try:
                with (route_dir / "output.json").open(encoding="utf-8") as handle:
                    payload = json.load(handle)
                if not isinstance(payload, dict):
                    raise ValueError("route output must be a JSON object")
                record["scientific_validation"] = validate_p3d_solve_output(payload, row)
                route_payloads[route] = payload
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                record["process_returncode"] = int(record["returncode"])
                record["returncode"] = 86
                record["scientific_validation"] = {"status": "failed", "reason": str(exc)}
        records.append(record)
        _write_json(case_root / "run_records.json", records)
    route_order = row["route_order"].split("|")
    if len(route_payloads) == len(route_order) and all(
        int(record["returncode"]) == 0 for record in records
    ):
        route_rows: dict[str, object] = {}
        for route in route_order:
            payload = route_payloads[route]
            if payload.get("total_time_reduction") != "mpi_collective_max":
                records.append(
                    {
                        "route": "block_validation",
                        "returncode": 86,
                        "reason": f"{route} lacks collective-max total timing",
                    }
                )
                break
            route_rows[route] = {
                "output_json": str(Path(route) / "output.json"),
                "state_npz": str(Path(route) / "state.npz"),
                "collective_max_wall_time_s": float(payload["total_time"]),
                "per_rank_wall_times_s": list(payload["total_time_by_rank_s"]),
                "timing_rank_count": len(payload["total_time_by_rank_s"]),
                "timing_provenance": "solver_allgather_then_MPI_MAX",
                "status": str(payload.get("status", "")),
                "solver_success": bool(payload.get("solver_success")),
            }
        else:
            _write_json(
                run_dir / "block_result.json",
                {
                    "schema_version": 1,
                    "experiment_id": "EXP-ROUTE-001",
                    "tier": row["tier"],
                    "status": "routes_completed_pending_endpoint_analysis",
                    "comparison_id": row["comparison_id"],
                    "block_repetition": int(row["block_repetition"]),
                    "route_order": route_order,
                    "route_order_policy": row["route_order_policy"],
                    "timing_reduction": "mpi_collective_max",
                    "job_metadata": {
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
                        "slurm_cluster_name": os.environ.get("SLURM_CLUSTER_NAME", ""),
                    },
                    "routes": route_rows,
                    "timing_claim_released": False,
                },
            )
        _write_json(case_root / "run_records.json", records)
    if not (run_dir / "block_result.json").exists():
        _write_json(
            run_dir / "block_result.json",
            {
                "schema_version": 1,
                "experiment_id": "EXP-ROUTE-001",
                "tier": row["tier"],
                "status": "censored_or_invalid",
                "comparison_id": row["comparison_id"],
                "block_repetition": int(row["block_repetition"]),
                "route_order": route_order,
                "route_order_policy": row["route_order_policy"],
                "timing_reduction": "mpi_collective_max",
                "routes": {},
                "timing_claim_released": False,
            },
        )
    return records


def _required_finite(value: object, *, field: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{field} must be a finite number")
    return converted


def validate_p3d_solve_output(
    payload: dict[str, object], row: dict[str, str]
) -> dict[str, object]:
    """Fail closed on the Riesz stopping evidence required by all P3D jobs."""

    if row.get("runner") not in {"p3d_solve", "p3d_solve_block"}:
        raise ValueError("Riesz output validation is only defined for P3D solve rows")
    if row.get("convergence_metric") != "reference_elastic_energy":
        raise ValueError("matrix row does not request reference_elastic_energy")
    if payload.get("convergence_metric_requested") != "reference_elastic_energy":
        raise ValueError("root output did not request reference_elastic_energy")
    if payload.get("convergence_metric") != "reference_elastic_energy":
        raise ValueError("root output did not use reference_elastic_energy effectively")
    convergence = dict(payload.get("nonlinear_convergence") or {})
    configuration = dict(convergence.get("configuration") or {})
    if configuration.get("selection") != "reference_elastic_energy":
        raise ValueError("output selected coefficient or unknown convergence stopping")
    if configuration.get("correction_normalization") != "metric_current_state":
        raise ValueError("output did not use metric-current-state correction normalization")

    metric = dict(convergence.get("metric") or {})
    provenance = dict(metric.get("provenance") or {})
    certificate = dict(provenance.get("spd_certificate") or {})
    inertia = dict(certificate.get("inertia") or {})
    if certificate.get("certified_spd") is not True:
        raise ValueError("reference elastic metric lacks a positive-definite certificate")
    negative = int(inertia.get("negative", -1))
    zero = int(inertia.get("zero", -1))
    positive = int(inertia.get("positive", -1))
    free_dofs = int(provenance.get("free_dofs", -1))
    owned_sum = int(dict(payload.get("parallel_setup") or {}).get("owned_free_dofs_sum", -1))
    if negative != 0 or zero != 0 or positive <= 0:
        raise ValueError("reference elastic metric inertia is not strictly positive")
    if positive != free_dofs or positive != owned_sum:
        raise ValueError(
            "positive inertia must equal provenance and distributed free-DOF counts"
        )
    if str(certificate.get("factor_solver_type", "")).lower() != "mumps":
        raise ValueError("SPD certification did not use the frozen MUMPS factor route")

    requested = dict(payload.get("riesz_solver_requested") or {})
    exact_requested = {
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "rtol": 1.0e-10,
        "atol": 1.0e-14,
        "max_it": 1000,
        "true_residual_rtol": 1.0e-8,
        "spd_factor_solver_type": "mumps",
        "symmetry_relative_tolerance": 1.0e-12,
    }
    for key, expected in exact_requested.items():
        actual = requested.get(key)
        if isinstance(expected, float):
            actual = _required_finite(actual, field=f"requested Riesz {key}")
            if not math.isclose(actual, expected, rel_tol=1.0e-14, abs_tol=0.0):
                raise ValueError(f"requested Riesz {key} changed from the frozen value")
        elif actual != expected:
            raise ValueError(f"requested Riesz {key} changed from the frozen value")
    metric_expected = {
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "requested_rtol": 1.0e-10,
        "requested_atol": 1.0e-14,
        "requested_max_it": 1000,
        "effective_rtol": 1.0e-10,
        "effective_atol": 1.0e-14,
        "effective_max_it": 1000,
        "true_residual_rtol_gate": 1.0e-8,
    }
    for key, expected in metric_expected.items():
        actual = metric.get(key)
        if isinstance(expected, float):
            actual = _required_finite(actual, field=f"metric Riesz {key}")
            if not math.isclose(actual, expected, rel_tol=1.0e-14, abs_tol=0.0):
                raise ValueError(f"effective metric Riesz {key} is not frozen")
        elif actual != expected:
            raise ValueError(f"effective metric Riesz {key} is not frozen")

    initial = _required_finite(
        dict(convergence.get("initial_absolute_dual_residual") or {}).get("value"),
        field="initial absolute dual residual",
    )
    terminal = _required_finite(
        dict(convergence.get("absolute_dual_residual") or {}).get("value"),
        field="terminal absolute dual residual",
    )
    if initial < 0.0 or terminal < 0.0:
        raise ValueError("dual residual norms cannot be negative")
    state_norm = _required_finite(
        dict(convergence.get("state_norm") or {}).get("value"),
        field="terminal state norm",
    )
    relative_correction = _required_finite(
        dict(convergence.get("relative_correction") or {}).get("value"),
        field="terminal relative correction",
    )
    coefficient_gradient = _required_finite(
        convergence.get("coefficient_gradient_l2"),
        field="terminal coefficient-gradient L2 norm",
    )
    root_gradient = _required_finite(
        payload.get("final_grad_norm"), field="root final coefficient-gradient norm"
    )
    if min(state_norm, relative_correction, coefficient_gradient, root_gradient) < 0.0:
        raise ValueError("terminal state/correction/gradient diagnostics cannot be negative")
    last_riesz = dict(convergence.get("last_riesz_solve") or {})
    if last_riesz.get("riesz_solve") != "iterative":
        raise ValueError("terminal Riesz norm must come from the iterative audited solve")
    reason = int(last_riesz.get("reason", 0))
    relative_true = _required_finite(
        last_riesz.get("relative_true_residual"), field="Riesz relative true residual"
    )
    true_gate = _required_finite(
        last_riesz.get("true_residual_rtol_gate"), field="Riesz true-residual gate"
    )
    if reason <= 0 or relative_true > true_gate:
        raise ValueError("terminal Riesz solve failed its convergence/true-residual gate")
    last_expected = {
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "requested_rtol": 1.0e-10,
        "requested_atol": 1.0e-14,
        "requested_max_it": 1000,
        "effective_rtol": 1.0e-10,
        "effective_atol": 1.0e-14,
        "effective_max_it": 1000,
        "true_residual_rtol_gate": 1.0e-8,
    }
    for key, expected in last_expected.items():
        actual = last_riesz.get(key)
        if isinstance(expected, float):
            actual = _required_finite(actual, field=f"endpoint Riesz {key}")
            if not math.isclose(actual, expected, rel_tol=1.0e-14, abs_tol=0.0):
                raise ValueError(f"endpoint Riesz {key} is not frozen")
        elif actual != expected:
            raise ValueError(f"endpoint Riesz {key} is not frozen")
    rhs_norm = _required_finite(last_riesz.get("rhs_norm"), field="endpoint Riesz rhs norm")
    if rhs_norm < 0.0 or not math.isclose(
        rhs_norm, coefficient_gradient, rel_tol=1.0e-12, abs_tol=1.0e-14
    ) or not math.isclose(
        rhs_norm, root_gradient, rel_tol=1.0e-12, abs_tol=1.0e-14
    ):
        raise ValueError(
            "endpoint Riesz rhs norm is stale or disagrees with coefficient-gradient diagnostics"
        )
    residual_gate = dict(convergence.get("residual_gate") or {})
    if (
        bool(payload.get("solver_success")) or payload.get("status") == "completed"
    ) and residual_gate.get("passed") is not True:
        raise ValueError("a completed solver row did not pass the Riesz residual gate")
    return {
        "status": "passed",
        "selection": "reference_elastic_energy",
        "positive_inertia": positive,
        "initial_absolute_dual_residual": initial,
        "terminal_absolute_dual_residual": terminal,
        "terminal_state_norm": state_norm,
        "terminal_relative_correction": relative_correction,
        "terminal_coefficient_gradient_l2": coefficient_gradient,
        "terminal_riesz_reason": reason,
        "terminal_relative_true_residual": relative_true,
        "terminal_true_residual_gate": true_gate,
        "solver_residual_gate_passed": bool(residual_gate.get("passed")),
    }


def execute(row: dict[str, str], *, out_root: Path, python: str) -> list[dict[str, object]]:
    if row["runner"] == "p3d_fixed_state_block":
        return execute_fixed_state_block(row, out_root=out_root, python=python)
    if row["runner"] == "p3d_solve_block":
        return execute_p3d_solve_block(row, out_root=out_root, python=python)
    case_root = out_root / "cases" / row["case_id"] / f"job_{os.environ.get('SLURM_JOB_ID', 'local')}"
    case_root.mkdir(parents=True, exist_ok=True)
    _write_json(case_root / "matrix_row.json", row)

    if row["runner"] in {"p3d_fixed_state", "route_factor_microbench"}:
        schedule = [("measure", 1)]
    else:
        schedule = [
            *(('warmup', index + 1) for index in range(int(row["warmups"]))),
            *(('measure', index + 1) for index in range(int(row["repetitions"]))),
        ]
    records: list[dict[str, object]] = []
    for kind, index in schedule:
        run_dir = case_root / f"{kind}_{index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(row, python=python, run_dir=run_dir)
        if row["tier"] == "optional_fixed_policy_p3d":
            command = [
                "timeout",
                "--signal=TERM",
                "--kill-after=30s",
                "30m",
                *command,
            ]
        (run_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
        record = {
            "kind": kind,
            "index": index,
            "command": shlex.join(command),
            **_run(
                command,
                stdout=run_dir / "stdout.txt",
                stderr=run_dir / "stderr.txt",
                timeout_s=None,
            ),
        }
        if row["runner"] == "p3d_solve" and int(record["returncode"]) == 0:
            output_path = run_dir / "output.json"
            try:
                with output_path.open(encoding="utf-8") as handle:
                    payload = json.load(
                        handle,
                        parse_constant=lambda token: (_ for _ in ()).throw(
                            ValueError(f"nonfinite JSON token {token!r}")
                        ),
                    )
                if not isinstance(payload, dict):
                    raise ValueError("p3d_solve output must be a JSON object")
                record["scientific_validation"] = validate_p3d_solve_output(payload, row)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                record["process_returncode"] = int(record["returncode"])
                record["returncode"] = 86
                record["scientific_validation"] = {
                    "status": "failed",
                    "reason": str(exc),
                }
        if row["experiment_id"] == "EXP-DISC-001" and record["returncode"] == 0:
            state_path = run_dir / "state.npz"
            if not state_path.is_file():
                record["process_returncode"] = int(record["returncode"])
                record["returncode"] = 86
                record["quadrature_reference_validation"] = {
                    "status": "failed",
                    "reason": "mandatory saved state is missing",
                }
            else:
                reference = quadrature_command(row, python=python, run_dir=run_dir)
                (run_dir / "quadrature_command.txt").write_text(
                    shlex.join(reference) + "\n", encoding="utf-8"
                )
                reference_result = _run(
                    reference,
                    stdout=run_dir / "quadrature_stdout.txt",
                    stderr=run_dir / "quadrature_stderr.txt",
                    timeout_s=None,
                )
                record["quadrature_reference"] = reference_result
                reference_output = run_dir / "quadrature_reference.json"
                reference_failure: str | None = None
                if int(reference_result["returncode"]) != 0:
                    reference_failure = (
                        "mandatory quadrature-reference evaluator returned "
                        f"{int(reference_result['returncode'])}"
                    )
                elif not reference_output.is_file():
                    reference_failure = (
                        "mandatory quadrature-reference evaluator did not write its output"
                    )
                else:
                    try:
                        with reference_output.open(encoding="utf-8") as handle:
                            reference_payload = json.load(
                                handle,
                                parse_constant=lambda token: (_ for _ in ()).throw(
                                    ValueError(f"nonfinite JSON token {token!r}")
                                ),
                            )
                        if not isinstance(reference_payload, dict):
                            raise ValueError("quadrature-reference output is not an object")
                        if reference_payload.get("status") != "completed":
                            raise ValueError(
                                "quadrature-reference output does not report completed status"
                            )
                    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        reference_failure = str(exc)
                if reference_failure is None:
                    record["quadrature_reference_validation"] = {"status": "passed"}
                else:
                    record["process_returncode"] = int(record["returncode"])
                    record["returncode"] = 86
                    record["quadrature_reference_validation"] = {
                        "status": "failed",
                        "reason": reference_failure,
                    }
        records.append(record)
        _write_json(case_root / "run_records.json", records)
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--python", default=os.environ.get("PYTHON", "./.venv/bin/python"))
    parser.add_argument("--print-command", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    row = load_row(Path(args.matrix).resolve(), str(args.case_id))
    if args.print_command:
        print(
            shlex.join(
                build_command(
                    row,
                    python=str(args.python),
                    run_dir=Path(args.out_root).resolve() / "command_preview",
                )
            )
        )
        return
    records = execute(
        row,
        out_root=Path(args.out_root).resolve(),
        python=str(args.python),
    )
    if any(int(record["returncode"]) != 0 for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
