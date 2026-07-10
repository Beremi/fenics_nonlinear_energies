#!/usr/bin/env python3
"""Run the fixed-state HyperElasticity rank-count gate from EXP-DIST-001.

The controller holds mesh source, construction, distribution, local assembly,
state, direction, and linear solver fixed while changing only the MPI rank
count over the prescribed one-, two-, and four-rank levels.  It is a
correctness experiment with descriptive phase timings; it is not a scaling
benchmark and it does not exercise the nonlinear solved-endpoint gate from
EXP-DIST-001.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import jax
import jaxlib
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import scipy
from scipy import sparse

from src.core.benchmark.run_record import (
    RUN_RECORD_SCHEMA_ID,
    RUN_RECORD_SCHEMA_VERSION,
    atomic_write_json,
    atomic_write_run_record,
    check_experiment_preflight,
    sha256_file,
    utc_now_iso,
)
from src.problems.hyperelasticity.jax_petsc.reordered_element_assembler import (
    HEReorderedElementAssembler,
)
from src.problems.hyperelasticity.support.mesh import (
    build_procedural_hyperelasticity_export_params,
    expand_tet_connectivity_to_dofs,
    generate_structured_element_data_for_indices,
    generate_structured_elements_for_indices,
    load_rank_local_hyperelasticity,
    local_dirichlet_values_from_reference,
    total_dofs_to_reordered_free,
)
from src.problems.hyperelasticity.support.rotate_boundary import (
    rotate_right_face_from_reference,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "EXP-DIST-001"
CAMPAIGN_ID = "paper_revision_2026_07_10"
CONTRACT_ID = "EXP-DIST-001-fixed-state-he-v1"
RANK_COUNTS = (1, 2, 4)
REFERENCE_RANK_COUNT = 1
PUBLICATION_CONFIGURATION = {
    "level": 1,
    "angle": 0.15,
    "repetitions": 3,
    "ksp_rtol": 1.0e-12,
    "linear_residual_tolerance": 1.0e-10,
    "derivative_tolerance": 1.0e-8,
    "solve_tolerance": 1.0e-8,
    "residual_scale_floor": 1.0,
}
FACTOR_CONFIGURATION = {
    "problem": "hyperelasticity",
    "mesh_source": "procedural",
    "problem_build_mode": "rank_local",
    "distribution_strategy": "overlap_p2p",
    "assembly_backend": "coo_local",
    "local_hessian_mode": "element",
    "element_reorder_mode": "block_xyz",
    "element_degree": 1,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "factor_solver_type": "mumps",
    "use_near_nullspace": False,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _artifact_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dirty_snapshot_sha256() -> str | None:
    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    if not status:
        return None
    digest = hashlib.sha256()
    digest.update(
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "diff", "--binary", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
    )
    untracked = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    for raw in sorted(part for part in untracked.split(b"\0") if part):
        digest.update(b"untracked\0")
        digest.update(raw)
        path = REPO_ROOT / os.fsdecode(raw)
        if path.is_file():
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def _cpu_model() -> str:
    path = Path("/proc/cpuinfo")
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "local CPU; model unavailable"


def _memory_model() -> str:
    path = Path("/proc/meminfo")
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                return line.replace("MemTotal:", "host MemTotal", 1).strip()
    return "local host memory; capacity unavailable"


def _relative_error(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(1.0, np.linalg.norm(a), np.linalg.norm(b)))


def _scalar_relative_error(left: float, right: float) -> float:
    return float(abs(float(left) - float(right)) / max(1.0, abs(float(left)), abs(float(right))))


def _critical(comm: MPI.Comm, value: float) -> float:
    return float(comm.allreduce(float(value), op=MPI.MAX))


def _gather_owned_vector(comm: MPI.Comm, vector: PETSc.Vec) -> np.ndarray | None:
    pieces = comm.gather(np.asarray(vector.array[:], dtype=np.float64).copy(), root=0)
    if comm.rank != 0:
        return None
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float64)


def _gather_matrix(comm: MPI.Comm, matrix: PETSc.Mat) -> sparse.csr_matrix | None:
    lo, hi = matrix.getOwnershipRange()
    indptr, indices, values = matrix.getValuesCSR()
    local = (
        int(lo),
        int(hi),
        np.asarray(indptr, dtype=np.int64),
        np.asarray(indices, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
    )
    parts = comm.gather(local, root=0)
    if comm.rank != 0:
        return None
    n_rows, n_cols = matrix.getSize()
    global_indptr = np.zeros(int(n_rows) + 1, dtype=np.int64)
    global_indices: list[np.ndarray] = []
    global_values: list[np.ndarray] = []
    offset = 0
    for part_lo, part_hi, part_indptr, part_indices, part_values in sorted(parts):
        if int(part_hi - part_lo) != int(part_indptr.size - 1):
            raise RuntimeError("PETSc local CSR row count does not match ownership range")
        counts = np.diff(part_indptr)
        global_indptr[part_lo + 1 : part_hi + 1] = offset + np.cumsum(counts)
        offset += int(part_values.size)
        global_indices.append(part_indices)
        global_values.append(part_values)
    result = sparse.csr_matrix(
        (
            np.concatenate(global_values),
            np.concatenate(global_indices),
            global_indptr,
        ),
        shape=(int(n_rows), int(n_cols)),
    )
    result.sum_duplicates()
    result.sort_indices()
    return result


def _canonical_twist_state(params: Mapping[str, Any], angle: float) -> np.ndarray:
    coordinates = np.asarray(params["_distributed_owned_block_coordinates"], dtype=np.float64)
    grid = params["_he_grid"]
    theta = float(angle) * coordinates[:, 0] / float(grid.x_max)
    cosine = np.cos(theta)
    sine = np.sin(theta)
    state = coordinates.copy()
    state[:, 1] = cosine * coordinates[:, 1] + sine * coordinates[:, 2]
    state[:, 2] = -sine * coordinates[:, 1] + cosine * coordinates[:, 2]
    return state.ravel()


def _canonical_direction(lo: int, hi: int) -> np.ndarray:
    index = np.arange(int(lo), int(hi), dtype=np.float64) + 1.0
    return np.sin(0.013 * index) + 0.25 * np.cos(0.031 * index)


def _validate_local_mesh(params: Mapping[str, Any]) -> dict[str, Any]:
    elem_idx = np.asarray(params["_distributed_local_elem_idx"], dtype=np.int64)
    grid = params["_he_grid"]
    expected_elems = generate_structured_elements_for_indices(elem_idx, grid)
    expected_dx, expected_dy, expected_dz, expected_vol = (
        generate_structured_element_data_for_indices(elem_idx, grid)
    )
    local_nodes = np.asarray(params["_distributed_local_total_nodes"], dtype=np.int64)
    expected_map = total_dofs_to_reordered_free(local_nodes, grid, "block_xyz", 1)
    elems_scalar = np.asarray(params["_distributed_local_elems_scalar_np"], dtype=np.int64)
    scalar_nodes, inverse = np.unique(expected_elems.ravel(), return_inverse=True)
    expected_local_scalar = inverse.reshape(expected_elems.shape)
    expected_total = expand_tet_connectivity_to_dofs(expected_elems)
    tests = {
        "connectivity": bool(
            np.array_equal(
                expected_elems,
                np.asarray(params["_distributed_local_elems_total"], dtype=np.int64)[
                    :, ::3
                ]
                // 3,
            )
        ),
        "expanded_connectivity": bool(
            np.array_equal(
                expected_total,
                np.asarray(params["_distributed_local_elems_total"], dtype=np.int64),
            )
        ),
        "local_scalar_numbering": bool(
            np.array_equal(expected_local_scalar, elems_scalar)
            and scalar_nodes.size * 3 == local_nodes.size
        ),
        "dphix": bool(
            np.allclose(expected_dx, params["_distributed_dphix"], rtol=0.0, atol=0.0)
        ),
        "dphiy": bool(
            np.allclose(expected_dy, params["_distributed_dphiy"], rtol=0.0, atol=0.0)
        ),
        "dphiz": bool(
            np.allclose(expected_dz, params["_distributed_dphiz"], rtol=0.0, atol=0.0)
        ),
        "volume": bool(
            np.allclose(expected_vol, params["_distributed_vol"], rtol=0.0, atol=0.0)
        ),
        "free_dof_map": bool(
            np.array_equal(
                expected_map,
                np.asarray(params["_distributed_local_total_to_free_reord"], dtype=np.int64),
            )
        ),
    }
    return {"passed": bool(all(tests.values())), "checks": tests}


def _worker_payload(args: argparse.Namespace) -> None:
    comm = MPI.COMM_WORLD
    rank = int(comm.rank)
    size = int(comm.size)
    started_at = _utc_now()
    total_start = time.perf_counter()

    comm.Barrier()
    construction_start = time.perf_counter()
    params, adjacency, _ = load_rank_local_hyperelasticity(
        int(args.level),
        comm=comm,
        reorder_mode="block_xyz",
        mesh_source="procedural",
        element_degree=1,
    )
    construction_s = _critical(comm, time.perf_counter() - construction_start)
    local_mesh_validation = _validate_local_mesh(params)

    comm.Barrier()
    setup_start = time.perf_counter()
    assembler = HEReorderedElementAssembler(
        params=params,
        comm=comm,
        adjacency=adjacency,
        ksp_rtol=float(args.ksp_rtol),
        ksp_type="preonly",
        pc_type="lu",
        ksp_max_it=1,
        use_near_nullspace=False,
        reorder_mode="block_xyz",
        local_hessian_mode="element",
        distribution_strategy="overlap_p2p",
        assembly_backend="coo_local",
    )
    setup_s = _critical(comm, time.perf_counter() - setup_start)
    assembler.update_dirichlet(local_dirichlet_values_from_reference(params, float(args.angle)))

    lo, hi = int(assembler.layout.lo), int(assembler.layout.hi)
    state_owned = _canonical_twist_state(params, float(args.angle))
    direction_owned = _canonical_direction(lo, hi)
    state = assembler.create_vec(state_owned)
    direction = assembler.create_vec(direction_owned)
    residual = state.duplicate()
    action = state.duplicate()

    repetitions: list[dict[str, float | int]] = []
    energy_value = 0.0
    try:
        for repetition in range(1, int(args.repetitions) + 1):
            comm.Barrier()
            callback_before = assembler.callback_summary()

            start = time.perf_counter()
            energy_value = float(assembler.energy_fn(state))
            energy_s = _critical(comm, time.perf_counter() - start)

            start = time.perf_counter()
            assembler.gradient_fn(state, residual)
            gradient_s = _critical(comm, time.perf_counter() - start)

            start = time.perf_counter()
            hessian_detail = assembler.assemble_hessian(state.array[:])
            assembly_s = _critical(comm, time.perf_counter() - start)

            start = time.perf_counter()
            assembler.A.mult(direction, action)
            action_s = _critical(comm, time.perf_counter() - start)

            callback_after = assembler.callback_summary()
            energy_comm_local = (
                float(callback_after["energy"]["ghost_exchange"])
                - float(callback_before["energy"]["ghost_exchange"])
                + float(callback_after["energy"]["allreduce"])
                - float(callback_before["energy"]["allreduce"])
            )
            gradient_comm_local = (
                float(callback_after["gradient"]["ghost_exchange"])
                - float(callback_before["gradient"]["ghost_exchange"])
            )
            assembly_comm_local = float(hessian_detail.get("ghost_exchange", 0.0))
            repetitions.append(
                {
                    "repetition": repetition,
                    "energy_s": energy_s,
                    "gradient_s": gradient_s,
                    "assembly_s": assembly_s,
                    "matrix_action_s": action_s,
                    "instrumented_communication_s": _critical(
                        comm,
                        energy_comm_local + gradient_comm_local + assembly_comm_local,
                    ),
                    "energy_ghost_exchange_s": _critical(
                        comm,
                        float(callback_after["energy"]["ghost_exchange"])
                        - float(callback_before["energy"]["ghost_exchange"]),
                    ),
                    "energy_allreduce_s": _critical(
                        comm,
                        float(callback_after["energy"]["allreduce"])
                        - float(callback_before["energy"]["allreduce"]),
                    ),
                    "gradient_ghost_exchange_s": _critical(comm, gradient_comm_local),
                    "assembly_ghost_exchange_s": _critical(comm, assembly_comm_local),
                    "assembly_element_hessian_s": _critical(
                        comm, float(hessian_detail.get("elem_hessian_compute", 0.0))
                    ),
                    "assembly_scatter_s": _critical(
                        comm, float(hessian_detail.get("scatter", 0.0))
                    ),
                    "assembly_coo_insert_s": _critical(
                        comm, float(hessian_detail.get("coo_assembly", 0.0))
                    ),
                }
            )

        ksp = assembler.ksp
        ksp.setOperators(assembler.A)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        ksp.setTolerances(rtol=float(args.ksp_rtol), max_it=1)

        rhs = residual.duplicate()
        residual.copy(rhs)
        rhs.scale(-1.0)
        correction = state.duplicate()
        true_residual = state.duplicate()
        try:
            comm.Barrier()
            start = time.perf_counter()
            ksp.setUp()
            pc_setup_s = _critical(comm, time.perf_counter() - start)
            comm.Barrier()
            start = time.perf_counter()
            ksp.solve(rhs, correction)
            linear_solve_s = _critical(comm, time.perf_counter() - start)
            assembler.A.mult(correction, true_residual)
            true_residual.axpy(-1.0, rhs)
            rhs_norm = float(rhs.norm())
            true_residual_norm = float(true_residual.norm())
            relative_true_residual = float(
                true_residual_norm / max(float(args.residual_scale_floor), rhs_norm)
            )
            ksp_reason = int(ksp.getConvergedReason())
            ksp_iterations = int(ksp.getIterationNumber())

            full_state = _gather_owned_vector(comm, state)
            full_direction = _gather_owned_vector(comm, direction)
            full_residual = _gather_owned_vector(comm, residual)
            full_action = _gather_owned_vector(comm, action)
            full_correction = _gather_owned_vector(comm, correction)
            full_matrix = _gather_matrix(comm, assembler.A)
        finally:
            true_residual.destroy()
            correction.destroy()
            rhs.destroy()

        setup_by_rank = comm.gather(assembler.setup_summary(), root=0)
        memory_by_rank = comm.gather(assembler.memory_summary(), root=0)
        validation_by_rank = comm.gather(local_mesh_validation, root=0)
        ownership_by_rank = comm.gather(
            {
                "rank": rank,
                "owned_lo": lo,
                "owned_hi": hi,
                "owned_dofs": hi - lo,
                "local_element_count": int(
                    np.asarray(params["_distributed_local_elem_idx"]).size
                ),
                "owned_element_count": int(
                    np.sum(np.asarray(params["_distributed_energy_weights"]))
                ),
                "local_overlap_dofs": int(
                    np.asarray(params["_distributed_local_total_nodes"]).size
                ),
                "ghost_receive_peers": int(len(assembler._ghost_recv)),
                "ghost_send_peers": int(len(assembler._ghost_send_offsets)),
                "local_element_index_sha256": _array_sha256(
                    np.asarray(params["_distributed_local_elem_idx"], dtype=np.int64)
                ),
                "local_overlap_dof_sha256": _array_sha256(
                    np.asarray(params["_distributed_local_total_nodes"], dtype=np.int64)
                ),
            },
            root=0,
        )
        local_mesh_parts = comm.gather(
            {
                "indices": np.asarray(
                    params["_distributed_local_elem_idx"], dtype=np.int64
                ),
                "weights": np.asarray(
                    params["_distributed_energy_weights"], dtype=np.float64
                ),
            },
            root=0,
        )
        rss_mib = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
        rss_by_rank = comm.gather(rss_mib, root=0)
        total_s_critical = _critical(comm, time.perf_counter() - total_start)

        if rank == 0:
            assert full_state is not None
            assert full_direction is not None
            assert full_residual is not None
            assert full_action is not None
            assert full_correction is not None
            assert full_matrix is not None
            export = build_procedural_hyperelasticity_export_params(int(args.level))
            coordinates = np.asarray(export["nodes2coord"], dtype=np.float64)
            connectivity = np.asarray(export["elems_scalar"], dtype=np.int64)
            freedofs = np.asarray(export["freedofs"], dtype=np.int64)
            right_nodes = np.asarray(export["right_nodes"], dtype=np.int64)
            affine_lift = rotate_right_face_from_reference(
                np.asarray(export["u_0_ref"], dtype=np.float64),
                coordinates,
                float(args.angle),
                right_nodes,
            )
            ownership_count = np.zeros(connectivity.shape[0], dtype=np.float64)
            coverage_count = np.zeros(connectivity.shape[0], dtype=np.int64)
            for part in local_mesh_parts:
                np.add.at(ownership_count, part["indices"], part["weights"])
                np.add.at(coverage_count, part["indices"], 1)
            mesh_gate = bool(
                all(bool(row["passed"]) for row in validation_by_rank)
                and np.array_equal(ownership_count, np.ones_like(ownership_count))
                and np.all(coverage_count >= 1)
            )
            topology_hashes = {
                "coordinates": _array_sha256(coordinates),
                "connectivity": _array_sha256(connectivity),
                "freedofs": _array_sha256(freedofs),
                "right_boundary_nodes": _array_sha256(right_nodes),
                "affine_lift": _array_sha256(affine_lift),
            }
            state_hashes = {
                "state": _array_sha256(full_state),
                "direction": _array_sha256(full_direction),
                "matrix_structure": _mapping_sha256(
                    {
                        "indptr": _array_sha256(full_matrix.indptr.astype(np.int64)),
                        "indices": _array_sha256(full_matrix.indices.astype(np.int64)),
                    }
                ),
            }
            output_npz = Path(args.worker_npz).resolve()
            output_npz.parent.mkdir(parents=True, exist_ok=True)
            state_output_start = time.perf_counter()
            np.savez_compressed(
                output_npz,
                state=full_state,
                direction=full_direction,
                residual=full_residual,
                matrix_action=full_action,
                correction=full_correction,
                matrix_indptr=full_matrix.indptr.astype(np.int64),
                matrix_indices=full_matrix.indices.astype(np.int64),
                matrix_data=full_matrix.data.astype(np.float64),
            )
            state_output_s = float(time.perf_counter() - state_output_start)
            median_timing = {
                key: float(np.median([float(row[key]) for row in repetitions]))
                for key in repetitions[0]
                if key != "repetition"
            }
            payload = {
                "schema": {
                    "id": "fenics-nonlinear-energies.exp-dist-he-worker",
                    "version": 1,
                },
                "experiment": EXPERIMENT_ID,
                "run_kind": str(args.run_kind),
                "status": "passed"
                if mesh_gate
                and ksp_reason > 0
                and relative_true_residual <= float(args.linear_residual_tolerance)
                else "failed",
                "rank_count": size,
                "configuration": {
                    **FACTOR_CONFIGURATION,
                    "mesh_level": int(args.level),
                    "canonical_twist_angle_rad": float(args.angle),
                    "repetitions": int(args.repetitions),
                    "ksp_rtol": float(args.ksp_rtol),
                    "linear_residual_tolerance": float(args.linear_residual_tolerance),
                    "residual_scale_floor": float(args.residual_scale_floor),
                },
                "problem": {
                    "total_dofs": int(coordinates.size),
                    "free_dofs": int(full_state.size),
                    "nodes": int(coordinates.shape[0]),
                    "elements": int(connectivity.shape[0]),
                },
                "mesh_semantics": {
                    "passed": mesh_gate,
                    "local_validation_by_rank": validation_by_rank,
                    "ownership_exactly_once": bool(
                        np.array_equal(ownership_count, np.ones_like(ownership_count))
                    ),
                    "overlap_covers_every_element": bool(np.all(coverage_count >= 1)),
                    "minimum_element_overlap_count": int(np.min(coverage_count)),
                    "maximum_element_overlap_count": int(np.max(coverage_count)),
                    "topology_hashes": topology_hashes,
                },
                "algebraic_objects": {
                    "energy": energy_value,
                    "residual_norm": float(np.linalg.norm(full_residual)),
                    "matrix_frobenius_norm": float(sparse.linalg.norm(full_matrix)),
                    "matrix_nnz": int(full_matrix.nnz),
                    "matrix_action_norm": float(np.linalg.norm(full_action)),
                    "correction_norm": float(np.linalg.norm(full_correction)),
                    "state_hashes": state_hashes,
                },
                "linear_solve": {
                    "ksp_converged_reason": ksp_reason,
                    "ksp_iterations": ksp_iterations,
                    "rhs_norm": rhs_norm,
                    "true_residual_norm": true_residual_norm,
                    "relative_true_residual": relative_true_residual,
                    "gate_passed": bool(
                        ksp_reason > 0
                        and relative_true_residual
                        <= float(args.linear_residual_tolerance)
                    ),
                },
                "timing": {
                    "mesh_construction_critical_s": construction_s,
                    "assembler_setup_critical_s": setup_s,
                    "preconditioner_setup_critical_s": pc_setup_s,
                    "linear_solve_critical_s": linear_solve_s,
                    "distributed_compute_critical_s": total_s_critical,
                    "state_output_s": state_output_s,
                    "repetitions": repetitions,
                    "median_critical_s": median_timing,
                    "interpretation": (
                        "descriptive local correctness timing only; MPI ranks share one "
                        "workstation and PETSc matvec/factorization communication is "
                        "embedded in phase totals"
                    ),
                },
                "ownership_by_rank": ownership_by_rank,
                "assembler_setup_by_rank": setup_by_rank,
                "assembler_memory_by_rank": memory_by_rank,
                "resources": {
                    "rank_ru_maxrss_mib": [float(value) for value in rss_by_rank],
                    "ru_maxrss_mib_max": float(max(rss_by_rank)),
                    "ru_maxrss_mib_total": float(sum(rss_by_rank)),
                },
                "artifacts": {"arrays_npz": _artifact_label(output_npz)},
                "started_at_utc": started_at,
                "finished_at_utc": _utc_now(),
                "total_s": float(time.perf_counter() - total_start),
            }
            atomic_write_json(Path(args.worker_json), payload)
    finally:
        action.destroy()
        residual.destroy()
        direction.destroy()
        state.destroy()
        assembler.cleanup()


def compare_worker_outputs(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    reference_arrays: Mapping[str, np.ndarray],
    candidate_arrays: Mapping[str, np.ndarray],
    *,
    derivative_tolerance: float,
    solve_tolerance: float,
) -> dict[str, Any]:
    exact_topology = {
        key: reference["mesh_semantics"]["topology_hashes"][key]
        == candidate["mesh_semantics"]["topology_hashes"][key]
        for key in reference["mesh_semantics"]["topology_hashes"]
    }
    exact_objects = {
        "state": bool(np.array_equal(reference_arrays["state"], candidate_arrays["state"])),
        "direction": bool(
            np.array_equal(reference_arrays["direction"], candidate_arrays["direction"])
        ),
        "matrix_indptr": bool(
            np.array_equal(
                reference_arrays["matrix_indptr"], candidate_arrays["matrix_indptr"]
            )
        ),
        "matrix_indices": bool(
            np.array_equal(
                reference_arrays["matrix_indices"], candidate_arrays["matrix_indices"]
            )
        ),
    }
    errors = {
        "energy_relative": _scalar_relative_error(
            reference["algebraic_objects"]["energy"],
            candidate["algebraic_objects"]["energy"],
        ),
        "residual_relative": _relative_error(
            reference_arrays["residual"], candidate_arrays["residual"]
        ),
        "matrix_relative": _relative_error(
            reference_arrays["matrix_data"], candidate_arrays["matrix_data"]
        )
        if exact_objects["matrix_indptr"] and exact_objects["matrix_indices"]
        else sys.float_info.max,
        "matrix_action_relative": _relative_error(
            reference_arrays["matrix_action"], candidate_arrays["matrix_action"]
        ),
        "linear_correction_relative": _relative_error(
            reference_arrays["correction"], candidate_arrays["correction"]
        ),
    }
    derivative_gates = {
        key: bool(value <= float(derivative_tolerance))
        for key, value in errors.items()
        if key != "linear_correction_relative"
    }
    solve_gates = {
        "linear_correction": bool(
            errors["linear_correction_relative"] <= float(solve_tolerance)
        ),
        "reference_true_residual": bool(reference["linear_solve"]["gate_passed"]),
        "candidate_true_residual": bool(candidate["linear_solve"]["gate_passed"]),
    }
    gate = bool(
        reference["status"] == "passed"
        and candidate["status"] == "passed"
        and all(exact_topology.values())
        and all(exact_objects.values())
        and all(derivative_gates.values())
        and all(solve_gates.values())
    )
    return {
        "algebraic_gate_passed": gate,
        "exact_topology_gates": exact_topology,
        "exact_object_gates": exact_objects,
        "relative_errors": errors,
        "derivative_gates": derivative_gates,
        "linear_solve_gates": solve_gates,
        "derivative_tolerance": float(derivative_tolerance),
        "solve_tolerance": float(solve_tolerance),
    }


def aggregate_rank_comparisons(
    comparisons: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine the prescribed np1-reference comparisons without hiding a failure."""
    expected = {f"np{rank}" for rank in RANK_COUNTS if rank != REFERENCE_RANK_COUNT}
    if set(comparisons) != expected:
        raise ValueError(
            "rank comparisons must contain exactly "
            f"{sorted(expected)}, got {sorted(comparisons)}"
        )
    ordered = [
        comparisons[f"np{rank}"]
        for rank in RANK_COUNTS
        if rank != REFERENCE_RANK_COUNT
    ]
    first = ordered[0]
    derivative_tolerance = float(first["derivative_tolerance"])
    solve_tolerance = float(first["solve_tolerance"])
    for comparison in ordered[1:]:
        if float(comparison["derivative_tolerance"]) != derivative_tolerance:
            raise ValueError("rank comparisons use different derivative tolerances")
        if float(comparison["solve_tolerance"]) != solve_tolerance:
            raise ValueError("rank comparisons use different solve tolerances")

    def _all_boolean_block(name: str) -> dict[str, bool]:
        keys = set(first[name])
        if any(set(comparison[name]) != keys for comparison in ordered[1:]):
            raise ValueError(f"rank comparisons expose inconsistent {name} keys")
        return {
            key: bool(all(bool(comparison[name][key]) for comparison in ordered))
            for key in sorted(keys)
        }

    error_keys = set(first["relative_errors"])
    if any(set(comparison["relative_errors"]) != error_keys for comparison in ordered[1:]):
        raise ValueError("rank comparisons expose inconsistent relative-error keys")
    return {
        "algebraic_gate_passed": bool(
            all(bool(comparison["algebraic_gate_passed"]) for comparison in ordered)
        ),
        "exact_topology_gates": _all_boolean_block("exact_topology_gates"),
        "exact_object_gates": _all_boolean_block("exact_object_gates"),
        "relative_errors": {
            key: float(max(float(comparison["relative_errors"][key]) for comparison in ordered))
            for key in sorted(error_keys)
        },
        "derivative_gates": _all_boolean_block("derivative_gates"),
        "linear_solve_gates": _all_boolean_block("linear_solve_gates"),
        "derivative_tolerance": derivative_tolerance,
        "solve_tolerance": solve_tolerance,
        "aggregation": "maximum error and conjunction of np1-vs-np2 and np1-vs-np4 gates",
    }


def build_worker_command(
    *,
    ranks: int,
    output_json: Path,
    output_npz: Path,
    level: int,
    angle: float,
    repetitions: int,
    ksp_rtol: float,
    linear_residual_tolerance: float,
    residual_scale_floor: float = 1.0,
    run_kind: str = "pilot",
) -> list[str]:
    return [
        "mpiexec",
        "--oversubscribe",
        "-n",
        str(int(ranks)),
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-kind",
        str(run_kind),
        "--worker",
        "--worker-json",
        str(output_json),
        "--worker-npz",
        str(output_npz),
        "--level",
        str(int(level)),
        "--angle",
        str(float(angle)),
        "--repetitions",
        str(int(repetitions)),
        "--ksp-rtol",
        str(float(ksp_rtol)),
        "--linear-residual-tolerance",
        str(float(linear_residual_tolerance)),
        "--residual-scale-floor",
        str(float(residual_scale_floor)),
    ]


def _run_worker(command: Sequence[str], log_path: Path, timeout_s: float) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
        }
    )
    started = time.perf_counter()
    process = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=float(timeout_s),
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(process.stdout or "", encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(
            f"MPI worker failed with exit code {process.returncode} after "
            f"{time.perf_counter() - started:.1f}s; see {log_path}"
        )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _median_phase(worker: Mapping[str, Any], phase: str) -> float:
    return float(worker["timing"]["median_critical_s"][phase])


def _phase_ratios(workers: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    if set(workers) != set(RANK_COUNTS):
        raise ValueError(f"phase timings require workers for ranks {list(RANK_COUNTS)}")

    def _value(worker: Mapping[str, Any], phase: str) -> float:
        direct = {
            "mesh_construction": "mesh_construction_critical_s",
            "assembler_setup": "assembler_setup_critical_s",
            "preconditioner_setup": "preconditioner_setup_critical_s",
            "linear_solve": "linear_solve_critical_s",
        }
        if phase in direct:
            return float(worker["timing"][direct[phase]])
        return _median_phase(worker, f"{phase}_s")

    result: dict[str, Any] = {}
    for phase in (
        "mesh_construction",
        "assembler_setup",
        "energy",
        "gradient",
        "assembly",
        "matrix_action",
        "instrumented_communication",
        "preconditioner_setup",
        "linear_solve",
    ):
        reference = _value(workers[REFERENCE_RANK_COUNT], phase)
        row: dict[str, float | None] = {"np1_s": reference}
        for rank in RANK_COUNTS[1:]:
            value = _value(workers[rank], phase)
            row[f"np{rank}_s"] = value
            row[f"np{rank}_over_np1"] = float(value / reference) if reference > 0.0 else None
        result[phase] = row
    return result


def _render_report(payload: Mapping[str, Any]) -> str:
    comparison = payload["comparison"]
    publication = payload.get("run_kind") == "publication"
    evidence_text = (
        "This clean managed execution is eligible for publication finalization as a "
        "fixed-state correctness result. Its timings remain descriptive and are not "
        "publication performance evidence."
        if publication
        else
        "This local pilot is not publication evidence. Its timings are descriptive "
        "diagnostics only."
    )
    lines = [
        "# EXP-DIST-001 Hyperelasticity Fixed-State Rank-Count Gate",
        "",
        f"Status: **{payload['status']}**.",
        "",
        "## Controlled design",
        "",
        "The one-, two-, and four-rank runs use the same procedural P1 mesh definition, "
        "rank-local construction, block-XYZ canonical ordering, point-to-point overlap "
        "exchange, owned-row local COO assembly, twist state, deterministic direction, "
        "and MUMPS LU solve. Only the MPI ownership partition changes.",
        "",
        evidence_text,
        "All rank counts share one workstation, and communication inside PETSc "
        "matrix-vector products and MUMPS is embedded rather than separately attributed.",
        "",
        "## Algebraic gate",
        "",
        f"- Overall fixed-state gate: `{comparison['algebraic_gate_passed']}`.",
        f"- Derivative tolerance: `{comparison['derivative_tolerance']:.1e}`.",
        f"- Linear-correction tolerance: `{comparison['solve_tolerance']:.1e}`.",
        "- The coordinate, connectivity, boundary, constrained/free-map, affine-lift, "
        "canonical-state, canonical-direction, and matrix-pattern checks use exact "
        "hashes or exact integer/FP64 arrays.",
        "",
        "| Comparison | Quantity | Relative error | Gate |",
        "| --- | --- | ---: | :---: |",
    ]
    for candidate in ("np2", "np4"):
        rank_comparison = payload["rank_comparisons"][candidate]
        for key, value in rank_comparison["relative_errors"].items():
            gate = (
                rank_comparison["linear_solve_gates"]["linear_correction"]
                if key == "linear_correction_relative"
                else rank_comparison["derivative_gates"][key]
            )
            lines.append(
                f"| np1 vs {candidate} | {key.replace('_', ' ')} | "
                f"{float(value):.3e} | {gate} |"
            )
    lines.extend(
        [
            "",
            "The fixed-state Newton correction is included as a linear algebra gate. "
            "It is not a converged nonlinear endpoint; the EXP-DIST solved-endpoint gate "
            "remains outstanding.",
            "",
            "## Descriptive phase timings",
            "",
            "| Phase | np1 (s) | np2 (s) | np2/np1 | np4 (s) | np4/np1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for phase, row in payload["descriptive_phase_timings"].items():
        ratio2 = row["np2_over_np1"]
        ratio4 = row["np4_over_np1"]
        ratio2_text = "n/a" if ratio2 is None else f"{float(ratio2):.3f}"
        ratio4_text = "n/a" if ratio4 is None else f"{float(ratio4):.3f}"
        lines.append(
            f"| {phase.replace('_', ' ')} | {float(row['np1_s']):.6f} | "
            f"{float(row['np2_s']):.6f} | {ratio2_text} | "
            f"{float(row['np4_s']):.6f} | {ratio4_text} |"
        )
    lines.extend(
        [
            "",
            "`instrumented communication` includes explicit overlap exchanges and the "
            "energy reduction only. It is a lower-bound attribution, not total MPI cost.",
            "",
            "## Remaining EXP-DIST work",
            "",
            "1. Independently factor HDF5/procedural source, "
            "replicated/rank-local construction, all-gather/P2P exchange, and "
            "global/local COO assembly one factor at a time.",
            "2. Run the calibrated nonlinear solved-endpoint gate and compare weighted "
            "states, independent residuals, energy, and physical observables.",
            "3. Only after those gates pass may timing or memory advantages be interpreted.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_run_record(
    *,
    worker: Mapping[str, Any],
    comparison: Mapping[str, Any],
    rank_count: int,
    preflight: Any,
    dirty_patch_sha256: str | None,
    command: Sequence[str],
    raw_json: Path,
    arrays_npz: Path,
    log_path: Path,
    report_path: Path,
    code_hashes: Mapping[str, str],
) -> dict[str, Any]:
    gate = bool(comparison["algebraic_gate_passed"])
    phase = worker["timing"]["median_critical_s"]
    memory = worker["resources"]
    configuration_hash = _mapping_sha256(
        {
            "worker_configuration": dict(worker["configuration"]),
            "rank_counts": list(RANK_COUNTS),
            "reference_rank_count": REFERENCE_RANK_COUNT,
            "derivative_tolerance": float(comparison["derivative_tolerance"]),
            "solve_tolerance": float(comparison["solve_tolerance"]),
        }
    )
    status = "success" if gate else "failure"
    correction_error = (
        0.0
        if int(rank_count) == 1
        else float(comparison["relative_errors"]["linear_correction_relative"])
    )
    jit_warmup_s = max(
        float(row.get("warmup", 0.0))
        for row in worker["assembler_setup_by_rank"]
    )
    return {
        "schema": {"id": RUN_RECORD_SCHEMA_ID, "version": RUN_RECORD_SCHEMA_VERSION},
        "record_id": f"paper-revision-2026-07-10-exp-dist-001-he-np{rank_count}-r01",
        "run_kind": str(preflight.run_kind),
        "identifiers": {
            "campaign": CAMPAIGN_ID,
            "experiment": EXPERIMENT_ID,
            "case": f"hyperelasticity-p1-l{worker['configuration']['mesh_level']}-np{rank_count}",
            "method": "fixed-state-distributed-equivalence",
            "route": "rank-local-procedural-p2p-local-coo",
            "repetition": 1,
        },
        "problem": {
            "name": "three-dimensional compressible hyperelasticity",
            "mesh": f"procedural structured level {worker['configuration']['mesh_level']}",
            "degree": 1,
            "quadrature": "P1 tetrahedral exact element rule",
            "total_degrees_of_freedom": int(worker["problem"]["total_dofs"]),
            "free_degrees_of_freedom": int(worker["problem"]["free_dofs"]),
            "notes": "Canonical twist state; fixed-state algebraic gate only.",
        },
        "solver": {
            "algorithm": "fixed-state residual, tangent, action, and Newton correction",
            "implementation": _artifact_label(Path(__file__)),
            "parameters": dict(worker["configuration"]),
            "preconditioner": {"type": "MUMPS distributed LU"},
            "stopping_contract": CONTRACT_ID,
        },
        "termination": {
            "status": status,
            "reason": (
                "one-, two-, and four-rank fixed-state algebraic gates passed"
                if gate
                else "one or more prescribed rank-count algebraic gates failed"
            ),
            "exit_code": 0 if gate else 2,
            "started_at_utc": str(worker["started_at_utc"]),
            "finished_at_utc": str(worker["finished_at_utc"]),
            "limit_kind": None,
            "limit_value": None,
            "censored": False,
        },
        "accuracy": {
            "contract_id": CONTRACT_ID,
            "gate_passed": gate,
            "absolute_residual": float(worker["linear_solve"]["true_residual_norm"]),
            "relative_residual": float(worker["linear_solve"]["relative_true_residual"]),
            "scaled_residual": None,
            "relative_correction": correction_error,
            "energy_change": None,
            "custom_metrics": dict(comparison["relative_errors"]),
            "notes": "Residual is the independently recomputed linear-system residual; no nonlinear endpoint was solved.",
        },
        "counts": {
            "nonlinear_iterations": 0,
            "krylov_iterations": int(worker["linear_solve"]["ksp_iterations"]),
            "function_evaluations": int(worker["configuration"]["repetitions"]),
            "gradient_evaluations": int(worker["configuration"]["repetitions"]),
            "hessian_evaluations": int(worker["configuration"]["repetitions"]),
            "hvp_evaluations": int(worker["configuration"]["repetitions"]),
            "preconditioner_setups": 1,
            "notes": "Counts exclude constructor JIT warm-up and include fixed-state measured repetitions.",
        },
        "timing": {
            "aggregation": "maximum rank wall clock per phase; median across fixed-state repetitions",
            "cold_process": True,
            "barrier_policy": "MPI barrier before construction, setup, each repetition, factor setup, and solve",
            "synchronization_policy": "JAX values block before timing ends; collective energy and PETSc operations synchronize as required",
            "phases_overlap": False,
            "relation_to_total": "phase times are non-overlapping within a repetition but do not sum to the worker total through NPZ output",
            "process_startup_s": None,
            "jit_compilation_s": jit_warmup_s,
            "coloring_s": None,
            "derivative_evaluation_s": float(phase["energy_s"]) + float(phase["gradient_s"]),
            "constitutive_contraction_s": float(phase["assembly_element_hessian_s"]),
            "assembly_s": float(phase["assembly_s"]),
            "communication_s": float(phase["instrumented_communication_s"]),
            "preconditioner_setup_s": float(worker["timing"]["preconditioner_setup_critical_s"]),
            "krylov_solve_s": float(worker["timing"]["linear_solve_critical_s"]),
            "globalization_s": None,
            "state_output_s": float(worker["timing"]["state_output_s"]),
            "total_s": float(worker["total_s"]),
            "notes": (
                "Descriptive local correctness timing, not performance evidence; "
                "communication omits MPI embedded in PETSc matvec and MUMPS, and "
                "total excludes final JSON/log serialization."
            ),
            "mesh_construction_s": float(worker["timing"]["mesh_construction_critical_s"]),
            "matrix_action_s": float(phase["matrix_action_s"]),
        },
        "resources": {
            "nodes": 1,
            "ranks": int(rank_count),
            "threads_per_rank": 1,
            "peak_memory_per_rank_bytes": int(float(memory["ru_maxrss_mib_max"]) * 1024**2),
            "peak_memory_per_node_bytes": int(float(memory["ru_maxrss_mib_total"]) * 1024**2),
            "tracked_allocations_bytes": None,
            "measurement_method": "per-rank getrusage(RUSAGE_SELF).ru_maxrss gathered by MPI",
            "notes": "All ranks share one local workstation; values are diagnostic, not scaling evidence.",
        },
        "diagnostics": {
            "state": {
                "hashes": dict(worker["algebraic_objects"]["state_hashes"]),
                "topology_hashes": dict(worker["mesh_semantics"]["topology_hashes"]),
                "twist_angle_rad": float(worker["configuration"]["canonical_twist_angle_rad"]),
            },
            "branch": {},
            "feasibility": {"mesh_semantics_passed": bool(worker["mesh_semantics"]["passed"])},
            "kkt": {},
        },
        "environment": {
            "python": sys.version.split()[0],
            "packages": {
                "jax": jax.__version__,
                "jaxlib": jaxlib.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "platform": platform.platform(),
            "jax": jax.__version__,
            "xla": jaxlib.__version__,
            "jax_enable_x64": bool(jax.config.x64_enabled),
            "petsc": ".".join(str(value) for value in PETSc.Sys.getVersion()),
            "mpi": MPI.Get_library_version().strip().replace("\n", " "),
            "compiler": "prebuilt Python/JAX/PETSc runtimes; compiler not separately captured",
            "blas": "NumPy linked BLAS; provider not separately captured",
            "cpu_model": _cpu_model(),
            "node_model": platform.node() or "local host",
            "memory_model": _memory_model(),
            "scheduler": "local",
            "scheduler_job_id": None,
            "affinity": "OMP/BLAS/JAX CPU threads fixed to one; mpiexec --oversubscribe",
        },
        "provenance": {
            **preflight.provenance_fields(),
            "command_argv": [str(item) for item in command],
            "working_directory": str(REPO_ROOT),
            "code_hashes": dict(code_hashes),
            "configuration_hashes": {CONTRACT_ID: configuration_hash},
            "input_hashes": {},
            "dirty_patch_sha256": dirty_patch_sha256,
            "seed": None,
            "deterministic_policy": "Closed-form mesh, twist state, direction by global canonical DOF index, fixed rank counts and FP64.",
            "recorded_at_utc": utc_now_iso(),
            "preflight_checked_at_utc": str(preflight.checked_at_utc),
        },
        "artifacts": {
            "raw_outputs": [_artifact_label(raw_json)],
            "states": [_artifact_label(arrays_npz)],
            "logs": [_artifact_label(log_path)],
            "tables": [],
            "figures": [],
            "reports": [_artifact_label(report_path)],
        },
    }


def _controller(args: argparse.Namespace) -> None:
    preflight = check_experiment_preflight(
        REPO_ROOT,
        run_kind=str(args.run_kind),
        pilot_dirty_override=bool(args.pilot_dirty_override),
        pilot_override_reason=args.pilot_override_reason,
    )
    dirty_digest = _dirty_snapshot_sha256()
    root = Path(args.output_dir).resolve()
    if str(args.run_kind) == "publication" and root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"publication output directory must be fresh and empty: {root}"
        )
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    commands: dict[int, list[str]] = {}
    paths: dict[int, dict[str, Path]] = {}
    for ranks in RANK_COUNTS:
        paths[ranks] = {
            "json": raw_dir / f"hyperelasticity_np{ranks}.json",
            "npz": raw_dir / f"hyperelasticity_np{ranks}_arrays.npz",
            "log": raw_dir / f"hyperelasticity_np{ranks}.log",
        }
        commands[ranks] = build_worker_command(
            ranks=ranks,
            output_json=paths[ranks]["json"],
            output_npz=paths[ranks]["npz"],
            level=int(args.level),
            angle=float(args.angle),
            repetitions=int(args.repetitions),
            ksp_rtol=float(args.ksp_rtol),
            linear_residual_tolerance=float(args.linear_residual_tolerance),
            residual_scale_floor=float(args.residual_scale_floor),
            run_kind=str(args.run_kind),
        )
        _run_worker(commands[ranks], paths[ranks]["log"], float(args.timeout_s))

    workers = {rank: _load_json(paths[rank]["json"]) for rank in RANK_COUNTS}
    arrays = {rank: _load_npz(paths[rank]["npz"]) for rank in RANK_COUNTS}
    rank_comparisons = {
        f"np{rank}": compare_worker_outputs(
            workers[REFERENCE_RANK_COUNT],
            workers[rank],
            arrays[REFERENCE_RANK_COUNT],
            arrays[rank],
            derivative_tolerance=float(args.derivative_tolerance),
            solve_tolerance=float(args.solve_tolerance),
        )
        for rank in RANK_COUNTS
        if rank != REFERENCE_RANK_COUNT
    }
    comparison = aggregate_rank_comparisons(rank_comparisons)
    payload = {
        "schema": {
            "id": "fenics-nonlinear-energies.exp-dist-he-comparison",
            "version": 1,
        },
        "experiment": EXPERIMENT_ID,
        "run_kind": str(args.run_kind),
        "publication_evidence": False,
        "status": "passed" if comparison["algebraic_gate_passed"] else "failed",
        "controlled_factors": FACTOR_CONFIGURATION,
        "varied_factor": {"name": "mpi_ranks", "levels": list(RANK_COUNTS)},
        "comparison": comparison,
        "rank_comparisons": rank_comparisons,
        "descriptive_phase_timings": _phase_ratios(workers),
        "nonlinear_solved_endpoint_gate": {
            "status": "not_run",
            "reason": "stopping calibration remains a prerequisite",
        },
        "timing_claim_admissible": False,
        "timing_claim_blockers": [
            "one local workstation with ranks sharing hardware",
            "correctness repetitions are not a performance sampling design",
            "PETSc and MUMPS communication is embedded in phase totals",
            "nonlinear solved-endpoint gate not run",
            "mesh source, construction, distribution, and assembly variants not factorized",
        ],
        "workers": {
            f"np{rank}": {
                "raw_json": _artifact_label(paths[rank]["json"]),
                "arrays_npz": _artifact_label(paths[rank]["npz"]),
                "log": _artifact_label(paths[rank]["log"]),
                "status": workers[rank]["status"],
            }
            for rank in RANK_COUNTS
        },
    }
    comparison_path = root / "distribution_equivalence.json"
    atomic_write_json(comparison_path, payload)
    report_path = root / "pilot_report.md"
    report_path.write_text(_render_report(payload), encoding="utf-8")

    protocol_path = REPO_ROOT / "paper" / "protocols" / "EXP-DIST-001.md"
    source_paths = [
        Path(__file__).resolve(),
        protocol_path,
        REPO_ROOT
        / "src/problems/hyperelasticity/jax_petsc/reordered_element_assembler.py",
        REPO_ROOT / "src/problems/hyperelasticity/support/mesh.py",
        REPO_ROOT / "src/core/petsc/reordered_element_base.py",
    ]
    code_hashes = {
        _artifact_label(path): sha256_file(path) for path in source_paths if path.is_file()
    }
    run_record_paths: list[Path] = []
    for rank in RANK_COUNTS:
        record_path = root / f"run_record_np{rank}.json"
        record = _build_run_record(
            worker=workers[rank],
            comparison=comparison,
            rank_count=rank,
            preflight=preflight,
            dirty_patch_sha256=dirty_digest,
            command=commands[rank],
            raw_json=paths[rank]["json"],
            arrays_npz=paths[rank]["npz"],
            log_path=paths[rank]["log"],
            report_path=report_path,
            code_hashes=code_hashes,
        )
        atomic_write_run_record(record_path, record)
        run_record_paths.append(record_path)

    artifact_paths = [
        comparison_path,
        report_path,
        *run_record_paths,
        *(path for rank_paths in paths.values() for path in rank_paths.values()),
    ]
    manifest = {
        "schema": {
            "id": "fenics-nonlinear-energies.pilot-manifest",
            "version": 1,
        },
        "campaign": CAMPAIGN_ID,
        "experiment": EXPERIMENT_ID,
        "status": payload["status"],
        "publication_evidence": False,
        "run_kind": str(args.run_kind),
        "reason": (
            "managed clean raw correctness evidence pending finalization and admission"
            if str(args.run_kind) == "publication"
            else "local controlled pilot not eligible for publication"
        ),
        "preflight": preflight.provenance_fields(),
        "dirty_patch_sha256": dirty_digest,
        "commands": {f"np{rank}": commands[rank] for rank in RANK_COUNTS},
        "artifacts": [
            {"path": _artifact_label(path), "sha256": sha256_file(path)}
            for path in artifact_paths
        ],
        "comparison": comparison,
        "rank_comparisons": rank_comparisons,
        "recorded_at_utc": utc_now_iso(),
    }
    atomic_write_json(root / "pilot_manifest.json", manifest)
    print(json.dumps({"status": payload["status"], **comparison}, indent=2, allow_nan=False))
    if not comparison["algebraic_gate_passed"]:
        raise SystemExit(2)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-kind", choices=("pilot", "publication"), default="pilot")
    parser.add_argument("--pilot-dirty-override", action="store_true")
    parser.add_argument("--pilot-override-reason")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--angle", type=float, default=0.15)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--ksp-rtol", type=float, default=1.0e-12)
    parser.add_argument("--linear-residual-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--derivative-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--solve-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--residual-scale-floor", type=float, default=1.0)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-json", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-npz", type=Path, help=argparse.SUPPRESS)
    return parser


def _validate_publication_configuration(args: argparse.Namespace) -> None:
    if str(args.run_kind) != "publication":
        return
    mismatches: list[str] = []
    for name, expected in PUBLICATION_CONFIGURATION.items():
        actual = getattr(args, name)
        if isinstance(expected, int):
            matches = int(actual) == expected
        else:
            matches = float(actual) == expected
        if not matches:
            mismatches.append(f"--{name.replace('_', '-')}={actual!r} (expected {expected!r})")
    if mismatches:
        raise ValueError(
            "publication execution must use the frozen EXP-DIST-001 configuration: "
            + "; ".join(mismatches)
        )


def main() -> None:
    args = _parser().parse_args()
    if int(args.level) < 1:
        raise ValueError("--level must be at least one")
    if int(args.repetitions) < 1:
        raise ValueError("--repetitions must be at least one")
    if float(args.timeout_s) <= 0.0:
        raise ValueError("--timeout-s must be positive")
    _validate_publication_configuration(args)
    if args.worker:
        if args.worker_json is None or args.worker_npz is None:
            raise ValueError("worker mode requires --worker-json and --worker-npz")
        _worker_payload(args)
        return
    if args.output_dir is None:
        raise ValueError("controller mode requires --output-dir")
    _controller(args)


if __name__ == "__main__":
    main()
