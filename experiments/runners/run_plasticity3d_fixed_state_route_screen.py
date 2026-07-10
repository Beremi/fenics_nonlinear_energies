#!/usr/bin/env python3
"""Time Plasticity3D Hessian routes repeatedly at one deterministic FE state.

The state is an analytic displacement field evaluated on the selected mesh and
then restricted to the free degrees of freedom.  Consequently every derivative
route and every MPI decomposition receives the same global coefficient vector.
The runner writes the state and one deterministic tangent action so route
equivalence can be checked directly after the campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import platform
import resource
import shlex
import subprocess
import sys
import time

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from src.core.benchmark.run_record import atomic_write_json
from experiments.runners.run_plasticity3d_backend_mix_case import (
    _build_local_assembly_backend,
)
from src.problems.slope_stability_3d.support.fixed_state import (
    prescribed_analytic_displacement,
)


ROUTE_SETTINGS = {
    "element_ad": ("element", "element"),
    "colored_sfd": ("sfd_local", "element"),
    "constitutive_ad": ("element", "constitutive"),
}


def _git_metadata(repo_root: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--short")),
    }


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _analytic_state(backend, amplitude: float) -> np.ndarray:
    coords = np.asarray(backend.coords_ref, dtype=np.float64)
    full = prescribed_analytic_displacement(coords, amplitude=float(amplitude))
    free_original = full.reshape(-1)[np.asarray(backend.freedofs, dtype=np.int64)]
    return np.asarray(free_original[np.asarray(backend.perm, dtype=np.int64)], dtype=np.float64)


def _set_owned_from_global(vec: PETSc.Vec, values: np.ndarray) -> None:
    lo, hi = (int(v) for v in vec.getOwnershipRange())
    vec.array[:] = np.asarray(values[lo:hi], dtype=np.float64)
    vec.assemble()


def _deterministic_probe(vec: PETSc.Vec, probe_index: int = 0) -> None:
    lo, hi = (int(v) for v in vec.getOwnershipRange())
    indices = np.arange(lo, hi, dtype=np.float64)
    probe_index = int(probe_index)
    first = 0.173 + 0.037 * probe_index
    second = 0.071 + 0.019 * probe_index
    phase = 0.13 * probe_index
    vec.array[:] = np.sin(first * (indices + 1.0) + phase) + (
        0.25 + 0.05 * probe_index
    ) * np.cos(second * (indices + 1.0) - phase)
    norm = float(vec.norm(PETSc.NormType.NORM_2))
    if norm <= 0.0 or not np.isfinite(norm):
        raise RuntimeError("deterministic tangent probe has invalid norm")
    vec.scale(1.0 / norm)


def _relative_gap(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    scale = np.maximum.reduce(
        (
            np.abs(left),
            np.abs(right),
            np.full_like(np.asarray(left, dtype=np.float64), 1.0e-14),
        )
    )
    return np.abs(left - right) / scale


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS and the BSD family report bytes.
    return value if sys.platform == "darwin" else 1024 * value


def _branch_diagnostics(backend, state: PETSc.Vec) -> dict[str, object]:
    """Classify the active Mohr--Coulomb branch on owned quadrature points."""

    assembler = backend.assembler
    v_local, _exchange = assembler._owned_to_local(  # noqa: SLF001 - experiment diagnostic
        np.asarray(state.array[:], dtype=np.float64)
    )
    local = assembler.local_data
    element_values = np.asarray(v_local, dtype=np.float64)[local.elems_local_np]
    ux = element_values[:, 0::3]
    uy = element_values[:, 1::3]
    uz = element_values[:, 2::3]
    data = local.local_elem_data
    dphix = np.asarray(data["dphix"], dtype=np.float64)
    dphiy = np.asarray(data["dphiy"], dtype=np.float64)
    dphiz = np.asarray(data["dphiz"], dtype=np.float64)
    exx = np.einsum("en,eqn->eq", ux, dphix, optimize=True)
    eyy = np.einsum("en,eqn->eq", uy, dphiy, optimize=True)
    ezz = np.einsum("en,eqn->eq", uz, dphiz, optimize=True)
    gxy = np.einsum("en,eqn->eq", ux, dphiy, optimize=True) + np.einsum(
        "en,eqn->eq", uy, dphix, optimize=True
    )
    gyz = np.einsum("en,eqn->eq", uy, dphiz, optimize=True) + np.einsum(
        "en,eqn->eq", uz, dphiy, optimize=True
    )
    gxz = np.einsum("en,eqn->eq", ux, dphiz, optimize=True) + np.einsum(
        "en,eqn->eq", uz, dphix, optimize=True
    )
    strain = np.empty((*exx.shape, 3, 3), dtype=np.float64)
    strain[..., 0, 0] = exx
    strain[..., 1, 1] = eyy
    strain[..., 2, 2] = ezz
    strain[..., 0, 1] = strain[..., 1, 0] = 0.5 * gxy
    strain[..., 1, 2] = strain[..., 2, 1] = 0.5 * gyz
    strain[..., 0, 2] = strain[..., 2, 0] = 0.5 * gxz
    eigvals = np.linalg.eigvalsh(strain)
    eig3, eig2, eig1 = eigvals[..., 0], eigvals[..., 1], eigvals[..., 2]
    invariant1 = exx + eyy + ezz

    c_bar = np.asarray(data["c_bar_q"], dtype=np.float64)
    sin_phi = np.asarray(data["sin_phi_q"], dtype=np.float64)
    shear = np.asarray(data["shear_q"], dtype=np.float64)
    bulk = np.asarray(data["bulk_q"], dtype=np.float64)
    lame = np.asarray(data["lame_q"], dtype=np.float64)
    f_term = 2.0 * shear * ((1.0 + sin_phi) * eig1 - (1.0 - sin_phi) * eig3)
    volumetric_term = 2.0 * lame * sin_phi * invariant1
    f_trial = f_term + volumetric_term - c_bar
    gamma_sl = (eig1 - eig2) / np.maximum(1.0e-15, 1.0 + sin_phi)
    gamma_sr = (eig2 - eig3) / np.maximum(1.0e-15, 1.0 - sin_phi)
    gamma_la = (eig1 + eig2 - 2.0 * eig3) / np.maximum(1.0e-15, 3.0 - sin_phi)
    gamma_ra = (2.0 * eig1 - eig2 - eig3) / np.maximum(1.0e-15, 3.0 + sin_phi)
    denominator_s = 4.0 * lame * sin_phi**2 + 4.0 * shear * (1.0 + sin_phi**2)
    denominator_l = (
        4.0 * lame * sin_phi**2
        + shear * (1.0 + sin_phi) ** 2
        + 2.0 * shear * (1.0 - sin_phi) ** 2
    )
    denominator_r = (
        4.0 * lame * sin_phi**2
        + 2.0 * shear * (1.0 + sin_phi) ** 2
        + shear * (1.0 - sin_phi) ** 2
    )
    lambda_s = f_trial / denominator_s
    lambda_l = (
        shear * ((1.0 + sin_phi) * (eig1 + eig2) - 2.0 * (1.0 - sin_phi) * eig3)
        + volumetric_term
        - c_bar
    ) / denominator_l
    lambda_r = (
        shear * (2.0 * (1.0 + sin_phi) * eig1 - (1.0 - sin_phi) * (eig2 + eig3))
        + volumetric_term
        - c_bar
    ) / denominator_r

    elastic = f_trial <= 0.0
    shear_return = (~elastic) & (lambda_s <= np.minimum(gamma_sl, gamma_sr))
    left_return = (
        (~(elastic | shear_return))
        & (gamma_sl < gamma_sr)
        & (lambda_l >= gamma_sl)
        & (lambda_l <= gamma_la)
    )
    right_return = (
        (~(elastic | shear_return | left_return))
        & (gamma_sl > gamma_sr)
        & (lambda_r >= gamma_sr)
        & (lambda_r <= gamma_ra)
    )
    branch = np.full(f_trial.shape, 4, dtype=np.int8)
    branch[elastic] = 0
    branch[shear_return] = 1
    branch[left_return] = 2
    branch[right_return] = 3

    f_scale = np.maximum.reduce(
        (np.abs(f_term), np.abs(volumetric_term), np.abs(c_bar), np.full_like(f_trial, 1.0e-14))
    )
    f_margin = np.abs(f_trial) / f_scale
    margin = np.asarray(f_margin, dtype=np.float64)
    margin[shear_return] = np.minimum.reduce(
        (
            f_margin[shear_return],
            _relative_gap(lambda_s, gamma_sl)[shear_return],
            _relative_gap(lambda_s, gamma_sr)[shear_return],
        )
    )
    margin[left_return] = np.minimum.reduce(
        (
            f_margin[left_return],
            _relative_gap(gamma_sl, gamma_sr)[left_return],
            _relative_gap(lambda_l, gamma_sl)[left_return],
            _relative_gap(lambda_l, gamma_la)[left_return],
        )
    )
    margin[right_return] = np.minimum.reduce(
        (
            f_margin[right_return],
            _relative_gap(gamma_sl, gamma_sr)[right_return],
            _relative_gap(lambda_r, gamma_sr)[right_return],
            _relative_gap(lambda_r, gamma_ra)[right_return],
        )
    )

    owned_elements = np.asarray(local.energy_weights, dtype=np.float64) > 0.5
    owned_qp = np.broadcast_to(owned_elements[:, None], branch.shape)
    local_counts = np.asarray(
        [np.count_nonzero(owned_qp & (branch == code)) for code in range(5)],
        dtype=np.int64,
    )
    counts = np.asarray(backend.comm.allreduce(local_counts, op=MPI.SUM), dtype=np.int64)
    total = int(np.sum(counts))
    local_near = int(np.count_nonzero(owned_qp & (margin < 1.0e-8)))
    near = int(backend.comm.allreduce(local_near, op=MPI.SUM))
    local_min = float(np.min(margin[owned_qp])) if np.any(owned_qp) else float("inf")
    min_margin = float(backend.comm.allreduce(local_min, op=MPI.MIN))
    labels = ("elastic", "shear", "left_edge", "right_edge", "apex")
    return {
        "definition": "mohr_coulomb_return_branch_v1",
        "owned_quadrature_points": total,
        "counts": {label: int(value) for label, value in zip(labels, counts, strict=True)},
        "fractions": {
            label: (float(value) / float(total) if total else 0.0)
            for label, value in zip(labels, counts, strict=True)
        },
        "plastic_fraction": (float(total - counts[0]) / float(total) if total else 0.0),
        "normalized_boundary_margin_min": min_margin,
        "near_boundary_threshold": 1.0e-8,
        "near_boundary_fraction": (float(near) / float(total) if total else 0.0),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    route = str(args.route)
    local_hessian_mode, tangent_mode = ROUTE_SETTINGS[route]
    backend = _build_local_assembly_backend(
        mesh_name=str(args.mesh_name),
        elem_degree=int(args.element_degree),
        constraint_variant=str(args.constraint_variant),
        quadrature_rule_id=str(args.quadrature_rule),
        lambda_target=float(args.lambda_target),
        local_hessian_mode=local_hessian_mode,
        autodiff_tangent_mode=tangent_mode,
        ksp_rtol=float(args.ksp_rtol),
        ksp_max_it=int(args.ksp_max_it),
    )
    comm = MPI.COMM_WORLD
    x = backend.create_vec()
    probe = backend.create_vec()
    action = backend.create_vec()
    gradient = backend.create_vec()
    try:
        state_global = _analytic_state(backend, float(args.state_amplitude))
        _set_owned_from_global(x, state_global)
        _deterministic_probe(probe, 0)

        # Compile both first- and second-derivative paths before warm repetitions.
        backend.vec_gradient(x, gradient)
        backend.vec_tangent(x)
        comm.Barrier()

        for _ in range(int(args.warmup_repetitions)):
            backend.vec_tangent(x)
        comm.Barrier()

        timings: list[float] = []
        timings_by_rank: list[list[float]] = []
        for _ in range(int(args.measured_repetitions)):
            started = time.perf_counter()
            tangent = backend.vec_tangent(x)
            local_elapsed = float(time.perf_counter() - started)
            rank_elapsed = [float(value) for value in comm.allgather(local_elapsed)]
            collective_max = float(comm.allreduce(local_elapsed, op=MPI.MAX))
            if not np.isclose(
                collective_max,
                max(rank_elapsed),
                rtol=1.0e-13,
                atol=1.0e-15,
            ):
                raise RuntimeError("MPI_MAX timing disagrees with gathered rank timings")
            timings.append(collective_max)
            timings_by_rank.append(rank_elapsed)

        action_rows: list[np.ndarray] = []
        action_hashes: list[str] = []
        action_norms: list[float] = []
        for probe_index in range(int(args.probe_count)):
            _deterministic_probe(probe, probe_index)
            tangent.mult(probe, action)
            action_global_row = backend.global_from_vec(action)
            action_rows.append(np.asarray(action_global_row, dtype=np.float64))
            action_hashes.append(_sha256_array(action_global_row))
            action_norms.append(float(action.norm(PETSc.NormType.NORM_2)))
        action_global = action_rows[0]
        actions_global = np.stack(action_rows, axis=0)
        gradient_global = backend.global_from_vec(gradient)
        state_global_check = backend.global_from_vec(x)
        if not np.array_equal(state_global, state_global_check):
            raise RuntimeError("distributed analytic state does not match its global definition")

        callbacks = dict(backend.assembler.callback_summary())
        setup = dict(backend.assembler.setup_summary())
        memory = dict(backend.assembler.memory_summary())
        local_elements = int(memory.get("local_elements", 0))
        local_overlap = int(memory.get("local_overlap_dofs", 0))
        owned = int(x.getLocalSize())
        owned_nnz = int(memory.get("owned_nnz", 0))
        tracked_bytes = int(
            round(float(memory.get("tracked_total_gib", 0.0)) * float(1024**3))
        )
        local_color_count = int(getattr(backend.assembler, "_sfd_n_colors", 0))
        element_dofs = int(backend.assembler.local_data.elems_local_np.shape[1])
        quadrature_points = int(
            np.asarray(backend.assembler.local_data.local_elem_data["dphix"]).shape[1]
        )
        owned_element_count = int(
            np.count_nonzero(
                np.asarray(backend.assembler.local_data.energy_weights, dtype=np.float64)
                > 0.5
            )
        )
        rank_rows = comm.gather(
            {
                "rank": int(comm.rank),
                "owned_dofs": owned,
                "local_elements": local_elements,
                "owned_elements": owned_element_count,
                "overlap_dofs": local_overlap,
                "owned_matrix_nonzeros": owned_nnz,
                "local_color_count": local_color_count,
                "tracked_allocation_bytes": tracked_bytes,
                "peak_rss_bytes": _peak_rss_bytes(),
            },
            root=0,
        )
        maximum_colors = int(comm.allreduce(local_color_count, op=MPI.MAX))
        total_owned_elements = int(comm.allreduce(owned_element_count, op=MPI.SUM))
        payload: dict[str, object] = {
            "schema_version": 1,
            "experiment_id": "EXP-ROUTE-001",
            "tier": str(args.tier),
            "status": "completed",
            "route": route,
            "mesh_name": str(args.mesh_name),
            "element_degree": int(args.element_degree),
            "quadrature_rule_id": str(args.quadrature_rule),
            "constraint_variant": str(args.constraint_variant),
            "lambda_target": float(args.lambda_target),
            "state_family": "analytic_mesh_field_v1",
            "state_label": str(args.state_label),
            "state_amplitude": float(args.state_amplitude),
            "state_sha256": _sha256_array(state_global),
            "probe_definition": "normalized_sin_cos_global_index_family_v2",
            "probe_count": int(args.probe_count),
            "action_sha256": _sha256_array(action_global),
            "action_sha256_by_probe": action_hashes,
            "action_norm_2": action_norms[0],
            "action_norm_2_by_probe": action_norms,
            "gradient_sha256": _sha256_array(gradient_global),
            "gradient_residual_identity": "energy_gradient_is_discrete_residual",
            "tangent_frobenius_norm": float(tangent.norm(PETSc.NormType.FROBENIUS)),
            "gradient_norm_2": float(gradient.norm(PETSc.NormType.NORM_2)),
            "branch_diagnostics": _branch_diagnostics(backend, x),
            "model_covariates": {
                "element_dofs": element_dofs,
                "constitutive_dimension": 6,
                "quadrature_points_per_element": quadrature_points,
                "maximum_local_color_count": (
                    maximum_colors if route == "colored_sfd" else None
                ),
                "total_owned_elements": total_owned_elements,
                "global_free_dofs": int(state_global.size),
                "rank_count": int(comm.size),
                "assembly_backend": str(memory.get("assembly_backend", "")),
                "matrix_type": str(memory.get("matrix_type", "")),
            },
            "mpi_ranks": int(comm.size),
            "warmup_repetitions": int(args.warmup_repetitions),
            "measured_repetitions": int(args.measured_repetitions),
            "wall_times_s": timings,
            "wall_times_by_rank_s": timings_by_rank,
            "wall_time_reduction": "mpi_collective_max",
            "wall_time_median_s": float(np.median(np.asarray(timings))),
            "wall_time_min_s": float(np.min(np.asarray(timings))),
            "wall_time_max_s": float(np.max(np.asarray(timings))),
            "assembly_callbacks": callbacks,
            "assembler_setup": setup,
            "assembler_memory": memory,
            "rank_summaries": rank_rows or [],
            "memory_measurement": {
                "tracked_allocations": "explicit NumPy/JAX/PETSc-owned arrays in assembler summary",
                "peak_rss": "resource.getrusage(RUSAGE_SELF).ru_maxrss; rank-local process high-water mark",
                "aggregation": "rank records retained; use rank maximum for capacity claims",
            },
            "runtime": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "jax_platforms": os.environ.get("JAX_PLATFORMS", ""),
            },
            "command": shlex.join([sys.executable, *sys.argv]),
            "git": _git_metadata(Path(__file__).resolve().parents[2]),
            "job_metadata": {
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
                "slurm_cluster_name": os.environ.get("SLURM_CLUSTER_NAME", ""),
                "workstation_run_id": os.environ.get("WORKSTATION_RUN_ID", ""),
            },
            "comparison_design": {
                "comparison_id": str(args.comparison_id),
                "block_repetition": int(args.block_repetition),
                "route_order_position": int(args.route_order_position),
                "route_order_policy": str(args.route_order_policy),
                "timing_reduction": "mpi_collective_max",
                "independent_process_block": True,
            },
        }
        if comm.rank == 0:
            record_dir = Path(args.output).resolve().parent
            action_out = Path(args.action_out).resolve()
            try:
                action_relative = action_out.relative_to(record_dir)
            except ValueError as exc:
                raise ValueError("--action-out must remain inside the output record directory") from exc
            action_out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                action_out,
                state=state_global,
                tangent_action=action_global,
                tangent_actions=actions_global,
                gradient=gradient_global,
                route=np.asarray(route),
                state_label=np.asarray(str(args.state_label)),
            )
            payload["action_out"] = str(action_relative)
            if bool(args.save_direct_matrix) and int(comm.size) == 1:
                indptr, indices, values = tangent.getValuesCSR()
                matrix_out = action_out.with_name("tangent_matrix_csr.npz")
                np.savez_compressed(
                    matrix_out,
                    indptr=np.asarray(indptr, dtype=np.int64),
                    indices=np.asarray(indices, dtype=np.int64),
                    values=np.asarray(values, dtype=np.float64),
                    shape=np.asarray(tangent.getSize(), dtype=np.int64),
                    route=np.asarray(route),
                )
                payload["direct_matrix_out"] = str(matrix_out.relative_to(record_dir))
                payload["direct_matrix_value_sha256"] = _sha256_array(values)
                payload["direct_matrix_nonzeros"] = int(np.asarray(values).size)
            else:
                payload["direct_matrix_out"] = ""
        return payload
    finally:
        gradient.destroy()
        action.destroy()
        probe.destroy()
        x.destroy()
        backend.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=tuple(ROUTE_SETTINGS), required=True)
    parser.add_argument(
        "--tier",
        choices=("fixed_state_screen", "factorized_quadrature"),
        default="fixed_state_screen",
    )
    parser.add_argument("--mesh-name", default="hetero_ssr_L1")
    parser.add_argument("--element-degree", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument(
        "--quadrature-rule",
        choices=("tetra_1point", "tetra_11point", "tetra_24point", "tetra_duffy_125point"),
        default="tetra_11point",
    )
    parser.add_argument("--constraint-variant", default="glued_bottom")
    parser.add_argument("--lambda-target", type=float, default=1.55)
    parser.add_argument("--state-label", required=True)
    parser.add_argument("--state-amplitude", type=float, required=True)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--measured-repetitions", type=int, default=5)
    parser.add_argument("--probe-count", type=int, default=1)
    parser.add_argument("--comparison-id", default="legacy_unpaired")
    parser.add_argument("--block-repetition", type=int, default=0)
    parser.add_argument("--route-order-position", type=int, default=0)
    parser.add_argument("--route-order-policy", default="legacy_unpaired")
    parser.add_argument("--save-direct-matrix", action="store_true")
    parser.add_argument("--ksp-rtol", type=float, default=1.0e-2)
    parser.add_argument("--ksp-max-it", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-out", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if int(args.warmup_repetitions) < 1 or int(args.measured_repetitions) < 5:
        raise SystemExit("publication screening requires >=1 warmup and >=5 measured repetitions")
    if int(args.probe_count) < 1:
        raise SystemExit("--probe-count must be positive")
    if bool(args.save_direct_matrix) and MPI.COMM_WORLD.size != 1:
        raise SystemExit("--save-direct-matrix is supported only for one-rank feasibility checks")
    payload = run(args)
    if MPI.COMM_WORLD.rank == 0:
        output = Path(args.output).resolve()
        atomic_write_json(output, payload)
        print(output)


if __name__ == "__main__":
    main()
