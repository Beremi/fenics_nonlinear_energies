#!/usr/bin/env python3
"""Run one full HE Newton step with experimental element distribution strategies."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import jax
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from src.core.petsc.minimizers import newton
from src.problems.hyperelasticity.support.mesh import MeshHyperElasticity3D
from src.problems.hyperelasticity.support.rotate_boundary import (
    rotate_right_face_from_reference,
)
from experiments.diagnostics.bench_he_element_distribution import (
    _build_global_layout,
    _build_local_strategy_data,
    _configure_thread_env,
    _create_matrix,
    _make_local_element_kernels,
    _perm_block_metis,
    _perm_block_rcm,
    _perm_block_xyz,
    _perm_identity,
    _local_vec_from_full,
)


def _select_perm(reorder: str, params, adjacency, n_parts: int) -> np.ndarray:
    block_size = 3
    if reorder == "none":
        return _perm_identity(len(params["freedofs"]))
    if reorder == "block_rcm":
        return _perm_block_rcm(adjacency, block_size)
    if reorder == "block_xyz":
        return _perm_block_xyz(params["nodes2coord"], params["freedofs"], block_size)
    if reorder == "block_metis":
        return _perm_block_metis(adjacency, block_size, n_parts)
    raise ValueError(f"Unknown reorder {reorder}")


def _build_gamg_coordinates(layout, params):
    freedofs = np.asarray(params["freedofs"], dtype=np.int64)
    owned_orig_free = layout.perm[layout.lo:layout.hi]
    owned_total_dofs = freedofs[owned_orig_free]
    blocks = owned_total_dofs.reshape(-1, 3)
    node_ids = blocks[:, 0] // 3
    return np.asarray(params["nodes2coord"][node_ids], dtype=np.float64)


@dataclass
class ScatterData:
    grad_reord: np.ndarray
    grad_mask_owned: np.ndarray
    grad_mask_free: np.ndarray
    hess_e: np.ndarray
    hess_i: np.ndarray
    hess_j: np.ndarray
    hess_positions: np.ndarray


class ExperimentalElementStep:
    def __init__(self, args):
        self.args = args
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.size = self.comm.size
        self.block_size = 3
        self.total_runtime_start = time.perf_counter()

        mesh = MeshHyperElasticity3D(args.level)
        params, adjacency, u_init = mesh.get_data()
        self.params = params
        self.adjacency = adjacency
        self.u_init = np.asarray(u_init, dtype=np.float64)

        perm = _select_perm(args.reorder, params, adjacency, self.size)
        self.layout = _build_global_layout(params, adjacency, perm, self.comm, self.block_size)
        self.local_data = _build_local_strategy_data(params, self.layout, args.strategy, self.comm)
        self.energy_local_fn, self.elem_grad_fn, self.elem_hess_fn = _make_local_element_kernels(
            self.local_data, params
        )

        self._scatter = self._build_scatter_data()
        self._gather_sizes = np.asarray(self.comm.allgather(self.layout.hi - self.layout.lo), dtype=np.int64)
        self._gather_displs = np.zeros_like(self._gather_sizes)
        if len(self._gather_displs) > 1:
            self._gather_displs[1:] = np.cumsum(self._gather_sizes[:-1])

        setup_t0 = time.perf_counter()
        self.A, self.nullspace = _create_matrix(self.layout, params, self.comm)
        self.ksp = PETSc.KSP().create(self.comm)
        self.ksp.setType(args.ksp_type)
        pc = self.ksp.getPC()
        pc.setType(args.pc_type)
        opts = PETSc.Options()
        opts["pc_gamg_threshold"] = float(args.gamg_threshold)
        opts["pc_gamg_agg_nsmooths"] = int(args.gamg_agg_nsmooths)
        self.ksp.setTolerances(rtol=float(args.ksp_rtol), max_it=int(args.ksp_max_it))
        self.ksp.setFromOptions()
        self.gamg_coords = _build_gamg_coordinates(self.layout, params)
        self.force_pc_setup_next = True
        self.setup_time_matrix = time.perf_counter() - setup_t0

        self.x = PETSc.Vec().createMPI((self.layout.hi - self.layout.lo, self.layout.n_free), comm=self.comm)
        init_reordered = self.u_init[self.layout.perm]
        self.x.array[:] = init_reordered[self.layout.lo:self.layout.hi]
        self.x.assemble()

        self.linear_timing_records = []
        self.step_angle = float(args.start_step) * (4.0 * 2.0 * np.pi / float(args.total_steps))
        self.dirichlet_full = rotate_right_face_from_reference(
            params["u_0_ref"],
            params["nodes2coord"],
            self.step_angle,
            params["right_nodes"],
        )

        self.setup_time_jit = self._warmup()
        self.setup_time = self.setup_time_matrix + self.setup_time_jit

    def _warmup(self) -> float:
        t0 = time.perf_counter()
        v_local = np.asarray(self.dirichlet_full[self.local_data.local_total_nodes], dtype=np.float64)
        v_local_j = jax.numpy.asarray(v_local, dtype=jax.numpy.float64)
        _ = self.energy_local_fn(v_local_j).block_until_ready()
        _ = self.elem_grad_fn(v_local_j).block_until_ready()
        _ = self.elem_hess_fn(v_local_j).block_until_ready()
        return time.perf_counter() - t0

    def _build_scatter_data(self) -> ScatterData:
        elems_reordered = self.local_data.elems_reordered
        grad_reord = elems_reordered.reshape(-1)
        grad_mask_owned = (grad_reord >= self.layout.lo) & (grad_reord < self.layout.hi)
        grad_mask_free = grad_reord >= 0

        rows = elems_reordered[:, :, None]
        cols = elems_reordered[:, None, :]
        if self.args.strategy == "overlap_allgather":
            valid = (rows >= self.layout.lo) & (rows < self.layout.hi) & (cols >= 0)
            vi = np.where(valid)
            row_vals = elems_reordered[vi[0], vi[1]]
            col_vals = elems_reordered[vi[0], vi[2]]
            key_to_pos = self.layout.owned_key_to_pos
        else:
            valid = (rows >= 0) & (cols >= 0)
            vi = np.where(valid)
            row_vals = elems_reordered[vi[0], vi[1]]
            col_vals = elems_reordered[vi[0], vi[2]]
            key_to_pos = self.layout.global_key_to_pos
        keys = row_vals.astype(np.int64) * np.int64(self.layout.n_free) + col_vals.astype(np.int64)
        positions = np.fromiter((key_to_pos[int(k)] for k in keys), dtype=np.int64, count=len(keys))
        return ScatterData(
            grad_reord=grad_reord,
            grad_mask_owned=grad_mask_owned,
            grad_mask_free=grad_mask_free,
            hess_e=np.asarray(vi[0], dtype=np.int64),
            hess_i=np.asarray(vi[1], dtype=np.int64),
            hess_j=np.asarray(vi[2], dtype=np.int64),
            hess_positions=positions,
        )

    def _allgather_full(self, vec: PETSc.Vec):
        full = np.empty(self.layout.n_free, dtype=np.float64)
        t0 = time.perf_counter()
        self.comm.Allgatherv(
            np.asarray(vec.array[:], dtype=np.float64),
            [full, self._gather_sizes, self._gather_displs, MPI.DOUBLE],
        )
        return full, time.perf_counter() - t0

    def _build_v_local(self, full_reordered: np.ndarray):
        t0 = time.perf_counter()
        v_local = _local_vec_from_full(
            full_reordered,
            self.layout.total_to_free_reord,
            self.local_data.local_total_nodes,
            self.dirichlet_full,
        )
        return v_local, time.perf_counter() - t0

    def energy_fn(self, vec: PETSc.Vec):
        full, t_comm = self._allgather_full(vec)
        v_local, t_build = self._build_v_local(full)
        t0 = time.perf_counter()
        val_local = float(self.energy_local_fn(jax.numpy.asarray(v_local)).block_until_ready())
        val = float(self.comm.allreduce(val_local, op=MPI.SUM))
        t_eval = time.perf_counter() - t0
        self._last_energy_timing = {
            "state_allgather": t_comm,
            "state_build": t_build,
            "energy_eval": t_eval,
        }
        return val

    def gradient_fn(self, vec: PETSc.Vec, g: PETSc.Vec):
        full, t_comm = self._allgather_full(vec)
        v_local, t_build = self._build_v_local(full)

        t0 = time.perf_counter()
        elem_grad = np.asarray(self.elem_grad_fn(jax.numpy.asarray(v_local)).block_until_ready())
        grad_vals = elem_grad.reshape(-1)
        if self.args.strategy == "overlap_allgather":
            grad_owned = np.zeros(self.layout.hi - self.layout.lo, dtype=np.float64)
            np.add.at(
                grad_owned,
                self._scatter.grad_reord[self._scatter.grad_mask_owned] - self.layout.lo,
                grad_vals[self._scatter.grad_mask_owned],
            )
        else:
            grad_full_local = np.zeros(self.layout.n_free, dtype=np.float64)
            np.add.at(
                grad_full_local,
                self._scatter.grad_reord[self._scatter.grad_mask_free],
                grad_vals[self._scatter.grad_mask_free],
            )
            grad_full = np.zeros_like(grad_full_local)
            self.comm.Allreduce(grad_full_local, grad_full, op=MPI.SUM)
            grad_owned = grad_full[self.layout.lo:self.layout.hi]
        t_grad = time.perf_counter() - t0

        g.array[:] = grad_owned
        g.assemble()
        self._last_grad_timing = {
            "state_allgather": t_comm,
            "state_build": t_build,
            "grad_compute": t_grad,
        }

    def hessian_solve_fn(self, vec: PETSc.Vec, rhs: PETSc.Vec, sol: PETSc.Vec):
        full, t_comm = self._allgather_full(vec)
        v_local, t_build = self._build_v_local(full)

        t0 = time.perf_counter()
        elem_hess = np.asarray(self.elem_hess_fn(jax.numpy.asarray(v_local)).block_until_ready())
        contrib = elem_hess[self._scatter.hess_e, self._scatter.hess_i, self._scatter.hess_j]
        if self.args.strategy == "overlap_allgather":
            values = np.zeros(self.layout.owned_rows.size, dtype=np.float64)
            np.add.at(values, self._scatter.hess_positions, contrib)
        else:
            values_local = np.zeros(self.layout.coo_rows.size, dtype=np.float64)
            np.add.at(values_local, self._scatter.hess_positions, contrib)
            values_global = np.zeros_like(values_local)
            self.comm.Allreduce(values_local, values_global, op=MPI.SUM)
            values = values_global[self.layout.owned_mask]
        t_assemble_compute = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.A.setValuesCOO(values.astype(PETSc.ScalarType), addv=PETSc.InsertMode.INSERT_VALUES)
        self.A.assemble()
        t_mat = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.ksp.setOperators(self.A)
        if self.gamg_coords is not None:
            self.ksp.getPC().setCoordinates(self.gamg_coords)
            self.gamg_coords = None
        t_setop = time.perf_counter() - t0

        t0 = time.perf_counter()
        if self.args.pc_setup_on_ksp_cap:
            if self.force_pc_setup_next:
                self.ksp.setUp()
                self.force_pc_setup_next = False
        else:
            self.ksp.setUp()
        t_setup = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.ksp.solve(rhs, sol)
        t_solve = time.perf_counter() - t0
        ksp_its = int(self.ksp.getIterationNumber())
        if self.args.pc_setup_on_ksp_cap and ksp_its >= int(self.args.ksp_max_it):
            self.force_pc_setup_next = True

        self.linear_timing_records.append(
            {
                "assemble_state_allgather": float(t_comm),
                "assemble_state_build": float(t_build),
                "assemble_elem_compute": float(t_assemble_compute),
                "assemble_matrix_time": float(t_mat),
                "assemble_total_time": float(t_comm + t_build + t_assemble_compute + t_mat),
                "setop_time": float(t_setop),
                "pc_setup_time": float(t_setup),
                "solve_time": float(t_solve),
                "linear_total_time": float(t_comm + t_build + t_assemble_compute + t_mat + t_setop + t_setup + t_solve),
                "ksp_its": ksp_its,
            }
        )
        return ksp_its

    def run(self):
        t0 = time.perf_counter()
        result = newton(
            energy_fn=self.energy_fn,
            gradient_fn=self.gradient_fn,
            hessian_solve_fn=self.hessian_solve_fn,
            x=self.x,
            tolf=float(self.args.tolf),
            tolg=float(self.args.tolg),
            tolg_rel=float(self.args.tolg_rel),
            linesearch_tol=float(self.args.linesearch_tol),
            linesearch_interval=(float(self.args.linesearch_a), float(self.args.linesearch_b)),
            maxit=int(self.args.maxit),
            tolx_rel=float(self.args.tolx_rel),
            tolx_abs=float(self.args.tolx_abs),
            require_all_convergence=True,
            fail_on_nonfinite=True,
            verbose=(not self.args.quiet),
            comm=self.comm,
            ghost_update_fn=None,
            hessian_matvec_fn=lambda _x, vin, vout: self.A.mult(vin, vout),
            save_history=bool(self.args.save_history),
            trust_region=False,
        )
        step_time = time.perf_counter() - t0
        step_record = {
            "step": int(self.args.start_step),
            "angle": float(self.step_angle),
            "time": float(round(step_time, 6)),
            "nit": int(result["nit"]),
            "linear_iters": int(sum(int(r["ksp_its"]) for r in self.linear_timing_records)),
            "energy": float(result["fun"]),
            "message": str(result["message"]),
            "history": result.get("history", []) if self.args.save_history else [],
            "linear_timing": list(self.linear_timing_records) if self.args.save_linear_timing else [],
        }
        return {
            "mesh_level": int(self.args.level),
            "total_dofs": int(len(self.params["u_0"])),
            "free_dofs": int(self.layout.n_free),
            "setup_time": float(round(self.setup_time, 6)),
            "solve_time_total": float(round(step_record["time"], 6)),
            "total_time": float(round(time.perf_counter() - self.total_runtime_start, 6)),
            "steps": [step_record],
            "metadata": {
                "nprocs": int(self.size),
                "reorder": self.args.reorder,
                "strategy": self.args.strategy,
                "linear_solver": {
                    "ksp_type": self.args.ksp_type,
                    "pc_type": self.args.pc_type,
                    "ksp_rtol": float(self.args.ksp_rtol),
                    "ksp_max_it": int(self.args.ksp_max_it),
                    "pc_setup_on_ksp_cap": bool(self.args.pc_setup_on_ksp_cap),
                    "gamg_threshold": float(self.args.gamg_threshold),
                    "gamg_agg_nsmooths": int(self.args.gamg_agg_nsmooths),
                },
                "distribution": {
                    "elem_duplication_factor": float(self.comm.allreduce(len(self.local_data.local_elem_idx), op=MPI.SUM) / len(self.params["elems"])),
                    "local_elems_sum": int(self.comm.allreduce(len(self.local_data.local_elem_idx), op=MPI.SUM)),
                },
                "newton": {
                    "tolf": float(self.args.tolf),
                    "tolg": float(self.args.tolg),
                    "tolg_rel": float(self.args.tolg_rel),
                    "tolx_rel": float(self.args.tolx_rel),
                    "tolx_abs": float(self.args.tolx_abs),
                    "maxit": int(self.args.maxit),
                    "linesearch_interval": [float(self.args.linesearch_a), float(self.args.linesearch_b)],
                    "linesearch_tol": float(self.args.linesearch_tol),
                    "require_all_convergence": True,
                },
            },
        }

    def cleanup(self):
        self.ksp.destroy()
        self.nullspace.destroy()
        self.A.destroy()
        self.x.destroy()


def _build_parser():
    p = argparse.ArgumentParser(description="Full HE Newton step with experimental element distribution")
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--start-step", type=int, default=1)
    p.add_argument("--total-steps", type=int, default=96)
    p.add_argument("--reorder", choices=("none", "block_rcm", "block_xyz", "block_metis"), required=True)
    p.add_argument("--strategy", choices=("overlap_allgather", "nonoverlap_allreduce"), required=True)
    p.add_argument("--ksp-type", type=str, default="gmres")
    p.add_argument("--pc-type", type=str, default="gamg")
    p.add_argument("--ksp-rtol", type=float, default=1e-1)
    p.add_argument("--ksp-max-it", type=int, default=30)
    p.add_argument("--pc-setup-on-ksp-cap", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gamg-threshold", type=float, default=0.05)
    p.add_argument("--gamg-agg-nsmooths", type=int, default=1)
    p.add_argument("--tolf", type=float, default=1e-4)
    p.add_argument("--tolg", type=float, default=1e-3)
    p.add_argument("--tolg-rel", type=float, default=1e-3)
    p.add_argument("--tolx-rel", type=float, default=1e-3)
    p.add_argument("--tolx-abs", type=float, default=1e-10)
    p.add_argument("--maxit", type=int, default=100)
    p.add_argument("--linesearch-a", type=float, default=-0.5)
    p.add_argument("--linesearch-b", type=float, default=2.0)
    p.add_argument("--linesearch-tol", type=float, default=1e-3)
    p.add_argument("--save-history", action="store_true")
    p.add_argument("--save-linear-timing", action="store_true")
    p.add_argument("--nproc-threads", type=int, default=1)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--out", type=str, default="")
    return p


def main():
    args = _build_parser().parse_args()
    _configure_thread_env(args.nproc_threads)

    runner = ExperimentalElementStep(args)
    try:
        result = runner.run()
    finally:
        runner.cleanup()

    if MPI.COMM_WORLD.rank == 0:
        text = json.dumps(result, indent=2)
        print(text)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text)


if __name__ == "__main__":
    main()
