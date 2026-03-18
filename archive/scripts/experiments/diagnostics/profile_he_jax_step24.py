#!/usr/bin/env python3
import argparse
import json
import time
import numpy as np
import scipy.sparse.linalg as spla
import pyamg

from tools.minimizers import newton
from tools.jax_diff import EnergyDerivator
from HyperElasticity3D_jax.jax_energy import J
from HyperElasticity3D_jax.mesh import MeshHyperElasticity3D
from HyperElasticity3D_jax.rotate_boundary import rotate_boundary


class SolverAMGElasticityTracked:
    def __init__(self, H, elastic_kernel=None, tol=1e-3, maxiter=1000, tracker=None):
        self.H = H
        self.tol = tol
        self.maxiter = maxiter
        self.tracker = tracker
        ml = pyamg.smoothed_aggregation_solver(self.H.tocsr(), B=elastic_kernel, smooth="energy")
        self.M_lin = ml.aspreconditioner()

    def solve(self, x):
        iteration_count = [0]

        def callback(_xk):
            iteration_count[0] += 1

        sol = spla.cg(
            self.H,
            x.copy(),
            rtol=self.tol,
            M=self.M_lin,
            callback=callback,
            maxiter=self.maxiter,
        )[0]

        if self.tracker is not None:
            self.tracker.append(int(iteration_count[0]))

        return sol


class HessSolverGeneratorTracked:
    def __init__(self, ddf, elastic_kernel=None, tol=1e-3, maxiter=1000, tracker=None):
        self.ddf = ddf
        self.elastic_kernel = elastic_kernel
        self.tol = tol
        self.maxiter = maxiter
        self.tracker = tracker

    def __call__(self, x):
        sparse_matrix_scipy = self.ddf(x)
        return SolverAMGElasticityTracked(
            sparse_matrix_scipy,
            elastic_kernel=self.elastic_kernel,
            tol=self.tol,
            maxiter=self.maxiter,
            tracker=self.tracker,
        )


def run(level: int, target_step: int, lin_tol: float):
    rotation_per_iter = 4 * 2 * np.pi / 24

    mesh = MeshHyperElasticity3D(mesh_level=level)
    params, adjacency, u_init = mesh.get_data_jax()
    energy = EnergyDerivator(J, params, adjacency, u_init)

    out = None

    for step in range(1, target_step + 1):
        params = rotate_boundary(params, angle=rotation_per_iter)
        energy.params = params
        F, dF, ddF = energy.get_derivatives()

        lin_tracker = []
        ddf_with_solver = HessSolverGeneratorTracked(
            ddf=ddF,
            elastic_kernel=mesh.elastic_kernel,
            tol=lin_tol,
            maxiter=1000,
            tracker=lin_tracker,
        )

        t0 = time.perf_counter()
        res = newton(
            F,
            dF,
            ddf_with_solver,
            u_init,
            verbose=False,
            tolf=1e-4,
            linesearch_tol=1e-3,
            maxit=100,
        )
        elapsed = time.perf_counter() - t0
        u_init = res["x"]

        if step == target_step:
            out = {
                "solver": "jax",
                "mesh_level": level,
                "step": step,
                "angle": float(step * rotation_per_iter),
                "time": float(round(elapsed, 4)),
                "newton_iters": int(res["nit"]),
                "energy": float(res["fun"]),
                "message": res["message"],
                "linear_iters_per_newton": lin_tracker,
                "linear_iters_total": int(sum(lin_tracker)),
                "linear_iters_max": int(max(lin_tracker) if lin_tracker else 0),
            }

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--step", type=int, default=24)
    parser.add_argument("--lin_tol", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    result = run(args.level, args.step, args.lin_tol)
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
