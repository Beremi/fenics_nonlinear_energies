import argparse
import json
import time
import numpy as np

from tools.minimizers import newton
from tools.sparse_solvers import HessSolverGenerator
from tools.jax_diff import EnergyDerivator
from HyperElasticity3D_jax.jax_energy import J
from HyperElasticity3D_jax.mesh import MeshHyperElasticity3D
from HyperElasticity3D_jax.rotate_boundary import rotate_boundary


def run_level(level: int, steps: int):
    rotation_per_iter = 4 * 2 * np.pi / 24

    mesh = MeshHyperElasticity3D(mesh_level=level)
    params, adjacency, u_init = mesh.get_data_jax()
    energy = EnergyDerivator(J, params, adjacency, u_init)

    results = []

    for step in range(1, steps + 1):
        angle = step * rotation_per_iter
        params = rotate_boundary(params, angle=rotation_per_iter)
        energy.params = params

        F, dF, ddF = energy.get_derivatives()
        ddf_with_solver = HessSolverGenerator(
            ddf=ddF,
            solver_type="amg",
            elastic_kernel=mesh.elastic_kernel,
            verbose=False,
            tol=1e-3,
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

        results.append(
            {
                "step": step,
                "angle": float(angle),
                "time": round(float(elapsed), 4),
                "iters": int(res["nit"]),
                "energy": float(res["fun"]),
                "message": res["message"],
            }
        )

    return {
        "solver": "jax",
        "mesh_level": level,
        "steps": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    out = run_level(args.level, args.steps)
    print(json.dumps(out, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
