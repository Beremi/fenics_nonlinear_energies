#!/usr/bin/env python3
import argparse
import json
import numpy as np

from src.core.serial.jax_diff import EnergyDerivator
from src.core.serial.minimizers import newton
from src.core.serial.sparse_solvers import HessSolverGenerator
from src.problems.hyperelasticity.jax.jax_energy import J
from src.problems.hyperelasticity.jax.mesh import MeshHyperElasticity3D
from src.problems.hyperelasticity.jax.rotate_boundary import rotate_boundary


def run(level: int, steps: int):
    rotation_per_iter = 4 * 2 * np.pi / 24

    mesh = MeshHyperElasticity3D(mesh_level=level)
    params, adjacency, u_init = mesh.get_data_jax()
    energy = EnergyDerivator(J, params, adjacency, u_init)

    coords = np.array(mesh.params["nodes2coord"], dtype=np.float64)
    freedofs = np.array(mesh.params["dofsMinim"]).ravel().astype(np.int64)

    u_full_steps = []
    records = []

    for step in range(1, steps + 1):
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

        u_free = np.array(res["x"], dtype=np.float64)
        u_full = np.array(params["u_0"], dtype=np.float64)
        u_full[freedofs] = u_free
        u_full = u_full.reshape((-1, 3))

        u_full_steps.append(u_full)
        records.append(
            {
                "step": step,
                "angle": float(step * rotation_per_iter),
                "energy": float(res["fun"]),
                "iters": int(res["nit"]),
                "message": res["message"],
            }
        )

        u_init = u_free

    return {
        "coords": coords,
        "u_full_steps": np.stack(u_full_steps, axis=0),
        "records": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--out_npz", type=str, required=True)
    parser.add_argument("--out_json", type=str, default="")
    args = parser.parse_args()

    out = run(args.level, args.steps)

    np.savez_compressed(
        args.out_npz,
        coords=out["coords"],
        u_full_steps=out["u_full_steps"],
    )

    summary = {
        "mesh_level": args.level,
        "steps": out["records"],
        "npz": args.out_npz,
    }
    print(json.dumps(summary, indent=2))

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2)
