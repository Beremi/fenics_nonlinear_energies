import json
import numpy as np
from src.core.serial.jax_diff import EnergyDerivator
from src.core.serial.minimizers import newton
from src.core.serial.sparse_solvers import HessSolverGenerator
from src.problems.hyperelasticity.jax.jax_energy import J
from src.problems.hyperelasticity.jax.mesh import MeshHyperElasticity3D
from src.problems.hyperelasticity.jax.rotate_boundary import rotate_boundary

rotation_per_iter = 4 * 2 * np.pi / 24
out = []
for lvl in [1, 2]:
    mesh = MeshHyperElasticity3D(mesh_level=lvl)
    params, adjacency, u_init = mesh.get_data_jax()
    energy = EnergyDerivator(J, params, adjacency, u_init)
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
    res = newton(F, dF, ddf_with_solver, u_init, verbose=False, tolf=1e-4, linesearch_tol=1e-3)
    out.append(
        {
            "mesh_level": lvl,
            "energy": float(res["fun"]),
            "iters": int(res["nit"]),
            "time": float(res["time"]),
            "message": res["message"],
        }
    )

print(json.dumps({"solver": "jax", "steps": 1, "results": out}, indent=2))
