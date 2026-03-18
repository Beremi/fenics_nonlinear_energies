"""
p-Laplace 2D — FEniCS SNES Newton solver logic.

Provides ``run_level()`` using DOLFINx built-in SNES Newton solver.
CLI entry point is in ``solve_pLaplace_snes_newton.py``.
"""

import time

import h5py
import ufl
import numpy as np
import basix.ufl
from mpi4py import MPI
from dolfinx import fem, mesh
from petsc4py import PETSc
from dolfinx.fem.petsc import NonlinearProblem, set_bc
from petsc4py.PETSc import ScalarType

from src.core.petsc.fenics_tools import ghost_update as _ghost_update
from src.core.problem_data.hdf5 import mesh_data_path


def run_level(mesh_level):
    """Run p-Laplace SNES Newton solver for a single mesh level.

    Returns dict with: mesh_level, total_dofs, time, iters, energy, converged_reason
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        with h5py.File(mesh_data_path("pLaplace", f"pLaplace_level{mesh_level}.h5"), "r",
                       driver="core", backing_store=False) as f:
            points = f["nodes"][:]
            triangles = f["elems"][:].astype(np.int64)
    else:
        points = np.empty((0, 2), dtype=np.float64)
        triangles = np.empty((0, 3), dtype=np.int64)
    c_el = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    msh = mesh.create_mesh(comm, triangles, c_el, points)

    V = fem.functionspace(msh, ("Lagrange", 1))
    total_dofs = V.dofmap.index_map.size_global

    msh.topology.create_connectivity(1, 2)
    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    bc_dofs = fem.locate_dofs_topological(V, 1, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0), bc_dofs, V)

    p = 3
    f_const = fem.Constant(msh, ScalarType(-10.0))

    u = fem.Function(V)

    # Initial guess (small random, fixed seed)
    np.random.seed(42)
    vec = u.x.petsc_vec
    idx_local = range(*vec.getOwnershipRange())
    vec.setValues(idx_local, 1e-2 * np.random.rand(len(idx_local)))
    vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    vec.assemble()
    set_bc(vec, [bc])

    v = ufl.TestFunction(V)
    F_energy = (1 / p) * ufl.inner(ufl.grad(u), ufl.grad(u))**(p / 2) * ufl.dx - f_const * u * ufl.dx
    J_form = ufl.derivative(F_energy, u, v)

    petsc_opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "basic",
        "snes_atol": 1e-6,
        "snes_rtol": 1e-8,
        "snes_max_it": 20,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1e-1,
    }

    problem = NonlinearProblem(
        J_form, u,
        petsc_options_prefix=f"lvl{mesh_level}_",
        bcs=[bc],
        petsc_options=petsc_opts,
    )

    start_time = time.time()
    problem.solve()
    total_time = time.time() - start_time

    snes = problem.solver
    n_iters = snes.getIterationNumber()
    reason = snes.getConvergedReason()
    snes.destroy()

    final_energy = fem.assemble_scalar(fem.form(F_energy))
    final_energy = msh.comm.allreduce(final_energy, op=MPI.SUM)

    return {
        "mesh_level": mesh_level,
        "total_dofs": total_dofs,
        "time": round(total_time, 4),
        "iters": n_iters,
        "energy": round(final_energy, 4),
        "converged_reason": int(reason),
    }
