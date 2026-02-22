#!/usr/bin/env python3
"""
Run JAX Ginzburg-Landau solver at levels 5-9, 3 repetitions each, report median.
"""
import sys, time
import numpy as np
from jax import config
config.update("jax_enable_x64", True)

from tools.minimizers import newton
from tools.sparse_solvers import HessSolverGenerator
from tools.jax_diff import EnergyDerivator
from GinzburgLandau2D_jax.jax_energy import J
from GinzburgLandau2D_jax.mesh import MeshGL2D

for lvl in [5, 6, 7, 8, 9]:
    try:
        mesh = MeshGL2D(mesh_level=lvl)
    except Exception as e:
        print(f"L{lvl}: SKIP (no mesh: {e})")
        continue

    params, adjacency, u_init = mesh.get_data_jax()
    ndofs_free = len(u_init)
    ndofs_total = len(mesh.params["u_0"])

    # First run: includes JIT compilation
    energy_GL = EnergyDerivator(J, params, adjacency, u_init)
    F, dF, ddF = energy_GL.get_derivatives()
    ddf_solver = HessSolverGenerator(ddf=ddF, solver_type="amg", verbose=False, tol=1e-3)
    # warmup / JIT
    res0 = newton(F, dF, ddf_solver, u_init, verbose=False, tolf=1e-6, tolg=1e-5, linesearch_tol=1e-3)

    results = []
    for run in range(3):
        t0 = time.perf_counter()
        res = newton(F, dF, ddf_solver, u_init, verbose=False, tolf=1e-6, tolg=1e-5, linesearch_tol=1e-3)
        dt = time.perf_counter() - t0
        results.append((dt, res["nit"], res["fun"]))

    results.sort(key=lambda x: x[0])
    dt, nit, fun = results[1]  # median
    print(f"L{lvl}: dofs_total={ndofs_total} dofs_free={ndofs_free} time={dt:.4f} iters={nit} J={fun:.7f}", flush=True)
