"""FEniCS KSP benchmark — unit square, no h5py, for parallel comparison."""
import time
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import mesh, fem
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from dolfinx.fem.petsc import (
    assemble_matrix, assemble_vector, create_matrix, set_bc, apply_lifting,
)
from tools_petsc4py.minimizers import newton

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=441, help="Grid dimension (DOFs ≈ (N+1)^2)")
args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
N = args.N

msh = mesh.create_unit_square(comm, N, N)
V = fem.functionspace(msh, ("Lagrange", 1))
total_dofs = V.dofmap.index_map.size_global

msh.topology.create_connectivity(1, 2)
bf = mesh.exterior_facet_indices(msh.topology)
bc_dofs = fem.locate_dofs_topological(V, 1, bf)
bc = fem.dirichletbc(ScalarType(0), bc_dofs, V)

u = fem.Function(V)
v = ufl.TestFunction(V)
w = ufl.TrialFunction(V)
p = 3.0
f_rhs = fem.Constant(msh, ScalarType(-10.0))

J_energy = (1 / p) * ufl.inner(ufl.grad(u), ufl.grad(u))**(p / 2) * ufl.dx - f_rhs * u * ufl.dx
dJ = ufl.derivative(J_energy, u, v)
ddJ = ufl.derivative(dJ, u, w)
energy_form = fem.form(J_energy)
grad_form = fem.form(dJ)
hessian_form = fem.form(ddJ)

# energy at arbitrary point
u_ls = fem.Function(V)
energy_ls = ufl.replace(J_energy, {u: u_ls})
energy_ls_form = fem.form(energy_ls)

# initial guess
np.random.seed(42)
x = u.x.petsc_vec
lo, hi = x.getOwnershipRange()
x.setValues(range(lo, hi), 1e-2 * np.random.rand(hi - lo))
x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
x.assemble()
set_bc(x, [bc])

A = create_matrix(hessian_form)
ksp = PETSc.KSP().create(comm)
ksp.setType("cg")
ksp.getPC().setType("hypre")
ksp.setTolerances(rtol=1e-3)


def _ghost_update(v): return v.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


_hess_timings = []


def energy_fn(vec):
    vec.copy(u_ls.x.petsc_vec)
    u_ls.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    return comm.allreduce(fem.assemble_scalar(energy_ls_form), op=MPI.SUM)


def gradient_fn(vec, g):
    with g.localForm() as g_loc:
        g_loc.set(0.0)
    assemble_vector(g, grad_form)
    apply_lifting(g, [hessian_form], [[bc]], x0=[vec])
    g.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(g, [bc], vec)


def hessian_solve_fn(vec, rhs, sol):
    t0 = time.perf_counter()
    A.zeroEntries()
    assemble_matrix(A, hessian_form, bcs=[bc])
    A.assemble()
    t_asm = time.perf_counter() - t0
    t0 = time.perf_counter()
    ksp.setOperators(A)
    ksp.solve(rhs, sol)
    ksp_its = ksp.getIterationNumber()
    t_ksp = time.perf_counter() - t0
    _hess_timings.append({"assembly": t_asm, "ksp": t_ksp, "ksp_its": ksp_its})
    return ksp_its


if rank == 0:
    print(f"FEniCS Custom Newton | {size} procs | {total_dofs} DOFs")

t_start = time.perf_counter()
result = newton(
    energy_fn, gradient_fn, hessian_solve_fn, x,
    tolf=1e-5, tolg=1e-3, linesearch_tol=1e-3,
    linesearch_interval=(-0.5, 2.0), maxit=100,
    verbose=True, comm=comm, ghost_update_fn=_ghost_update,
    save_history=True,
)
total = time.perf_counter() - t_start

if rank == 0:
    print(f"\nPer-iteration breakdown (hessian_solve_fn):")
    print(f"  {'It':>3s} {'assembly':>10s} {'KSP':>10s} {'KSP it':>7s}")
    print("  " + "-" * 34)
    for i, d in enumerate(_hess_timings):
        print(f"  {i:3d} {d['assembly']:10.4f} {d['ksp']:10.4f} {d['ksp_its']:7d}")
    asm_sum = sum(d["assembly"] for d in _hess_timings)
    ksp_sum = sum(d["ksp"] for d in _hess_timings)
    print("  " + "-" * 34)
    print(f"  SUM {asm_sum:10.4f} {ksp_sum:10.4f}")

    # Newton-level breakdown (from history)
    hist = result.get("history", [])
    if hist:
        print(f"\n  Newton-level breakdown:")
        print(f"  {'It':>3s} {'grad':>8s} {'hess':>8s} {'LS':>8s} {'update':>8s} {'ls_ev':>6s} {'iter':>8s}")
        print("  " + "-" * 52)
        s_grad = s_hess = s_ls = s_update = 0.0
        for h in hist:
            s_grad += h["t_grad"]
            s_hess += h["t_hess"]
            s_ls += h["t_ls"]
            s_update += h["t_update"]
            print(f"  {h['it']:3d} {h['t_grad']:8.4f} {h['t_hess']:8.4f} "
                  f"{h['t_ls']:8.4f} {h['t_update']:8.4f} {h['ls_evals']:6d} {h['t_iter']:8.4f}")
        print("  " + "-" * 52)
        print(f"  SUM {s_grad:8.4f} {s_hess:8.4f} {s_ls:8.4f} {s_update:8.4f}")
        print(f"\n  grad={s_grad:.4f}s  hess={s_hess:.4f}s  LS={s_ls:.4f}s  "
              f"update={s_update:.4f}s  solve={total:.4f}s")

    print(f"  iters={result['nit']}, J(u)={result['fun']:.6f}")

ksp.destroy()
A.destroy()
