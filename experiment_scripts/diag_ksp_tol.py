"""Diagnostic: check if snes.setFromOptions() resets KSP tolerances."""
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.fem.petsc import create_matrix
from petsc4py import PETSc
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from HyperElasticity3D_fenics.solve_HE_snes_newton import build_nullspace

comm = MPI.COMM_WORLD
msh = mesh.create_box(comm, [[0.0, -0.005, -0.005], [0.4, 0.005, 0.005]],
                      [80, 2, 2], cell_type=mesh.CellType.tetrahedron)
V = fem.functionspace(msh, ("Lagrange", 1, (3,)))
u = fem.Function(V)
v = ufl.TestFunction(V)
w = ufl.TrialFunction(V)
d = len(u)
I = ufl.Identity(d)
F_def = I + ufl.grad(u)
J_det = ufl.det(F_def)
I1 = ufl.inner(F_def, F_def)
C1, D1 = 38461538.461538464, 83333333.33333333
W = C1 * (I1 - 3 - 2 * ufl.ln(J_det)) + D1 * (J_det - 1)**2
ddJ = ufl.derivative(ufl.derivative(W * ufl.dx, u, v), u, w)
A = create_matrix(fem.form(ddJ))

ns = build_nullspace(V, A)
A.setNearNullSpace(ns)

snes = PETSc.SNES().create(comm)
snes.setType("newtonls")
snes.setOptionsPrefix("he_")
ksp = snes.getKSP()
ksp.setType("cg")
pc = ksp.getPC()
pc.setType("hypre")
pc.setHYPREType("boomeramg")

opts = PETSc.Options()
opts["he_pc_hypre_boomeramg_nodal_coarsen"] = 6
opts["he_pc_hypre_boomeramg_vec_interp_variant"] = 3
ksp.setTolerances(rtol=1e-1, max_it=30)

print(f"BEFORE setFromOptions: rtol={ksp.getTolerances()}", flush=True)
snes.setFromOptions()
print(f"AFTER  setFromOptions: rtol={ksp.getTolerances()}", flush=True)

# Also check with ksp_view:
ksp.view()
