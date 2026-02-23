"""Diagnose build_nullspace SEGV: compare vector sizes vs DOF counts."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.fem.petsc import create_matrix
import ufl
import numpy as np
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.rank

for level in [1, 2]:
    Nx = 80 * 2**(level - 1)
    Ny = 2 * 2**(level - 1)
    msh = mesh.create_box(comm, [[0.0, -0.005, -0.005], [0.4, 0.005, 0.005]],
                          [Nx, Ny, Ny], cell_type=mesh.CellType.tetrahedron)
    V = fem.functionspace(msh, ("Lagrange", 1, (3,)))
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    w = ufl.TrialFunction(V)
    d = len(u)
    I = ufl.Identity(d)
    F_def = I + ufl.grad(u)
    J_det = ufl.det(F_def)
    I1 = ufl.inner(F_def, F_def)
    C1, D1 = 38461538.0, 83333333.0
    W = C1 * (I1 - 3 - 2 * ufl.ln(J_det)) + D1 * (J_det - 1)**2
    ddJ = ufl.derivative(ufl.derivative(W * ufl.dx, u, v), u, w)
    A = create_matrix(fem.form(ddJ))

    im = V.dofmap.index_map
    size_local = im.size_local
    size_ghosts = im.num_ghosts
    vec = A.createVecLeft()
    vec_local_size = vec.getLocalSize()

    x = V.tabulate_dof_coordinates()
    x_shape = x.shape

    print(f"[rank={rank}] level={level}  size_local={size_local}  ghosts={size_ghosts}"
          f"  vec_local={vec_local_size}  tabulate_shape={x_shape}"
          f"  3*size_local={3*size_local}  vec_local==3*size_local={vec_local_size == 3*size_local}",
          flush=True)

    # Try localForm
    with vec.localForm() as loc:
        loc_size = len(loc.array)
        print(f"[rank={rank}] level={level}  localForm array size={loc_size}  "
              f"expected={3*size_local}  match={loc_size == 3*size_local}", flush=True)

    comm.Barrier()
    if rank == 0:
        print(f"--- level {level} OK ---", flush=True)
    comm.Barrier()
