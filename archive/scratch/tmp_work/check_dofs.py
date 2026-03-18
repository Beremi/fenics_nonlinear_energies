import dolfinx
from mpi4py import MPI
import ufl
from petsc4py import PETSc

import sys
sys.path.append(".")
from tools.dolfinx_utils import ensure_mesh

comm = MPI.COMM_WORLD
mesh, _ = ensure_mesh(3)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (3,)))
owned_dofs = V.dofmap.index_map.size_local
print(f"Rank {comm.rank} has {owned_dofs} local nodes ({owned_dofs*3} DOFs)")
