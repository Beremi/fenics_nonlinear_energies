import dolfinx
from mpi4py import MPI
import sys
import os
sys.path.insert(0, "/home/michal/repos/fenics_nonlinear_energies")

from HyperElasticity3D_jax_petsc.mesh import MeshHyperElasticity3D

comm = MPI.COMM_WORLD
mesh_obj = MeshHyperElasticity3D(comm, 3)
mesh, V = mesh_obj.mesh, mesh_obj.V

num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
num_cells_ghost = mesh.topology.index_map(mesh.topology.dim).num_ghosts
num_dofs_local = V.dofmap.index_map.size_local

print(f"Rank {comm.rank:2d}: Cells={num_cells_local:5d}  Ghosts={num_cells_ghost:5d}  DOFs={num_dofs_local:5d}")
