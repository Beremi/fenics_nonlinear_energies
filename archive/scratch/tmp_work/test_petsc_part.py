import sys
from petsc4py import PETSc
from mpi4py import MPI
import scipy.sparse as sp

sys.path.insert(0, ".")
from HyperElasticity3D_jax_petsc.mesh import MeshHyperElasticity3D

m = MeshHyperElasticity3D(1)
p, adj, u = m.get_data()

adj_csr = adj.tocsr()
A = PETSc.Mat().createAIJ(size=adj.shape, csr=(adj_csr.indptr, adj_csr.indices, adj_csr.data), comm=PETSc.COMM_SELF)
A.assemble()

part = PETSc.MatPartitioning().create(comm=PETSc.COMM_SELF)
part.setAdjacency(A)
part.setNParts(4)
# Try chaco or scotch if metis is unavailable
try:
    part.setType(PETSc.MatPartitioning.Type.METIS)
except:
    part.setType(PETSc.MatPartitioning.Type.SCOTCH)

is_part = part.apply()
print("Success:", is_part.getIndices()[:10], len(is_part.getIndices()))
print(is_part.getIndices().max())
