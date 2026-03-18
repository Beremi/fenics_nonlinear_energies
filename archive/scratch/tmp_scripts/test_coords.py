import dolfinx
from mpi4py import MPI
import basix.ufl

mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
element = basix.ufl.element("Lagrange", mesh.topology.cell_name(), 1, shape=(3,))
V = dolfinx.fem.functionspace(mesh, element)

x = V.tabulate_dof_coordinates()
index_map = V.dofmap.index_map
x_owned = x[:index_map.size_local, :]

print("local blocks =", index_map.size_local)
print("x shape =", x.shape)
print("x_owned shape =", x_owned.shape)
