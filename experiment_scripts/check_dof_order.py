from dolfinx.mesh import create_box, CellType
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

mesh = create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [3, 3, 3], CellType.tetrahedron)
V = fem.functionspace(mesh, ("Lagrange", 1, (3,)))
imap = V.dofmap.index_map
bs = V.dofmap.index_map_bs

print('index_map.size_local:', imap.size_local)
print('index_map_bs (block size):', bs)
print('total local DOFs (size_local * bs):', imap.size_local * bs)

coords = V.tabulate_dof_coordinates()
print('tabulate_dof_coordinates shape:', coords.shape)
print()
print('First 10 rows of coords (DOFs 0-9):')
for i in range(min(10, len(coords))):
    print(f'  DOF {i}: {coords[i]}')

# --- Build nullspace same as build_nullspace in solve_HE_snes_newton.py ---
print()
print('=== Building nullspace vectors ===')
x = V.tabulate_dof_coordinates()
x_owned = x[:imap.size_local, :]

# Create a dummy PETSc Vec of the same size (size_local * bs)
n_local = imap.size_local * bs
v = PETSc.Vec().createSeq(n_local)
v.set(0.0)
arrays = []

names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
vecs_data = [np.zeros(n_local) for _ in range(6)]

for i in range(3):
    vecs_data[i][i::3] = 1.0

# rot_x: rotation about x-axis: (-) y-displacement = z-coord, z-displacement = y-coord
vecs_data[3][1::3] = -x_owned[:, 2]
vecs_data[3][2::3] = x_owned[:, 1]
# rot_y: rotation about y-axis
vecs_data[4][0::3] = x_owned[:, 2]
vecs_data[4][2::3] = -x_owned[:, 0]
# rot_z: rotation about z-axis
vecs_data[5][0::3] = -x_owned[:, 1]
vecs_data[5][1::3] = x_owned[:, 0]

print('Nullspace vector norms (should be non-zero):')
for i, (name, arr) in enumerate(zip(names, vecs_data)):
    norm = np.linalg.norm(arr)
    print(f'  {name}: norm = {norm:.6f}, nonzero count = {np.count_nonzero(arr)}')

print()
print('Dot products (orthogonality check, should be 0 for different modes):')
for i in range(6):
    for j in range(i + 1, 6):
        dot = np.dot(vecs_data[i], vecs_data[j])
        print(f'  {names[i]} . {names[j]} = {dot:.4f}')

# Check first 12 scalar DOFs of rot_z vector to verify layout
print()
print('rot_z vector first 12 scalar DOFs (should be [-y0, x0, 0, -y1, x1, 0, ...]):')
for i in range(min(12, n_local)):
    node = i // 3
    comp = i % 3
    print(
        f'  scalar_dof {i} (node={node}, comp={comp}): val={vecs_data[5][i]:.4f}, node_coord=({x_owned[node, 0]:.3f},{x_owned[node, 1]:.3f},{x_owned[node, 2]:.3f})')
