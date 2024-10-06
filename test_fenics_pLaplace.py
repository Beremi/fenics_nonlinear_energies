import time
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh, log
from dolfinx.io import XDMFFile
from petsc4py import PETSc
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, set_bc, create_matrix, create_vector
from petsc4py.PETSc import ScalarType  # type: ignore
import numpy as np

# Load mesh
with XDMFFile(MPI.COMM_WORLD, "pLaplace_fenics_mesh/mesh_level_9.xdmf", "r") as xdmf_file:
    msh = xdmf_file.read_mesh(name="mesh")

# Function space, boundary conditions, and parameters
V = fem.functionspace(msh, ("Lagrange", 1))
msh.topology.create_connectivity(1, 2)  # Ensure facet connectivity for boundary facets
boundary_facets = mesh.exterior_facet_indices(msh.topology)
dofs = fem.locate_dofs_topological(V, 1, boundary_facets)
bc = fem.dirichletbc(ScalarType(0), dofs, V)

p = 3
f = fem.Constant(msh, ScalarType(-10.0))

# Define variational problem
u = fem.Function(V)

# Initial guess
indexes_local = range(*u.vector.getOwnershipRange())  # type: ignore
u.vector.setValues(indexes_local, 1e-2 * np.random.rand(len(indexes_local)))  # type: ignore
u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
u.vector.assemble()  # type: ignore
set_bc(u.vector, [bc])  # type: ignore

v = ufl.TestFunction(V)
F = (1 / p) * ufl.inner(ufl.grad(u), ufl.grad(u))**(p / 2) * ufl.dx - f * u * ufl.dx
J = ufl.derivative(F, u, v)

# Define solver
problem = NonlinearProblem(J, u, bcs=[bc])
solver = NewtonSolver(msh.comm, problem)
solver.atol = 1e-6
solver.rtol = 1e-8
solver.max_it = 20
solver.report = True
# log.set_log_level(log.LogLevel.INFO)

# Set KSP options
ksp = solver.krylov_solver
ksp.setType(PETSc.KSP.Type.CG)
ksp.getPC().setType(PETSc.PC.Type.HYPRE)
ksp.setTolerances(rtol=1e-1)

# Solve
start_time = time.time()
n_newton_iterations, _ = solver.solve(u)
total_time = time.time() - start_time

# Output final energy
final_energy = fem.assemble_scalar(fem.form(F))
final_energy = msh.comm.allreduce(final_energy, op=MPI.SUM)
if MPI.COMM_WORLD.rank == 0:
    print(f"Final energy: {final_energy:.3e}")
    print(f"Total run time: {total_time:.3f} s")
    print(f"Number of Newton iterations: {n_newton_iterations}")
