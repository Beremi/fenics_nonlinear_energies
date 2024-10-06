import numpy as np
import time
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType  # type: ignore
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, set_bc, create_matrix, create_vector
from dolfinx.io import XDMFFile


def zlatyrez(f, a, b, tol):
    """
    Find the minimum of a function f using the Golden section method.

    Parameters
    ----------
    f : Callable
        Function to find the minimum of.
    a : float
        Left endpoint of the interval.
    b : float
        Right endpoint of the interval.
    tol : float
        Tolerance of the method.

    Returns
    -------
    tuple[float, int]
        Tuple of the argument of minimum and number of iterations.
    """

    # Golden ratio
    gamma = 1 / 2 + np.sqrt(5) / 2

    # Initial values
    a0 = a
    b0 = b
    d0 = (b0 - a0) / gamma + a0
    c0 = a0 + b0 - d0

    # Iteration counter
    it = 0

    # Store the values of the interval and the function
    an = a0
    bn = b0
    cn = c0
    dn = d0
    fcn = f(cn)
    fdn = f(dn)

    while bn - an > tol:
        # Store the values of the interval and the function
        a = an
        b = bn
        c = cn
        d = dn
        fc = fcn
        fd = fdn

        if fc < fd:
            # Update the interval
            an = a
            bn = d
            dn = c
            cn = an + bn - dn

            # Update the function value
            fcn = f(cn)
            fdn = fc
        else:
            # Update the interval
            an = c
            bn = b
            cn = d
            dn = an + bn - cn

            # Update the function value
            fcn = fd
            fdn = f(dn)

        # Increment the iteration counter
        it += 1

    # Return the result
    t = (an + bn) / 2
    return t, it


all_start_time = time.time()

# Specify mesh level
mesh_level = 9  # Change this to the desired level

# Load the mesh from the XDMF file
with XDMFFile(MPI.COMM_WORLD, f"pLaplace_fenics_mesh/mesh_level_{mesh_level}.xdmf", "r") as xdmf_file:
    msh = xdmf_file.read_mesh(name="mesh")

# Define a continuous Galerkin (CG) function space of degree 1
V = fem.functionspace(msh, ("Lagrange", 1))

# --- Define Boundary Conditions ---
msh.topology.create_connectivity(1, 2)  # Ensure facet connectivity for boundary facets
boundary_facets = mesh.exterior_facet_indices(msh.topology)

dofs = fem.locate_dofs_topological(V, 1, boundary_facets)

# print(dofs.shape)

# # Define the Dirichlet boundary condition (u = 0 on boundary nodes)
bc = fem.dirichletbc(ScalarType(0), dofs, V)

# --- Define Variational Problem ---
# Trial and test functions
u = fem.Function(V)
v = ufl.TestFunction(V)
u_trial = ufl.TrialFunction(V)

# Initial guess
indexes_local = range(*u.vector.getOwnershipRange())  # type: ignore
u.vector.setValues(indexes_local, 1e-2 * np.random.rand(len(indexes_local)))  # type: ignore
u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
u.vector.assemble()  # type: ignore
set_bc(u.vector, [bc])  # type: ignore

# Define parameters
p = 3.0  # p-Laplacian parameter
f = fem.Constant(msh, ScalarType(-10.0))  # Constant RHS

# Define the energy functional
energy = (1 / p) * ufl.inner(ufl.grad(u), ufl.grad(u))**(p / 2) * ufl.dx - f * u * ufl.dx  # type: ignore

# Gradient (variational form)
grad_energy = ufl.derivative(energy, u, v)

# Hessian (second derivative)
hessian = ufl.derivative(grad_energy, u, u_trial)

# Forms
energy_form = fem.form(energy)

# Create a new function on the same function space V
u_new = fem.Function(V)

# Define a copy of the energy functional using the new function
energy_new = ufl.replace(energy, {u: u_new})  # Create a copy of the energy functional without rewriting its formula

# Create a new energy form with the new energy functional
energy_form_new = fem.form(energy_new)

grad_form = fem.form(grad_energy)
hessian_form = fem.form(hessian)

# Function to compute energy for given alpha


def compute_energy_alpha(alpha):
    u_new.vector.waxpy(alpha, du.vector, u.vector)  # u_new = u + alpha * du
    u_new.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # Update ghost values
    energy_value = fem.assemble_scalar(energy_form_new)  # Assemble energy
    energy_value = msh.comm.allreduce(energy_value, op=MPI.SUM)  # Reduce to rank 0
    return energy_value


# Initial energy
initial_energy = fem.assemble_scalar(energy_form)  # type: ignore
# reduce energy to rank 0
initial_energy = msh.comm.allreduce(initial_energy, op=MPI.SUM)
if MPI.COMM_WORLD.rank == 0:
    print(f"Initial energy: {initial_energy:.3e}")

# Create matrix and vector for linear problem
du = fem.Function(V)
A = create_matrix(hessian_form)  # type: ignore
L = create_vector(grad_form)  # type: ignore
solver = PETSc.KSP().create(msh.comm)  # type: ignore
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.CG)    # type: ignore # Conjugate Gradient
solver.getPC().setType(PETSc.PC.Type.HYPRE)    # type: ignore # Algebraic Multigrid Preconditioner
solver.setTolerances(rtol=1e-1)  # Set accuracy for linear solver

# Main Newton iteration loop
i = 0
max_iterations = 40
du_norm = []

while i < max_iterations:
    # Assemble Jacobian and residual
    start_time = time.time()
    # assemble Hessian
    A.zeroEntries()
    assemble_matrix(A, hessian_form, bcs=[bc])
    A.assemble()
    # assemble gradient
    with L.localForm() as loc_L:  # zero out L gradient
        loc_L.set(0)
    assemble_vector(L, grad_form)
    apply_lifting(L, [hessian_form], [[bc]], x0=[u.vector])  # type: ignore
    L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
    L.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
    set_bc(L, [bc])
    L.assemble()
    L.scale(-1)
    assembly_time = time.time() - start_time

    # Solve linear problem
    solve_start_time = time.time()
    solver.solve(L, du.vector)  # type: ignore
    du.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
    solve_time = time.time() - solve_start_time

    grad_inf_norm = L.norm(PETSc.NormType.NORM_INFINITY)  # type: ignore

    # Line search using Golden section method
    line_search_start_time = time.time()
    alpha, _ = zlatyrez(compute_energy_alpha, 0, 1, 1e-1)
    # alpha = 1.0
    line_search_time = time.time() - line_search_start_time

    # Update u_{i+1} = u_i + alpha * du_i
    u.vector.axpy(alpha, du.vector)  # type: ignore
    u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore

    i += 1

    # Compute norm of update
    du_norm.append(grad_inf_norm)

    # Debug info (print only on rank 0)
    if MPI.COMM_WORLD.rank == 0:
        print((f"IT {i}: Grad inf-norm = {grad_inf_norm:.3e}, Alpha = {alpha:.3e}, TIMES: Assembly = {assembly_time:.3f} s, " +
               f"Solve = {solve_time:.3f} s, Line search = {line_search_time:.3f} s, Solver its = {solver.getIterationNumber()}"))

    if grad_inf_norm < 1e-6:
        if MPI.COMM_WORLD.rank == 0:
            print("Converged!")
        break

# Final energy (print only on rank 0)
final_energy = fem.assemble_scalar(energy_form)  # type: ignore

all_run_time = time.time() - all_start_time

# reduce energy to rank 0
final_energy = msh.comm.allreduce(final_energy, op=MPI.SUM)
if MPI.COMM_WORLD.rank == 0:
    print(f"Final energy: {final_energy:.3e}")
    print(f"Total run time: {all_run_time:.3f} s")
