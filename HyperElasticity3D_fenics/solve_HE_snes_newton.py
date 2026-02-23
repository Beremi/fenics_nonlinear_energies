"""
HyperElasticity 3D solver — SNES Newton with PETSc linear algebra.

Usage:
  Serial:   python3 HyperElasticity3D_fenics/solve_HE_snes_newton.py
  Parallel: mpirun -n <nprocs> python3 HyperElasticity3D_fenics/solve_HE_snes_newton.py
"""
import time
import json
import argparse
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_matrix,
    set_bc,
)

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
C1 = 38461538.461538464
D1 = 83333333.33333333

# ---------------------------------------------------------------------------
# Helper: PETSc ghost update for DOLFINx vectors
# ---------------------------------------------------------------------------


def _ghost_update(v):
    """INSERT-mode forward scatter (owned → ghosts)."""
    v.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


def build_nullspace(V, A, gram_schmidt=False):
    """Build the 6 rigid body modes for 3D elasticity.

    Parameters
    ----------
    gram_schmidt : bool
        If True, apply coordinate centering + Gram-Schmidt orthonormalisation
        before passing vectors to HYPRE.  If False (default), pass the raw
        un-centered RBMs exactly as the custom solver does — this is required
        for HYPRE nodal_coarsen=6 / vec_interp_variant=3 to work reliably.
    """
    x = V.tabulate_dof_coordinates()
    index_map = V.dofmap.index_map
    x_owned = x[:index_map.size_local, :]

    if gram_schmidt:
        # Center coordinates so translations are orthogonal to rotations
        x_mean = np.zeros(3)
        for d in range(3):
            local_sum = float(np.sum(x_owned[:, d]))
            local_count = float(len(x_owned))
            global_sum = A.comm.tompi4py().allreduce(local_sum)
            global_count = A.comm.tompi4py().allreduce(local_count)
            x_mean[d] = global_sum / global_count
        xc = x_owned - x_mean
    else:
        xc = x_owned

    vecs = [A.createVecLeft() for _ in range(6)]

    for vec in vecs:
        vec.getArray()[:] = 0.0

    for i in range(3):
        vecs[i].getArray()[i::3] = 1.0

    vecs[3].getArray()[1::3] = -xc[:, 2]
    vecs[3].getArray()[2::3] = xc[:, 1]

    vecs[4].getArray()[0::3] = xc[:, 2]
    vecs[4].getArray()[2::3] = -xc[:, 0]

    vecs[5].getArray()[0::3] = -xc[:, 1]
    vecs[5].getArray()[1::3] = xc[:, 0]

    if gram_schmidt:
        # Gram-Schmidt orthonormalization
        for i, v in enumerate(vecs):
            for j in range(i):
                alpha = v.dot(vecs[j])
                v.axpy(-alpha, vecs[j])
            v.normalize()

    return PETSc.NullSpace().create(vectors=vecs)

# ---------------------------------------------------------------------------
# Solver for a single mesh level
# ---------------------------------------------------------------------------


def run_level(mesh_level, num_steps=1, total_steps=None, snes_type="newtonls", linesearch="basic",
              ksp_type="gmres", pc_type="hypre", ksp_rtol=1e-3, ksp_max_it=10000,
              snes_atol=1e-5, use_objective=False, verbose=True,
              use_near_nullspace=True, hypre_nodal_coarsen=-1, hypre_vec_interp_variant=-1,
              nullspace_gram_schmidt=False, stop_on_fail=False):
    """Run SNES Newton solver for one HE mesh level.

    total_steps controls the rotation per iteration: the full 4*2*pi rotation
    is divided into total_steps equal steps (default: num_steps, i.e. the steps
    run span the entire rotation). Set total_steps > num_steps to use a finer
    step size and run only the first num_steps of them.
    """
    """Run SNES Newton solver for one HE mesh level."""
    comm = MPI.COMM_WORLD
    rank = comm.rank

    Nx = 80 * 2**(mesh_level - 1)
    Ny = 2 * 2**(mesh_level - 1)
    Nz = 2 * 2**(mesh_level - 1)

    msh = mesh.create_box(
        comm, [[0.0, -0.005, -0.005], [0.4, 0.005, 0.005]], [Nx, Ny, Nz],
        cell_type=mesh.CellType.tetrahedron,
    )

    V = fem.functionspace(msh, ("Lagrange", 1, (3,)))
    total_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    left_facets = mesh.locate_entities_boundary(msh, 2, left_boundary)
    left_dofs = fem.locate_dofs_topological(V, 2, left_facets)
    bc_left = fem.dirichletbc(np.zeros(3, dtype=ScalarType), left_dofs, V)

    def right_boundary(x):
        return np.isclose(x[0], 0.4)

    right_facets = mesh.locate_entities_boundary(msh, 2, right_boundary)
    right_dofs = fem.locate_dofs_topological(V, 2, right_facets)

    u_right = fem.Function(V)
    bc_right = fem.dirichletbc(u_right, right_dofs)

    bcs = [bc_left, bc_right]

    u = fem.Function(V)
    v = ufl.TestFunction(V)
    w = ufl.TrialFunction(V)

    d = len(u)
    I = ufl.Identity(d)
    F_def = I + ufl.grad(u)
    J_det = ufl.det(F_def)
    I1 = ufl.inner(F_def, F_def)

    W = C1 * (I1 - 3 - 2 * ufl.ln(J_det)) + D1 * (J_det - 1)**2
    J_energy = W * ufl.dx

    dJ = ufl.derivative(J_energy, u, v)
    ddJ = ufl.derivative(dJ, u, w)

    energy_form = fem.form(J_energy)
    grad_form = fem.form(dJ)
    hessian_form = fem.form(ddJ)

    u.x.array[:] = 0.0
    x = u.x.petsc_vec
    set_bc(x, bcs)
    _ghost_update(x)
    x.assemble()

    A = create_matrix(hessian_form)
    nullspace = None
    if use_near_nullspace:
        nullspace = build_nullspace(V, A, gram_schmidt=nullspace_gram_schmidt)
        A.setNearNullSpace(nullspace)

    b = x.duplicate()

    # ---- SNES setup ----
    snes = PETSc.SNES().create(comm)
    snes.setType(snes_type)
    snes.setOptionsPrefix("he_")

    opts = PETSc.Options()
    if snes_type == "newtonls":
        opts["he_snes_linesearch_type"] = linesearch

    snes.setTolerances(atol=snes_atol, rtol=1e-50, stol=1e-50, max_it=100)

    ksp = snes.getKSP()
    ksp.setType(ksp_type)
    pc = ksp.getPC()
    pc.setType(pc_type)
    if pc_type == "hypre":
        pc.setHYPREType("boomeramg")
        if hypre_nodal_coarsen >= 0:
            opts["he_pc_hypre_boomeramg_nodal_coarsen"] = hypre_nodal_coarsen
        if hypre_vec_interp_variant >= 0:
            opts["he_pc_hypre_boomeramg_vec_interp_variant"] = hypre_vec_interp_variant
    ksp.setTolerances(rtol=ksp_rtol, max_it=ksp_max_it)

    snes.setFromOptions()

    # ---- Callbacks ----
    def snes_objective(snes, vec):
        vec.copy(u.x.petsc_vec)
        _ghost_update(u.x.petsc_vec)
        local_val = fem.assemble_scalar(energy_form)
        return comm.allreduce(local_val, op=MPI.SUM)

    def snes_residual(snes, vec, f):
        vec.copy(u.x.petsc_vec)
        _ghost_update(u.x.petsc_vec)
        with f.localForm() as f_loc:
            f_loc.set(0.0)
        assemble_vector(f, grad_form)
        apply_lifting(f, [hessian_form], [bcs], x0=[u.x.petsc_vec], alpha=1.0)
        f.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(f, bcs, u.x.petsc_vec, alpha=1.0)

    def snes_jacobian(snes, vec, J_mat, P_mat):
        vec.copy(u.x.petsc_vec)
        _ghost_update(u.x.petsc_vec)
        J_mat.zeroEntries()
        assemble_matrix(J_mat, hessian_form, bcs=bcs)
        J_mat.assemble()
        if P_mat.handle != J_mat.handle:
            P_mat.zeroEntries()
            assemble_matrix(P_mat, hessian_form, bcs=bcs)
            P_mat.assemble()
        if nullspace is not None:
            J_mat.setNearNullSpace(nullspace)
            if P_mat.handle != J_mat.handle:
                P_mat.setNearNullSpace(nullspace)

    snes.setFunction(snes_residual, b)
    snes.setJacobian(snes_jacobian, A, A)
    if use_objective:
        snes.setObjective(snes_objective)

    # ---- time evolution ----
    effective_total = total_steps if total_steps is not None else num_steps
    rotation_per_iter = 4 * 2 * np.pi / effective_total
    results = []

    for step in range(1, num_steps + 1):
        angle = step * rotation_per_iter

        def right_rotation(x_coords):
            vals = np.zeros_like(x_coords)
            vals[0] = 0.0
            vals[1] = np.cos(angle) * x_coords[1] + np.sin(angle) * x_coords[2] - x_coords[1]
            vals[2] = -np.sin(angle) * x_coords[1] + np.cos(angle) * x_coords[2] - x_coords[2]
            return vals

        u_right.interpolate(right_rotation)

        set_bc(x, bcs)
        _ghost_update(x)
        x.assemble()

        if rank == 0 and verbose:
            print(f"--- Step {step}/{num_steps}, Angle: {angle:.4f} ---")

        t_start = time.perf_counter()
        snes.solve(None, x)
        total_time = time.perf_counter() - t_start

        final_energy = snes_objective(snes, x)

        results.append({
            "step": step,
            "angle": angle,
            "time": round(total_time, 4),
            "iters": snes.getIterationNumber(),
            "linear_iters": snes.getLinearSolveIterations(),
            "energy": round(final_energy, 6),
            "reason": snes.getConvergedReason(),
        })

        if rank == 0 and verbose:
            print(
                f"Step {step} finished: Energy = {final_energy:.6f}, "
                f"Iters = {snes.getIterationNumber()}, "
                f"Reason = {snes.getConvergedReason()}"
            )

        if stop_on_fail and snes.getConvergedReason() < 0:
            if rank == 0 and verbose:
                print(f"Stopping at step {step} due to divergence (reason={snes.getConvergedReason()}).")
            break

    snes.destroy()
    A.destroy()
    b.destroy()
    if nullspace is not None:
        nullspace.destroy()

    return {
        "mesh_level": mesh_level,
        "total_dofs": total_dofs,
        "config": {
            "snes_type": snes_type,
            "linesearch": linesearch,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": ksp_rtol,
            "ksp_max_it": ksp_max_it,
            "snes_atol": snes_atol,
            "use_objective": use_objective,
            "use_near_nullspace": use_near_nullspace,
            "hypre_nodal_coarsen": hypre_nodal_coarsen,
            "hypre_vec_interp_variant": hypre_vec_interp_variant,
            "nullspace_gram_schmidt": nullspace_gram_schmidt,
        },
        "steps": results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--snes_type", type=str, default="newtonls")
    parser.add_argument("--linesearch", type=str, default="basic")
    parser.add_argument("--ksp_type", type=str, default="gmres")
    parser.add_argument("--pc_type", type=str, default="hypre")
    parser.add_argument("--ksp_rtol", type=float, default=1e-3)
    parser.add_argument("--ksp_max_it", type=int, default=10000)
    parser.add_argument("--snes_atol", type=float, default=1e-5)
    parser.add_argument("--use_objective", action="store_true")
    parser.add_argument("--no_near_nullspace", action="store_true")
    parser.add_argument("--hypre_nodal_coarsen", type=int, default=-1,
                        help="BoomerAMG nodal coarsen (-1 to skip setting)")
    parser.add_argument("--hypre_vec_interp_variant", type=int, default=-1,
                        help="BoomerAMG vec interp variant (-1 to skip setting)")
    parser.add_argument("--nullspace_gram_schmidt", action="store_true",
                        help="Apply centering + Gram-Schmidt to RBM nullspace vectors")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--stop_on_fail", action="store_true",
                        help="Stop loading sequence at first diverged step")
    parser.add_argument("--total_steps", type=int, default=None,
                        help="Total steps spanning full 4*2pi rotation (controls step size); "
                             "default: same as --steps")
    args, _ = parser.parse_known_args()

    res = run_level(args.level, num_steps=args.steps, total_steps=args.total_steps,
                    snes_type=args.snes_type, linesearch=args.linesearch,
                    ksp_type=args.ksp_type, pc_type=args.pc_type,
                    ksp_rtol=args.ksp_rtol, ksp_max_it=args.ksp_max_it,
                    snes_atol=args.snes_atol,
                    use_objective=args.use_objective, verbose=not args.quiet,
                    use_near_nullspace=not args.no_near_nullspace,
                    hypre_nodal_coarsen=args.hypre_nodal_coarsen,
                    hypre_vec_interp_variant=args.hypre_vec_interp_variant,
                    nullspace_gram_schmidt=args.nullspace_gram_schmidt,
                    stop_on_fail=args.stop_on_fail)

    if MPI.COMM_WORLD.rank == 0:
        print(json.dumps(res, indent=2))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(res, f, indent=2)
