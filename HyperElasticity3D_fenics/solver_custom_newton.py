"""
HyperElasticity 3D — FEniCS custom Newton solver logic.

Provides ``run_level()`` using the JAX-version Newton algorithm
(golden-section line search) on top of DOLFINx assembly.
CLI entry point is in ``solve_HE_custom_jaxversion.py``.
"""

import time

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

from tools_petsc4py.minimizers import newton
from tools_petsc4py.fenics_tools import ghost_update as _ghost_update

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
C1 = 38461538.461538464
D1 = 83333333.33333333


def _owned_bc_dofs(*bcs):
    """Return the owned local DOF indices constrained by the given BCs."""
    owned = []
    for bc in bcs:
        dofs, pos = bc.dof_indices()
        if pos:
            owned.append(np.asarray(dofs[:pos], dtype=np.int32))
    if not owned:
        return np.empty(0, dtype=np.int32)
    return np.unique(np.concatenate(owned))


def build_nullspace(V, A, constrained_dofs=None):
    """Build the 6 rigid body modes for 3D elasticity."""
    x = V.tabulate_dof_coordinates()
    index_map = V.dofmap.index_map
    x_owned = x[:index_map.size_local, :]
    constrained = np.empty(0, dtype=np.int32)
    if constrained_dofs is not None:
        constrained = np.asarray(constrained_dofs, dtype=np.int32)

    vecs = [A.createVecLeft() for _ in range(6)]

    for vec in vecs:
        vec.getArray()[:] = 0.0

    # Translations
    for i in range(3):
        vecs[i].getArray()[i::3] = 1.0

    # Rotations
    vecs[3].getArray()[1::3] = -x_owned[:, 2]
    vecs[3].getArray()[2::3] = x_owned[:, 1]

    vecs[4].getArray()[0::3] = x_owned[:, 2]
    vecs[4].getArray()[2::3] = -x_owned[:, 0]

    vecs[5].getArray()[0::3] = -x_owned[:, 1]
    vecs[5].getArray()[1::3] = x_owned[:, 0]

    if constrained.size:
        n_local = vecs[0].getLocalSize()
        constrained = constrained[(constrained >= 0) & (constrained < n_local)]
        for vec in vecs:
            vec.getArray()[constrained] = 0.0

    return PETSc.NullSpace().create(vectors=vecs)


def _set_initial_from_jax_npz(V, u, npz_path, init_step):
    data = np.load(npz_path)
    coords_jax = data["coords"]
    u_steps = data["u_full_steps"]

    if init_step < 1 or init_step > u_steps.shape[0]:
        raise ValueError(f"init_step={init_step} out of range [1, {u_steps.shape[0]}]")

    u_nodes_jax = u_steps[init_step - 1]
    if coords_jax.shape[0] != u_nodes_jax.shape[0]:
        raise ValueError("JAX test data inconsistent: coords and displacement node counts differ")

    disp_nodes = u_nodes_jax - coords_jax
    mapping = {tuple(np.round(c, 12)): disp_nodes[i] for i, c in enumerate(coords_jax)}

    def _interp_from_map(x):
        vals = np.zeros((3, x.shape[1]), dtype=np.float64)
        missing = 0
        for i in range(x.shape[1]):
            key = tuple(np.round([x[0, i], x[1, i], x[2, i]], 12))
            if key not in mapping:
                missing += 1
                continue
            vals[:, i] = mapping[key]
        if missing > 0:
            raise RuntimeError(f"Failed to map {missing} interpolation points from JAX data")
        return vals

    u.interpolate(_interp_from_map)
    u.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT,
        mode=PETSc.ScatterMode.FORWARD,
    )


def run_level(mesh_level, num_steps=1, verbose=True, maxit=100, start_step=1,
              init_npz="", init_step=0, linesearch_interval=(-0.5, 2.0),
              use_abs_det=False, ksp_type="gmres", pc_type="hypre",
              ksp_rtol=1e-3, ksp_max_it=10000, use_near_nullspace=True,
              total_steps=24, hypre_nodal_coarsen=6, hypre_vec_interp_variant=3,
              hypre_strong_threshold=None, hypre_coarsen_type="",
              tolf=1e-4, tolg=1e-3, tolg_rel=0.0, tolx_rel=1e-6, tolx_abs=1e-10,
              save_history=False, save_linear_timing=False,
              pc_setup_on_ksp_cap=False,
              gamg_threshold=-1.0, gamg_agg_nsmooths=1,
              gamg_set_coordinates=True,
              require_all_convergence=False,
              fail_fast=True, retry_on_nonfinite=True,
              retry_on_maxit=True,
              nonfinite_retry_rtol_factor=0.1, nonfinite_retry_linesearch_b=1.0,
              retry_ksp_max_it_factor=2.0,
              use_trust_region=False,
              trust_radius_init=1.0,
              trust_radius_min=1e-8,
              trust_radius_max=1e6,
              trust_shrink=0.5,
              trust_expand=1.5,
              trust_eta_shrink=0.05,
              trust_eta_expand=0.75,
              trust_max_reject=6):
    """Run JAX-version Newton solver for one HE mesh level.

    Returns dict with: mesh_level, total_dofs, time, iters, energy, message
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    total_runtime_start = time.perf_counter()
    setup_start = time.perf_counter()

    # ---- mesh: structured grid on [0, 0.4] x [-0.005, 0.005] x [-0.005, 0.005] ----
    Nx = 80 * 2**(mesh_level - 1)
    Ny = 2 * 2**(mesh_level - 1)
    Nz = 2 * 2**(mesh_level - 1)

    msh = mesh.create_box(
        comm, [[0.0, -0.005, -0.005], [0.4, 0.005, 0.005]], [Nx, Ny, Nz],
        cell_type=mesh.CellType.tetrahedron,
    )

    V = fem.functionspace(msh, ("Lagrange", 1, (3,)))
    total_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

    # ---- Dirichlet BC ----
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
    constrained_dofs = _owned_bc_dofs(*bcs)

    # ---- variational forms ----
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    w = ufl.TrialFunction(V)

    d = len(u)
    I = ufl.Identity(d)
    F_def = I + ufl.grad(u)
    J_det = ufl.det(F_def)
    J_det_for_energy = abs(J_det) if use_abs_det else J_det
    I1 = ufl.inner(F_def, F_def)

    W = C1 * (I1 - 3 - 2 * ufl.ln(J_det_for_energy)) + D1 * (J_det_for_energy - 1)**2
    J_energy = W * ufl.dx

    dJ = ufl.derivative(J_energy, u, v)
    ddJ = ufl.derivative(dJ, u, w)

    energy_form = fem.form(J_energy)
    grad_form = fem.form(dJ)
    hessian_form = fem.form(ddJ)

    u_ls = fem.Function(V)
    energy_ls = ufl.replace(J_energy, {u: u_ls})
    energy_ls_form = fem.form(energy_ls)

    # ---- initial guess ----
    if init_npz:
        if init_step <= 0:
            raise ValueError("init_step must be >= 1 when init_npz is provided")
        _set_initial_from_jax_npz(V, u, init_npz, init_step)
    else:
        u.x.array[:] = 0.0
    x = u.x.petsc_vec
    set_bc(x, bcs)
    _ghost_update(x)
    x.assemble()

    # ---- pre-allocate Hessian matrix and KSP ----
    if rank == 0 and verbose:
        print("Creating matrix...", flush=True)
    A = create_matrix(hessian_form)
    use_hypre_system_amg = (
        pc_type == "hypre"
        and hypre_nodal_coarsen >= 0
        and hypre_vec_interp_variant >= 0
    )
    if pc_type != "hypre" or use_hypre_system_amg:
        # Keep the old fast HYPRE-default path scalar unless the user explicitly
        # requests systems AMG (or we are using a different PC that benefits from
        # the natural vector block size).
        A.setBlockSize(3)

    nullspace = None
    if use_near_nullspace:
        if rank == 0 and verbose:
            print("Building nullspace...", flush=True)
        nullspace = build_nullspace(V, A, constrained_dofs=constrained_dofs)
        if rank == 0 and verbose:
            print("Setting near nullspace...", flush=True)
        A.setNearNullSpace(nullspace)

    if rank == 0 and verbose:
        print("Creating KSP...", flush=True)
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setType(ksp_type)
    pc = ksp.getPC()
    pc.setType(pc_type)
    if pc_type == "hypre":
        pc.setHYPREType("boomeramg")

    opts = PETSc.Options()
    if pc_type == "hypre":
        if hypre_nodal_coarsen >= 0:
            opts["pc_hypre_boomeramg_nodal_coarsen"] = hypre_nodal_coarsen
        if hypre_vec_interp_variant >= 0:
            opts["pc_hypre_boomeramg_vec_interp_variant"] = hypre_vec_interp_variant
        if hypre_strong_threshold is not None:
            opts["pc_hypre_boomeramg_strong_threshold"] = hypre_strong_threshold
        if hypre_coarsen_type:
            opts["pc_hypre_boomeramg_coarsen_type"] = hypre_coarsen_type
        if (
            use_near_nullspace
            and rank == 0
            and verbose
            and (hypre_nodal_coarsen < 0 or hypre_vec_interp_variant < 0)
        ):
            print(
                "Warning: HYPRE BoomerAMG ignores MatSetNearNullSpace() unless "
                "both hypre_nodal_coarsen and hypre_vec_interp_variant are set.",
                flush=True,
            )

    gamg_coords = None
    gamg_coords_kept = None
    if pc_type == "gamg":
        opts["pc_gamg_threshold"] = gamg_threshold
        opts["pc_gamg_agg_nsmooths"] = gamg_agg_nsmooths
        if gamg_set_coordinates:
            index_map = V.dofmap.index_map
            gamg_coords = V.tabulate_dof_coordinates()[:index_map.size_local, :]

    ksp.setFromOptions()
    ksp.setTolerances(rtol=ksp_rtol, max_it=ksp_max_it)

    # ------------------------------------------------------------------
    # Callbacks for tools_petsc4py.minimizers.newton
    # ------------------------------------------------------------------
    linear_timing_records = []
    force_pc_setup_next = True

    def energy_fn(vec):
        """J(u) at an arbitrary PETSc Vec (globally reduced scalar)."""
        vec.copy(u_ls.x.petsc_vec)
        u_ls.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        local_val = fem.assemble_scalar(energy_ls_form)
        return comm.allreduce(local_val, op=MPI.SUM)

    def gradient_fn(vec, g):
        """Assemble ∇J into *g* (BCs applied, ghost-updated)."""
        with g.localForm() as g_loc:
            g_loc.set(0.0)
        assemble_vector(g, grad_form)
        apply_lifting(g, [hessian_form], [bcs], x0=[vec])
        g.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(g, bcs, vec)

    def hessian_solve_fn(vec, rhs, sol):
        """Assemble Hessian, solve H · sol = rhs. Return KSP iters."""
        nonlocal force_pc_setup_next, gamg_coords, gamg_coords_kept
        if rank == 0 and verbose:
            print("Assembling Hessian...", flush=True)
        t0 = time.perf_counter()
        A.zeroEntries()
        assemble_matrix(A, hessian_form, bcs=bcs)
        A.assemble()
        if nullspace is not None:
            # Mirror the SNES Jacobian path and make the intent explicit:
            # the assembled operator passed to KSP always carries the RBM modes.
            A.setNearNullSpace(nullspace)
        t1 = time.perf_counter()
        if rank == 0 and verbose:
            print("Setting operators...", flush=True)
        ksp.setOperators(A)
        if gamg_coords is not None:
            pc.setCoordinates(gamg_coords)
            gamg_coords_kept = gamg_coords
            gamg_coords = None
        t2 = time.perf_counter()
        if pc_setup_on_ksp_cap:
            if force_pc_setup_next:
                ksp.setUp()
                force_pc_setup_next = False
            t3 = time.perf_counter()
        else:
            ksp.setUp()
            t3 = time.perf_counter()
        if rank == 0 and verbose:
            print("Solving KSP...", flush=True)
        ksp.solve(rhs, sol)
        t4 = time.perf_counter()
        if rank == 0 and verbose:
            print("KSP solved.", flush=True)

        ksp_its = ksp.getIterationNumber()
        if pc_setup_on_ksp_cap and ksp_its >= ksp_max_it:
            force_pc_setup_next = True

        if save_linear_timing:
            linear_timing_records.append(
                {
                    "assemble_time": round(t1 - t0, 6),
                    "setop_time": round(t2 - t1, 6),
                    "pc_setup_time": round(t3 - t2, 6),
                    "solve_time": round(t4 - t3, 6),
                    "ksp_its": ksp_its,
                    "linear_total_time": round(t4 - t0, 6),
                }
            )
        return ksp_its

    # ---- time evolution ----
    rotation_per_iter = 4 * 2 * np.pi / total_steps

    results = []
    x_step_start = x.duplicate()
    setup_time = time.perf_counter() - setup_start

    for step in range(start_step, start_step + num_steps):
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
        x.copy(x_step_start)

        if rank == 0 and verbose:
            print(f"--- Step {step}/{num_steps}, Angle: {angle:.4f} ---")

        attempt_configs = [
            ("primary", linesearch_interval, ksp_rtol, ksp_max_it),
        ]
        if retry_on_nonfinite or retry_on_maxit:
            ls_a, ls_b = linesearch_interval
            retry_ls_b = min(ls_b, nonfinite_retry_linesearch_b)
            if retry_ls_b <= ls_a:
                retry_ls_b = ls_b
            retry_rtol = max(1e-8, ksp_rtol * nonfinite_retry_rtol_factor)
            retry_kmax = max(ksp_max_it + 1, int(round(ksp_max_it * retry_ksp_max_it_factor)))
            if retry_ls_b < ls_b or retry_rtol < ksp_rtol or retry_kmax > ksp_max_it:
                attempt_configs.append(
                    ("repair", (ls_a, retry_ls_b), retry_rtol, retry_kmax)
                )

        result = None
        total_time = 0.0
        used_attempt = "primary"
        used_linesearch = linesearch_interval
        used_ksp_rtol = ksp_rtol
        used_ksp_max_it = ksp_max_it

        for attempt_idx, (attempt_name, ls_interval, ksp_rtol_attempt,
                          ksp_max_it_attempt) in enumerate(attempt_configs):
            x_step_start.copy(x)
            set_bc(x, bcs)
            _ghost_update(x)
            x.assemble()

            force_pc_setup_next = True
            ksp.setTolerances(rtol=ksp_rtol_attempt, max_it=ksp_max_it_attempt)
            if save_linear_timing:
                linear_timing_records.clear()

            t_start = time.perf_counter()
            result = newton(
                energy_fn,
                gradient_fn,
                hessian_solve_fn,
                x,
                tolf=tolf,
                tolg=tolg,
                tolg_rel=tolg_rel,
                linesearch_tol=1e-3,
                linesearch_interval=ls_interval,
                maxit=maxit,
                tolx_rel=tolx_rel,
                tolx_abs=tolx_abs,
                require_all_convergence=require_all_convergence,
                fail_on_nonfinite=True,
                verbose=verbose,
                comm=comm,
                ghost_update_fn=_ghost_update,
                project_fn=lambda vec: set_bc(vec, bcs),
                hessian_matvec_fn=lambda _x, vin, vout: A.mult(vin, vout),
                save_history=save_history,
                trust_region=use_trust_region,
                trust_radius_init=trust_radius_init,
                trust_radius_min=trust_radius_min,
                trust_radius_max=trust_radius_max,
                trust_shrink=trust_shrink,
                trust_expand=trust_expand,
                trust_eta_shrink=trust_eta_shrink,
                trust_eta_expand=trust_eta_expand,
                trust_max_reject=trust_max_reject,
            )
            total_time = time.perf_counter() - t_start
            used_attempt = attempt_name
            used_linesearch = ls_interval
            used_ksp_rtol = ksp_rtol_attempt
            used_ksp_max_it = ksp_max_it_attempt

            msg_lower = result["message"].lower()
            has_nonfinite = "non-finite" in msg_lower or "nan" in msg_lower
            hit_newton_maxit = "maximum number of iterations reached" in msg_lower
            need_retry = (retry_on_nonfinite and has_nonfinite) or (retry_on_maxit and hit_newton_maxit)
            if need_retry and attempt_idx + 1 < len(attempt_configs):
                if rank == 0 and verbose:
                    reason = "non-finite state" if has_nonfinite else "max Newton iterations"
                    print(
                        f"Step {step}: {attempt_name} failed with {reason}, "
                        f"retrying with tighter linear solve / safer line-search."
                    )
                continue
            break

        ksp.setTolerances(rtol=ksp_rtol, max_it=ksp_max_it)

        final_energy = result["fun"]

        step_record = {
            "step": step,
            "angle": angle,
            "time": round(total_time, 4),
            "iters": result["nit"],
            "energy": round(final_energy, 6),
            "message": result["message"],
            "attempt": used_attempt,
            "ksp_rtol_used": used_ksp_rtol,
            "ksp_max_it_used": int(used_ksp_max_it),
            "linesearch_interval_used": [float(used_linesearch[0]), float(used_linesearch[1])],
        }
        if save_history:
            step_record["history"] = result.get("history", [])
        if save_linear_timing:
            step_record["linear_timing"] = linear_timing_records.copy()
            linear_timing_records.clear()
        results.append(step_record)

        if rank == 0 and verbose:
            print(f"Step {step} finished: Energy = {final_energy:.6f}, Iters = {result['nit']}")
        if fail_fast and "converged" not in result["message"].lower():
            if rank == 0 and verbose:
                print(f"Stopping early at step {step} due to solver message: {result['message']}")
            break

    # ---- clean up PETSc objects ----
    x_step_start.destroy()
    ksp.destroy()
    A.destroy()
    if nullspace is not None:
        nullspace.destroy()

    return {
        "mesh_level": mesh_level,
        "total_dofs": total_dofs,
        "setup_time": round(setup_time, 6),
        "solve_time_total": round(sum(step["time"] for step in results), 6),
        "total_time": round(time.perf_counter() - total_runtime_start, 6),
        "steps": results
    }
