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

from src.core.benchmark.repair import build_retry_attempts, needs_solver_repair
from src.core.fenics.nullspace import build_elasticity_nullspace
from src.core.petsc.fenics_tools import ghost_update as _ghost_update
from src.core.petsc.load_step_driver import (
    attempts_from_tuples,
    build_load_step_result,
    run_load_steps,
)
from src.core.petsc.minimizers import newton
from src.core.petsc.trust_ksp import ksp_cg_set_radius

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
              linesearch_tol=1e-3,
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
              trust_max_reject=6,
              trust_subproblem_line_search=False,
              step_time_limit_s=None):
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
        nullspace = build_elasticity_nullspace(
            V,
            A,
            constrained_dofs=constrained_dofs,
        )
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
    trust_ksp_subproblem = (
        use_trust_region and str(ksp_type).lower() in {"stcg", "nash", "gltr"}
    )

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

    def _assemble_and_solve(rhs, sol, trust_radius=None):
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
        if trust_radius is not None:
            ksp_cg_set_radius(ksp, float(trust_radius))
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
                    "trust_radius": (
                        None if trust_radius is None else round(float(trust_radius), 12)
                    ),
                }
            )
        return ksp_its

    def hessian_solve_fn(vec, rhs, sol):
        return _assemble_and_solve(rhs, sol, trust_radius=None)

    def trust_subproblem_solve_fn(vec, rhs, sol, trust_radius):
        return _assemble_and_solve(rhs, sol, trust_radius=trust_radius)

    # ---- time evolution ----
    rotation_per_iter = 4 * 2 * np.pi / total_steps

    x_step_start = x.duplicate()
    setup_time = time.perf_counter() - setup_start
    attempt_specs = attempts_from_tuples(
        build_retry_attempts(
            retry_on_nonfinite=retry_on_nonfinite,
            retry_on_maxit=retry_on_maxit,
            linesearch_interval=linesearch_interval,
            ksp_rtol=ksp_rtol,
            ksp_max_it=ksp_max_it,
            retry_rtol_factor=nonfinite_retry_rtol_factor,
            retry_linesearch_b=nonfinite_retry_linesearch_b,
            retry_ksp_max_it_factor=retry_ksp_max_it_factor,
            min_rtol=1e-8,
        )
    )

    def prepare_step(step_ctx):
        def right_rotation(x_coords):
            vals = np.zeros_like(x_coords)
            vals[0] = 0.0
            vals[1] = (
                np.cos(step_ctx.angle) * x_coords[1]
                + np.sin(step_ctx.angle) * x_coords[2]
                - x_coords[1]
            )
            vals[2] = (
                -np.sin(step_ctx.angle) * x_coords[1]
                + np.cos(step_ctx.angle) * x_coords[2]
                - x_coords[2]
            )
            return vals

        u_right.interpolate(right_rotation)

        set_bc(x, bcs)
        _ghost_update(x)
        x.assemble()
        x.copy(x_step_start)

        if rank == 0 and verbose:
            print(
                f"--- Step {step_ctx.step}/{num_steps}, Angle: {step_ctx.angle:.4f} ---"
            )

    def build_attempts(_step_ctx):
        return attempt_specs

    def solve_attempt(step_ctx, attempt):
        nonlocal force_pc_setup_next

        x_step_start.copy(x)
        set_bc(x, bcs)
        _ghost_update(x)
        x.assemble()

        force_pc_setup_next = True
        ksp.setTolerances(
            rtol=float(attempt.linear_rtol),
            max_it=int(attempt.linear_max_it),
        )
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
            linesearch_tol=linesearch_tol,
            linesearch_interval=attempt.linesearch_interval,
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
            trust_subproblem_solve_fn=(
                trust_subproblem_solve_fn if trust_ksp_subproblem else None
            ),
            trust_subproblem_line_search=trust_subproblem_line_search,
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
            step_time_limit_s=step_time_limit_s,
        )
        step_ctx.state["step_time_raw"] = time.perf_counter() - t_start
        step_ctx.state["last_result"] = result
        return result, float(step_ctx.state["step_time_raw"])

    def should_retry(result, _step_ctx):
        return needs_solver_repair(
            result,
            retry_on_nonfinite=retry_on_nonfinite,
            retry_on_maxit=retry_on_maxit,
        )

    def build_step_record(step_ctx, result, step_time, attempt):
        step_record = {
            "step": int(step_ctx.step),
            "angle": float(step_ctx.angle),
            "time": round(step_time, 4),
            "iters": int(result["nit"]),
            "energy": round(float(result["fun"]), 6),
            "message": str(result["message"]),
            "attempt": str(attempt.name),
            "ksp_rtol_used": float(attempt.linear_rtol),
            "ksp_max_it_used": int(attempt.linear_max_it),
            "linesearch_interval_used": [
                float(attempt.linesearch_interval[0]),
                float(attempt.linesearch_interval[1]),
            ],
        }
        if step_time_limit_s is not None:
            step_record["step_time_limit_s"] = float(step_time_limit_s)
            step_record["kill_switch_exceeded"] = bool(
                step_time > float(step_time_limit_s)
            )
        if save_history:
            step_record["history"] = result.get("history", [])
        if save_linear_timing:
            step_record["linear_timing"] = linear_timing_records.copy()
        return step_record

    def on_retry(step_ctx, attempt, _attempt_idx, _total_attempts):
        if rank == 0 and verbose:
            msg_lower = str(step_ctx.state["last_result"]["message"]).lower()
            reason = "non-finite state" if "maximum number of iterations reached" not in msg_lower else "max Newton iterations"
            print(
                f"Step {step_ctx.step}: {attempt.name} failed with {reason}, "
                f"retrying with tighter linear solve / safer line-search."
            )

    def on_step_complete(step_record, _step_ctx):
        ksp.setTolerances(rtol=ksp_rtol, max_it=ksp_max_it)
        if rank == 0 and verbose:
            print(
                f"Step {step_record['step']} finished: Energy = "
                f"{float(step_record['energy']):.6f}, Iters = {step_record['iters']}"
            )

    def should_stop(step_record, result, step_ctx):
        if step_time_limit_s is not None and bool(step_record.get("kill_switch_exceeded")):
            if rank == 0 and verbose:
                print(
                    f"Stopping early at step {step_ctx.step}: step time "
                    f"{float(step_ctx.state['step_time_raw']):.3f}s exceeded limit "
                    f"{float(step_time_limit_s):.3f}s"
                )
            return True
        if fail_fast and "converged" not in str(result["message"]).lower():
            if rank == 0 and verbose:
                print(
                    f"Stopping early at step {step_ctx.step} due to solver message: "
                    f"{result['message']}"
                )
            return True
        return False

    results = []
    try:
        results = run_load_steps(
            start_step=int(start_step),
            num_steps=int(num_steps),
            rotation_per_step=float(rotation_per_iter),
            prepare_step=prepare_step,
            build_attempts=build_attempts,
            solve_attempt=solve_attempt,
            should_retry=should_retry,
            build_step_record=build_step_record,
            should_stop=should_stop,
            on_retry=on_retry,
            on_step_complete=on_step_complete,
        )
    finally:
        x_step_start.destroy()
        ksp.destroy()
        A.destroy()
        if nullspace is not None:
            nullspace.destroy()

    return build_load_step_result(
        mesh_level=mesh_level,
        total_dofs=total_dofs,
        setup_time=setup_time,
        total_runtime_start=total_runtime_start,
        steps=results,
    )
