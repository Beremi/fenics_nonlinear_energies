"""
HyperElasticity 3D solver — custom Newton (JAX-version algorithm) with PETSc linear algebra.

Re-implements the JAX Newton minimiser (tools/minimizers.py) on top of PETSc
objects so it runs under MPI with DOLFINx assembly.

Energy:
    W(F) = C1 * (I1 - 3 - 2*log(det(F))) + D1 * (det(F) - 1)^2
    J(u) = ∫_Ω W(F(u)) dx

with homogeneous Dirichlet BCs on x=0 and prescribed rotation on x=0.4.

Usage:
  Serial:   python3 HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py
  Parallel: mpirun -n <nprocs> python3 HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py
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

from tools_petsc4py.minimizers import newton

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


def build_nullspace(V, A):
    """Build the 6 rigid body modes for 3D elasticity."""
    x = V.tabulate_dof_coordinates()
    index_map = V.dofmap.index_map
    x_owned = x[:index_map.size_local, :]

    vecs = [A.createVecLeft() for _ in range(6)]

    for vec in vecs:
        vec.getArray()[:] = 0.0

    # Translations
    for i in range(3):
        vecs[i].getArray()[i::3] = 1.0

    # Rotations
    # Mode 3: rotation about x-axis
    vecs[3].getArray()[1::3] = -x_owned[:, 2]
    vecs[3].getArray()[2::3] = x_owned[:, 1]

    # Mode 4: rotation about y-axis
    vecs[4].getArray()[0::3] = x_owned[:, 2]
    vecs[4].getArray()[2::3] = -x_owned[:, 0]

    # Mode 5: rotation about z-axis
    vecs[5].getArray()[0::3] = -x_owned[:, 1]
    vecs[5].getArray()[1::3] = x_owned[:, 0]

    return PETSc.NullSpace().create(vectors=vecs)


def _snes_reason_label(reason_code):
    """Return PETSc SNES converged/diverged reason label for an integer code."""
    reason_int = int(reason_code)
    for name, value in PETSc.SNES.ConvergedReason.__dict__.items():
        if name.startswith("_"):
            continue
        if isinstance(value, int) and int(value) == reason_int:
            return name
    return "UNKNOWN"

# ---------------------------------------------------------------------------
# Solver for a single mesh level
# ---------------------------------------------------------------------------


def _set_initial_from_jax_npz(V, u, npz_path, init_step):
    data = np.load(npz_path)
    coords_jax = data["coords"]
    u_steps = data["u_full_steps"]

    if init_step < 1 or init_step > u_steps.shape[0]:
        raise ValueError(f"init_step={init_step} out of range [1, {u_steps.shape[0]}]")

    u_nodes_jax = u_steps[init_step - 1]
    if coords_jax.shape[0] != u_nodes_jax.shape[0]:
        raise ValueError("JAX test data inconsistent: coords and displacement node counts differ")

    # JAX stores full deformed coordinates; FEniCS unknown is displacement.
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
              tolf=1e-4, tolg=1e-3, tolg_rel=1e-3, tolx_rel=1e-3, tolx_abs=1e-10,
              save_history=False, save_linear_timing=False,
              pc_setup_on_ksp_cap=False,
              gamg_threshold=-1.0, gamg_agg_nsmooths=1,
              gamg_set_coordinates=True,
              fail_fast=True, retry_on_nonfinite=True,
              retry_on_maxit=True,
              nonfinite_retry_rtol_factor=0.1, nonfinite_retry_linesearch_b=1.0,
              retry_ksp_max_it_factor=2.0,
              use_trust_region=True,
              trust_radius_init=1.0,
              trust_radius_min=1e-8,
              trust_radius_max=1e6,
              trust_shrink=0.5,
              trust_expand=1.5,
              trust_eta_shrink=0.05,
              trust_eta_expand=0.75,
              trust_max_reject=6,
              trust_region_retry_on_maxit=False,
              trust_region_retry_maxit=100,
              trust_region_retry_rtol_factor=0.1,
              trust_region_retry_ksp_max_it_factor=2.0):
    """Run JAX-version Newton solver for one HE mesh level.

    Returns dict with: mesh_level, total_dofs, time, iters, energy, message
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

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
    # Left face (x = 0): u = 0
    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    left_facets = mesh.locate_entities_boundary(msh, 2, left_boundary)
    left_dofs = fem.locate_dofs_topological(V, 2, left_facets)
    bc_left = fem.dirichletbc(np.zeros(3, dtype=ScalarType), left_dofs, V)

    # Right face (x = 0.4): prescribed rotation
    def right_boundary(x):
        return np.isclose(x[0], 0.4)

    right_facets = mesh.locate_entities_boundary(msh, 2, right_boundary)
    right_dofs = fem.locate_dofs_topological(V, 2, right_facets)

    u_right = fem.Function(V)
    bc_right = fem.dirichletbc(u_right, right_dofs)

    bcs = [bc_left, bc_right]

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

    # Energy density
    W = C1 * (I1 - 3 - 2 * ufl.ln(J_det_for_energy)) + D1 * (J_det_for_energy - 1)**2
    J_energy = W * ufl.dx

    dJ = ufl.derivative(J_energy, u, v)
    ddJ = ufl.derivative(dJ, u, w)

    energy_form = fem.form(J_energy)
    grad_form = fem.form(dJ)
    hessian_form = fem.form(ddJ)

    # ---- form for energy evaluation at arbitrary point ----
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

    # Set block size for vector problems (3 DOFs per node for 3D elasticity)
    A.setBlockSize(3)

    nullspace = None
    if use_near_nullspace:
        if rank == 0 and verbose:
            print("Building nullspace...", flush=True)
        nullspace = build_nullspace(V, A)
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

    # Tell HYPRE it's 3D elasticity
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

    # GAMG-specific settings for elasticity
    gamg_coords = None
    gamg_coords_kept = None
    if pc_type == "gamg":
        opts["pc_gamg_threshold"] = gamg_threshold
        opts["pc_gamg_agg_nsmooths"] = gamg_agg_nsmooths
        # Store coordinates for deferred PCSetCoordinates (needs operators set first)
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
        t1 = time.perf_counter()
        if rank == 0 and verbose:
            print("Setting operators...", flush=True)
        ksp.setOperators(A)
        # Set GAMG coordinates after operators are set (only needed once)
        if gamg_coords is not None:
            pc.setCoordinates(gamg_coords)
            gamg_coords_kept = gamg_coords  # attach to outer scope to prevent GC!
            gamg_coords = None  # only set once
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

    def hessian_matvec_fn(_x_current, vin, vout):
        """Apply currently assembled Hessian to a vector."""
        A.mult(vin, vout)

    def trust_region_retry_fn(ksp_rtol_attempt, ksp_max_it_attempt, maxit_attempt):
        """Fallback solve via SNES trust-region Newton."""
        tr_prefix = "hetr_"

        # Dedicated SNES objects so custom-Newton KSP state is untouched.
        A_tr = create_matrix(hessian_form)
        A_tr.setBlockSize(3)
        if nullspace is not None:
            A_tr.setNearNullSpace(nullspace)
        b_tr = x.duplicate()

        snes = PETSc.SNES().create(comm)
        snes.setType("newtontr")
        snes.setOptionsPrefix(tr_prefix)

        g0 = x.duplicate()
        gradient_fn(x, g0)
        grad0 = g0.norm(PETSc.NormType.NORM_2)
        g0.destroy()
        grad_target = max(tolg, tolg_rel * grad0) if tolg_rel > 0.0 else tolg
        snes.setTolerances(atol=grad_target, rtol=1e-50, stol=1e-50, max_it=maxit_attempt)

        ksp_tr = snes.getKSP()
        ksp_tr.setType(ksp_type)
        pc_tr = ksp_tr.getPC()
        pc_tr.setType(pc_type)
        if pc_type == "hypre":
            pc_tr.setHYPREType("boomeramg")
            if hypre_nodal_coarsen >= 0:
                opts[f"{tr_prefix}pc_hypre_boomeramg_nodal_coarsen"] = hypre_nodal_coarsen
            if hypre_vec_interp_variant >= 0:
                opts[f"{tr_prefix}pc_hypre_boomeramg_vec_interp_variant"] = hypre_vec_interp_variant
            if hypre_strong_threshold is not None:
                opts[f"{tr_prefix}pc_hypre_boomeramg_strong_threshold"] = hypre_strong_threshold
            if hypre_coarsen_type:
                opts[f"{tr_prefix}pc_hypre_boomeramg_coarsen_type"] = hypre_coarsen_type
        elif pc_type == "gamg":
            opts[f"{tr_prefix}pc_gamg_threshold"] = float(gamg_threshold)
            opts[f"{tr_prefix}pc_gamg_agg_nsmooths"] = int(gamg_agg_nsmooths)
        ksp_tr.setTolerances(rtol=ksp_rtol_attempt, max_it=ksp_max_it_attempt)

        tr_coords = None
        tr_coords_pending = False
        if pc_type == "gamg" and gamg_set_coordinates:
            index_map = V.dofmap.index_map
            tr_coords = V.tabulate_dof_coordinates()[:index_map.size_local, :]
            tr_coords_pending = True

        def snes_residual(_snes, vec, f):
            vec.copy(u.x.petsc_vec)
            _ghost_update(u.x.petsc_vec)
            with f.localForm() as f_loc:
                f_loc.set(0.0)
            assemble_vector(f, grad_form)
            apply_lifting(f, [hessian_form], [bcs], x0=[u.x.petsc_vec], alpha=1.0)
            f.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            set_bc(f, bcs, u.x.petsc_vec, alpha=1.0)

        def snes_jacobian(_snes, vec, J_mat, P_mat):
            nonlocal tr_coords_pending
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
            if tr_coords_pending and tr_coords is not None:
                pc_tr.setCoordinates(tr_coords)
                tr_coords_pending = False

        snes.setFunction(snes_residual, b_tr)
        snes.setJacobian(snes_jacobian, A_tr, A_tr)
        snes.setFromOptions()

        t_start = time.perf_counter()
        snes.solve(None, x)
        total_time = time.perf_counter() - t_start

        final_energy = energy_fn(x)
        snes_reason = int(snes.getConvergedReason())
        if snes_reason > 0:
            message = "Converged (trust-region SNES)"
        elif snes_reason == int(PETSc.SNES.ConvergedReason.DIVERGED_MAX_IT):
            message = "Maximum number of iterations reached (trust-region SNES)"
        else:
            reason_label = _snes_reason_label(snes_reason)
            message = f"Trust-region SNES failed ({reason_label}, reason={snes_reason})"

        out = {
            "x": x,
            "fun": final_energy,
            "nit": int(snes.getIterationNumber()),
            "time": float(total_time),
            "message": message,
            "history": [],
            "success": snes_reason > 0,
            "linear_iters": int(snes.getLinearSolveIterations()),
            "snes_reason": snes_reason,
        }

        snes.destroy()
        A_tr.destroy()
        b_tr.destroy()
        return out

    # ---- time evolution ----
    rotation_per_iter = 4 * 2 * np.pi / total_steps

    results = []
    x_step_start = x.duplicate()

    for step in range(start_step, start_step + num_steps):
        angle = step * rotation_per_iter

        # Update right boundary condition
        def right_rotation(x_coords):
            vals = np.zeros_like(x_coords)
            # x_coords[0] is x, x_coords[1] is y, x_coords[2] is z
            vals[0] = 0.0
            vals[1] = np.cos(angle) * x_coords[1] + np.sin(angle) * x_coords[2] - x_coords[1]
            vals[2] = -np.sin(angle) * x_coords[1] + np.cos(angle) * x_coords[2] - x_coords[2]
            return vals

        u_right.interpolate(right_rotation)

        # Apply BCs to current guess
        set_bc(x, bcs)
        _ghost_update(x)
        x.assemble()
        x.copy(x_step_start)

        if rank == 0 and verbose:
            print(f"--- Step {step}/{num_steps}, Angle: {angle:.4f} ---")

        attempt_configs = [
            {
                "name": "primary",
                "mode": "newton",
                "linesearch_interval": linesearch_interval,
                "ksp_rtol": ksp_rtol,
                "ksp_max_it": int(ksp_max_it),
                "maxit": int(maxit),
            },
        ]
        if trust_region_retry_on_maxit and retry_on_maxit:
            tr_retry_rtol = max(1e-8, ksp_rtol * trust_region_retry_rtol_factor)
            tr_retry_kmax = max(ksp_max_it + 1, int(round(ksp_max_it * trust_region_retry_ksp_max_it_factor)))
            attempt_configs.append(
                {
                    "name": "trust_region",
                    "mode": "snes_newtontr",
                    "linesearch_interval": None,
                    "ksp_rtol": tr_retry_rtol,
                    "ksp_max_it": int(tr_retry_kmax),
                    "maxit": int(max(1, trust_region_retry_maxit)),
                }
            )
        if retry_on_nonfinite or retry_on_maxit:
            ls_a, ls_b = linesearch_interval
            retry_ls_b = min(ls_b, nonfinite_retry_linesearch_b)
            if retry_ls_b <= ls_a:
                retry_ls_b = ls_b
            retry_rtol = max(1e-8, ksp_rtol * nonfinite_retry_rtol_factor)
            retry_kmax = max(ksp_max_it + 1, int(round(ksp_max_it * retry_ksp_max_it_factor)))
            if retry_ls_b < ls_b or retry_rtol < ksp_rtol or retry_kmax > ksp_max_it:
                attempt_configs.append(
                    {
                        "name": "repair",
                        "mode": "newton",
                        "linesearch_interval": (ls_a, retry_ls_b),
                        "ksp_rtol": retry_rtol,
                        "ksp_max_it": int(retry_kmax),
                        "maxit": int(maxit),
                    }
                )

        result = None
        total_time = 0.0
        used_attempt = "primary"
        used_mode = "newton"
        used_linesearch = linesearch_interval
        used_ksp_rtol = ksp_rtol
        used_ksp_max_it = ksp_max_it
        attempt_log = []

        for attempt_idx, attempt in enumerate(attempt_configs):
            attempt_name = attempt["name"]
            attempt_mode = attempt["mode"]
            ls_interval = attempt.get("linesearch_interval")
            ksp_rtol_attempt = float(attempt["ksp_rtol"])
            ksp_max_it_attempt = int(attempt["ksp_max_it"])
            maxit_attempt = int(attempt.get("maxit", maxit))

            # Restore step-start state for each attempt.
            x_step_start.copy(x)
            set_bc(x, bcs)
            _ghost_update(x)
            x.assemble()

            if attempt_mode == "newton":
                # Ensure the first Newton linear solve in each attempt performs setup.
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
                    maxit=maxit_attempt,
                    tolx_rel=tolx_rel,
                    tolx_abs=tolx_abs,
                    require_all_convergence=True,
                    fail_on_nonfinite=True,
                    verbose=verbose,
                    comm=comm,
                    ghost_update_fn=_ghost_update,
                    project_fn=lambda vec: set_bc(vec, bcs),
                    hessian_matvec_fn=hessian_matvec_fn,
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
            elif attempt_mode == "snes_newtontr":
                if save_linear_timing:
                    linear_timing_records.clear()
                result = trust_region_retry_fn(
                    ksp_rtol_attempt=ksp_rtol_attempt,
                    ksp_max_it_attempt=ksp_max_it_attempt,
                    maxit_attempt=maxit_attempt,
                )
                total_time = float(result["time"])
            else:
                raise RuntimeError(f"Unknown attempt mode: {attempt_mode}")

            used_attempt = attempt_name
            used_mode = attempt_mode
            used_linesearch = ls_interval
            used_ksp_rtol = ksp_rtol_attempt
            used_ksp_max_it = ksp_max_it_attempt
            attempt_log.append(
                {
                    "name": attempt_name,
                    "mode": attempt_mode,
                    "time": round(float(total_time), 4),
                    "iters": int(result.get("nit", 0)),
                    "message": str(result.get("message", "")),
                    "ksp_rtol": float(ksp_rtol_attempt),
                    "ksp_max_it": int(ksp_max_it_attempt),
                }
            )

            msg_lower = result["message"].lower()
            converged = "converged" in msg_lower
            need_retry = not converged
            if need_retry and attempt_idx + 1 < len(attempt_configs):
                if rank == 0 and verbose:
                    print(f"Step {step}: {attempt_name} did not converge, retrying with next fallback.")
                continue
            break

        # Restore baseline KSP tolerance for the next load step.
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
            "solver_mode": used_mode,
            "ksp_rtol_used": used_ksp_rtol,
            "ksp_max_it_used": int(used_ksp_max_it),
            "attempt_log": attempt_log,
        }
        if used_linesearch is not None:
            step_record["linesearch_interval_used"] = [float(used_linesearch[0]), float(used_linesearch[1])]
        if "linear_iters" in result:
            step_record["linear_iters"] = int(result["linear_iters"])
        if "snes_reason" in result:
            step_record["snes_reason"] = int(result["snes_reason"])
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
        "steps": results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=1, help="Mesh level (1-4)")
    parser.add_argument("--steps", type=int, default=1, help="Number of time steps")
    parser.add_argument("--start_step", type=int, default=1, help="Global step index to start from")
    parser.add_argument("--maxit", type=int, default=100, help="Maximum Newton iterations")
    parser.add_argument("--init_npz", type=str, default="", help="NPZ from JAX test data with coords and u_full_steps")
    parser.add_argument("--init_step", type=int, default=0, help="Step index in init_npz to use as initial guess")
    parser.add_argument("--linesearch_a", type=float, default=-0.5, help="Line-search interval lower bound")
    parser.add_argument("--linesearch_b", type=float, default=2.0, help="Line-search interval upper bound")
    parser.add_argument("--use_abs_det", action="store_true", help="Use abs(det(F)) in energy (JAX-compatible)")
    parser.add_argument("--ksp_type", type=str, default="gmres", help="PETSc KSP type")
    parser.add_argument("--pc_type", type=str, default="hypre", help="PETSc PC type")
    parser.add_argument("--ksp_rtol", type=float, default=1e-3, help="KSP relative tolerance")
    parser.add_argument("--ksp_max_it", type=int, default=10000, help="KSP maximum iterations per Newton step")
    parser.add_argument("--tolf", type=float, default=1e-4, help="Newton energy-change tolerance")
    parser.add_argument("--tolg", type=float, default=1e-3, help="Newton gradient-norm tolerance")
    parser.add_argument("--tolg_rel", type=float, default=1e-3,
                        help="Newton relative gradient tolerance (scaled by initial gradient)")
    parser.add_argument("--tolx_rel", type=float, default=1e-3, help="Newton relative step-size tolerance")
    parser.add_argument("--tolx_abs", type=float, default=1e-10, help="Newton absolute step-size tolerance")
    parser.add_argument("--no_near_nullspace", action="store_true", help="Disable elasticity near-nullspace on Hessian")
    parser.add_argument("--hypre_nodal_coarsen", type=int, default=6,
                        help="BoomerAMG nodal coarsen (-1 to skip setting)")
    parser.add_argument("--hypre_vec_interp_variant", type=int, default=3,
                        help="BoomerAMG vector interpolation variant (-1 to skip setting)")
    parser.add_argument("--hypre_strong_threshold", type=float, default=None, help="BoomerAMG strong threshold")
    parser.add_argument("--hypre_coarsen_type", type=str, default="", help="BoomerAMG coarsen type (e.g. HMIS, PMIS)")
    parser.add_argument(
        "--save_history",
        action="store_true",
        help="Include per-iteration Newton profile in output JSON")
    parser.add_argument(
        "--save_linear_timing",
        action="store_true",
        help="Include per-Newton linear timing breakdown in output JSON")
    parser.add_argument(
        "--pc_setup_on_ksp_cap",
        action="store_true",
        help="Only run KSP/PC setup when previous linear solve hit ksp_max_it (first solve always sets up)")
    parser.add_argument("--out", type=str, default="", help="Output JSON file")
    parser.add_argument("--total_steps", type=int, default=24,
                        help="Total steps that span the full 4×2π rotation (controls step size)")
    parser.add_argument("--gamg_threshold", type=float, default=-1.0,
                        help="GAMG threshold for filtering graph (-1 = keep all, most robust)")
    parser.add_argument("--gamg_agg_nsmooths", type=int, default=1,
                        help="GAMG number of smoothing steps for SA prolongation (1=smoothed, 0=unsmoothed)")
    parser.add_argument("--no_gamg_coordinates", action="store_true",
                        help="Disable PCSetCoordinates for GAMG")
    parser.add_argument("--no_fail_fast", action="store_true",
                        help="Do not stop trajectory early on Newton non-convergence")
    parser.add_argument("--no_retry_on_nonfinite", action="store_true",
                        help="Disable non-finite repair attempt with tighter settings")
    parser.add_argument("--nonfinite_retry_rtol_factor", type=float, default=0.1,
                        help="Multiplier for KSP rtol in non-finite repair attempt")
    parser.add_argument("--nonfinite_retry_linesearch_b", type=float, default=1.0,
                        help="Upper bound of line-search interval in non-finite repair attempt")
    parser.add_argument("--no_retry_on_maxit", action="store_true",
                        help="Disable repair attempt when Newton reaches max iterations")
    parser.add_argument("--retry_ksp_max_it_factor", type=float, default=2.0,
                        help="Multiplier for KSP max_it in repair attempt")
    parser.add_argument("--no_trust_region", action="store_true",
                        help="Disable trust-region control inside custom Newton")
    parser.add_argument("--trust_radius_init", type=float, default=1.0,
                        help="Initial trust-region radius for custom Newton")
    parser.add_argument("--trust_radius_min", type=float, default=1e-8,
                        help="Minimum trust-region radius for custom Newton")
    parser.add_argument("--trust_radius_max", type=float, default=1e6,
                        help="Maximum trust-region radius for custom Newton")
    parser.add_argument("--trust_shrink", type=float, default=0.5,
                        help="Trust-region shrink factor after rejected trial")
    parser.add_argument("--trust_expand", type=float, default=1.5,
                        help="Trust-region expansion factor after successful trial")
    parser.add_argument("--trust_eta_shrink", type=float, default=0.05,
                        help="Minimum rho acceptance threshold for trust-region step")
    parser.add_argument("--trust_eta_expand", type=float, default=0.75,
                        help="Rho threshold for trust-region radius expansion")
    parser.add_argument("--trust_max_reject", type=int, default=6,
                        help="Maximum rejected trust-region trials per Newton iteration")
    parser.add_argument("--trust_region_retry_on_maxit", action="store_true",
                        help="Retry non-converged Newton step with SNES trust-region (newtontr)")
    parser.add_argument("--trust_region_retry_maxit", type=int, default=100,
                        help="Maximum trust-region SNES iterations in fallback attempt")
    parser.add_argument("--trust_region_retry_rtol_factor", type=float, default=0.1,
                        help="Multiplier for KSP rtol in trust-region fallback")
    parser.add_argument("--trust_region_retry_ksp_max_it_factor", type=float, default=2.0,
                        help="Multiplier for KSP max_it in trust-region fallback")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    res = run_level(
        args.level,
        num_steps=args.steps,
        verbose=not args.quiet,
        maxit=args.maxit,
        start_step=args.start_step,
        init_npz=args.init_npz,
        init_step=args.init_step,
        linesearch_interval=(args.linesearch_a, args.linesearch_b),
        use_abs_det=args.use_abs_det,
        ksp_type=args.ksp_type,
        pc_type=args.pc_type,
        ksp_rtol=args.ksp_rtol,
        ksp_max_it=args.ksp_max_it,
        tolf=args.tolf,
        tolg=args.tolg,
        tolg_rel=args.tolg_rel,
        tolx_rel=args.tolx_rel,
        tolx_abs=args.tolx_abs,
        use_near_nullspace=not args.no_near_nullspace,
        total_steps=args.total_steps,
        hypre_nodal_coarsen=args.hypre_nodal_coarsen,
        hypre_vec_interp_variant=args.hypre_vec_interp_variant,
        hypre_strong_threshold=args.hypre_strong_threshold,
        hypre_coarsen_type=args.hypre_coarsen_type,
        save_history=args.save_history,
        save_linear_timing=args.save_linear_timing,
        pc_setup_on_ksp_cap=args.pc_setup_on_ksp_cap,
        gamg_threshold=args.gamg_threshold,
        gamg_agg_nsmooths=args.gamg_agg_nsmooths,
        gamg_set_coordinates=not args.no_gamg_coordinates,
        fail_fast=not args.no_fail_fast,
        retry_on_nonfinite=not args.no_retry_on_nonfinite,
        retry_on_maxit=not args.no_retry_on_maxit,
        nonfinite_retry_rtol_factor=args.nonfinite_retry_rtol_factor,
        nonfinite_retry_linesearch_b=args.nonfinite_retry_linesearch_b,
        retry_ksp_max_it_factor=args.retry_ksp_max_it_factor,
        use_trust_region=not args.no_trust_region,
        trust_radius_init=args.trust_radius_init,
        trust_radius_min=args.trust_radius_min,
        trust_radius_max=args.trust_radius_max,
        trust_shrink=args.trust_shrink,
        trust_expand=args.trust_expand,
        trust_eta_shrink=args.trust_eta_shrink,
        trust_eta_expand=args.trust_eta_expand,
        trust_max_reject=args.trust_max_reject,
        trust_region_retry_on_maxit=args.trust_region_retry_on_maxit,
        trust_region_retry_maxit=args.trust_region_retry_maxit,
        trust_region_retry_rtol_factor=args.trust_region_retry_rtol_factor,
        trust_region_retry_ksp_max_it_factor=args.trust_region_retry_ksp_max_it_factor,
    )

    if MPI.COMM_WORLD.rank == 0:
        print(json.dumps(res, indent=2))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(res, f, indent=2)
