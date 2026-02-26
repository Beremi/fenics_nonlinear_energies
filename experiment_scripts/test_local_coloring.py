#!/usr/bin/env python3
"""
Experiment: local (per-partition) graph coloring vs global graph coloring.

Tests whether colouring only a subgraph of A² (restricted to a partition's
neighbourhood) produces exact Hessian values for the owned rows.

Three variants are tested:

  **Reference** (dense Hessian):
      Full dense Hessian via ``jax.hessian`` in free-DOF space.
      This is the ground truth.

  **Global SFD** (current code):
      Colour the full A² (n_free × n_free), build full-domain SFD Hessian.
      Should match dense Hessian exactly (up to floating point).

  **Variant A — two rings** (user's thesis):
      For owned DOFs S = [lo, hi):
        J = S ∪ N_A(S)       (one ring of neighbours)
        K = J ∪ N_A(J)       (two rings)
      Take all elements inside K, colour A²|_K, build Hessian on K.
      Extract rows corresponding to S.

  **Variant B — one ring** (simplified):
      For owned DOFs S = [lo, hi):
        J = S ∪ N_A(S)       (one ring of neighbours)
      Take all elements inside J, colour A²|_J, build Hessian on J.
      Extract rows corresponding to S.

Usage:
    python experiment_scripts/test_local_coloring.py [--level 3] [--nparts 4]
"""

from graph_coloring.coloring_custom import color_custom_random
from pLaplace2D_jax_petsc.mesh import MeshpLaplace2D
from jax import config
import jax.numpy as jnp
import jax
import sys
import os
import time
import numpy as np
import scipy.sparse as sp
import argparse

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

config.update("jax_enable_x64", True)


# =========================================================================
# Helpers
# =========================================================================

def petsc_ownership_range(n, rank, size):
    """PETSc-style block distribution [lo, hi)."""
    q, r = divmod(n, size)
    if rank < r:
        lo = rank * (q + 1)
        hi = lo + q + 1
    else:
        lo = rank * q + r
        hi = lo + q
    return lo, hi


def neighbours_csr(A_csr, dof_set):
    """Return set of all neighbours of dof_set in adjacency A (CSR)."""
    dof_arr = np.array(sorted(dof_set), dtype=np.int64)
    nbrs = set()
    for d in dof_arr:
        nbrs.update(A_csr.indices[A_csr.indptr[d]:A_csr.indptr[d + 1]])
    return nbrs


def validate_coloring(A2_csr, coloring):
    """Check that no two adjacent nodes in A² have the same colour."""
    n = A2_csr.shape[0]
    for i in range(n):
        for j in A2_csr.indices[A2_csr.indptr[i]:A2_csr.indptr[i + 1]]:
            if i != j and coloring[i] == coloring[j]:
                return False, (i, j, coloring[i])
    return True, None


# =========================================================================
# Build dense reference Hessian in free-DOF space
# =========================================================================

def build_dense_hessian_freedof(v_full, freedofs, elems, dvx, dvy, vol, p):
    """
    Compute the exact dense Hessian of E(v) projected to free-DOF space.

    Returns (n_free, n_free) dense numpy array.

    Method: for each free DOF j, compute HVP(v, e_j) where e_j is unit
    vector at freedofs[j] in total-node space.  Extract free-DOF rows.
    """
    n_total = len(v_full)
    n_free = len(freedofs)

    elems_j = jnp.array(elems, dtype=jnp.int32)
    dvx_j = jnp.array(dvx, dtype=jnp.float64)
    dvy_j = jnp.array(dvy, dtype=jnp.float64)
    vol_j = jnp.array(vol, dtype=jnp.float64)
    v_j = jnp.array(v_full, dtype=jnp.float64)

    def energy_fn(v):
        v_e = v[elems_j]
        Fx = jnp.sum(v_e * dvx_j, axis=1)
        Fy = jnp.sum(v_e * dvy_j, axis=1)
        return jnp.sum((1.0 / p) * (Fx**2 + Fy**2) ** (p / 2.0) * vol_j)

    def hvp_fn(v, t):
        return jax.jvp(jax.grad(energy_fn), (v,), (t,))[1]
    hvp_jit = jax.jit(hvp_fn)

    # Warm up
    _ = hvp_jit(v_j, jnp.zeros(n_total, dtype=jnp.float64)).block_until_ready()

    H = np.zeros((n_free, n_free), dtype=np.float64)
    for j in range(n_free):
        e_j = np.zeros(n_total, dtype=np.float64)
        e_j[freedofs[j]] = 1.0
        col = np.asarray(
            hvp_jit(v_j, jnp.array(e_j)).block_until_ready(), dtype=np.float64
        )
        H[:, j] = col[freedofs]

    return H


# =========================================================================
# SFD Hessian extraction on a local subdomain
# =========================================================================

def sfd_hessian_on_subdomain(v_full, freedofs, elems, dvx, dvy, vol, p,
                             A_full_csr, A2_full_csr, dof_set, elem_mask,
                             coloring_local, n_colors,
                             owned_freedofs):
    """
    Build Hessian rows for ``owned_freedofs`` using local subdomain + local coloring.

    Parameters
    ----------
    v_full : (n_total,) full node values
    freedofs : (n_free,) free-DOF → total-node map
    elems : (n_elems, npe) full element connectivity
    dvx, dvy, vol : element data
    p : float
    A_full_csr : CSR of A in free-DOF space (n_free × n_free) — for extraction
    A2_full_csr : CSR of A² in free-DOF space (n_free × n_free) — for coloring validation
    dof_set : sorted array of free-DOF indices in this local domain
    elem_mask : bool mask of which elements are in this local domain
    coloring_local : (len(dof_set),) colour per DOF in dof_set
    n_colors : int
    owned_freedofs : sorted array of owned free-DOF indices (subset of dof_set)

    Returns
    -------
    H_dict : dict  {(i_global, j_global): value}  for owned rows
    """
    n_total = len(v_full)
    n_free = len(freedofs)

    # --- Build local mesh ---
    local_elems_total = elems[elem_mask]
    local_total_nodes, inverse = np.unique(
        local_elems_total.ravel(), return_inverse=True
    )
    n_loc = len(local_total_nodes)
    elems_loc = inverse.reshape(-1, elems.shape[1]).astype(np.int32)

    dvx_loc = dvx[elem_mask]
    dvy_loc = dvy[elem_mask]
    vol_loc = vol[elem_mask]
    v_loc = v_full[local_total_nodes]

    # --- Index mappings ---
    total_to_local = np.full(n_total, -1, dtype=np.int64)
    total_to_local[local_total_nodes] = np.arange(n_loc, dtype=np.int64)

    # dof_set (global free-DOF indices) → local-node indices
    dof_to_local_node = total_to_local[freedofs[dof_set]]
    assert np.all(dof_to_local_node >= 0), "Some DOFs in dof_set not in local mesh!"

    # dof_set index lookup: global free-DOF → index in dof_set
    dof_to_setidx = np.full(n_free, -1, dtype=np.int64)
    dof_to_setidx[dof_set] = np.arange(len(dof_set), dtype=np.int64)

    # owned free-DOFs → indices in dof_set
    owned_setidx = dof_to_setidx[owned_freedofs]
    assert np.all(owned_setidx >= 0), "Some owned DOFs not in dof_set!"

    # --- Build A_local (for extraction) and A²_local (for coloring) ---
    n_local_free = len(dof_set)
    # A_local: Hessian sparsity, only A-neighbours
    rows_a, cols_a = [], []
    for k, d in enumerate(dof_set):
        for j in A_full_csr.indices[A_full_csr.indptr[d]:A_full_csr.indptr[d + 1]]:
            jj = dof_to_setidx[j]
            if jj >= 0:
                rows_a.append(k)
                cols_a.append(jj)
    A_local = sp.csr_matrix(
        (np.ones(len(rows_a), dtype=np.float64),
         (np.array(rows_a), np.array(cols_a))),
        shape=(n_local_free, n_local_free)
    )

    # A²_local: for coloring (superset of A_local)
    rows_l, cols_l = [], []
    for k, d in enumerate(dof_set):
        for j in A2_full_csr.indices[A2_full_csr.indptr[d]:A2_full_csr.indptr[d + 1]]:
            jj = dof_to_setidx[j]
            if jj >= 0:
                rows_l.append(k)
                cols_l.append(jj)
    A2_local = sp.csr_matrix(
        (np.ones(len(rows_l), dtype=np.float64),
         (np.array(rows_l), np.array(cols_l))),
        shape=(n_local_free, n_local_free)
    )

    # --- Compile local energy / HVP ---
    elems_lj = jnp.array(elems_loc, dtype=jnp.int32)
    dvx_lj = jnp.array(dvx_loc, dtype=jnp.float64)
    dvy_lj = jnp.array(dvy_loc, dtype=jnp.float64)
    vol_lj = jnp.array(vol_loc, dtype=jnp.float64)
    v_lj = jnp.array(v_loc, dtype=jnp.float64)

    def energy_loc(v):
        v_e = v[elems_lj]
        Fx = jnp.sum(v_e * dvx_lj, axis=1)
        Fy = jnp.sum(v_e * dvy_lj, axis=1)
        return jnp.sum((1.0 / p) * (Fx**2 + Fy**2) ** (p / 2.0) * vol_lj)

    hvp_loc = jax.jit(
        lambda v, t: jax.jvp(jax.grad(energy_loc), (v,), (t,))[1]
    )
    _ = hvp_loc(v_lj, jnp.zeros(n_loc, dtype=jnp.float64)).block_until_ready()

    # --- SFD extraction colour by colour ---
    H_dict = {}

    for c in range(n_colors):
        # Indicator: 1.0 at local-node positions of DOFs with colour c
        indicator = np.zeros(n_loc, dtype=np.float64)
        dofs_c_setidx = np.where(coloring_local == c)[0]
        for k in dofs_c_setidx:
            indicator[dof_to_local_node[k]] = 1.0

        hvp_res = np.asarray(
            hvp_loc(v_lj, jnp.array(indicator)).block_until_ready(),
            dtype=np.float64,
        )

        # For each owned row, find the (unique) colour-c A-neighbour (not A²!)
        for oi in owned_setidx:
            i_node = dof_to_local_node[oi]
            val = hvp_res[i_node]

            # Find which A_local neighbours of oi have colour c
            for jj in A_local.indices[A_local.indptr[oi]:A_local.indptr[oi + 1]]:
                if coloring_local[jj] == c:
                    i_global = int(dof_set[oi])
                    j_global = int(dof_set[jj])
                    H_dict[(i_global, j_global)] = val

    return H_dict


# =========================================================================
# Main experiment
# =========================================================================

def run_experiment(level, nparts, coloring_trials=20):
    print(f"{'=' * 70}")
    print(f"Local coloring experiment: level={level}, nparts={nparts}")
    print(f"{'=' * 70}\n")

    # ------------------------------------------------------------------
    # 1. Load mesh
    # ------------------------------------------------------------------
    print("Loading mesh...", end=" ", flush=True)
    t0 = time.perf_counter()
    mesh = MeshpLaplace2D(level)
    params_jax, adjacency, u_init = mesh.get_data_jax()
    params_np = mesh.params
    print(f"done ({time.perf_counter() - t0:.2f}s)")

    n_free = len(params_np["freedofs"])
    n_total = len(params_np["u_0"])
    n_elems = len(params_np["elems"])
    print(f"  n_free={n_free:,}, n_total={n_total:,}, n_elems={n_elems:,}")

    freedofs = np.asarray(params_np["freedofs"], dtype=np.int64)
    elems = np.asarray(params_np["elems"], dtype=np.int64)
    u_0 = np.asarray(params_np["u_0"], dtype=np.float64)
    dvx = np.asarray(params_np["dvx"], dtype=np.float64)
    dvy = np.asarray(params_np["dvy"], dtype=np.float64)
    vol = np.asarray(params_np["vol"], dtype=np.float64)
    p = float(params_np["p"])

    A_csr = sp.csr_matrix(adjacency)

    # A²
    print("Computing A²...", end=" ", flush=True)
    t0 = time.perf_counter()
    A2_full = sp.csr_matrix(A_csr @ A_csr)
    print(f"done ({time.perf_counter() - t0:.2f}s), nnz(A²)={A2_full.nnz:,}")

    # Build v_full
    u_np = np.asarray(u_init, dtype=np.float64)
    v_full = u_0.copy()
    v_full[freedofs] = u_np

    # ------------------------------------------------------------------
    # 2. Dense reference Hessian (ground truth)
    # ------------------------------------------------------------------
    print("\n--- Dense Hessian (ground truth) ---")
    t0 = time.perf_counter()
    H_dense = build_dense_hessian_freedof(v_full, freedofs, elems, dvx, dvy, vol, p)
    t_dense = time.perf_counter() - t0
    print(f"  Shape: {H_dense.shape}, time: {t_dense:.2f}s")
    print(f"  Symmetry check: max|H - H^T| = {np.max(np.abs(H_dense - H_dense.T)):.2e}")

    # ------------------------------------------------------------------
    # 3. Global SFD Hessian (sanity check)
    # ------------------------------------------------------------------
    print("\n--- Global SFD Hessian ---")
    t0 = time.perf_counter()
    nc_global, coloring_global = color_custom_random(A2_full, seed=42, is_A2=True)
    print(f"  Coloring: {nc_global} colors, ", end="")
    ok, info = validate_coloring(sp.csr_matrix(A2_full), coloring_global)
    print(f"valid={ok}")

    # Build full-domain SFD via column-by-column HVP in total-node space
    n_total = len(v_full)
    elems_j = jnp.array(elems, dtype=jnp.int32)
    dvx_j = jnp.array(dvx, dtype=jnp.float64)
    dvy_j = jnp.array(dvy, dtype=jnp.float64)
    vol_j = jnp.array(vol, dtype=jnp.float64)
    v_j = jnp.array(v_full, dtype=jnp.float64)

    def energy_full(v):
        v_e = v[elems_j]
        Fx = jnp.sum(v_e * dvx_j, axis=1)
        Fy = jnp.sum(v_e * dvy_j, axis=1)
        return jnp.sum((1.0 / p) * (Fx**2 + Fy**2) ** (p / 2.0) * vol_j)

    hvp_full = jax.jit(
        lambda v, t: jax.jvp(jax.grad(energy_full), (v,), (t,))[1]
    )
    _ = hvp_full(v_j, jnp.zeros(n_total, dtype=jnp.float64)).block_until_ready()

    # SFD colour-by-colour — extract using A (not A²!) since only A-neighbors
    # have nonzero Hessian entries. The A² coloring guarantees that all
    # A-neighbors of any vertex have distinct colors (they're pairwise at
    # distance ≤ 2 in A, hence A²-adjacent), so extraction from A is sound.
    A2_csr = sp.csr_matrix(A2_full)
    H_sfd = np.zeros((n_free, n_free), dtype=np.float64)

    for c in range(nc_global):
        indicator = np.zeros(n_total, dtype=np.float64)
        dofs_c = np.where(coloring_global == c)[0]
        indicator[freedofs[dofs_c]] = 1.0

        hvp_res = np.asarray(
            hvp_full(v_j, jnp.array(indicator)).block_until_ready(),
            dtype=np.float64,
        )

        # Extract using A-neighbors (not A²-neighbors!)
        for i in range(n_free):
            for j in A_csr.indices[A_csr.indptr[i]:A_csr.indptr[i + 1]]:
                if coloring_global[j] == c:
                    H_sfd[i, j] = hvp_res[freedofs[i]]

    err_sfd_vs_dense = np.max(np.abs(H_sfd - H_dense))
    t_sfd = time.perf_counter() - t0
    print(f"  max|SFD - Dense| = {err_sfd_vs_dense:.2e}  ({t_sfd:.2f}s)")

    # Diagnostic: print first 5 mismatched entries
    if err_sfd_vs_dense > 1e-10:
        print("  DIAGNOSTIC — first 5 large-error entries:")
        count = 0
        first_bad_i = None
        for i in range(n_free):
            for j in A2_csr.indices[A2_csr.indptr[i]:A2_csr.indptr[i + 1]]:
                err = abs(H_sfd[i, j] - H_dense[i, j])
                if err > 0.01:
                    if first_bad_i is None:
                        first_bad_i = i
                    if i == first_bad_i:
                        print(f"    H[{i},{j}]: dense={H_dense[i, j]:.6f}, "
                              f"sfd={H_sfd[i, j]:.6f}, color[{j}]={coloring_global[j]}")
                        count += 1
            if count >= 10:
                break

        # For the first bad DOF, show its A-neighbors and A²-neighbors
        i0 = first_bad_i
        A_nbrs = set(A_csr.indices[A_csr.indptr[i0]:A_csr.indptr[i0 + 1]])
        A2_nbrs = set(A2_csr.indices[A2_csr.indptr[i0]:A2_csr.indptr[i0 + 1]])
        print(f"\n  DOF {i0}: A-neighbors  ({len(A_nbrs)}): {sorted(A_nbrs)}")
        print(f"  DOF {i0}: A²-neighbors ({len(A2_nbrs)}): {sorted(A2_nbrs)}")
        print(f"  DOF {i0}: distance-2 only: {sorted(A2_nbrs - A_nbrs)}")

        # Colors of A-neighbors
        print(f"  Colors of A-neighbors:  ", end="")
        for j in sorted(A_nbrs):
            print(f"{j}:c{coloring_global[j]} ", end="")
        print()

        # Check: does A ⊂ A²?
        A_coo = A_csr.tocoo()
        n_not_in_A2 = 0
        for ii, jj in zip(A_coo.row, A_coo.col):
            if A2_full[ii, jj] == 0:
                n_not_in_A2 += 1
        print(f"\n  A ⊂ A² check: {n_not_in_A2} edges in A not in A²")

        # Check A.data values
        print(f"  A.data unique values (first 20): {np.unique(A_csr.data)[:20]}")
        print(f"  A.nnz={A_csr.nnz}, A² nnz={A2_csr.nnz}")

        # For DOF i0, show dense row
        nz_dense = np.where(np.abs(H_dense[i0, :]) > 1e-15)[0]
        print(f"\n  H_dense[{i0}, :] nonzero at: {nz_dense.tolist()}")
        print(f"  These are A-neighbors? {set(nz_dense.tolist()) == A_nbrs}")

    # Dirichlet mask for element selection
    dirichlet_mask = np.ones(n_total, dtype=bool)
    dirichlet_mask[freedofs] = False

    # ------------------------------------------------------------------
    # 4. Test each partition with local coloring
    # ------------------------------------------------------------------
    max_err = {"A": 0.0, "B": 0.0}
    max_rel = {"A": 0.0, "B": 0.0}
    total_missing = {"A": 0, "B": 0}
    total_checked = {"A": 0, "B": 0}

    for part_id in range(nparts):
        lo, hi = petsc_ownership_range(n_free, part_id, nparts)
        S = set(range(lo, hi))
        owned_arr = np.array(sorted(S), dtype=np.int64)

        J = S | neighbours_csr(A_csr, S)         # one ring
        K = J | neighbours_csr(A_csr, J)          # two rings

        print(f"\n--- Partition {part_id}: owned=[{lo},{hi}), "
              f"|S|={len(S)}, |J|={len(J)}, |K|={len(K)} ---")

        for tag, dof_set_py in [("A", K), ("B", J)]:
            dof_set = np.array(sorted(dof_set_py), dtype=np.int64)

            # Element selection: all elements whose vertices are
            # either free DOFs in dof_set or Dirichlet nodes
            node_ok = dirichlet_mask.copy()
            node_ok[freedofs[dof_set]] = True
            elem_mask = np.all(node_ok[elems], axis=1)

            n_elems_local = int(elem_mask.sum())

            # Colour A²|_dof_set
            # Build restricted A²
            dof_to_setidx = np.full(n_free, -1, dtype=np.int64)
            dof_to_setidx[dof_set] = np.arange(len(dof_set), dtype=np.int64)
            rows_l, cols_l = [], []
            for k, d in enumerate(dof_set):
                for j in A2_csr.indices[A2_csr.indptr[d]:A2_csr.indptr[d + 1]]:
                    jj = dof_to_setidx[j]
                    if jj >= 0:
                        rows_l.append(k)
                        cols_l.append(jj)
            A2_local = sp.csr_matrix(
                (np.ones(len(rows_l)), (rows_l, cols_l)),
                shape=(len(dof_set), len(dof_set))
            )

            # Multi-start coloring
            best_nc = len(dof_set) + 1
            best_cols = None
            for seed in range(coloring_trials):
                nc, cols = color_custom_random(A2_local, seed=seed, is_A2=True)
                if nc < best_nc:
                    best_nc = nc
                    best_cols = cols

            ok, _ = validate_coloring(A2_local, best_cols)

            print(f"  Variant {tag}: |dof_set|={len(dof_set)}, "
                  f"elems={n_elems_local}, colors={best_nc}, valid={ok}")

            # SFD on local subdomain
            H_dict = sfd_hessian_on_subdomain(
                v_full, freedofs, elems, dvx, dvy, vol, p,
                A_csr, A2_csr, dof_set, elem_mask,
                best_cols, best_nc,
                owned_arr,
            )

            # Compare against dense reference
            n_checked = 0
            n_miss = 0
            me = 0.0
            mre = 0.0

            for i in range(lo, hi):
                for j in A_csr.indices[A_csr.indptr[i]:A_csr.indptr[i + 1]]:
                    ref_val = H_dense[i, j]
                    key = (i, j)
                    if key in H_dict:
                        err = abs(H_dict[key] - ref_val)
                        re = err / max(abs(ref_val), 1e-30)
                        me = max(me, err)
                        mre = max(mre, re)
                        n_checked += 1
                    else:
                        n_miss += 1

            print(f"    checked={n_checked}, missing={n_miss}, "
                  f"max|err|={me:.2e}, max|rel|={mre:.2e}")

            max_err[tag] = max(max_err[tag], me)
            max_rel[tag] = max(max_rel[tag], mre)
            total_missing[tag] += n_miss
            total_checked[tag] += n_checked

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"SUMMARY (level={level}, nparts={nparts})")
    print(f"{'=' * 70}")
    print(f"  Global SFD vs Dense:  max|err| = {err_sfd_vs_dense:.2e}")
    for tag, name in [("A", "two rings"), ("B", "one ring")]:
        status = "PASS" if max_err[tag] < 1e-10 else "FAIL"
        print(f"  Variant {tag} ({name}): max|err|={max_err[tag]:.2e}, "
              f"checked={total_checked[tag]}, missing={total_missing[tag]}  "
              f"[{status}]")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test local graph coloring variants")
    parser.add_argument("--level", type=int, default=3,
                        help="Mesh refinement level (default: 3, small for quick test)")
    parser.add_argument("--nparts", type=int, default=4,
                        help="Number of partitions to simulate (default: 4)")
    parser.add_argument("--coloring-trials", type=int, default=20,
                        help="Coloring multi-start trials per subgraph (default: 20)")
    args = parser.parse_args()

    run_experiment(args.level, args.nparts, args.coloring_trials)
