"""
MPI-parallel energy, gradient, and HVP using DOF-based domain decomposition.

Key differences from the previous element-partition approach:
  1. Each rank owns a DOF range [lo, hi) matching PETSc's block distribution.
  2. Each rank has ALL elements touching its owned DOFs (overlapping at boundaries).
  3. Gradient/HVP at owned DOFs are EXACT from local computation alone.
     → NO Allreduce on the full gradient/HVP vector!
  4. Only communication:
       - P2P ghost exchange (~8-32 KB) before each computation
       - Allreduce of a SCALAR for energy                     ~8 bytes
  5. XLA thread control to prevent oversubscription with MPI.

Usage
-----
  mpiexec -n 4 python3 src/problems/plaplace/jax_petsc/mpi_dof_partitioned.py
"""

from src.problems.plaplace.jax_petsc.dof_partition import DOFPartition
from jax import config
import jax.numpy as jnp
import jax
import os
import time
import numpy as np
from mpi4py import MPI

# ---- Prevent XLA thread oversubscription ----
# Each MPI rank should use at most ncpus/nprocs threads for XLA.
_nprocs = MPI.COMM_WORLD.Get_size()
_ncpus = os.cpu_count() or 1
_threads = max(1, _ncpus // _nprocs)
os.environ.setdefault("XLA_FLAGS",
                      f"--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1")
os.environ.setdefault("OMP_NUM_THREADS", str(_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(_threads))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_threads))


config.update("jax_enable_x64", True)


class MPIDOFPartitionedEnergy:
    """MPI-parallel energy / gradient / HVP using DOF-based decomposition.

    Each rank:
      - owns free DOFs [lo, hi)
      - has local elements = all elements touching at least one owned DOF
      - JIT-compiles one local JAX energy on the local problem
      - gradient/HVP at owned DOFs need NO Allreduce (exact from local compute)

    Parameters
    ----------
    params : dict
        JAX mesh params (u_0, freedofs, elems, dvx, dvy, vol, p, f).
    comm : MPI.Comm
        MPI communicator.
    adjacency : scipy.sparse matrix or None
        DOF-DOF adjacency for RCM reordering. Required on rank 0; others may pass None.
    reorder : bool
        Whether to apply RCM reordering for spatial locality (default True).
    """

    def __init__(self, params, comm, adjacency=None, reorder=True):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.timings = {}

        # ---- 1. Build DOF partition ----
        self.part = DOFPartition(params, comm, adjacency=adjacency, reorder=reorder)
        self.timings.update(self.part.timings)

        # ---- 2. Convert local arrays to JAX ----
        t0 = time.perf_counter()
        self._elems = jnp.array(self.part.elems_local_np, dtype=jnp.int32)
        self._dvx = jnp.array(self.part.dvx_np, dtype=jnp.float64)
        self._dvy = jnp.array(self.part.dvy_np, dtype=jnp.float64)
        self._vol = jnp.array(self.part.vol_np, dtype=jnp.float64)
        self._vol_weighted = jnp.array(
            self.part.vol_np * self.part.elem_weights, dtype=jnp.float64
        )
        self._owned_idx = jnp.array(self.part.owned_local_indices, dtype=jnp.int32)
        self._f_owned = jnp.array(self.part.f_owned, dtype=jnp.float64)
        self.timings["jax_convert"] = time.perf_counter() - t0

        # ---- 3. JIT compile local JAX functions ----
        t0 = time.perf_counter()
        self._compile_jax()
        self.timings["jit"] = time.perf_counter() - t0

    def _compile_jax(self):
        """JIT-compile energy, gradient, HVP for the local problem."""
        p = self.part.p
        elems = self._elems
        dvx = self._dvx
        dvy = self._dvy
        vol = self._vol            # unweighted (all local elements)
        vol_w = self._vol_weighted  # weighted (unique assignment for energy)

        # --- Energy: weighted (unique element assignment) ---
        def energy_weighted(v_local):
            v_e = v_local[elems]
            Fx = jnp.sum(v_e * dvx, axis=1)
            Fy = jnp.sum(v_e * dvy, axis=1)
            intgrds = (1.0 / p) * (Fx**2 + Fy**2) ** (p / 2.0)
            return jnp.sum(intgrds * vol_w)

        self._energy_jit = jax.jit(energy_weighted)

        # --- Gradient: unweighted (all local elements → exact at owned DOFs) ---
        def energy_full(v_local):
            v_e = v_local[elems]
            Fx = jnp.sum(v_e * dvx, axis=1)
            Fy = jnp.sum(v_e * dvy, axis=1)
            intgrds = (1.0 / p) * (Fx**2 + Fy**2) ** (p / 2.0)
            return jnp.sum(intgrds * vol)

        self._grad_jit = jax.jit(jax.grad(energy_full))

        # --- HVP: unweighted, forward-over-reverse ---
        def hvp_fn(v_local, tangent_local):
            return jax.jvp(jax.grad(energy_full), (v_local,), (tangent_local,))[1]

        self._hvp_jit = jax.jit(hvp_fn)

        # Warmup (trigger compilation)
        v_dummy = jnp.zeros(self.part.n_local, dtype=jnp.float64)
        _ = self._energy_jit(v_dummy).block_until_ready()
        _ = self._grad_jit(v_dummy).block_until_ready()
        _ = self._hvp_jit(v_dummy, v_dummy).block_until_ready()

    # ------------------------------------------------------------------
    # Internal: build local vectors from owned DOFs
    # ------------------------------------------------------------------

    def _get_v_local(self, u_owned):
        """Build local v_local from owned DOFs via P2P ghost exchange."""
        v_np = self.part.build_v_local_p2p(np.asarray(u_owned, dtype=np.float64))
        return jnp.array(v_np)

    def _get_v_and_t_local(self, u_owned, tangent_owned):
        """Build local v_local and t_local via P2P ghost exchange."""
        u_np = np.asarray(u_owned, dtype=np.float64)
        t_np = np.asarray(tangent_owned, dtype=np.float64)
        v_np = self.part.build_v_local_p2p(u_np)
        t_np_local = self.part.build_zero_local_p2p(t_np)
        return jnp.array(v_np), jnp.array(t_np_local)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def energy(self, u_owned):
        """Compute energy J(u).

        u_owned: (n_owned,) array of this rank's free DOFs.
        Returns scalar (same on all ranks).
        """
        v_local = self._get_v_local(u_owned)

        local_e = float(self._energy_jit(v_local).block_until_ready())
        total_e = self.comm.allreduce(local_e, op=MPI.SUM)

        # f·u term: each rank contributes its owned part
        local_fu = float(np.dot(np.asarray(self._f_owned),
                                np.asarray(u_owned, dtype=np.float64)))
        total_fu = self.comm.allreduce(local_fu, op=MPI.SUM)

        return total_e - total_fu

    def gradient(self, u_owned):
        """Compute gradient at owned free DOFs.

        Returns (n_owned,) numpy array — NO Allreduce needed!
        """
        v_local = self._get_v_local(u_owned)

        g_local = self._grad_jit(v_local).block_until_ready()

        # Extract owned free DOF part (exact!) and subtract f term
        g_owned = np.asarray(g_local[self._owned_idx]) - np.asarray(self._f_owned)
        return g_owned

    def hess_vec(self, u_owned, tangent_owned):
        """Compute HVP H(u)·tangent at owned free DOFs.

        Returns (n_owned,) numpy array — NO Allreduce needed!
        """
        v_local, t_local = self._get_v_and_t_local(u_owned, tangent_owned)

        h_local = self._hvp_jit(v_local, t_local).block_until_ready()

        # Extract owned free DOF part (exact!)
        h_owned = np.asarray(h_local[self._owned_idx])
        return h_owned

    def gradient_full(self, u_owned):
        """Compute gradient and return full free-DOF vector on all ranks.

        Uses Allgatherv to collect owned gradients from all ranks.
        Useful for correctness verification.
        """
        g_owned = self.gradient(u_owned)
        g_full = np.empty(self.part.n_free, dtype=np.float64)
        self.comm.Allgatherv(
            np.ascontiguousarray(g_owned, dtype=np.float64),
            [g_full, self.part._gather_sizes, self.part._gather_displs, MPI.DOUBLE],
        )
        return g_full

    def hess_vec_full(self, u_owned, tangent_owned):
        """Compute HVP and return full free-DOF vector on all ranks.

        Uses Allgatherv to collect owned HVPs from all ranks.
        Useful for correctness verification.
        """
        h_owned = self.hess_vec(u_owned, tangent_owned)
        h_full = np.empty(self.part.n_free, dtype=np.float64)
        self.comm.Allgatherv(
            np.ascontiguousarray(h_owned, dtype=np.float64),
            [h_full, self.part._gather_sizes, self.part._gather_displs, MPI.DOUBLE],
        )
        return h_full
