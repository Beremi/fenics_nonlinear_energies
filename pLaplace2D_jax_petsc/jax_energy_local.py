"""
Local (per-partition) energy, gradient, and HVP for pLaplace, with assembly.

The local energy function ``J_local`` computes element integrals for one
partition. Gradient and HVP are obtained via JAX AD. The assembled functions
combine partition results with scatter-add into global vectors.

Usage
-----
>>> from pLaplace2D_jax_petsc.jax_energy_local import (
...     PartitionedEnergy, make_partitioned_energy)
>>> pe = make_partitioned_energy(params, n_partitions=4, padded=False)
>>> energy = pe.energy(u)
>>> grad = pe.gradient(u)
>>> hvp = pe.hess_vec(u, tangent)
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import config
from functools import partial

config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Local energy function (pure JAX, no u_0/freedofs/f handling)
# ---------------------------------------------------------------------------

def J_local(v_local, elems_local, dvx_local, dvy_local, vol_local, p):
    """Element-integral energy for one partition.

    Parameters
    ----------
    v_local : (n_local_dofs,) — local DOF values
    elems_local : (n_local_elems, npe) int — local element connectivity
    dvx_local, dvy_local : (n_local_elems, npe) float
    vol_local : (n_local_elems,) float
    p : float — p-Laplace exponent

    Returns scalar energy (element-integral part only).
    """
    v_elems = v_local[elems_local]
    F_x = jnp.sum(v_elems * dvx_local, axis=1)
    F_y = jnp.sum(v_elems * dvy_local, axis=1)
    intgrds = (1.0 / p) * (F_x**2 + F_y**2) ** (p / 2.0)
    return jnp.sum(intgrds * vol_local)


# ---------------------------------------------------------------------------
# JIT-compiled derivative closures for a specific partition shape
# ---------------------------------------------------------------------------

def _make_jit_fns(elems_local, dvx_local, dvy_local, vol_local, p):
    """Create JIT-compiled energy, gradient, HVP for fixed partition data.

    Returns (f_jit, grad_jit, hvp_jit) — all take v_local as first arg.
    """
    # Freeze partition data into closures via partial
    def energy_fn(v_local):
        return J_local(v_local, elems_local, dvx_local, dvy_local, vol_local, p)

    f_jit = jax.jit(energy_fn)
    grad_fn = jax.jit(jax.grad(energy_fn))

    def hvp_fn(v_local, tangent):
        return jax.jvp(jax.grad(energy_fn), (v_local,), (tangent,))[1]

    hvp_jit = jax.jit(hvp_fn)

    return f_jit, grad_fn, hvp_jit


# ---------------------------------------------------------------------------
# Compiled partition bundle
# ---------------------------------------------------------------------------

class CompiledPartition:
    """One partition with JIT-compiled JAX functions and index maps."""

    def __init__(self, partition_data, p, do_warmup=True, warmup_n_dofs=None):
        from pLaplace2D_jax_petsc.element_partition import PartitionData

        pd = partition_data
        self.local_to_global = pd.local_to_global
        self.n_local_dofs = pd.n_local_dofs
        self.n_local_elems = pd.n_local_elems

        # Convert to JAX arrays
        elems_jnp = jnp.array(pd.elems_local, dtype=jnp.int32)
        dvx_jnp = jnp.array(pd.dvx_local, dtype=jnp.float64)
        dvy_jnp = jnp.array(pd.dvy_local, dtype=jnp.float64)
        vol_jnp = jnp.array(pd.vol_local, dtype=jnp.float64)

        self.f_jit, self.grad_jit, self.hvp_jit = _make_jit_fns(
            elems_jnp, dvx_jnp, dvy_jnp, vol_jnp, p
        )

        if do_warmup:
            n = warmup_n_dofs if warmup_n_dofs is not None else pd.n_local_dofs
            v_dummy = jnp.zeros(pd.n_local_dofs, dtype=jnp.float64)
            _ = self.f_jit(v_dummy)
            _ = self.grad_jit(v_dummy)
            _ = self.hvp_jit(v_dummy, v_dummy)

    def energy(self, v_local):
        return float(self.f_jit(v_local))

    def gradient(self, v_local):
        return self.grad_jit(v_local)

    def hess_vec(self, v_local, tangent_local):
        return self.hvp_jit(v_local, tangent_local)


# ---------------------------------------------------------------------------
# Partitioned energy: assembled computation over all partitions
# ---------------------------------------------------------------------------

class PartitionedEnergy:
    """Assembled energy / gradient / HVP from element partitions.

    Parameters
    ----------
    compiled_parts : list of CompiledPartition
    u_0 : (n_total,) jnp array — full DOF vector with BCs
    freedofs : (n_free,) jnp int array
    f_rhs : (n_total,) jnp array — RHS vector
    p : float
    n_total : int — total number of DOFs
    """

    def __init__(self, compiled_parts, u_0, freedofs, f_rhs, p, n_total):
        self.parts = compiled_parts
        self.u_0 = u_0
        self.freedofs = freedofs
        self.f_rhs = f_rhs
        self.p = p
        self.n_total = n_total
        self.n_free = len(freedofs)

    def _scatter_u(self, u):
        """Scatter free DOFs into full vector: v = u_0.at[freedofs].set(u)."""
        return np.array(self.u_0).copy()

    def energy(self, u):
        """Compute assembled energy J(u) = Σ_i J_local_i - f·v."""
        v_full = np.array(self.u_0)
        v_full[np.array(self.freedofs)] = np.array(u)

        total = 0.0
        for part in self.parts:
            v_local = jnp.array(v_full[part.local_to_global])
            total += part.energy(v_local)

        # Subtract linear term
        total -= float(jnp.dot(self.f_rhs, jnp.array(v_full)))
        return total

    def gradient(self, u):
        """Compute assembled gradient ∇J(u) w.r.t. free DOFs."""
        v_full = np.array(self.u_0)
        v_full[np.array(self.freedofs)] = np.array(u)

        # Accumulate gradient in full DOF space
        grad_full = np.zeros(self.n_total, dtype=np.float64)

        for part in self.parts:
            v_local = jnp.array(v_full[part.local_to_global])
            g_local = np.array(part.gradient(v_local))
            np.add.at(grad_full, part.local_to_global, g_local)

        # Subtract linear term gradient: d/dv (f·v) = f
        grad_full -= np.array(self.f_rhs)

        # Extract free-DOF gradient
        freedofs_np = np.array(self.freedofs)
        return grad_full[freedofs_np]

    def hess_vec(self, u, tangent):
        """Compute assembled Hessian-vector product H(u) · tangent.

        tangent is in free-DOF space (length n_free).
        Returns HVP in free-DOF space.
        """
        v_full = np.array(self.u_0)
        v_full[np.array(self.freedofs)] = np.array(u)

        # Expand tangent to full DOF space (zero on Dirichlet DOFs)
        tangent_full = np.zeros(self.n_total, dtype=np.float64)
        freedofs_np = np.array(self.freedofs)
        tangent_full[freedofs_np] = np.array(tangent)

        # Accumulate HVP in full DOF space
        hvp_full = np.zeros(self.n_total, dtype=np.float64)

        for part in self.parts:
            v_local = jnp.array(v_full[part.local_to_global])
            t_local = jnp.array(tangent_full[part.local_to_global])
            h_local = np.array(part.hess_vec(v_local, t_local))
            np.add.at(hvp_full, part.local_to_global, h_local)

        # Linear term contributes zero to Hessian
        return hvp_full[freedofs_np]


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def make_partitioned_energy(params, n_partitions, padded=False, do_warmup=True):
    """Build a PartitionedEnergy from mesh params.

    Parameters
    ----------
    params : dict — from mesh.get_data_jax() (u_0, freedofs, elems, dvx, dvy, vol, p, f)
    n_partitions : int
    padded : bool — if True, use uniform-padded partitions
    do_warmup : bool — if True, JIT-warmup each partition

    Returns PartitionedEnergy
    """
    from pLaplace2D_jax_petsc.element_partition import (
        build_all_partitions, build_all_partitions_padded
    )

    elems_np = np.array(params["elems"])
    dvx_np = np.array(params["dvx"])
    dvy_np = np.array(params["dvy"])
    vol_np = np.array(params["vol"])
    p = float(params["p"])

    if padded:
        part_data_list = build_all_partitions_padded(
            elems_np, dvx_np, dvy_np, vol_np, n_partitions
        )
    else:
        part_data_list = build_all_partitions(
            elems_np, dvx_np, dvy_np, vol_np, n_partitions
        )

    compiled = []
    for pd in part_data_list:
        compiled.append(CompiledPartition(pd, p, do_warmup=do_warmup))

    return PartitionedEnergy(
        compiled_parts=compiled,
        u_0=params["u_0"],
        freedofs=params["freedofs"],
        f_rhs=params["f"],
        p=p,
        n_total=len(params["u_0"]),
    )
