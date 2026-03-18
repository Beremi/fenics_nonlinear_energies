"""Shared nullspace builders for FEniCS elasticity operators."""

from __future__ import annotations

import numpy as np
from petsc4py import PETSc


def build_elasticity_nullspace(
    V,
    A,
    *,
    constrained_dofs=None,
    gram_schmidt: bool = False,
) -> PETSc.NullSpace:
    """Build the six rigid-body modes for 3D elasticity."""
    x = V.tabulate_dof_coordinates()
    index_map = V.dofmap.index_map
    x_owned = np.asarray(x[: index_map.size_local, :], dtype=np.float64)
    constrained = np.empty(0, dtype=np.int32)
    if constrained_dofs is not None:
        constrained = np.asarray(constrained_dofs, dtype=np.int32)

    if gram_schmidt:
        x_mean = np.zeros(3, dtype=np.float64)
        for dim in range(3):
            local_sum = float(np.sum(x_owned[:, dim]))
            local_count = float(len(x_owned))
            global_sum = A.comm.tompi4py().allreduce(local_sum)
            global_count = A.comm.tompi4py().allreduce(local_count)
            x_mean[dim] = global_sum / global_count
        xc = x_owned - x_mean
    else:
        xc = x_owned

    vecs = [A.createVecLeft() for _ in range(6)]
    for vec in vecs:
        vec.getArray()[:] = 0.0

    for dim in range(3):
        vecs[dim].getArray()[dim::3] = 1.0

    vecs[3].getArray()[1::3] = -xc[:, 2]
    vecs[3].getArray()[2::3] = xc[:, 1]
    vecs[4].getArray()[0::3] = xc[:, 2]
    vecs[4].getArray()[2::3] = -xc[:, 0]
    vecs[5].getArray()[0::3] = -xc[:, 1]
    vecs[5].getArray()[1::3] = xc[:, 0]

    if constrained.size:
        n_local = vecs[0].getLocalSize()
        constrained = constrained[(constrained >= 0) & (constrained < n_local)]
        for vec in vecs:
            vec.getArray()[constrained] = 0.0

    if gram_schmidt:
        for i, vec in enumerate(vecs):
            for j in range(i):
                alpha = vec.dot(vecs[j])
                vec.axpy(-alpha, vecs[j])
            vec.normalize()

    return PETSc.NullSpace().create(vectors=vecs)
