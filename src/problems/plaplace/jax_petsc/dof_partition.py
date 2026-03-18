"""
pLaplace-specific DOFPartition wrapper.

Backward-compatible wrapper that accepts the pLaplace ``params`` dict and
delegates to the generic ``tools_petsc4py.dof_partition.DOFPartition``.

For new code, use the generic DOFPartition directly.
"""

import numpy as np
from src.core.petsc.dof_partition import DOFPartition as _GenericDOFPartition
from src.core.petsc.dof_partition import petsc_ownership_range  # noqa: F401


class DOFPartition(_GenericDOFPartition):
    """pLaplace DOFPartition that accepts ``params`` dict.

    Adds convenience properties ``dvx_np``, ``dvy_np``, ``vol_np``, ``p``
    for backward compatibility with code that accesses these attributes.
    """

    def __init__(self, params, comm, adjacency=None, reorder=True):
        self.p = float(params["p"])
        super().__init__(
            freedofs=np.asarray(params["freedofs"]),
            elems=np.asarray(params["elems"]),
            u_0=np.asarray(params["u_0"]),
            comm=comm,
            adjacency=adjacency,
            reorder=reorder,
            f=np.asarray(params["f"]),
            elem_data={
                "dvx": np.asarray(params["dvx"]),
                "dvy": np.asarray(params["dvy"]),
                "vol": np.asarray(params["vol"]),
            },
        )

    @property
    def dvx_np(self):
        return self.local_elem_data["dvx"]

    @property
    def dvy_np(self):
        return self.local_elem_data["dvy"]

    @property
    def vol_np(self):
        return self.local_elem_data["vol"]
