"""Generic FEniCS/DOLFINx utility helpers."""

from petsc4py import PETSc


def ghost_update(v):
    """INSERT-mode forward scatter (owned → ghosts)."""
    v.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
