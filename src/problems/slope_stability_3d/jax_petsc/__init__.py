"""PETSc/JAX solvers for the 3D heterogeneous slope-stability benchmark."""

from src.problems.slope_stability_3d.jax_petsc.reordered_element_assembler import (
    SlopeStability3DReorderedElementAssembler,
)

__all__ = [
    "SlopeStability3DReorderedElementAssembler",
]
