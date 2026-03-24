"""Support helpers for the plaplace_u3 problem family."""

from src.problems.plaplace_u3.support.mesh import (
    GEOMETRY_SQUARE_HOLE_PI,
    GEOMETRY_SQUARE_PI,
    INIT_ABS_SINE_3X3,
    INIT_ABS_SINE_Y2,
    INIT_SINE,
    INIT_SINE_X2,
    INIT_SINE_Y2,
    INIT_SKEW,
    MeshPLaplaceU32D,
    SUPPORTED_GEOMETRIES,
    SUPPORTED_INIT_MODES,
    build_problem_data,
)

__all__ = [
    "GEOMETRY_SQUARE_HOLE_PI",
    "GEOMETRY_SQUARE_PI",
    "INIT_ABS_SINE_3X3",
    "INIT_ABS_SINE_Y2",
    "INIT_SINE",
    "INIT_SINE_X2",
    "INIT_SINE_Y2",
    "INIT_SKEW",
    "MeshPLaplaceU32D",
    "SUPPORTED_GEOMETRIES",
    "SUPPORTED_INIT_MODES",
    "build_problem_data",
]
