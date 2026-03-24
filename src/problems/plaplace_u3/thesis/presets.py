"""Frozen thesis defaults and reproduction presets."""

from __future__ import annotations

from src.problems.plaplace_u3.support.mesh import (
    GEOMETRY_SQUARE_HOLE_PI,
    GEOMETRY_SQUARE_PI,
    INIT_ABS_SINE_3X3,
    INIT_ABS_SINE_Y2,
    INIT_SINE,
    INIT_SINE_X2,
    INIT_SINE_Y2,
    INIT_SKEW,
)
from src.problems.plaplace_u3.thesis.mesh1d import GEOMETRY_INTERVAL_PI


THESIS_DIRECTION_EXACT = "d"
THESIS_DIRECTION_VH = "d_vh"
THESIS_DIRECTION_RN = "d_rn"
THESIS_DIRECTIONS = (
    THESIS_DIRECTION_EXACT,
    THESIS_DIRECTION_VH,
    THESIS_DIRECTION_RN,
)

THESIS_METHODS = ("rmpa", "oa1", "oa2", "mpa")
THESIS_P_SWEEP = tuple(value / 6.0 for value in range(9, 19))
THESIS_P_SWEEP_MPA = tuple(value / 6.0 for value in range(10, 19))
THESIS_MAIN_LEVELS = (5, 6, 7)
THESIS_FINE_LEVEL_1D = 11
THESIS_TOL_MAIN = 1.0e-5
THESIS_TOL_COMPARE = 1.0e-4
THESIS_MAXIT_RMPA_OA = 500
THESIS_MAXIT_MPA = 1000

THESIS_CANDIDATE_RMPA_DELTA0 = (1.0, 0.5, 0.25)
THESIS_CANDIDATE_OA_DELTA_HAT = (1.0, 0.5, 0.25)
THESIS_CANDIDATE_MPA_SEGMENT_TOL_FACTORS = (0.25, 0.125, 0.0625)

# Frozen defaults chosen for the reproduction layer.
THESIS_RMPA_DELTA0 = 1.0
THESIS_OA_DELTA_HAT = 1.0
THESIS_OA_GOLDEN_TOL = 1.0e-5
THESIS_MPA_SEGMENT_TOL_FACTOR = 0.125
THESIS_MPA_NUM_NODES = 50
THESIS_MPA_RHO = 1.0

THESIS_SQUARE_DEFAULT = {
    "geometry": GEOMETRY_SQUARE_PI,
    "init_mode": INIT_SINE,
    "direction": THESIS_DIRECTION_VH,
    "tolerance": THESIS_TOL_MAIN,
}

THESIS_SQUARE_MULTIBRANCH_SEEDS = (
    INIT_SINE,
    INIT_SINE_X2,
    INIT_SINE_Y2,
    INIT_SKEW,
)

THESIS_SQUARE_HOLE_DEFAULT = {
    "geometry": GEOMETRY_SQUARE_HOLE_PI,
    "init_mode": INIT_SINE,
    "direction": THESIS_DIRECTION_VH,
    "tolerance": THESIS_TOL_COMPARE,
}

THESIS_SQUARE_HOLE_MULTIBRANCH_SEEDS = (
    INIT_SINE,
    INIT_ABS_SINE_Y2,
    INIT_SKEW,
    INIT_ABS_SINE_3X3,
)

THESIS_INTERVAL_DEFAULT = {
    "geometry": GEOMETRY_INTERVAL_PI,
    "init_mode": INIT_SINE,
    "direction": THESIS_DIRECTION_VH,
    "tolerance": THESIS_TOL_COMPARE,
}
