"""Ray-profile diagnostics for the arctan-resonance family.

The helpers in this module sample one positive ray ``t -> J_p(t w)`` and
classify whether the sampled profile contains a stable interior maximum or
minimum.  The returned payloads are plain JSON-friendly dictionaries so the
runner can store them directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Literal

import numpy as np

from src.problems.plaplace_up_arctan.solver_common import ObjectiveBundle, ProblemInstance


RayMode = Literal["auto", "maximum", "minimum"]


@dataclass(frozen=True)
class RaySample:
    """One sampled ray point."""

    t: float
    value: float | None


@dataclass(frozen=True)
class RayCandidate:
    """One candidate interior extremum extracted from the sampled profile."""

    kind: str
    index: int | None
    t: float | None
    value: float | None
    prominence: float | None
    left_slope: float | None
    right_slope: float | None
    stable: bool


def _coerce_t_values(
    t_values: Iterable[float] | None,
    *,
    num_samples: int,
    t_max: float,
) -> np.ndarray:
    if t_values is None:
        if int(num_samples) < 3:
            raise ValueError("num_samples must be at least 3")
        if float(t_max) <= 0.0 or not math.isfinite(float(t_max)):
            raise ValueError("t_max must be finite and positive")
        return np.linspace(0.0, float(t_max), int(num_samples), dtype=np.float64)

    values = np.asarray(list(t_values), dtype=np.float64)
    if values.ndim != 1 or values.size < 3:
        raise ValueError("t_values must be a one-dimensional iterable with at least 3 samples")
    if not np.all(np.isfinite(values)):
        raise ValueError("t_values must be finite")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("t_values must be strictly increasing")
    return values


def sample_ray_profile(
    objective: ObjectiveBundle,
    base_free: np.ndarray,
    *,
    t_values: Iterable[float] | None = None,
    num_samples: int = 61,
    t_max: float = 6.0,
) -> dict[str, object]:
    """Sample the positive ray ``t -> objective(t * base_free)``."""

    base_free = np.asarray(base_free, dtype=np.float64)
    t_grid = _coerce_t_values(t_values, num_samples=num_samples, t_max=t_max)
    samples: list[RaySample] = []
    for t in t_grid:
        value = float(objective.value(float(t) * base_free))
        if not np.isfinite(value):
            samples.append(RaySample(t=float(t), value=None))
        else:
            samples.append(RaySample(t=float(t), value=float(value)))
    return {
        "objective_name": str(getattr(objective, "name", "objective")),
        "sample_count": int(len(samples)),
        "t_values": [float(item.t) for item in samples],
        "values": [None if item.value is None else float(item.value) for item in samples],
        "samples": [asdict(item) for item in samples],
    }


def _candidate_from_samples(
    t_values: np.ndarray,
    values: np.ndarray,
    *,
    kind: str,
    index: int | None,
    stability_window: int,
    rel_tol: float,
) -> RayCandidate:
    if index is None or index <= 0 or index >= t_values.size - 1:
        return RayCandidate(
            kind=kind,
            index=None,
            t=None,
            value=None,
            prominence=None,
            left_slope=None,
            right_slope=None,
            stable=False,
        )

    value = float(values[index])
    left_value = float(values[index - 1])
    right_value = float(values[index + 1])
    if kind == "minimum":
        left_slope = value - left_value
        right_slope = right_value - value
        prominence = min(left_value - value, right_value - value)
        window_left = values[max(0, index - int(stability_window)) : index + 1]
        window_right = values[index : min(values.size, index + int(stability_window) + 1)]
        monotone = bool(
            np.all(np.diff(window_left) <= rel_tol * max(1.0, abs(value)))
            and np.all(np.diff(window_right) >= -rel_tol * max(1.0, abs(value)))
        )
        stable = bool(prominence > rel_tol * max(1.0, abs(value)) and monotone)
    else:
        left_slope = value - left_value
        right_slope = right_value - value
        prominence = min(value - left_value, value - right_value)
        window_left = values[max(0, index - int(stability_window)) : index + 1]
        window_right = values[index : min(values.size, index + int(stability_window) + 1)]
        monotone = bool(
            np.all(np.diff(window_left) >= -rel_tol * max(1.0, abs(value)))
            and np.all(np.diff(window_right) <= rel_tol * max(1.0, abs(value)))
        )
        stable = bool(prominence > rel_tol * max(1.0, abs(value)) and monotone)

    return RayCandidate(
        kind=kind,
        index=int(index),
        t=float(t_values[index]),
        value=value,
        prominence=float(prominence),
        left_slope=float(left_slope),
        right_slope=float(right_slope),
        stable=stable,
    )


def _pick_candidate(candidates: list[RayCandidate]) -> RayCandidate:
    stable = [cand for cand in candidates if cand.index is not None and cand.stable and cand.prominence is not None]
    if stable:
        return max(stable, key=lambda cand: float(cand.prominence))
    interior = [cand for cand in candidates if cand.index is not None and cand.prominence is not None]
    if interior:
        return max(interior, key=lambda cand: float(cand.prominence))
    return RayCandidate(
        kind="none",
        index=None,
        t=None,
        value=None,
        prominence=None,
        left_slope=None,
        right_slope=None,
        stable=False,
    )


def audit_ray_profile(
    problem: ProblemInstance | None,
    objective: ObjectiveBundle,
    base_free: np.ndarray,
    *,
    t_values: Iterable[float] | None = None,
    num_samples: int = 61,
    t_max: float = 6.0,
    ray_mode: RayMode = "auto",
    stability_window: int = 2,
    rel_tol: float = 1.0e-10,
) -> dict[str, object]:
    """Diagnose the sampled ray profile for a single FE state."""

    base_free = np.asarray(base_free, dtype=np.float64)
    payload = sample_ray_profile(
        objective,
        base_free,
        t_values=t_values,
        num_samples=num_samples,
        t_max=t_max,
    )
    t_grid = np.asarray(payload["t_values"], dtype=np.float64)
    raw_values = list(payload["values"])

    if base_free.size == 0 or not np.all(np.isfinite(base_free)) or np.linalg.norm(base_free) <= 0.0:
        payload.update(
            {
                "status": "invalid_base",
                "ray_mode": str(ray_mode),
                "base_norm": float(np.linalg.norm(base_free)),
                "base_value": None,
                "best_kind": "none",
                "best_index": None,
                "best_t": None,
                "best_value": None,
                "stable_interior_extremum": False,
                "extremum_prominence": None,
                "left_slope": None,
                "right_slope": None,
                "summary": "Base vector is zero or non-finite.",
            }
        )
        return payload

    finite_mask = np.asarray([value is not None for value in raw_values], dtype=bool)
    if not np.any(finite_mask):
        payload.update(
            {
                "status": "nonfinite",
                "ray_mode": str(ray_mode),
                "base_norm": float(np.linalg.norm(base_free)),
                "base_value": None,
                "best_kind": "none",
                "best_index": None,
                "best_t": None,
                "best_value": None,
                "stable_interior_extremum": False,
                "extremum_prominence": None,
                "left_slope": None,
                "right_slope": None,
                "summary": "All sampled ray values were non-finite.",
            }
        )
        return payload

    values = np.asarray([float(v) if v is not None else np.nan for v in raw_values], dtype=np.float64)
    if np.any(~np.isfinite(values)):
        payload.update(
            {
                "status": "nonfinite",
                "ray_mode": str(ray_mode),
                "base_norm": float(np.linalg.norm(base_free)),
                "base_value": float(objective.value(base_free)),
                "best_kind": "none",
                "best_index": None,
                "best_t": None,
                "best_value": None,
                "stable_interior_extremum": False,
                "extremum_prominence": None,
                "left_slope": None,
                "right_slope": None,
                "summary": "At least one sampled ray value was non-finite.",
            }
        )
        return payload

    base_value = float(objective.value(base_free))
    candidate_min = _candidate_from_samples(
        t_grid,
        values,
        kind="minimum",
        index=int(np.argmin(values)),
        stability_window=int(stability_window),
        rel_tol=float(rel_tol),
    )
    candidate_max = _candidate_from_samples(
        t_grid,
        values,
        kind="maximum",
        index=int(np.argmax(values)),
        stability_window=int(stability_window),
        rel_tol=float(rel_tol),
    )

    if str(ray_mode) == "minimum":
        chosen = candidate_min
    elif str(ray_mode) == "maximum":
        chosen = candidate_max
    else:
        chosen = _pick_candidate([candidate_min, candidate_max])

    endpoint_gap = None
    if chosen.index is not None and chosen.value is not None:
        if chosen.kind == "minimum":
            endpoint_gap = float(min(values[0], values[-1]) - chosen.value)
        elif chosen.kind == "maximum":
            endpoint_gap = float(chosen.value - max(values[0], values[-1]))

    payload.update(
        {
            "status": "ok",
            "ray_mode": str(ray_mode),
            "base_norm": float(np.linalg.norm(base_free)),
            "base_value": float(base_value),
            "best_kind": str(chosen.kind),
            "best_index": None if chosen.index is None else int(chosen.index),
            "best_t": None if chosen.t is None else float(chosen.t),
            "best_value": None if chosen.value is None else float(chosen.value),
            "stable_interior_extremum": bool(chosen.stable),
            "extremum_prominence": None if chosen.prominence is None else float(chosen.prominence),
            "left_slope": None if chosen.left_slope is None else float(chosen.left_slope),
            "right_slope": None if chosen.right_slope is None else float(chosen.right_slope),
            "endpoint_gap": None if endpoint_gap is None else float(endpoint_gap),
            "summary": (
                f"stable interior {chosen.kind} detected"
                if chosen.stable and chosen.index is not None
                else "no stable interior extremum detected"
            ),
        }
    )
    if problem is not None:
        payload.update(
            {
                "geometry": str(problem.geometry),
                "mesh_level": int(problem.mesh_level),
                "p": float(problem.p),
                "lambda1": float(problem.lambda1),
                "lambda_level": int(problem.lambda_level),
            }
        )
    return payload


__all__ = [
    "RayMode",
    "RaySample",
    "RayCandidate",
    "sample_ray_profile",
    "audit_ray_profile",
]
