"""Helpers for converting PETSc converged-reason integers to names."""

from __future__ import annotations

from petsc4py import PETSc


def _enum_reason_map(enum_cls) -> dict[int, str]:
    reason_map: dict[int, str] = {}
    for name, value in vars(enum_cls).items():
        if name.startswith("_") or not isinstance(value, int):
            continue
        reason_map.setdefault(int(value), str(name))
    return reason_map


_KSP_REASON_MAP = _enum_reason_map(PETSc.KSP.ConvergedReason)


def ksp_reason_name(reason_code: int | None) -> str:
    if reason_code is None:
        return "UNKNOWN"
    return _KSP_REASON_MAP.get(int(reason_code), f"UNKNOWN_{int(reason_code)}")
