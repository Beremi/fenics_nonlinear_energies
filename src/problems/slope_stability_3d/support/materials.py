"""Material expansion helpers for the 3D heterogeneous slope-stability benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaterialSpec:
    """Source material-table row."""

    c0: float
    phi: float
    psi: float
    young: float
    poisson: float
    gamma_sat: float
    gamma_unsat: float

    @property
    def shear(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def bulk(self) -> float:
        return self.young / (3.0 * (1.0 - 2.0 * self.poisson))

    @property
    def lame(self) -> float:
        return self.bulk - (2.0 * self.shear) / 3.0


def _coerce_materials(materials: list[MaterialSpec] | list[dict] | dict) -> list[MaterialSpec]:
    if isinstance(materials, dict):
        materials = [materials]
    out: list[MaterialSpec] = []
    for entry in materials:
        if isinstance(entry, MaterialSpec):
            out.append(entry)
        else:
            out.append(
                MaterialSpec(
                    c0=float(entry["c0"]),
                    phi=float(entry["phi"]),
                    psi=float(entry["psi"]),
                    young=float(entry.get("young", entry.get("E"))),
                    poisson=float(entry.get("poisson", entry.get("nu"))),
                    gamma_sat=float(entry["gamma_sat"]),
                    gamma_unsat=float(entry["gamma_unsat"]),
                )
            )
    return out


def heterogenous_materials_qp(
    material_id: np.ndarray,
    *,
    n_q: int,
    materials: list[MaterialSpec] | list[dict] | dict,
    saturation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand source material rows to quadrature-point arrays shaped ``(n_elem, n_q)``."""

    mat_id = np.asarray(material_id, dtype=np.int64).ravel()
    if mat_id.ndim != 1:
        raise ValueError("material_id must be one-dimensional")

    mat_list = _coerce_materials(materials)
    if mat_id.size:
        max_mid = int(np.max(mat_id))
        if max_mid >= len(mat_list):
            raise IndexError(
                f"Material identifier {max_mid} requires at least {max_mid + 1} rows, "
                f"got {len(mat_list)}."
            )

    n_elem = int(mat_id.size)
    n_q = int(n_q)
    c0 = np.zeros((n_elem, n_q), dtype=np.float64)
    phi = np.zeros((n_elem, n_q), dtype=np.float64)
    psi = np.zeros((n_elem, n_q), dtype=np.float64)
    shear = np.zeros((n_elem, n_q), dtype=np.float64)
    bulk = np.zeros((n_elem, n_q), dtype=np.float64)
    lame = np.zeros((n_elem, n_q), dtype=np.float64)
    gamma_sat = np.zeros((n_elem, n_q), dtype=np.float64)
    gamma_unsat = np.zeros((n_elem, n_q), dtype=np.float64)

    for elem_idx, mid in enumerate(mat_id.tolist()):
        spec = mat_list[int(mid)]
        c0[elem_idx, :] = float(spec.c0)
        phi[elem_idx, :] = np.deg2rad(float(spec.phi))
        psi[elem_idx, :] = np.deg2rad(float(spec.psi))
        shear[elem_idx, :] = float(spec.shear)
        bulk[elem_idx, :] = float(spec.bulk)
        lame[elem_idx, :] = float(spec.lame)
        gamma_sat[elem_idx, :] = float(spec.gamma_sat)
        gamma_unsat[elem_idx, :] = float(spec.gamma_unsat)

    if saturation is None:
        gamma = gamma_sat
    else:
        sat = np.asarray(saturation, dtype=bool)
        if sat.shape != (n_elem, n_q):
            raise ValueError(
                f"saturation must have shape {(n_elem, n_q)}, got {sat.shape}"
            )
        gamma = np.where(sat, gamma_sat, gamma_unsat)

    return c0, phi, psi, shear, bulk, lame, gamma
