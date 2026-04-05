"""Helpers for exporting compact publication state files."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


def _npz_payload(
    arrays: Mapping[str, object],
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = dict(arrays)
    if metadata:
        for key, value in metadata.items():
            payload[key] = np.asarray(value) if isinstance(value, (list, tuple)) else value
    return payload


def export_scalar_function_state_npz(
    path: str | Path,
    V,
    u,
    *,
    mesh_level: int,
    problem_name: str,
    energy: float | None = None,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Export one serial scalar P1 function as coordinates, cells, and nodal values."""
    msh = V.mesh
    if msh.comm.size != 1:
        raise ValueError("Scalar publication state export currently supports serial runs only")

    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, 0)

    num_cells = msh.topology.index_map(tdim).size_local
    cell_dofs = np.vstack(
        [np.asarray(V.dofmap.cell_dofs(cell), dtype=np.int32) for cell in range(num_cells)]
    )
    coords = np.asarray(V.tabulate_dof_coordinates(), dtype=np.float64)
    if coords.ndim == 1:
        coords = coords.reshape((-1, msh.geometry.x.shape[1]))
    values = np.asarray(u.x.array, dtype=np.float64).copy()

    payload = _npz_payload(
        {
            "coords": coords,
            "triangles": cell_dofs,
            "u": values,
        },
        metadata={
            "mesh_level": int(mesh_level),
            "problem_name": str(problem_name),
            **({"energy": float(energy)} if energy is not None else {}),
            **dict(metadata or {}),
        },
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def export_scalar_mesh_state_npz(
    path: str | Path,
    *,
    coords: np.ndarray,
    triangles: np.ndarray,
    u: np.ndarray,
    mesh_level: int,
    problem_name: str,
    energy: float | None = None,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Export one scalar P1 state directly from mesh arrays and nodal values."""
    coords = np.asarray(coords, dtype=np.float64).reshape((-1, 2))
    triangles = np.asarray(triangles, dtype=np.int32)
    values = np.asarray(u, dtype=np.float64).reshape((-1,))

    payload = _npz_payload(
        {
            "coords": coords,
            "triangles": triangles,
            "u": values,
        },
        metadata={
            "mesh_level": int(mesh_level),
            "problem_name": str(problem_name),
            **({"energy": float(energy)} if energy is not None else {}),
            **dict(metadata or {}),
        },
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def export_hyperelasticity_state_npz(
    path: str | Path,
    *,
    coords_ref: np.ndarray,
    x_final: np.ndarray,
    tetrahedra: np.ndarray,
    mesh_level: int,
    total_steps: int,
    energy: float | None = None,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Export one HE state as reference/deformed coordinates plus tetra connectivity."""
    coords_ref = np.asarray(coords_ref, dtype=np.float64).reshape((-1, 3))
    x_final = np.asarray(x_final, dtype=np.float64).reshape((-1, 3))
    tetrahedra = np.asarray(tetrahedra, dtype=np.int32)
    displacement = x_final - coords_ref

    payload = _npz_payload(
        {
            "coords_ref": coords_ref,
            "coords_final": x_final,
            "displacement": displacement,
            "tetrahedra": tetrahedra,
        },
        metadata={
            "mesh_level": int(mesh_level),
            "total_steps": int(total_steps),
            **({"energy": float(energy)} if energy is not None else {}),
            **dict(metadata or {}),
        },
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def export_planestrain_state_npz(
    path: str | Path,
    *,
    coords_ref: np.ndarray,
    x_final: np.ndarray,
    triangles: np.ndarray,
    case_name: str,
    lambda_target: float,
    energy: float | None = None,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Export one 2D vector state as reference/deformed coordinates and connectivity."""
    coords_ref = np.asarray(coords_ref, dtype=np.float64).reshape((-1, 2))
    x_final = np.asarray(x_final, dtype=np.float64).reshape((-1, 2))
    triangles = np.asarray(triangles, dtype=np.int32)
    displacement = x_final - coords_ref

    payload = _npz_payload(
        {
            "coords_ref": coords_ref,
            "coords_final": x_final,
            "displacement": displacement,
            "triangles": triangles,
        },
        metadata={
            "case_name": str(case_name),
            "lambda_target": float(lambda_target),
            **({"energy": float(energy)} if energy is not None else {}),
            **dict(metadata or {}),
        },
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def export_plasticity3d_state_npz(
    path: str | Path,
    *,
    coords_ref: np.ndarray,
    x_final: np.ndarray,
    tetrahedra: np.ndarray,
    surface_faces: np.ndarray,
    boundary_label: np.ndarray,
    mesh_name: str,
    element_degree: int,
    lambda_target: float,
    energy: float | None = None,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Export one 3D plasticity state as reference/deformed coordinates plus topology."""
    coords_ref = np.asarray(coords_ref, dtype=np.float64).reshape((-1, 3))
    x_final = np.asarray(x_final, dtype=np.float64).reshape((-1, 3))
    tetrahedra = np.asarray(tetrahedra, dtype=np.int32)
    surface_faces = np.asarray(surface_faces, dtype=np.int32)
    boundary_label = np.asarray(boundary_label, dtype=np.int32).reshape((-1,))
    displacement = x_final - coords_ref

    payload = _npz_payload(
        {
            "coords_ref": coords_ref,
            "coords_final": x_final,
            "displacement": displacement,
            "tetrahedra": tetrahedra,
            "surface_faces": surface_faces,
            "boundary_label": boundary_label,
        },
        metadata={
            "mesh_name": str(mesh_name),
            "element_degree": int(element_degree),
            "lambda_target": float(lambda_target),
            **({"energy": float(energy)} if energy is not None else {}),
            **dict(metadata or {}),
        },
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
