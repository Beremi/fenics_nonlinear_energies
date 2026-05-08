#!/usr/bin/env python3
"""Generate scalar level-10 meshes by uniformly refining the checked-in level 9.

The level-10 HDF5 outputs are intentionally generated local artifacts, not
checked-in inputs. Generate them before running the full globalization-method
comparison:

    ./.venv/bin/python experiments/runners/generate_scalar_uniform_l10_meshes.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ScalarMeshSpec:
    key: str
    parent_path: Path
    output_path: Path
    adjacency_data_dtype: np.dtype
    freedofs_dtype: np.dtype


SPECS = {
    "plaplace": ScalarMeshSpec(
        key="plaplace",
        parent_path=REPO_ROOT / "data/meshes/pLaplace/pLaplace_level9.h5",
        output_path=REPO_ROOT / "data/meshes/pLaplace/pLaplace_level10.h5",
        adjacency_data_dtype=np.dtype(np.bool_),
        freedofs_dtype=np.dtype(np.int64),
    ),
    "gl": ScalarMeshSpec(
        key="gl",
        parent_path=REPO_ROOT / "data/meshes/GinzburgLandau/GL_level9.h5",
        output_path=REPO_ROOT / "data/meshes/GinzburgLandau/GL_level10.h5",
        adjacency_data_dtype=np.dtype(np.float64),
        freedofs_dtype=np.dtype(np.int32),
    ),
}


def _uniform_refine_triangles(nodes: np.ndarray, elems: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    elems = np.asarray(elems, dtype=np.int64)
    n_elems = int(elems.shape[0])
    edge_pairs = np.empty((3 * n_elems, 2), dtype=np.int64)
    edge_pairs[:n_elems] = elems[:, [0, 1]]
    edge_pairs[n_elems : 2 * n_elems] = elems[:, [1, 2]]
    edge_pairs[2 * n_elems :] = elems[:, [2, 0]]
    edge_pairs.sort(axis=1)
    unique_edges, inverse = np.unique(edge_pairs, axis=0, return_inverse=True)

    midpoint_ids = np.arange(nodes.shape[0], nodes.shape[0] + unique_edges.shape[0], dtype=np.int64)
    refined_nodes = np.vstack(
        [
            np.asarray(nodes, dtype=np.float64),
            0.5 * (nodes[unique_edges[:, 0]] + nodes[unique_edges[:, 1]]),
        ]
    )
    m01 = midpoint_ids[inverse[:n_elems]]
    m12 = midpoint_ids[inverse[n_elems : 2 * n_elems]]
    m20 = midpoint_ids[inverse[2 * n_elems :]]
    a = elems[:, 0]
    b = elems[:, 1]
    c = elems[:, 2]

    refined_elems = np.empty((4 * n_elems, 3), dtype=np.int32)
    refined_elems[0::4] = np.column_stack([a, m01, m20])
    refined_elems[1::4] = np.column_stack([m01, b, m12])
    refined_elems[2::4] = np.column_stack([m20, m12, c])
    refined_elems[3::4] = np.column_stack([m01, m12, m20])
    return refined_nodes, refined_elems


def _triangle_derivatives(nodes: np.ndarray, elems: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p0 = nodes[elems[:, 0]]
    p1 = nodes[elems[:, 1]]
    p2 = nodes[elems[:, 2]]
    twice_area = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (
        p1[:, 1] - p0[:, 1]
    )
    if np.any(twice_area <= 0.0):
        raise ValueError("Uniform refinement produced non-positive triangle orientation.")

    dvx = np.empty((elems.shape[0], 3), dtype=np.float64)
    dvy = np.empty((elems.shape[0], 3), dtype=np.float64)
    dvx[:, 0] = (p1[:, 1] - p2[:, 1]) / twice_area
    dvx[:, 1] = (p2[:, 1] - p0[:, 1]) / twice_area
    dvx[:, 2] = (p0[:, 1] - p1[:, 1]) / twice_area
    dvy[:, 0] = (p2[:, 0] - p1[:, 0]) / twice_area
    dvy[:, 1] = (p0[:, 0] - p2[:, 0]) / twice_area
    dvy[:, 2] = (p1[:, 0] - p0[:, 0]) / twice_area
    return dvx, dvy, 0.5 * twice_area


def _free_dofs(problem: str, nodes: np.ndarray, dtype: np.dtype) -> np.ndarray:
    x = nodes[:, 0]
    y = nodes[:, 1]
    eps = 1.0e-12
    if problem == "plaplace":
        mask = (
            (x > eps)
            & (x < 2.0 - eps)
            & (y > eps)
            & (y < 2.0 - eps)
            & ~((x >= 1.0 - eps) & (y >= 1.0 - eps))
        )
    elif problem == "gl":
        mask = (
            (x > float(x.min()) + eps)
            & (x < float(x.max()) - eps)
            & (y > float(y.min()) + eps)
            & (y < float(y.max()) - eps)
        )
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"Unknown scalar problem {problem!r}")
    return np.flatnonzero(mask).astype(dtype, copy=False)


def _lumped_plaplace_force(n_nodes: int, elems: np.ndarray, vol: np.ndarray) -> np.ndarray:
    force = np.zeros(n_nodes, dtype=np.float64)
    elem_load = -np.asarray(vol, dtype=np.float64) / 3.0
    for local_node in range(3):
        np.add.at(force, elems[:, local_node], elem_load)
    return force


def _free_adjacency(elems: np.ndarray, freedofs: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    free_index = np.full(n_nodes, -1, dtype=np.int64)
    free_index[np.asarray(freedofs, dtype=np.int64)] = np.arange(freedofs.size, dtype=np.int64)

    n_elems = int(elems.shape[0])
    edge_nodes = np.empty((3 * n_elems, 2), dtype=np.int64)
    edge_nodes[:n_elems] = elems[:, [0, 1]]
    edge_nodes[n_elems : 2 * n_elems] = elems[:, [1, 2]]
    edge_nodes[2 * n_elems :] = elems[:, [2, 0]]
    edge_free = free_index[edge_nodes]
    valid = (edge_free[:, 0] >= 0) & (edge_free[:, 1] >= 0)
    edge_free = edge_free[valid]
    edge_free.sort(axis=1)
    edge_free = np.unique(edge_free, axis=0)

    diag = np.arange(freedofs.size, dtype=np.int32)
    edge_a = edge_free[:, 0].astype(np.int32, copy=False)
    edge_b = edge_free[:, 1].astype(np.int32, copy=False)
    row = np.concatenate([diag, edge_a, edge_b])
    col = np.concatenate([diag, edge_b, edge_a])
    order = np.lexsort((col, row))
    return row[order], col[order]


def _write_dataset(handle: h5py.File | h5py.Group, name: str, value: np.ndarray | float) -> None:
    if np.asarray(value).shape == ():
        handle.create_dataset(name, data=value)
    else:
        handle.create_dataset(name, data=value, compression="gzip", compression_opts=9)


def generate(spec: ScalarMeshSpec, *, force: bool = False) -> Path:
    if spec.output_path.exists() and not force:
        return spec.output_path

    with h5py.File(spec.parent_path, "r") as parent:
        parent_nodes = np.asarray(parent["nodes"][:], dtype=np.float64)
        parent_elems = np.asarray(parent["elems"][:], dtype=np.int64)
        attrs = {key: parent[key][()] for key in parent if key not in {"adjacency", "nodes", "elems", "dvx", "dvy", "vol", "freedofs", "f", "u_0"}}

    nodes, elems = _uniform_refine_triangles(parent_nodes, parent_elems)
    dvx, dvy, vol = _triangle_derivatives(nodes, elems)
    freedofs = _free_dofs(spec.key, nodes, spec.freedofs_dtype)
    adj_row, adj_col = _free_adjacency(elems, freedofs, nodes.shape[0])

    tmp_path = spec.output_path.with_suffix(spec.output_path.suffix + ".tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    if tmp_path.exists():
        tmp_path.unlink()
    with h5py.File(tmp_path, "w") as out:
        _write_dataset(out, "nodes", nodes)
        _write_dataset(out, "elems", elems)
        _write_dataset(out, "freedofs", freedofs)
        _write_dataset(out, "dvx", dvx)
        _write_dataset(out, "dvy", dvy)
        _write_dataset(out, "vol", vol)
        _write_dataset(out, "u_0", np.zeros(nodes.shape[0], dtype=np.float64))
        if spec.key == "plaplace":
            _write_dataset(out, "f", _lumped_plaplace_force(nodes.shape[0], elems, vol))
        for key, value in attrs.items():
            _write_dataset(out, key, value)
        adj = out.create_group("adjacency")
        _write_dataset(adj, "row", adj_row)
        _write_dataset(adj, "col", adj_col)
        _write_dataset(adj, "data", np.ones(adj_row.shape[0], dtype=spec.adjacency_data_dtype))
        _write_dataset(adj, "shape", np.asarray([freedofs.size, freedofs.size], dtype=np.int64))
    tmp_path.replace(spec.output_path)
    return spec.output_path


def validate(path: Path) -> dict[str, int]:
    with h5py.File(path, "r") as handle:
        return {
            "nodes": int(handle["nodes"].shape[0]),
            "elements": int(handle["elems"].shape[0]),
            "freedofs": int(handle["freedofs"].shape[0]),
            "adjacency_nnz": int(handle["adjacency/row"].shape[0]),
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("all", *SPECS.keys()), default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    keys = list(SPECS) if args.problem == "all" else [str(args.problem)]
    for key in keys:
        spec = SPECS[key]
        path = spec.output_path
        if not args.validate_only:
            path = generate(spec, force=bool(args.force))
        stats = validate(path)
        print(
            f"{key}: {path} nodes={stats['nodes']} elements={stats['elements']} "
            f"freedofs={stats['freedofs']} adjacency_nnz={stats['adjacency_nnz']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
