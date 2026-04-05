"""CLI for importing source 3D heterogeneous SSR meshes into HDF5 snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.problems.slope_stability_3d.support.mesh import (
    DEFAULT_MESH_NAME,
    build_case_data_from_raw_mesh,
    ensure_same_mesh_case_hdf5,
    raw_mesh_path_for_name,
    same_mesh_case_hdf5_path,
    write_case_hdf5,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import a source 3D heterogeneous SSR .msh mesh into a same-mesh HDF5 snapshot"
    )
    parser.add_argument(
        "--mesh_name",
        type=str,
        default=DEFAULT_MESH_NAME,
        help="Logical mesh name, e.g. hetero_ssr_L1",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default="",
        help="Optional explicit .msh path; defaults to the vendored raw mesh for --mesh_name",
    )
    parser.add_argument(
        "--case",
        type=str,
        default="",
        help="Optional explicit case/output prefix; defaults to --mesh_name",
    )
    parser.add_argument(
        "--degree",
        type=int,
        choices=(1, 2, 4),
        required=True,
        help="Same-mesh Lagrange degree to generate",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output HDF5 path; defaults to the canonical data/meshes location",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output HDF5 snapshot",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    mesh_name = str(args.case or args.mesh_name)
    mesh_path = Path(args.mesh) if str(args.mesh).strip() else raw_mesh_path_for_name(args.mesh_name)
    out_path = (
        Path(args.out)
        if str(args.out).strip()
        else same_mesh_case_hdf5_path(mesh_name, int(args.degree))
    )

    if out_path.exists() and not args.overwrite and not str(args.out).strip():
        ensure_same_mesh_case_hdf5(mesh_name, int(args.degree))
        print(str(out_path))
        return
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{out_path} already exists; pass --overwrite to replace it"
        )

    case_data = build_case_data_from_raw_mesh(
        mesh_path,
        mesh_name=mesh_name,
        degree=int(args.degree),
    )
    write_case_hdf5(out_path, case_data)
    print(str(out_path))


if __name__ == "__main__":
    main()
