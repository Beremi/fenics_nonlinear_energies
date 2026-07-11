#!/usr/bin/env python3
"""Create one clean, prescribed Plasticity3D state for fixed-state studies.

The state is analytic rather than a nonlinear-solver endpoint.  This keeps the
quadrature and derivative checks independent of globalization and stopping
behavior while ensuring that every degree uses the same dimensionless field.
The managed publication receipt binds the resulting NPZ, this runner, and the
manifested mesh input to one clean experiment commit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.core.benchmark.run_record import (
    atomic_write_json,
    check_experiment_preflight,
    sha256_file,
    utc_now_iso,
)
from src.core.benchmark.state_export import export_plasticity3d_state_npz
from src.problems.slope_stability_3d.support.fixed_state import (
    prescribed_analytic_displacement,
)
from src.problems.slope_stability_3d.support.mesh import (
    default_tetra_quadrature_rule_id,
    load_same_mesh_case_hdf5_light,
    manifested_same_mesh_case_provenance,
    same_mesh_case_hdf5_path,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ID = "plasticity3d-prescribed-fixed-state"
SCHEMA_VERSION = 1


def prepare(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    if output == manifest:
        raise ValueError("state and manifest paths must be distinct")
    for path in (output, manifest):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite {path}")

    preflight = check_experiment_preflight(
        REPO_ROOT,
        run_kind=str(args.run_kind),
        pilot_dirty_override=bool(args.pilot_dirty_override),
        pilot_override_reason=args.pilot_override_reason,
    )
    degree = int(args.degree)
    quadrature_rule_id = default_tetra_quadrature_rule_id(degree)
    case_path = same_mesh_case_hdf5_path(
        str(args.mesh_name),
        degree,
        str(args.constraint_variant),
        quadrature_rule_id=quadrature_rule_id,
    )
    publication_mesh_manifest = getattr(args, "publication_mesh_manifest", None)
    mesh_input = (
        manifested_same_mesh_case_provenance(
            case_path,
            manifest_path=publication_mesh_manifest,
        )
        if publication_mesh_manifest is not None
        else None
    )
    case = load_same_mesh_case_hdf5_light(
        str(args.mesh_name),
        degree,
        constraint_variant=str(args.constraint_variant),
        quadrature_rule_id=quadrature_rule_id,
    )
    if mesh_input is None:
        mesh_input = {
            "path": str(case_path.resolve()),
            "sha256": sha256_file(case_path),
            "bytes": int(case_path.stat().st_size),
            "manifest": None,
        }
    coords_ref = np.asarray(case["nodes"], dtype=np.float64)
    prescribed = prescribed_analytic_displacement(
        coords_ref,
        amplitude=float(args.amplitude),
    ).reshape(-1)
    freedofs = np.asarray(case["freedofs"], dtype=np.int64).reshape(-1)
    constrained_reference = np.asarray(case["u_0"], dtype=np.float64).reshape(-1)
    if constrained_reference.size != prescribed.size:
        raise ValueError("case reference state and prescribed field are not aligned")
    displacement = constrained_reference.copy()
    displacement[freedofs] = prescribed[freedofs]
    if not np.all(np.isfinite(displacement)):
        raise FloatingPointError("constrained prescribed state is nonfinite")

    output.parent.mkdir(parents=True, exist_ok=True)
    export_plasticity3d_state_npz(
        output,
        coords_ref=coords_ref,
        x_final=coords_ref + displacement.reshape((-1, 3)),
        tetrahedra=np.asarray(case["elems_scalar"], dtype=np.int32),
        surface_faces=np.asarray(case["surf"], dtype=np.int32),
        boundary_label=np.asarray(case["boundary_label"], dtype=np.int32),
        mesh_name=str(args.mesh_name),
        element_degree=degree,
        lambda_target=float(args.lambda_target),
        metadata={
            "solver_family": "prescribed_fixed_state",
            "state_kind": "analytic_not_solved",
            "state_label": str(args.state_label),
            "state_amplitude": float(args.amplitude),
            "constraint_variant": str(args.constraint_variant),
            "quadrature_rule_id": quadrature_rule_id,
            "run_kind": str(args.run_kind),
            "git_commit": preflight.git_commit,
            "git_clean": bool(preflight.git_clean),
        },
    )
    payload: dict[str, object] = {
        "schema": {"id": SCHEMA_ID, "version": SCHEMA_VERSION},
        "status": "completed",
        "run_kind": str(args.run_kind),
        "created_at_utc": utc_now_iso(),
        "provenance": {
            **preflight.provenance_fields(),
            "producer": "experiments/runners/prepare_plasticity3d_fixed_state.py",
            "deterministic_policy": (
                "analytic trigonometric field on normalized coordinates; constrained "
                "coefficients replaced by the case reference lift"
            ),
        },
        "identifiers": {
            "mesh_name": str(args.mesh_name),
            "element_degree": degree,
            "constraint_variant": str(args.constraint_variant),
            "quadrature_rule_id": quadrature_rule_id,
            "lambda_target": float(args.lambda_target),
            "state_label": str(args.state_label),
            "state_amplitude": float(args.amplitude),
            "state_kind": "analytic_not_solved",
        },
        "dimensions": {
            "nodes": int(coords_ref.shape[0]),
            "elements": int(np.asarray(case["elems_scalar"]).shape[0]),
            "degrees_of_freedom": int(displacement.size),
            "free_degrees_of_freedom": int(freedofs.size),
        },
        "mesh_input": mesh_input,
        "state": {
            "path": output.name,
            "sha256": sha256_file(output),
            "coefficient_l2_norm": float(np.linalg.norm(displacement)),
            "coefficient_linf_norm": float(np.linalg.norm(displacement, ord=np.inf)),
            "constrained_coefficients_match_reference": bool(
                np.array_equal(
                    displacement[np.setdiff1d(np.arange(displacement.size), freedofs)],
                    constrained_reference[
                        np.setdiff1d(np.arange(displacement.size), freedofs)
                    ],
                )
            ),
        },
    }
    atomic_write_json(manifest, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--mesh-name", default="hetero_ssr_L1")
    parser.add_argument("--constraint-variant", default="glued_bottom")
    parser.add_argument("--lambda-target", type=float, default=1.55)
    parser.add_argument("--state-label", default="mixed")
    parser.add_argument("--amplitude", type=float, default=2.0e-2)
    parser.add_argument("--run-kind", choices=("publication", "pilot"), default="publication")
    parser.add_argument("--pilot-dirty-override", action="store_true")
    parser.add_argument("--pilot-override-reason")
    parser.add_argument("--publication-mesh-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    prepare(args)
    print(Path(args.output).resolve())


if __name__ == "__main__":
    main()
