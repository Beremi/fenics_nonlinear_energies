#!/usr/bin/env python3
"""Re-evaluate a saved Plasticity3D state with independent quadrature rules."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.problems.slope_stability_3d.support.fixed_state import (
    BRANCH_NAMES,
    FixedStateQuadratureDiagnostics,
    evaluate_fixed_state_quadrature_diagnostics,
)
from src.problems.slope_stability_3d.support.mesh import (
    TETRA_QUADRATURE_RULE_IDS,
    ensure_same_mesh_case_hdf5,
    load_case_hdf5_fields,
)


def _scalar(state: np.lib.npyio.NpzFile, key: str, default: object = None) -> object:
    if key not in state:
        return default
    value = np.asarray(state[key])
    return value.item() if value.shape == () else value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_content_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _save_array(
    path: Path,
    values: np.ndarray,
    *,
    dtype: np.dtype,
    content: str,
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npy")
    array = np.asarray(values, dtype=dtype)
    np.save(temporary, array, allow_pickle=False)
    temporary.replace(path)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "content_sha256": _array_content_sha256(array),
        "dtype": str(array.dtype),
        "shape": [int(value) for value in array.shape],
        "content": content,
    }


def _scaled_difference(
    value: float,
    reference: float,
    *,
    scale: float | None = None,
) -> dict[str, float]:
    absolute = abs(float(value) - float(reference))
    denominator = max(
        (
            max(abs(float(value)), abs(float(reference)))
            if scale is None
            else abs(float(scale))
        ),
        np.finfo(float).tiny,
    )
    return {
        "absolute_difference": float(absolute),
        "relative_difference": float(absolute / denominator),
    }


def _vector_difference(values: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = np.asarray(values, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    absolute_l2 = float(np.linalg.norm(delta))
    reference_l2 = float(np.linalg.norm(reference))
    values_l2 = float(np.linalg.norm(values))
    return {
        "absolute_l2_difference": absolute_l2,
        "relative_l2_difference": float(
            absolute_l2 / max(reference_l2, values_l2, np.finfo(float).tiny)
        ),
        "absolute_linf_difference": float(np.linalg.norm(delta, ord=np.inf)),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    state_path = Path(args.state).resolve()
    output_value = getattr(args, "output", None)
    experiment_root = (
        Path(output_value).resolve().parent if output_value is not None else None
    )
    state_record = str(state_path)
    if experiment_root is not None:
        try:
            state_record = state_path.relative_to(experiment_root).as_posix()
        except ValueError as exc:
            raise RuntimeError(
                "publication quadrature state must be contained in the output "
                "experiment directory"
            ) from exc
    if not state_path.is_file():
        raise ValueError("quadrature state must be an existing regular file")

    action_output_dir = (
        Path(args.action_output_dir).resolve()
        if getattr(args, "action_output_dir", None) is not None
        else None
    )
    if experiment_root is not None and action_output_dir is not None:
        try:
            action_output_dir.relative_to(experiment_root)
        except ValueError as exc:
            raise RuntimeError(
                "publication quadrature artifacts must be contained in the output "
                "experiment directory"
            ) from exc
    if action_output_dir is not None and action_output_dir.exists():
        if not action_output_dir.is_dir():
            raise ValueError("quadrature artifact output path must be a directory")

    with np.load(state_path, allow_pickle=False) as state:
        mesh_name = str(_scalar(state, "mesh_name", ""))
        element_degree = int(_scalar(state, "element_degree", 0))
        lambda_target = float(_scalar(state, "lambda_target", float("nan")))
        displacement = np.asarray(state["displacement"], dtype=np.float64)
        coords_ref = np.asarray(state["coords_ref"], dtype=np.float64)
        solve_quadrature_rule_id = str(_scalar(state, "quadrature_rule_id", "degree_default"))
        state_constraint_variant = str(_scalar(state, "constraint_variant", ""))
    if not mesh_name or element_degree not in {1, 2, 4} or not np.isfinite(lambda_target):
        raise ValueError(
            "state must contain mesh_name, element_degree, and finite lambda_target metadata"
        )

    constraint_variant = str(
        args.constraint_variant or state_constraint_variant or "glued_bottom"
    )
    case_path = ensure_same_mesh_case_hdf5(
        mesh_name,
        element_degree,
        constraint_variant=constraint_variant,
        quadrature_rule_id=solve_quadrature_rule_id,
    )
    case_data, _adjacency = load_case_hdf5_fields(
        case_path,
        fields=(
            "degree",
            "quadrature_rule_id",
            "nodes",
            "elems_scalar",
            "elems",
            "material_id",
            "freedofs",
        ),
        load_adjacency=False,
    )
    np.testing.assert_allclose(
        np.asarray(case_data["nodes"], dtype=np.float64),
        coords_ref,
        rtol=0.0,
        atol=float(args.coordinate_atol),
        err_msg="saved state and selected case do not share reference coordinates",
    )

    rules = tuple(
        part.strip()
        for part in str(args.quadrature_rules).split(",")
        if part.strip()
    )
    if not rules:
        raise ValueError("quadrature_rules must contain at least one rule ID")
    unknown = sorted(set(rules).difference(TETRA_QUADRATURE_RULE_IDS))
    if unknown:
        raise ValueError(f"unsupported quadrature rule IDs: {unknown}")
    if len(set(rules)) != len(rules):
        raise ValueError("quadrature_rules must not contain duplicate rule IDs")

    diagnostics: list[FixedStateQuadratureDiagnostics] = [
        evaluate_fixed_state_quadrature_diagnostics(
            case_data,
            displacement,
            lambda_target=lambda_target,
            quadrature_rule_id=rule_id,
            element_chunk_size=int(args.element_chunk_size),
        )
        for rule_id in rules
    ]
    evaluations = [dict(item.summary) for item in diagnostics]
    for item, row in zip(diagnostics, evaluations):
        action = np.asarray(item.hessian_action, dtype=np.float64)
        residual = np.asarray(item.full_residual, dtype=np.float64)
        branch_labels = np.asarray(item.branch_labels, dtype=np.int8)
        row["hessian_action_content_sha256"] = _array_content_sha256(action)
        row["residual_content_sha256"] = _array_content_sha256(residual)
        row["branch_map_content_sha256"] = _array_content_sha256(branch_labels)
        row["hessian_action_artifact"] = (
            _save_array(
                action_output_dir / f"{row['quadrature_rule_id']}_hessian_action.npy",
                action,
                dtype=np.dtype(np.float64),
                content=(
                    "full global Hessian action for the reported deterministic free-DOF "
                    "unit direction"
                ),
            )
            if action_output_dir is not None
            else None
        )
        row["residual_artifact"] = (
            _save_array(
                action_output_dir / f"{row['quadrature_rule_id']}_residual.npy",
                residual,
                dtype=np.dtype(np.float64),
                content="full global residual at the saved endpoint",
            )
            if action_output_dir is not None
            else None
        )
        row["branch_map_artifact"] = (
            _save_array(
                action_output_dir / f"{row['quadrature_rule_id']}_branch_map.npy",
                branch_labels,
                dtype=np.dtype(np.int8),
                content=(
                    "element-major quadrature-point branch labels using the published "
                    "elastic, shear, left-edge, right-edge, apex ordering"
                ),
            )
            if action_output_dir is not None
            else None
        )

    reference_diagnostics = diagnostics[-1]
    reference = evaluations[-1]
    tiny = np.finfo(float).tiny
    energy_scale = max(
        abs(float(reference["internal_energy"])),
        abs(float(reference["external_work"])),
        abs(float(reference["total_potential_energy"])),
        tiny,
    )
    scalar_metrics = (
        "internal_energy",
        "external_work",
        "total_potential_energy",
        "u_max",
        "full_residual_l2_norm",
        "full_residual_linf_norm",
        "free_residual_l2_norm",
        "free_residual_linf_norm",
        "full_hessian_action_l2_norm",
        "full_hessian_action_linf_norm",
        "free_hessian_action_l2_norm",
        "free_hessian_action_linf_norm",
        "minimum_normalized_active_branch_margin",
        "minimum_raw_principal_value_gap",
        "minimum_normalized_principal_value_gap",
        "minimum_normalized_constitutive_denominator",
        "quadrature_point_fraction_at_or_below_margin_gate",
    )
    for item, row in zip(diagnostics, evaluations):
        comparison: dict[str, dict[str, float]] = {}
        for metric in scalar_metrics:
            scale = (
                energy_scale
                if metric == "total_potential_energy"
                else None
            )
            comparison[metric] = _scaled_difference(
                float(row[metric]),
                float(reference[metric]),
                scale=scale,
            )
        row["comparison_to_last_rule"] = comparison
        row["relative_total_potential_difference_from_last_rule"] = float(
            comparison["total_potential_energy"]["relative_difference"]
        )
        row["full_residual_vector_comparison_to_last_rule"] = _vector_difference(
            item.full_residual,
            reference_diagnostics.full_residual,
        )
        row["free_residual_vector_comparison_to_last_rule"] = _vector_difference(
            item.full_residual[item.freedofs],
            reference_diagnostics.full_residual[reference_diagnostics.freedofs],
        )
        row["hessian_action_vector_comparison_to_last_rule"] = _vector_difference(
            item.hessian_action,
            reference_diagnostics.hessian_action,
        )
        row["free_hessian_action_vector_comparison_to_last_rule"] = _vector_difference(
            item.hessian_action[item.freedofs],
            reference_diagnostics.hessian_action[reference_diagnostics.freedofs],
        )
        point_fraction_differences = {
            name: float(
                abs(
                    float(row["branch_point_fractions"][name])
                    - float(reference["branch_point_fractions"][name])
                )
            )
            for name in BRANCH_NAMES
        }
        weight_fraction_differences = {
            name: float(
                abs(
                    float(row["branch_absolute_quadrature_weight_fractions"][name])
                    - float(
                        reference["branch_absolute_quadrature_weight_fractions"][name]
                    )
                )
            )
            for name in BRANCH_NAMES
        }
        row["branch_comparison_to_last_rule"] = {
            "point_fraction_absolute_differences": point_fraction_differences,
            "point_fraction_l1_difference": float(sum(point_fraction_differences.values())),
            "absolute_weight_fraction_absolute_differences": weight_fraction_differences,
            "absolute_weight_fraction_l1_difference": float(
                sum(weight_fraction_differences.values())
            ),
            "interpretation": (
                "point fractions use different sampling sets; absolute-weight fractions use "
                "physical quadrature weights in magnitude because the 11-point rule has a "
                "negative weight"
            ),
        }

    direction_hashes = {
        _array_content_sha256(item.deterministic_direction) for item in diagnostics
    }
    free_dof_sets_match = all(
        np.array_equal(item.freedofs, reference_diagnostics.freedofs)
        for item in diagnostics
    )
    if len(direction_hashes) != 1 or not free_dof_sets_match:
        raise RuntimeError("cross-rule diagnostics did not use one common direction/free-DOF set")

    if experiment_root is not None:
        for row in evaluations:
            for key in (
                "hessian_action_artifact",
                "residual_artifact",
                "branch_map_artifact",
            ):
                artifact = row.get(key)
                if isinstance(artifact, dict):
                    artifact_path = Path(str(artifact["path"])).resolve()
                    try:
                        artifact["path"] = artifact_path.relative_to(
                            experiment_root
                        ).as_posix()
                    except ValueError as exc:
                        raise RuntimeError(
                            "publication quadrature artifact escapes the output experiment "
                            "directory"
                        ) from exc
    return {
        "experiment_id": "EXP-DISC-001-P3D-FIXED-STATE-QUADRATURE",
        "status": "completed",
        "state_path": state_record,
        "case_hdf5": str(case_path),
        "mesh_name": mesh_name,
        "element_degree": element_degree,
        "constraint_variant": constraint_variant,
        "solve_quadrature_rule_id": str(case_data["quadrature_rule_id"]),
        "lambda_target": lambda_target,
        "reference_rule_id": str(reference["quadrature_rule_id"]),
        "reference_energy_scale": float(energy_scale),
        "common_direction_content_sha256": next(iter(direction_hashes)),
        "common_free_dof_set": bool(free_dof_sets_match),
        "comparison_scope": (
            "same saved coefficient state, free-DOF set, deterministic direction, rebuilt "
            "geometry/material/load, and named-rule-specific quadrature"
        ),
        "relative_difference_scale": (
            "maximum of compared norms/magnitudes and machine tiny; total potential uses "
            "the declared reference energy scale"
        ),
        "evaluations": evaluations,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--constraint-variant",
        choices=("componentwise_bottom", "glued_bottom"),
        default=None,
        help="Override the state metadata; defaults to its constraint variant",
    )
    parser.add_argument(
        "--quadrature-rules",
        default=",".join(TETRA_QUADRATURE_RULE_IDS),
        help="Comma-separated named rule IDs; the last rule is the comparison reference",
    )
    parser.add_argument("--element-chunk-size", type=int, default=256)
    parser.add_argument("--coordinate-atol", type=float, default=1.0e-10)
    parser.add_argument(
        "--action-output-dir",
        type=Path,
        default=None,
        help="Optionally save one full assembled Hessian-action .npy file per rule",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = run(args)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
