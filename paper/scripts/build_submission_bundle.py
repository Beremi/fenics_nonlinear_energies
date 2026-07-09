#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import REPO_ROOT


BUNDLE_ROOT = REPO_ROOT / "artifacts" / "reproduction" / "paper_submission_2026_07_08"
INPUT_ROOT = BUNDLE_ROOT / "inputs"

P3D_VALIDATION_ROOT = REPO_ROOT / "artifacts" / "raw_results" / "plasticity3d_validation"
P3D_ABLATION_ROOT = REPO_ROOT / "artifacts" / "raw_results" / "plasticity3d_derivative_ablation"
JAX_FEM_ROOT = REPO_ROOT / "artifacts" / "raw_results" / "jax_fem_hyperelastic_baseline"
P3D_SCALING_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "plasticity3d_p4_l1_2_mumps_pmg_step_grad_local_karolina_scaling"
)
P3D_DIRECT_BRANCH_ROOT = REPO_ROOT / "artifacts" / "raw_results" / "debug" / "p2_direct_branch_lambda1p6_merged"
SOURCE_BRANCH_ROOT = (
    REPO_ROOT
    / "tmp"
    / "source_compare"
    / "slope_stability_octave_ref"
    / "slope_stability"
    / "artifacts"
    / "compare_direct_branch_lambda1p6"
)


def _repo_rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _source_id(path: Path) -> str:
    resolved = path.resolve()
    try:
        relative_to_source_branch = resolved.relative_to(SOURCE_BRANCH_ROOT.resolve())
    except ValueError:
        return _repo_rel(path)
    return f"external_reference/slope_stability_octave_ref/compare_direct_branch_lambda1p6/{relative_to_source_branch.as_posix()}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sanitize_string(value: str) -> str:
    repo_prefix = REPO_ROOT.resolve().as_posix() + "/"
    text = value.replace(repo_prefix, "")
    bundle_prefix = _repo_rel(BUNDLE_ROOT)
    replacements = {
        "artifacts/raw_results/jax_fem_hyperelastic_baseline/comparison_summary.json": (
            f"{bundle_prefix}/inputs/jax_fem_hyperelastic_baseline/comparison_summary.json"
        ),
        "artifacts/raw_results/jax_fem_hyperelastic_baseline/parity/repo_serial_direct_state.npz": (
            f"{bundle_prefix}/inputs/jax_fem_hyperelastic_baseline/parity/repo_serial_direct_state.npz"
        ),
        "artifacts/raw_results/jax_fem_hyperelastic_baseline/parity/jax_fem_umfpack_serial_state.npz": (
            f"{bundle_prefix}/inputs/jax_fem_hyperelastic_baseline/parity/jax_fem_umfpack_serial_state.npz"
        ),
        "artifacts/raw_results/jax_fem_hyperelastic_baseline/parity/repo_serial_direct.json": (
            f"{bundle_prefix}/inputs/jax_fem_hyperelastic_baseline/parity/repo_serial_direct.json"
        ),
        "artifacts/raw_results/jax_fem_hyperelastic_baseline/parity/jax_fem_umfpack_serial.json": (
            f"{bundle_prefix}/inputs/jax_fem_hyperelastic_baseline/parity/jax_fem_umfpack_serial.json"
        ),
        "tmp/source_compare/slope_stability_octave_ref/slope_stability/artifacts/compare_direct_branch_lambda1p6/final_source_state.mat": (
            f"{bundle_prefix}/inputs/plasticity3d_validation/source_branch/final_source_state.mat"
        ),
        "tmp/source_compare/slope_stability_octave_ref/slope_stability/artifacts/compare_direct_branch_lambda1p6": (
            f"{bundle_prefix}/inputs/plasticity3d_validation/source_branch"
        ),
        "artifacts/raw_results/debug/p2_direct_branch_lambda1p6_merged": (
            f"{bundle_prefix}/inputs/plasticity3d_validation/maintained_branch"
        ),
        "tmp/source_compare/slope_stability_petsc4py": "external_reference/slope_stability_petsc4py",
        ".venv/bin/python": "python",
        ".venv/lib/python3.12/site-packages": "python-environment/site-packages",
        "tmp_work/jax_fem_0_0_10_py312/bin/python": "python",
        "tmp_work/jax_fem_0_0_10_py312": "external_environment/jax_fem_0_0_10_py312",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, str):
        return _sanitize_string(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _copy_json(source: Path, dest: Path, copied: list[dict[str, str]]) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    sanitized = _sanitize_json_value(payload)
    _write_json(dest, sanitized)
    copied.append(
        {
            "source_path": _source_id(source),
            "bundle_path": _repo_rel(dest),
            "source_sha256": _sha256(source),
            "bundle_sha256": _sha256(dest),
            "sanitization": "local paths rewritten to archive-neutral references; non-finite JSON numbers written as null",
        }
    )


def _copy_binary(source: Path, dest: Path, copied: list[dict[str, str]]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    copied.append(
        {
            "source_path": _source_id(source),
            "bundle_path": _repo_rel(dest),
            "source_sha256": _sha256(source),
            "bundle_sha256": _sha256(dest),
            "sanitization": "byte-for-byte copy",
        }
    )


def _copy_csv(source: Path, dest: Path, copied: list[dict[str, str]]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8", newline="") as in_handle:
        reader = csv.DictReader(in_handle)
        rows = [
            {key: _sanitize_string(value) if isinstance(value, str) else value for key, value in row.items()}
            for row in reader
        ]
        fieldnames = list(reader.fieldnames or [])
    with dest.open("w", encoding="utf-8", newline="") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    copied.append(
        {
            "source_path": _source_id(source),
            "bundle_path": _repo_rel(dest),
            "source_sha256": _sha256(source),
            "bundle_sha256": _sha256(dest),
            "sanitization": "local paths rewritten to archive-neutral references",
        }
    )


def main() -> None:
    if BUNDLE_ROOT.exists():
        shutil.rmtree(BUNDLE_ROOT)
    copied: list[dict[str, str]] = []
    _copy_json(
        P3D_VALIDATION_ROOT / "validation_manifest.json",
        INPUT_ROOT / "plasticity3d_validation" / "validation_manifest.json",
        copied,
    )
    _copy_json(
        P3D_VALIDATION_ROOT / "comparison_summary.json",
        INPUT_ROOT / "plasticity3d_validation" / "comparison_summary.json",
        copied,
    )
    _copy_json(
        SOURCE_BRANCH_ROOT / "branch_summary.json",
        INPUT_ROOT / "plasticity3d_validation" / "source_branch" / "branch_summary.json",
        copied,
    )
    _copy_binary(
        SOURCE_BRANCH_ROOT / "final_source_state.mat",
        INPUT_ROOT / "plasticity3d_validation" / "source_branch" / "final_source_state.mat",
        copied,
    )
    _copy_json(
        P3D_DIRECT_BRANCH_ROOT / "branch_summary.json",
        INPUT_ROOT / "plasticity3d_validation" / "maintained_branch" / "branch_summary.json",
        copied,
    )
    _copy_json(
        P3D_ABLATION_ROOT / "comparison_summary.json",
        INPUT_ROOT / "plasticity3d_derivative_ablation" / "comparison_summary.json",
        copied,
    )
    _copy_json(
        JAX_FEM_ROOT / "comparison_summary.json",
        INPUT_ROOT / "jax_fem_hyperelastic_baseline" / "comparison_summary.json",
        copied,
    )
    _copy_json(
        JAX_FEM_ROOT / "run_manifest.json",
        INPUT_ROOT / "jax_fem_hyperelastic_baseline" / "run_manifest.json",
        copied,
    )
    for name in (
        "repo_serial_direct.json",
        "jax_fem_umfpack_serial.json",
        "repo_serial_direct_state.npz",
        "jax_fem_umfpack_serial_state.npz",
    ):
        source = JAX_FEM_ROOT / "parity" / name
        dest = INPUT_ROOT / "jax_fem_hyperelastic_baseline" / "parity" / name
        if source.suffix == ".json":
            _copy_json(source, dest, copied)
        else:
            _copy_binary(source, dest, copied)
    for name in ("local_solver_total_scaling.csv", "karolina_rpn16_solver_total_scaling.csv"):
        _copy_csv(
            P3D_SCALING_ROOT / name,
            INPUT_ROOT / "plasticity3d_lambda155_scaling" / name,
            copied,
        )

    manifest = {
        "id": "paper_submission_2026_07_08",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "git_commit": _git_head(),
        "purpose": "Archive-neutral provenance bundle for manuscript-critical paper figure inputs.",
        "scope": [
            "Plasticity3D endpoint-surrogate validation inputs",
            "Plasticity3D derivative-route comparison summary",
            "Hyperelastic JAX-FEM comparison summary and terminal states",
            "Plasticity3D lambda=1.55 local/multi-node scaling summaries",
        ],
        "source_files": copied,
        "known_limitations": [
            "This bundle normalizes existing paper-critical provenance without rerunning MPI campaigns.",
            "Target-journal metadata, repository license, and permanent archive DOI remain outside this bundle.",
        ],
        "validation": {
            "paper_asset_check": "python paper/scripts/validate_paper_assets.py",
            "archive_neutral_check": "python paper/scripts/validate_paper_assets.py --archive-neutral",
        },
    }
    _write_json(BUNDLE_ROOT / "manifest.json", manifest)
    print(BUNDLE_ROOT / "manifest.json")


if __name__ == "__main__":
    main()
