#!/usr/bin/env python3
"""Generate the narrowed-scope local stopping table and presentation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_MANIFEST = (
    REPO_ROOT / "paper/tables/generated/stopping_local_manifest.json"
)
DEFAULT_OUT_DIR = REPO_ROOT / "paper/tables/generated"
TABLE_NAME = "stopping_submission_status.tex"
MANIFEST_NAME = "stopping_submission_manifest.json"
SCHEMA_ID = "fenics-nonlinear-energies.exp-stop-001-submission-table-manifest"
SCHEMA_VERSION = 1
SOURCE_SCHEMA_ID = "fenics-nonlinear-energies.exp-stop-001-local-table-manifest"

EXPECTED_FAMILIES = (
    ("ginzburg_landau", 8, 8, 5),
    ("hyperelasticity_reference_riesz", 6, 6, 6),
    ("hyperelasticity_nonlinear_stopping", 8, 8, 6),
    ("plasticity3d_fixed_state_linear", 15, 13, 8),
    ("plasticity3d_nonlinear_stopping", 8, 8, 3),
)

LABELS = {
    "ginzburg_landau": "Ginzburg--Landau endpoints",
    "hyperelasticity_reference_riesz": "Hyperelasticity metric checks",
    "hyperelasticity_nonlinear_stopping": "Hyperelasticity nonlinear endpoints",
    "plasticity3d_fixed_state_linear": "Mohr--Coulomb surrogate, fixed state",
    "plasticity3d_nonlinear_stopping": (
        "Mohr--Coulomb surrogate, nonlinear endpoint"
    ),
}


def read_strict_json(path: Path) -> dict[str, object]:
    def reject_constant(raw: str) -> object:
        raise ValueError(f"nonfinite JSON constant is forbidden: {raw}")

    payload = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"path lies outside the repository: {path}") from exc


def safe_repo_path(raw: object, *, label: str) -> Path:
    if (
        not isinstance(raw, str)
        or not raw
        or Path(raw).is_absolute()
        or ".." in Path(raw).parts
        or Path(raw).as_posix() != raw
    ):
        raise ValueError(f"{label} must be a canonical repository-relative path")
    path = (REPO_ROOT / raw).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"{label} resolves outside the repository") from exc
    return path


def load_family_rows(
    source_manifest: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    source_manifest = source_manifest.resolve()
    payload = read_strict_json(source_manifest)
    expected_top = {
        "schema_id": SOURCE_SCHEMA_ID,
        "schema_version": 1,
        "status": "admitted_local_calibration_cluster_deferred",
        "publication_evidence": True,
        "experiment_id": "EXP-STOP-001",
        "complete_exp_stop_pass": False,
        "timing_claim_admissible": False,
        "population_robustness_claim_admissible": False,
    }
    for key, expected in expected_top.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"source stopping manifest field {key} must equal {expected!r}"
            )

    source_analysis = payload.get("source_analysis")
    if not isinstance(source_analysis, Mapping):
        raise ValueError("source stopping manifest has no source_analysis binding")
    analysis_path = safe_repo_path(
        source_analysis.get("path"), label="source_analysis.path"
    )
    if (
        not analysis_path.is_file()
        or source_analysis.get("sha256") != sha256_file(analysis_path)
    ):
        raise ValueError("source stopping analysis is missing or has a stale hash")

    audit = payload.get("audit")
    adjudication = (
        audit.get("scientific_adjudication") if isinstance(audit, Mapping) else None
    )
    rows = (
        adjudication.get("family_summaries")
        if isinstance(adjudication, Mapping)
        else None
    )
    if not isinstance(rows, list):
        raise ValueError("source stopping family summaries are missing")
    by_family = {
        str(row.get("family")): dict(row)
        for row in rows
        if isinstance(row, Mapping)
    }
    if set(by_family) != {row[0] for row in EXPECTED_FAMILIES}:
        raise ValueError("source stopping family grid is not the frozen five-family grid")

    ordered: list[dict[str, object]] = []
    for family, completed, admitted, accepted in EXPECTED_FAMILIES:
        row = by_family[family]
        expected = {
            "required_local": completed,
            "completed_receipts": completed,
            "admitted_endpoints": admitted,
            "accepted_comparisons": accepted,
            "comparison_rows": completed,
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(
                    f"{family}.{key} must equal {value}, got {row.get(key)!r}"
                )
        ordered.append(row)

    if (
        sum(int(row["required_local"]) for row in ordered) != 45
        or sum(int(row["admitted_endpoints"]) for row in ordered) != 43
        or sum(int(row["accepted_comparisons"]) for row in ordered) != 28
    ):
        raise ValueError("source stopping aggregate counts differ from 45/43/28")
    return payload, ordered


def render_table(rows: list[dict[str, object]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        (
            r"  \caption{Deterministic local stopping-calibration outcomes. "
            r"An accepted comparison satisfies every family-specific "
            r"same-discretization gate in "
            r"Appendix~\ref{subsec:stopping-study-scope}.}"
        ),
        r"  \label{tab:stopping-local-status}",
        r"  \begin{tabularx}{\linewidth}{L{1.70}C{0.75}C{0.75}C{0.75}}",
        r"    \toprule",
        (
            r"    Problem family & Executions & Admitted records "
            r"& Accepted comparisons \\"
        ),
        r"    \midrule",
    ]
    for row in rows:
        family = str(row["family"])
        completed = int(row["completed_receipts"])
        admitted = int(row["admitted_endpoints"])
        accepted = int(row["accepted_comparisons"])
        lines.append(
            f"    {LABELS[family]} & {completed}/{completed} & "
            f"{admitted}/{completed} & {accepted}/{completed} \\\\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabularx}",
            r"  \begin{minipage}{0.96\linewidth}\small",
            (
                r"    The reported subset contains 45 of the 52 computations "
                r"in the frozen plan. Four degree-four nonlinear rows and "
                r"three MPI-consistency rows remain cluster-deferred; hence "
                r"the complete stopping protocol has not passed."
            ),
            r"  \end{minipage}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
    temporary.replace(path)


def generate(
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, object]:
    source_manifest = source_manifest.resolve()
    source, rows = load_family_rows(source_manifest)
    table = render_table(rows).encode("utf-8")
    table_hash = hashlib.sha256(table).hexdigest()
    analysis = source["source_analysis"]
    assert isinstance(analysis, Mapping)
    tools = {
        "generator": REPO_ROOT
        / "paper/scripts/generate_stopping_submission_status.py",
        "checker": REPO_ROOT
        / "paper/scripts/check_stopping_submission_manifest.py",
    }
    manifest: dict[str, object] = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "admitted_reported_local_subset",
        "publication_evidence": True,
        "experiment_id": "EXP-STOP-001",
        "claim_scope": "deterministic_same_discretization_local_subset",
        "reported_local_subset_complete": True,
        "complete_exp_stop_pass": False,
        "timing_claim_admissible": False,
        "population_robustness_claim_admissible": False,
        "source_commit": source["source_commit"],
        "source_manifest": {
            "path": repo_relative(source_manifest),
            "sha256": sha256_file(source_manifest),
        },
        "source_analysis": {
            "path": str(analysis["path"]),
            "sha256": str(analysis["sha256"]),
        },
        "presentation_counts": {
            "executions": 45,
            "admitted_records": 43,
            "accepted_comparisons": 28,
            "rejected_comparisons": 15,
            "endpoint_censored_comparisons": 2,
            "reference_self_comparisons": 11,
            "accepted_nonreference_candidates": 17,
        },
        "scope_exclusions": [
            "degree-four nonlinear endpoints",
            "rank-count stopping consistency",
            "timing, scaling, and performance selection",
            "population-level robustness",
        ],
        "tools": {
            name: {"path": repo_relative(path), "sha256": sha256_file(path)}
            for name, path in tools.items()
        },
        "outputs": {TABLE_NAME: table_hash},
        "allow_unreferenced_tables": False,
    }
    atomic_write(out_dir / TABLE_NAME, table)
    atomic_write(
        out_dir / MANIFEST_NAME,
        (
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8"),
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    try:
        manifest = generate(args.source_manifest, args.out_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
