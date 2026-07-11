from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from experiments.runners.paper_revision_karolina import tier_b_stopping as stopping


REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX = REPO_ROOT / "experiments/runners/paper_revision_karolina/campaign_matrix.csv"
ADJUDICATOR = REPO_ROOT / "experiments/runners/prepare_exp_stop_001_karolina.py"


def _tier_b_rows() -> list[dict[str, str]]:
    with MATRIX.open(newline="", encoding="utf-8") as handle:
        return [
            dict(row)
            for row in csv.DictReader(handle)
            if stopping.is_tier_b_row(row)
        ]


def _adjudication() -> dict[str, object]:
    policy = stopping.load_policy()
    local = dict(policy["local_calibration"])
    reference_id = "p3d_p4_nonlinear_1em07_cluster"
    comparison_ids = (
        "p3d_p4_nonlinear_1em02_cluster",
        "p3d_p4_nonlinear_1em04_cluster",
        "p3d_p4_nonlinear_1em06_cluster",
        reference_id,
        "ginzburg_landau_mpi_consistency_cluster",
        "hyperelasticity_mpi_consistency_cluster",
        "plasticity3d_mpi_consistency_cluster",
    )
    comparisons = {
        case_id: {
            "status": "accepted",
            "reference_row_id": reference_id,
            "gates": {"passed": True},
        }
        for case_id in comparison_ids
    }
    comparisons["p3d_p4_nonlinear_1em02_cluster"]["status"] = "rejected"
    return {
        "schema_id": "fenics-nonlinear-energies.exp-stop-001.final-adjudication",
        "schema_version": 3,
        "experiment_id": "EXP-STOP-001",
        "terminal_decision": "CALIBRATION_SCOPED_PASS_PENDING_DISCRETIZATION_GATE",
        "complete_exp_stop_pass": False,
        "calibration_scope_passed": True,
        "computation_source_commit": local["source_commit"],
        "adjudicator": {
            "source_commit": "0123456789abcdef0123456789abcdef01234567",
            "source_dirty": False,
            "path": "experiments/runners/prepare_exp_stop_001_karolina.py",
            "sha256": hashlib.sha256(ADJUDICATOR.read_bytes()).hexdigest(),
        },
        "local_analysis_sha256": local["analysis_sha256"],
        "cluster_archive_checksum_sha256": "a" * 64,
        "cluster_case_count": 7,
        "publication_timing_admissible": False,
        "comparisons": comparisons,
        "rejected_or_censored_cases": ["p3d_p4_nonlinear_1em02_cluster"],
        "required_gate_failures": [],
        "selected_policies": {
            "p3d_p4_nonlinear_cluster": {
                "status": "selected_loosest_accepted_same_discretization_policy",
                "row_id": "p3d_p4_nonlinear_1em04_cluster",
                "parameter": "relative_dual_residual_target",
                "tolerance": 1.0e-4,
            }
        },
    }


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def test_policy_and_all_thirty_matrix_rows_are_exact() -> None:
    policy = stopping.load_policy()
    assert policy["schema_version"] == 1
    rows = _tier_b_rows()
    assert len(rows) == 30
    contracts = [stopping.row_contract(row) for row in rows]
    assert sum(record["degree"] == 4 for record in contracts) == 20
    assert sum(record["degree"] == 1 for record in contracts) == 10
    assert {record["relative_dual_residual_target"] for record in contracts} == {
        1.0e-7,
        1.0e-6,
    }


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("ksp_rtol", "1e-7"),
        ("ksp_max_it", "999"),
        ("maxit", "79"),
        ("stop_tol", "1e-5"),
        ("grad_stop_tol", "1e-12"),
        ("convergence_metric", "coefficient_l2"),
    ),
)
def test_matrix_policy_mutations_are_rejected(field: str, replacement: str) -> None:
    row = dict(_tier_b_rows()[0])
    row[field] = replacement
    with pytest.raises(stopping.TierBStoppingError):
        stopping.row_contract(row)


def test_policy_numeric_or_extra_field_mutation_is_rejected(tmp_path: Path) -> None:
    policy = json.loads(stopping.POLICY_PATH.read_text(encoding="utf-8"))
    policy["riesz_solver"]["max_it"] = 4999
    with pytest.raises(stopping.TierBStoppingError, match="Riesz cap"):
        stopping.load_policy(_write(tmp_path / "bad_numeric.json", policy))

    policy = json.loads(stopping.POLICY_PATH.read_text(encoding="utf-8"))
    policy["nonlinear_solver"]["unreviewed"] = True
    with pytest.raises(stopping.TierBStoppingError, match="shape"):
        stopping.load_policy(_write(tmp_path / "bad_shape.json", policy))


def test_valid_adjudication_allows_rejected_loose_candidates(tmp_path: Path) -> None:
    path = _write(tmp_path / "adjudication.json", _adjudication())
    result = stopping.validate_stop_adjudication(path)
    assert result["schema_version"] == 3
    assert result["p4_reference_status"] == "accepted"
    assert result["cluster_archive_checksum_sha256"] == "a" * 64


@pytest.mark.parametrize(
    "mutation",
    (
        "reference_rejected",
        "mpi_rejected",
        "required_failure",
        "rejection_inventory",
        "adjudicator_hash",
    ),
)
def test_invalid_adjudication_is_rejected(tmp_path: Path, mutation: str) -> None:
    payload = _adjudication()
    comparisons = payload["comparisons"]
    assert isinstance(comparisons, dict)
    if mutation == "reference_rejected":
        comparisons["p3d_p4_nonlinear_1em07_cluster"]["status"] = "rejected"
    elif mutation == "mpi_rejected":
        comparisons["ginzburg_landau_mpi_consistency_cluster"]["status"] = "rejected"
    elif mutation == "required_failure":
        payload["required_gate_failures"] = ["p3d_p4_nonlinear_1em07_cluster"]
    elif mutation == "rejection_inventory":
        payload["rejected_or_censored_cases"] = []
    else:
        payload["adjudicator"]["sha256"] = "b" * 64
    path = _write(tmp_path / f"{mutation}.json", payload)
    with pytest.raises(stopping.TierBStoppingError):
        stopping.validate_stop_adjudication(path)
