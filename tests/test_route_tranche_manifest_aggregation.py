from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.analysis import aggregate_route_tranche_manifests as aggregation


COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _write_tranche(root: Path, tier: str, count: int) -> Path:
    root.mkdir(parents=True)
    matrix_sha256 = json.loads(aggregation.CONTRACT.read_text())["publication_model_input_gates"][
        "karolina_matrix_sha256"
    ]
    reviewed = root / "reviewed_artifacts" / "reviewed.txt"
    reviewed.parent.mkdir()
    reviewed.write_text("reviewed\n", encoding="utf-8")
    authorization = root / "release_authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "schema_id": "fenics-nonlinear-energies.human-release-authorization",
                "schema_version": 1,
                "status": "approved",
                "decision": "explicit_human_release_after_review",
                "matrix_sha256": matrix_sha256,
                "source_commit": COMMIT,
                "authorizes_experiment": "EXP-ROUTE-001",
                "authorizes_tiers": [tier],
                "reviewer": "fixture reviewer",
                "reviewed_artifacts": [
                    {
                        "path": str(reviewed.relative_to(root)),
                        "sha256": aggregation._sha256(reviewed),
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ledger = root / "submitted_jobs.jsonl"
    case_ids = sorted(aggregation._canonical_case_ids(matrix_sha256)[tier])
    assert len(case_ids) == count
    ledger.write_text(
        "".join(
            json.dumps(
                {
                    "case_id": case_id,
                    "command": f"sbatch --job-name {case_id} run.sbatch",
                    "returncode": 0,
                    "stdout": f"Submitted batch job {1000 + index}",
                    "stderr": "",
                }
            )
            + "\n"
            for index, case_id in enumerate(case_ids)
        ),
        encoding="utf-8",
    )
    manifest = {
        "status": "submitted",
        "matrix_sha256": matrix_sha256,
        "selected_experiments": ["EXP-ROUTE-001"],
        "selected_tiers": [tier],
        "case_count": count,
        "test_only_commands": False,
        "source_commit": COMMIT,
        "source_dirty": False,
        "release_authorization": {
            "schema_id": "fenics-nonlinear-energies.human-release-authorization",
            "path": authorization.name,
            "sha256": aggregation._sha256(authorization),
        },
    }
    path = root / "prepared_manifest.json"
    path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return path


def test_route_tranche_aggregation_requires_complete_disjoint_union(tmp_path: Path) -> None:
    manifests = [
        _write_tranche(tmp_path / "screen", "fixed_state_screen", 78),
        _write_tranche(tmp_path / "quadrature", "factorized_quadrature", 18),
        _write_tranche(tmp_path / "factor", "factorized_microbenchmark", 9),
    ]
    result = aggregation.aggregate(manifests, archive_root=tmp_path)
    assert result["status"] == "submitted_tranches_complete"
    assert result["case_count"] == 105
    assert len(result["case_ids"]) == 105
    assert result["source_commit"] == COMMIT
    assert {tier for row in result["tranches"] for tier in row["selected_tiers"]} == {
        "fixed_state_screen",
        "factorized_quadrature",
        "factorized_microbenchmark",
    }


def test_route_tranche_aggregation_rejects_incomplete_union(tmp_path: Path) -> None:
    manifests = [
        _write_tranche(tmp_path / "screen", "fixed_state_screen", 78),
        _write_tranche(tmp_path / "factor", "factorized_microbenchmark", 9),
    ]
    with pytest.raises(ValueError, match="union is incomplete"):
        aggregation.aggregate(manifests, archive_root=tmp_path)
