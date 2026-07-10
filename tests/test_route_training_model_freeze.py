from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.analysis import analyze_plasticity3d_route_cost_model as route_analysis
from experiments.analysis import freeze_route_training_model as freezer


def _contract() -> dict[str, object]:
    return route_analysis._read_json(freezer.DEFAULT_CONTRACT)


def _training_rows(*, rank_deficient: bool = False) -> list[dict[str, object]]:
    contract = _contract()
    routes = list(contract["route_order"])
    rng = np.random.default_rng(20260710)
    rows: list[dict[str, object]] = []
    for group in range(25):
        hardware = "workstation_local" if group < 5 else "karolina_cpu"
        ranks = 1 if hardware == "workstation_local" or group % 2 == 0 else 8
        group_routes = routes if group else routes[:2]
        for route_index, route in enumerate(group_routes):
            if rank_deficient:
                random_values = np.ones(7)
            else:
                random_values = rng.uniform(0.2, 3.0, size=7)
            covariates = {
                "route_work_proxy": float(10.0 ** random_values[0]),
                "owned_matrix_nonzeros": float(10.0 ** random_values[1]),
                "maximum_rank_overlap_dofs": float(10.0 ** random_values[2]),
                "rank_count": float(ranks),
                "plastic_fraction": float(random_values[3] / 4.0),
                "owned_element_imbalance": float(1.0 + random_values[4] / 4.0),
                "overlap_dof_imbalance": float(1.0 + random_values[5] / 4.0),
            }
            row: dict[str, object] = {
                "hardware_id": hardware,
                "configuration_id": f"configuration_{group:02d}",
                "state_id": f"state_{group:02d}",
                "rank_count": ranks,
                "route": route,
                "split": "training",
                "status": "admitted",
                "model_covariates": covariates,
                "publication_model_eligible": True,
                "source_commit": "0123456789abcdef0123456789abcdef01234567",
                "paired_block_repetitions": [1, 2, 3],
                "paired_block_medians_s": [1.0, 1.1, 0.9],
                "paired_block_route_positions": [
                    route_index,
                    (route_index + 1) % 3,
                    (route_index + 2) % 3,
                ],
            }
            features = route_analysis._feature_vector(
                row,
                list(contract["cost_model"]["features_in_order"]),
                factorized_gate={},
            )
            coefficients = np.linspace(0.01, 0.13, features.size)
            row["admitted_wall_time_median_s"] = float(np.exp(features @ coefficients))
            rows.append(row)
    assert len(rows) == freezer.EXPECTED_TRAINING_ROWS
    rows.sort(key=freezer._row_id)
    return rows


def test_training_fit_is_full_rank_and_uses_only_frozen_feature_order() -> None:
    contract = _contract()
    fit = freezer._fit_training_rows(_training_rows(), contract=contract)
    assert fit["feature_order"] == contract["cost_model"]["features_in_order"]
    assert list(fit["coefficients"]) == fit["feature_order"]
    assert fit["design_diagnostics"]["rows"] == 74
    assert fit["design_diagnostics"]["columns"] == 13
    assert fit["design_diagnostics"]["rank"] == 13
    assert fit["design_diagnostics"]["condition_number"] <= (
        contract["cost_model"]["maximum_design_condition_number"]
    )
    assert len(fit["training_row_ids"]) == 74
    assert fit["training_row_ids_sha256"] == freezer._ids_sha256(
        fit["training_row_ids"]
    )


def test_rank_deficient_training_design_fails_before_any_product_is_written() -> None:
    rows = _training_rows(rank_deficient=True)
    for row in rows:
        row["configuration_id"] = "one_configuration"
        row["state_id"] = "one_state"
    # Preserve unique identities while retaining an intentionally degenerate X.
    for index, row in enumerate(rows):
        row["configuration_id"] = f"unique_{index:03d}"
    with pytest.raises(freezer.TrainingFreezeError, match="rank deficient"):
        freezer._fit_training_rows(rows, contract=_contract())


def test_exact_training_plan_rejects_rank_32_before_results_are_scanned() -> None:
    contract = _contract()
    matrix_sha256 = contract["publication_model_input_gates"][
        "karolina_matrix_sha256"
    ]
    with route_analysis.REVIEWED_MATRIX.open(newline="", encoding="utf-8") as handle:
        matrix_rows = list(csv.DictReader(handle))
    rows = [
        dict(row)
        for row in matrix_rows
        if row["experiment_id"] == "EXP-ROUTE-001"
        and row["optional"] == "0"
        and int(row["total_ranks"]) in {1, 8}
    ]
    case_ids = freezer._validate_training_plan_rows(
        rows, matrix_sha256=matrix_sha256
    )
    assert len(case_ids) == 76
    assert freezer._ids_sha256(case_ids) == hashlib.sha256(
        json.dumps(sorted(case_ids), separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    holdout = next(
        dict(row)
        for row in matrix_rows
        if row["experiment_id"] == "EXP-ROUTE-001"
        and row["optional"] == "0"
        and int(row["total_ranks"]) == 32
    )
    changed = list(rows)
    changed[-1] = holdout
    with pytest.raises(freezer.TrainingFreezeError, match="exact 76"):
        freezer._validate_training_plan_rows(changed, matrix_sha256=matrix_sha256)


def test_written_products_have_versioned_pre_holdout_schemas_and_hash_binding(
    tmp_path: Path,
) -> None:
    contract = _contract()
    rows = _training_rows()
    fit = freezer._fit_training_rows(rows, contract=contract)
    case_ids = [f"training_case_{index:03d}" for index in range(76)]
    context = {
        "fit": fit,
        "contract": contract,
        "contract_path": freezer.DEFAULT_CONTRACT,
        "contract_sha256": freezer._sha256(freezer.DEFAULT_CONTRACT),
        "matrix_sha256": contract["publication_model_input_gates"][
            "karolina_matrix_sha256"
        ],
        "source_commit": "0123456789abcdef0123456789abcdef01234567",
        "training_case_ids": case_ids,
        "training_case_ids_sha256": freezer._ids_sha256(case_ids),
        "training_rows": rows,
        "row_evidence": [
            {
                "row_id": freezer._row_id(row),
                "source_records": [],
            }
            for row in rows
        ],
        "factor_records": [],
        "workstation": {"case_ids": [], "evidence": []},
        "karolina": {
            "jobs": {},
            "preflight": {
                "status": "passed",
                "mode": "offline_no_scheduler_access",
            },
            "release": {"reviewer": "test", "reviewed_artifacts": []},
            "evidence": [],
        },
    }
    outputs = freezer.write_training_products(context, tmp_path)
    analysis = json.loads(outputs["training_analysis"].read_text(encoding="utf-8"))
    model = json.loads(outputs["frozen_model"].read_text(encoding="utf-8"))
    assert analysis["schema_id"] == freezer.TRAINING_ANALYSIS_SCHEMA_ID
    assert analysis["schema_version"] == 1
    assert analysis["status"] == "training_fit_admitted"
    assert analysis["holdout_rows_seen"] == 0
    assert analysis["training_case_count"] == 76
    assert analysis["training_row_count"] == 74
    assert analysis["frozen_model"]["sha256"] == freezer._sha256(
        outputs["frozen_model"]
    )
    assert model["schema_id"] == freezer.FROZEN_MODEL_SCHEMA_ID
    assert model["schema_version"] == 1
    assert model["status"] == "frozen_before_holdout"
    assert model["holdout_rows_seen"] == 0
    assert model["training_rows"] == 74
    assert model["training_row_ids"] == fit["training_row_ids"]
    assert model["matrix_sha256"] == analysis["matrix_sha256"]
    assert model["source_commit"] == analysis["source_commit"]
    assert model["contract_sha256"] == analysis["contract"]["sha256"]


def test_utility_source_has_no_scheduler_execution_or_query_path() -> None:
    source = Path(freezer.__file__).read_text(encoding="utf-8")
    assert "import subprocess" not in source
    assert "subprocess." not in source
    assert "sacct(" not in source
    assert "squeue(" not in source
    assert "subprocess.run" not in source
