from __future__ import annotations

import csv
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.analysis import analyze_plasticity3d_route_cost_model as analysis
from experiments.analysis import aggregate_route_tranche_manifests as aggregator


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = REPO_ROOT / "paper/protocols/EXP-ROUTE-001-analysis-contract.json"


def _sha(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _write_record(
    root: Path,
    *,
    route: str,
    state: np.ndarray,
    action: np.ndarray,
    git_dirty: bool = False,
) -> Path:
    case = root / route
    case.mkdir(parents=True)
    npz = case / "tangent_action.npz"
    np.savez_compressed(
        npz,
        state=np.asarray(state, dtype=np.float64),
        tangent_action=np.asarray(action, dtype=np.float64),
        route=np.asarray(route),
        state_label=np.asarray("elastic"),
    )
    colors = 90 if route == "colored_sfd" else None
    payload = {
        "schema_version": 1,
        "experiment_id": "EXP-ROUTE-001",
        "tier": "fixed_state_screen",
        "status": "completed",
        "route": route,
        "constraint_variant": "glued_bottom",
        "lambda_target": 1.55,
        "warmup_repetitions": 1,
        "mesh_name": "hetero_ssr_L1",
        "element_degree": 1,
        "quadrature_rule_id": "tetra_1point",
        "state_family": "analytic_mesh_field_v1",
        "state_label": "elastic",
        "state_amplitude": 0.0002,
        "state_sha256": _sha(state),
        "action_sha256": _sha(action),
        "action_out": npz.name,
        "mpi_ranks": 1,
        "measured_repetitions": 5,
        "wall_times_s": [1.0, 1.01, 0.99, 1.02, 0.98],
        "wall_times_by_rank_s": [[1.0], [1.01], [0.99], [1.02], [0.98]],
        "wall_time_reduction": "mpi_collective_max",
        "wall_time_median_s": 1.0,
        "branch_diagnostics": {
            "counts": {
                "elastic": 10,
                "shear": 0,
                "left_edge": 0,
                "right_edge": 0,
                "apex": 0,
            },
            "plastic_fraction": 0.0,
            "normalized_boundary_margin_min": 0.5,
            "near_boundary_fraction": 0.0,
        },
        "model_covariates": {
            "element_dofs": 12,
            "constitutive_dimension": 6,
            "quadrature_points_per_element": 1,
            "maximum_local_color_count": colors,
            "total_owned_elements": 10,
            "global_free_dofs": int(state.size),
            "rank_count": 1,
        },
        "rank_summaries": [
            {
                "rank": 0,
                "owned_dofs": int(state.size),
                "local_elements": 10,
                "owned_matrix_nonzeros": 100,
                "overlap_dofs": int(state.size),
                "local_color_count": 90 if route == "colored_sfd" else 0,
                "peak_rss_bytes": 1000,
                "tracked_allocation_bytes": 500,
                "owned_elements": 10,
            }
        ],
        "git": {
            "commit": "0123456789abcdef0123456789abcdef01234567",
            "dirty": git_dirty,
        },
    }
    output = case / "output.json"
    output.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return output


def _contract() -> dict[str, object]:
    return analysis._read_json(CONTRACT)


def _observed(root: Path) -> dict[tuple[str, str, str, int, str], dict[str, object]]:
    observed, censors, invalid = analysis._scan_source(
        "workstation_local", root, contract=_contract()
    )
    assert not censors
    assert not invalid
    return observed


def test_contract_freezes_split_features_and_terminal_gates() -> None:
    contract = _contract()
    model = contract["cost_model"]
    assert contract["contract_version"] == 2
    assert contract["terminal_policy"] == {
        "selector_claim_requires_all_model_gates": True,
        "selector_admitted": "predictive_selector_admissible",
        "otherwise": "finite_empirical_map_only",
        "never_impute_censored_or_missing_timings": True,
    }
    assert contract["frozen_before_karolina_results"] is True
    assert contract["publication_model_input_gates"]["design_released_for_fitting"] is True
    assert (
        contract["publication_model_input_gates"][
            "endpoint_analysis_required_for_selector"
        ]
        is True
    )
    assert contract["publication_model_input_gates"]["endpoint_required_rows"] == 30
    assert model["training_rule"] == (
        "workstation_local rows and karolina_cpu ranks 1 or 8"
    )
    assert model["holdout_rule"] == "karolina_cpu rank 32 only"
    assert model["features_in_order"] == [
        "route_is_element_ad",
        "route_is_colored_sfd",
        "route_is_constitutive_ad",
        "karolina_route_is_element_ad",
        "karolina_route_is_colored_sfd",
        "karolina_route_is_constitutive_ad",
        "log1p_route_work_proxy",
        "log1p_owned_matrix_nonzeros",
        "log1p_maximum_rank_overlap_dofs",
        "log_rank_count",
        "plastic_fraction",
        "owned_element_imbalance",
        "overlap_dof_imbalance",
    ]
    assert model["median_absolute_percentage_error_max"] == 0.25
    assert model["p90_absolute_percentage_error_max"] == 0.5
    assert model["practical_ordering_tie_ratio"] == 1.1
    assert model["resolved_ordering_accuracy_min"] == 0.9
    assert model["paired_block_bootstrap_seed"] == 20260710
    assert model["paired_block_bootstrap_resamples"] == 10000
    assert model["paired_block_bootstrap_confidence_level"] == 0.95
    assert model["route_work_proxy"] == {
        "semantics": (
            "collective-max-aligned structural operation-shape count; "
            "not an exact FLOP or compiler-cost model"
        ),
        "element_ad": (
            "max_r(local_elements_r) * quadrature_points_per_element * "
            "element_dofs^2"
        ),
        "colored_sfd": (
            "max_r(local_elements_r * local_color_count_r) * "
            "quadrature_points_per_element * element_dofs"
        ),
        "constitutive_ad": (
            "max_r(local_elements_r) * quadrature_points_per_element * "
            "(constitutive_dimension^2 + constitutive_dimension^2 * "
            "element_dofs + constitutive_dimension * element_dofs^2)"
        ),
    }
    assert contract["factorized_calibration_policy"] == {
        "required_for_selector_claim": False,
        "current_status": "descriptive_replicated_synthetic_non_route_faithful_proxy",
        "independent_blocks_per_rank": 3,
        "single_rank_imbalance_policy": "mark_nonunit_imbalance_inapplicable_and_exclude_from_fit",
        "training_ranks": [1, 8],
        "validation_rank": 32,
        "response": "log_warm_median_stage_seconds",
        "fit": "ordinary_least_squares_main_effects",
        "stage_names": ["contraction", "color_hvp", "insertion", "communication"],
        "median_absolute_percentage_error_max": 0.5,
        "p90_absolute_percentage_error_max": 1.0,
        "fail_closed_reason": "factorized_mechanism_diagnostic_gate_failed",
    }


def test_route_work_proxies_follow_busiest_overlap_rank_and_contraction_shape() -> None:
    rank_rows = [
        {"local_elements": 7, "local_color_count": 4},
        {"local_elements": 5, "local_color_count": 9},
    ]
    common = {
        "rank_rows": rank_rows,
        "element_dofs": 12,
        "constitutive_dimension": 6,
        "quadrature_points": 2,
    }

    assert analysis._route_work_proxy("element_ad", **common) == 7 * 2 * 12**2
    assert analysis._route_work_proxy("colored_sfd", **common) == 5 * 9 * 2 * 12
    assert analysis._route_work_proxy("constitutive_ad", **common) == 7 * 2 * (
        6**2 + 6**2 * 12 + 6 * 12**2
    )


def test_exact_state_and_numerically_equivalent_action_admit_timing(tmp_path: Path) -> None:
    state = np.asarray([1.0, 2.0, 3.0])
    reference = np.asarray([4.0, 5.0, 6.0])
    _write_record(tmp_path, route="element_ad", state=state, action=reference)
    _write_record(tmp_path, route="colored_sfd", state=state, action=reference)
    _write_record(
        tmp_path,
        route="constitutive_ad",
        state=state,
        action=reference + np.asarray([1.0e-10, 0.0, 0.0]),
    )
    empirical = analysis.build_empirical_map(
        contract=_contract(),
        hardware_ids=["workstation_local"],
        observed=_observed(tmp_path),
        runtime_censors={},
    )
    p1 = [row for row in empirical if row["configuration_id"] == "p1_l1"]
    assert len(p1) == 6
    elastic = [row for row in p1 if row["state_id"] == "elastic"]
    assert {row["status"] for row in elastic} == {"admitted"}
    assert all(row["admitted_wall_time_median_s"] == 1.0 for row in elastic)
    model = analysis.fit_cost_model(empirical, _contract())
    assert model["status"] == "not_fit_insufficient_data"
    assert model["selector_claim_admissible"] is False


def test_fixed_state_record_tree_is_relocatable_and_rejects_path_escape(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original"
    state = np.asarray([1.0, 2.0, 3.0])
    action = np.asarray([4.0, 5.0, 6.0])
    for route in ("element_ad", "colored_sfd", "constitutive_ad"):
        _write_record(original, route=route, state=state, action=action)
    before = _observed(original)
    relocated = tmp_path / "relocated"
    original.rename(relocated)
    after = _observed(relocated)
    assert set(after) == set(before)
    for slot in before:
        assert np.array_equal(after[slot]["state"], before[slot]["state"])
        assert np.array_equal(after[slot]["actions"], before[slot]["actions"])

    output = relocated / "element_ad" / "output.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["action_out"] = "../colored_sfd/tangent_action.npz"
    output.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    _observed_rows, _censors, invalid = analysis._scan_source(
        "workstation_local", relocated, contract=_contract()
    )
    assert any("action_out escapes the output record directory" in row["reason"] for row in invalid)


def test_state_mismatch_blocks_every_timing_in_group(tmp_path: Path) -> None:
    reference = np.asarray([4.0, 5.0, 6.0])
    _write_record(
        tmp_path,
        route="element_ad",
        state=np.asarray([1.0, 2.0, 3.0]),
        action=reference,
    )
    _write_record(
        tmp_path,
        route="colored_sfd",
        state=np.asarray([1.0, 2.0, 3.0]),
        action=reference,
    )
    _write_record(
        tmp_path,
        route="constitutive_ad",
        state=np.asarray([1.0, 2.0, 3.5]),
        action=reference,
    )
    empirical = analysis.build_empirical_map(
        contract=_contract(),
        hardware_ids=["workstation_local"],
        observed=_observed(tmp_path),
        runtime_censors={},
    )
    rows = [
        row
        for row in empirical
        if row["configuration_id"] == "p1_l1" and row["state_id"] == "elastic"
    ]
    assert {row["status"] for row in rows} == {"equivalence_failed"}
    assert {row["reason"] for row in rows} == {"state_array_mismatch"}
    assert all(row["admitted_wall_time_median_s"] is None for row in rows)


def test_route_invariant_rank_structure_mismatch_blocks_group(tmp_path: Path) -> None:
    state = np.asarray([1.0, 2.0, 3.0])
    action = np.asarray([4.0, 5.0, 6.0])
    for route in ("element_ad", "colored_sfd", "constitutive_ad"):
        path = _write_record(tmp_path, route=route, state=state, action=action)
        if route == "constitutive_ad":
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["rank_summaries"][0]["owned_matrix_nonzeros"] += 1
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    empirical = analysis.build_empirical_map(
        contract=_contract(),
        hardware_ids=["workstation_local"],
        observed=_observed(tmp_path),
        runtime_censors={},
    )
    rows = [
        row
        for row in empirical
        if row["configuration_id"] == "p1_l1" and row["state_id"] == "elastic"
    ]
    assert {row["status"] for row in rows} == {"equivalence_failed"}
    assert {row["reason"] for row in rows} == {
        "route_invariant_rank_summary_mismatch"
    }


def test_action_mismatch_blocks_timing_and_p4_sfd_is_visible_censor(tmp_path: Path) -> None:
    state = np.asarray([1.0, 2.0, 3.0])
    reference = np.asarray([4.0, 5.0, 6.0])
    _write_record(tmp_path, route="element_ad", state=state, action=reference)
    _write_record(tmp_path, route="colored_sfd", state=state, action=reference)
    _write_record(
        tmp_path,
        route="constitutive_ad",
        state=state,
        action=reference + np.asarray([1.0, 0.0, 0.0]),
    )
    empirical = analysis.build_empirical_map(
        contract=_contract(),
        hardware_ids=["workstation_local", "karolina_cpu"],
        observed=_observed(tmp_path),
        runtime_censors={},
    )
    failed = [
        row
        for row in empirical
        if row["hardware_id"] == "workstation_local"
        and row["configuration_id"] == "p1_l1"
        and row["state_id"] == "elastic"
    ]
    assert {row["status"] for row in failed} == {"equivalence_failed"}
    assert {row["reason"] for row in failed} == {"tangent_action_mismatch"}
    censored = [
        row
        for row in empirical
        if row["hardware_id"] == "karolina_cpu"
        and row["configuration_id"] == "p4_l1"
        and row["route"] == "colored_sfd"
    ]
    assert len(censored) == 6
    assert {row["status"] for row in censored} == {"censored"}
    assert {row["reason"] for row in censored} == {
        "prespecified_not_attempted_memory_risk_no_threshold_claim"
    }


def test_dirty_records_enter_diagnostic_map_but_never_publication_model(tmp_path: Path) -> None:
    state = np.asarray([1.0, 2.0, 3.0])
    action = np.asarray([4.0, 5.0, 6.0])
    for route in ("element_ad", "colored_sfd", "constitutive_ad"):
        _write_record(
            tmp_path,
            route=route,
            state=state,
            action=action,
            git_dirty=True,
        )
    empirical = analysis.build_empirical_map(
        contract=_contract(),
        hardware_ids=["workstation_local"],
        observed=_observed(tmp_path),
        runtime_censors={},
    )
    rows = [
        row
        for row in empirical
        if row["configuration_id"] == "p1_l1" and row["state_id"] == "elastic"
    ]
    assert {row["status"] for row in rows} == {"admitted"}
    assert all(row["admitted_wall_time_median_s"] == 1.0 for row in rows)
    assert {row["publication_model_eligible"] for row in rows} == {False}
    assert {row["model_exclusion_reason"] for row in rows} == {
        "record_git_worktree_not_clean"
    }
    model = analysis.fit_cost_model(empirical, _contract())
    assert model["training_rows"] == 0
    assert model["selector_claim_admissible"] is False


def _write_valid_route_master_archive(
    archive: Path, contract: dict[str, object]
) -> dict[str, object]:
    matrix_sha = contract["publication_model_input_gates"]["karolina_matrix_sha256"]
    commit = "0123456789abcdef0123456789abcdef01234567"
    archive.mkdir(parents=True, exist_ok=True)
    tranche_paths: list[Path] = []
    canonical_by_tier = aggregator._canonical_case_ids(str(matrix_sha))
    for tier, canonical_ids in canonical_by_tier.items():
        count = len(canonical_ids)
        tranche = archive / tier
        reviewed_dir = tranche / "reviewed_artifacts"
        reviewed_dir.mkdir(parents=True)
        reviewed = reviewed_dir / "review.json"
        reviewed.write_text(json.dumps({"tier": tier}) + "\n", encoding="utf-8")
        reviewed_sha = hashlib.sha256(reviewed.read_bytes()).hexdigest()
        release = {
            "schema_id": "fenics-nonlinear-energies.human-release-authorization",
            "schema_version": 1,
            "status": "approved",
            "decision": "explicit_human_release_after_review",
            "matrix_sha256": matrix_sha,
            "source_commit": commit,
            "authorizes_experiment": "EXP-ROUTE-001",
            "authorizes_tiers": [tier],
            "reviewer": "relocation-test-reviewer",
            "reviewed_artifacts": [
                {
                    "path": "reviewed_artifacts/review.json",
                    "sha256": reviewed_sha,
                }
            ],
        }
        release_path = tranche / "release_authorization.json"
        release_path.write_text(json.dumps(release) + "\n", encoding="utf-8")
        manifest = {
            "status": "submitted",
            "matrix_sha256": matrix_sha,
            "selected_experiments": ["EXP-ROUTE-001"],
            "selected_tiers": [tier],
            "case_count": count,
            "test_only_commands": False,
            "source_commit": commit,
            "source_dirty": False,
            "release_authorization": {
                "schema_id": "fenics-nonlinear-energies.human-release-authorization",
                "path": "release_authorization.json",
                "sha256": hashlib.sha256(release_path.read_bytes()).hexdigest(),
            },
        }
        manifest_path = tranche / "prepared_manifest.json"
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        ledger = tranche / "submitted_jobs.jsonl"
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
                for index, case_id in enumerate(sorted(canonical_ids))
            ),
            encoding="utf-8",
        )
        tranche_paths.append(manifest_path)
    master = aggregator.aggregate(tranche_paths, archive_root=archive)
    master_path = archive / "route_campaign_master_manifest.json"
    master_path.write_text(json.dumps(master) + "\n", encoding="utf-8")
    return master


def test_karolina_model_rows_require_semantically_valid_submitted_master(
    tmp_path: Path,
) -> None:
    contract = _contract()
    missing = analysis._source_provenance_gate("karolina_cpu", tmp_path, contract)
    assert missing["eligible"] is False
    assert missing["reason"] == "karolina_route_campaign_master_manifest_missing"
    master = _write_valid_route_master_archive(tmp_path, contract)
    master_path = tmp_path / "route_campaign_master_manifest.json"
    admitted = analysis._source_provenance_gate("karolina_cpu", tmp_path, contract)
    assert admitted["eligible"] is True

    stale = copy.deepcopy(master)
    stale["matrix_sha256"] = "0" * 64
    master_path.write_text(json.dumps(stale) + "\n", encoding="utf-8")
    rejected = analysis._source_provenance_gate("karolina_cpu", tmp_path, contract)
    assert rejected["reason"] == "karolina_matrix_hash_mismatch"

    empty_tranche = tmp_path / str(master["tranches"][0]["manifest_path"])
    empty_tranche.write_text("{}\n", encoding="utf-8")
    master["tranches"][0]["manifest_sha256"] = analysis._sha256_file(empty_tranche)
    master_path.write_text(json.dumps(master) + "\n", encoding="utf-8")
    rejected = analysis._source_provenance_gate("karolina_cpu", tmp_path, contract)
    assert rejected["eligible"] is False
    assert rejected["reason"].startswith(
        "karolina_campaign_master_semantic_validation_failed"
    )


def test_route_master_archive_remains_valid_after_relocation(tmp_path: Path) -> None:
    contract = _contract()
    archive = tmp_path / "original_archive"
    master = _write_valid_route_master_archive(archive, contract)
    assert all(
        not Path(entry["manifest_path"]).is_absolute()
        for entry in master["tranches"]
    )

    relocated = tmp_path / "relocated_archive"
    archive.rename(relocated)
    gate = analysis._source_provenance_gate("karolina_cpu", relocated, contract)
    assert gate["eligible"] is True
    assert gate["reason"] == "reviewed_submitted_tranche_master_manifest"


def test_selector_endpoint_gate_is_hash_bound_and_semantically_deep(tmp_path: Path) -> None:
    contract = _contract()
    root = tmp_path / "karolina"
    master = _write_valid_route_master_archive(root, contract)
    endpoint_path = root / "analysis" / "tier_b_endpoints.json"
    endpoint_path.parent.mkdir()
    reason = contract["structural_censors"][0]["reason"]
    payload = {
        "schema": {
            "id": "fenics-nonlinear-energies.exp-route-001.tier-b-endpoints",
            "version": 1,
        },
        "experiment_id": "EXP-ROUTE-001",
        "matrix_sha256": contract["publication_model_input_gates"][
            "karolina_matrix_sha256"
        ],
        "analysis_contract_sha256": analysis._sha256_file(CONTRACT),
        "terminal_decision": "tier_b_descriptive_timing_only",
        "endpoint_correct_timing_admissible": True,
        "descriptive_timing_available": True,
        "comparative_ranking_admissible": False,
        "publication_admissible": True,
        "required_rows": 30,
        "admitted_rows": 30,
        "manifest": {"source_commit": master["source_commit"]},
        "blocks": [
            {
                "status": "timing_admitted",
                "routes": {
                    "element_ad": {"status": "timing_admitted"},
                    "constitutive_ad": {"status": "timing_admitted"},
                },
            }
            for _index in range(30)
        ],
        "structural_censors": [
            {
                "rank_count": ranks,
                "route": "colored_sfd",
                "status": "censored",
                "reason": reason,
                "timing_exposed": False,
            }
            for ranks in (8, 32)
        ],
    }
    endpoint_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    gate = analysis._endpoint_analysis_gate(
        endpoint_path,
        sources=[("karolina_cpu", root)],
        contract=contract,
    )
    assert gate["publication_admissible"] is True
    assert gate["required_rows"] == gate["admitted_rows"] == 30
    payload["comparative_ranking_admissible"] = True
    endpoint_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    mismatched = analysis._endpoint_analysis_gate(
        endpoint_path,
        sources=[("karolina_cpu", root)],
        contract=contract,
    )
    assert mismatched["publication_admissible"] is False
    payload["comparative_ranking_admissible"] = False
    payload["blocks"][0]["status"] = "invalid"
    endpoint_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    rejected = analysis._endpoint_analysis_gate(
        endpoint_path,
        sources=[("karolina_cpu", root)],
        contract=contract,
    )
    assert rejected["publication_admissible"] is False


def test_balanced_independent_blocks_are_publication_model_eligible() -> None:
    actions = np.arange(12, dtype=np.float64).reshape(4, 3) + 1.0
    gradient = np.asarray([2.0, 3.0, 4.0])
    records = []
    for block, position in ((1, 0), (2, 1), (3, 2)):
        records.append(
            {
                "path": f"block-{block}.json",
                "payload": {
                    "probe_count": 4,
                    "git": {
                        "commit": "0123456789abcdef0123456789abcdef01234567",
                        "dirty": False,
                    },
                    "comparison_design": {
                        "comparison_id": "paired",
                        "block_repetition": block,
                        "route_order_position": position,
                        "route_order_policy": "seeded_balanced_cyclic_v1",
                        "timing_reduction": "mpi_collective_max",
                    },
                },
                "state": np.asarray([1.0, 2.0, 3.0]),
                "action": actions[0],
                "actions": actions,
                "gradient": gradient,
                "median": float(block),
                "source_provenance": {"eligible": True},
            }
        )
    aggregated = analysis._aggregate_block_records(records, contract=_contract())
    assert aggregated["paired_block_design_passed"] is True
    assert aggregated["independent_block_count"] == 3
    assert aggregated["median"] == 2.0
    eligible, reason = analysis._publication_model_eligibility(aggregated)
    assert eligible is True
    assert reason == "clean_committed_record"


def test_karolina_fixed_record_is_bound_to_exact_reviewed_matrix_row(
    tmp_path: Path,
) -> None:
    with analysis.REVIEWED_MATRIX.open(newline="", encoding="utf-8") as handle:
        row = next(
            item
            for item in csv.DictReader(handle)
            if item["case_id"] == "route_block_p1l1_elastic_np1_b01"
        )
    route = row["route_order"].split("|")[0]
    job = tmp_path / "cases" / row["case_id"] / "job_123"
    output = job / "measure_01" / route / "output.json"
    output.parent.mkdir(parents=True)
    (job / "matrix_row.json").write_text(json.dumps(row), encoding="utf-8")
    payload = {
        "experiment_id": row["experiment_id"],
        "tier": row["tier"],
        "mesh_name": row["mesh_name"],
        "element_degree": int(row["element_degree"]),
        "quadrature_rule_id": row["quadrature_rule"],
        "state_label": row["state_label"],
        "state_amplitude": float(row["state_amplitude"]),
        "mpi_ranks": int(row["total_ranks"]),
        "probe_count": int(row["probe_count"]),
        "route": route,
        "constraint_variant": "glued_bottom",
        "lambda_target": 1.55,
        "warmup_repetitions": int(row["warmups"]),
        "measured_repetitions": int(row["repetitions"]),
        "comparison_design": {
            "comparison_id": row["comparison_id"],
            "block_repetition": int(row["block_repetition"]),
            "route_order_position": 0,
            "route_order_policy": row["route_order_policy"],
            "timing_reduction": row["timing_reduction"],
        },
    }
    bound = analysis._bind_fixed_record_to_reviewed_matrix(
        output,
        payload,
        contract=_contract(),
    )
    assert bound["case_id"] == row["case_id"]

    tampered = dict(row)
    tampered["state_label"] = "mixed"
    (job / "matrix_row.json").write_text(json.dumps(tampered), encoding="utf-8")
    try:
        analysis._bind_fixed_record_to_reviewed_matrix(
            output,
            payload,
            contract=_contract(),
        )
    except ValueError as exc:
        assert "differs from the reviewed row" in str(exc)
    else:
        raise AssertionError("tampered matrix_row.json was admitted")


def _synthetic_selector_rows() -> list[dict[str, object]]:
    contract = _contract()
    features = list(contract["cost_model"]["features_in_order"])
    routes = list(contract["route_order"])
    rng = np.random.default_rng(20260710)
    rows: list[dict[str, object]] = []
    groups: list[tuple[str, str, str, int, str]] = []
    for index in range(20):
        hardware = "workstation_local" if index < 8 else "karolina_cpu"
        ranks = 1 if hardware == "workstation_local" else (1 if index % 2 else 8)
        groups.append((hardware, f"train_{index:02d}", f"state_{index:02d}", ranks, "training"))
    for index in range(8):
        groups.append(
            (
                "karolina_cpu",
                f"holdout_{index:02d}",
                f"state_{index:02d}",
                32,
                "holdout",
            )
        )
    for group_index, (hardware, configuration, state, ranks, split) in enumerate(groups):
        winning_route = routes[group_index % len(routes)]
        for route_index, route in enumerate(routes):
            covariates = {
                "route_work_proxy": 1.0 if route == winning_route else 20.0,
                "owned_matrix_nonzeros": float(rng.uniform(100.0, 100000.0)),
                "maximum_rank_overlap_dofs": float(rng.uniform(20.0, 5000.0)),
                "rank_count": float(ranks),
                "plastic_fraction": float(rng.uniform(0.0, 0.9)),
                "owned_element_imbalance": float(rng.uniform(1.0, 1.8)),
                "overlap_dof_imbalance": float(rng.uniform(1.0, 1.8)),
            }
            row: dict[str, object] = {
                "hardware_id": hardware,
                "configuration_id": configuration,
                "state_id": state,
                "rank_count": ranks,
                "route": route,
                "split": split,
                "status": "admitted",
                "model_covariates": covariates,
                "publication_model_eligible": True,
                "source_commit": "0123456789abcdef0123456789abcdef01234567",
                "paired_block_repetitions": [1, 2, 3],
                "paired_block_route_positions": [
                    route_index,
                    (route_index + 1) % 3,
                    (route_index + 2) % 3,
                ],
            }
            x = analysis._feature_vector(row, features, factorized_gate={})
            coefficients = np.zeros(len(features), dtype=np.float64)
            coefficients[features.index("log1p_route_work_proxy")] = 1.0
            time_value = float(np.exp(x @ coefficients))
            row["admitted_wall_time_median_s"] = time_value
            row["paired_block_medians_s"] = [
                0.98 * time_value,
                time_value,
                1.02 * time_value,
            ]
            rows.append(row)
    return rows


def test_selector_full_path_passes_and_holdout_is_blind_until_gating() -> None:
    contract = _contract()
    rows = _synthetic_selector_rows()
    factor = {
        "passed": False,
        "calibration_integrated": False,
        "selector_use": "descriptive_replicated_synthetic_non_route_faithful_proxy",
    }
    blocked = analysis.fit_cost_model(rows, contract, factorized_gate=factor)
    assert blocked["selector_claim_admissible"] is False
    assert "tier_b_endpoint_analysis_gate_not_passed" in blocked["preflight_failures"]
    endpoint_gate = {"publication_admissible": True}
    passed = analysis.fit_cost_model(
        rows,
        contract,
        factorized_gate=factor,
        endpoint_gate=endpoint_gate,
    )
    assert passed["status"] == "selection_rule_passed"
    assert passed["selector_claim_admissible"] is True
    assert passed["design_rank"] == len(contract["cost_model"]["features_in_order"])
    assert passed["resolved_holdout_groups"] == 8
    assert passed["resolved_ordering_accuracy"] == 1.0
    assert len(passed["distinct_observed_holdout_winners"]) == 3
    assert set(passed["distinct_observed_holdout_winners"]) == {
        row["observed_winner"]
        for row in passed["holdout_ordering"]
        if row["uncertainty_resolved"] is True
    }
    assert all(
        row["observed_winner_interval_clears_tie_band"] is True
        and row["predicted_winner_interval_clears_tie_band"] is True
        for row in passed["holdout_ordering"]
    )

    changed = copy.deepcopy(rows)
    for row in changed:
        if row["split"] == "holdout":
            row["admitted_wall_time_median_s"] = float(
                row["admitted_wall_time_median_s"]
            ) * 4.0
            row["paired_block_medians_s"] = [
                float(value) * 4.0 for value in row["paired_block_medians_s"]
            ]
    failed = analysis.fit_cost_model(
        changed,
        contract,
        factorized_gate=factor,
        endpoint_gate=endpoint_gate,
    )
    assert failed["status"] == "fit_gate_failed"
    assert failed["selector_claim_admissible"] is False
    assert failed["gate_results"]["median_absolute_percentage_error"] is False
    assert failed["coefficients"] == passed["coefficients"]
    assert (
        failed["coefficient_bootstrap_confidence_intervals"]
        == passed["coefficient_bootstrap_confidence_intervals"]
    )
    publication_model = analysis._publication_safe_cost_model(failed)
    assert publication_model == {
        "status": "fit_gate_failed",
        "selector_claim_admissible": False,
        "feature_order": contract["cost_model"]["features_in_order"],
        "training_rows": 60,
        "holdout_rows": 24,
        "preflight_failures": [],
        "failed_gates": [
            "median_absolute_percentage_error",
            "p90_absolute_percentage_error",
        ],
    }
    assert not {
        "coefficients",
        "coefficient_bootstrap_confidence_intervals",
        "holdout_ordering",
        "distinct_observed_holdout_winners",
    } & set(publication_model)


def test_publication_release_admits_complete_finite_map_when_selector_or_factor_fails() -> None:
    contract = _contract()
    rows: list[dict[str, object]] = []
    for slot in sorted(
        analysis._expected_slots(("workstation_local", "karolina_cpu"), contract)
    ):
        hardware, configuration, state, ranks, route = slot
        reason = analysis._is_structural_censor(slot, contract)
        row: dict[str, object] = {
            "hardware_id": hardware,
            "configuration_id": configuration,
            "state_id": state,
            "rank_count": ranks,
            "route": route,
        }
        if reason is None:
            row.update(
                {
                    "status": "admitted",
                    "admitted_wall_time_median_s": 1.0,
                    "publication_model_eligible": True,
                    "split": analysis._split(hardware, ranks, contract),
                }
            )
        else:
            row.update(
                {
                    "status": "censored",
                    "reason": reason,
                    "admitted_wall_time_median_s": None,
                }
            )
        rows.append(row)
    failed_factor = {
        "passed": False,
        "failures": ["factorized calibration holdout gates failed"],
        "calibration_integrated": False,
        "selector_use": contract["factorized_calibration_policy"]["current_status"],
        "selector_blockers": [],
        "required_ranks": [1, 8, 32],
        "independent_blocks_per_rank": 3,
        "calibration_model": None,
    }
    assert analysis._factorized_diagnostic_integrity_errors(
        failed_factor, contract
    ) == []
    assert analysis._publication_evidence_is_admissible(
        clean_committed_analysis=True,
        terminal_decision="finite_empirical_map_only",
        empirical_rows=rows,
        cost_model={
            "status": "fit_gate_failed",
            "selector_claim_admissible": False,
            "feature_order": contract["cost_model"]["features_in_order"],
            "training_rows": 74,
            "holdout_rows": 22,
            "preflight_failures": [],
            "failed_gates": ["median_absolute_percentage_error"],
        },
        endpoint_gate={"publication_admissible": True},
        factorized_gate=failed_factor,
        invalid_records=[],
        contract=contract,
    )
    # The factor outcome is descriptive for either terminal, never a selector gate.
    assert analysis._publication_evidence_is_admissible(
        clean_committed_analysis=True,
        terminal_decision="predictive_selector_admissible",
        empirical_rows=rows,
        cost_model={
            "status": "selection_rule_passed",
            "selector_claim_admissible": True,
            "feature_order": contract["cost_model"]["features_in_order"],
            "training_rows": 74,
            "holdout_rows": 22,
        },
        endpoint_gate={"publication_admissible": True},
        factorized_gate=failed_factor,
        invalid_records=[],
        contract=contract,
    )
    rows[0]["status"] = "missing"
    assert not analysis._publication_evidence_is_admissible(
        clean_committed_analysis=True,
        terminal_decision="finite_empirical_map_only",
        empirical_rows=rows,
        cost_model={
            "status": "fit_gate_failed",
            "selector_claim_admissible": False,
            "feature_order": contract["cost_model"]["features_in_order"],
            "training_rows": 74,
            "holdout_rows": 22,
            "preflight_failures": [],
            "failed_gates": ["median_absolute_percentage_error"],
        },
        endpoint_gate={"publication_admissible": True},
        factorized_gate=failed_factor,
        invalid_records=[],
        contract=contract,
    )


def test_cluster_batch_evidence_requires_settled_accounting_and_logs(tmp_path: Path) -> None:
    case_id = "route_case"
    job_id = "123"
    commit = "0123456789abcdef0123456789abcdef01234567"
    batch = tmp_path / "jobs" / case_id / f"job_{job_id}"
    slurm = tmp_path / "slurm"
    batch.mkdir(parents=True)
    slurm.mkdir()
    (batch / "job_metadata.env").write_text(
        "\n".join(
            (
                f"case_id={case_id}",
                f"job_id={job_id}",
                "account=fta-26-40",
                "qos=3571_6328",
                "cluster=karolina",
                f"git_commit={commit}",
                "git_dirty=false",
                "allocation_revalidated=YES",
                "account_qos_revalidated=YES",
                "allocation_valid_until=2026-12-31",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (batch / "environment.txt").write_text("reviewed environment\n", encoding="utf-8")
    (batch / "execute.log").write_text("completed\n", encoding="utf-8")
    (slurm / f"{case_id}-{job_id}.out").write_text("stdout\n", encoding="utf-8")
    (slurm / f"{case_id}-{job_id}.err").write_text("", encoding="utf-8")
    raw = "raw parsable2 evidence\n"
    accounting = {
        "schema_id": "fenics-nonlinear-energies.slurm-accounting-snapshot",
        "schema_version": 1,
        "job_id": job_id,
        "source": {
            "raw_parsable2": raw,
            "sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "byte_count": len(raw.encode()),
        },
        "allocation": {
            "job_id_raw": job_id,
            "cluster": "karolina",
            "account": "fta-26-40",
            "qos": "3571_6328",
            "state": "COMPLETED",
            "exit_code": "0:0",
        },
    }
    accounting_path = batch / "sacct_final.json"
    accounting_path.write_text(json.dumps(accounting) + "\n", encoding="utf-8")
    evidence = analysis._cluster_batch_evidence(
        campaign_root=tmp_path,
        case_id=case_id,
        job_id=job_id,
        expected_commit=commit,
    )
    assert {row["role"] for row in evidence} == {
        "batch_job_metadata",
        "batch_environment",
        "batch_execute_log",
        "settled_slurm_accounting",
        "slurm_stdout",
        "slurm_stderr",
    }
    accounting["allocation"]["state"] = "FAILED"
    accounting_path.write_text(json.dumps(accounting) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="successful Karolina job"):
        analysis._cluster_batch_evidence(
            campaign_root=tmp_path,
            case_id=case_id,
            job_id=job_id,
            expected_commit=commit,
        )
