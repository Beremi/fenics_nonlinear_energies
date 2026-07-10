#!/usr/bin/env python3
"""Prepare and adjudicate the seven cluster-deferred EXP-STOP-001 rows.

Preparation is scheduler-free.  It binds a complete clean local calibration,
freezes four 32-rank P4 nonlinear endpoints and 16/32/32-rank GL/HE/P3D MPI
consistency checks, and emits a reviewed Karolina command inventory.  Final
adjudication requires the checksum-sealed cluster archive plus the copied local
plan/analysis; it never queries a scheduler.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.analysis import finalize_reviewed_karolina_archive as archive
from experiments.analysis.finalize_karolina_campaign_archive import verify_archive
from experiments.runners import karolina_reviewed_campaign as reviewed
from experiments.runners import run_exp_stop_001_local_calibration as local


PROTOCOL = REPO_ROOT / "paper/protocols/EXP-STOP-001.md"
CAMPAIGN_ID = "exp_stop_001_karolina_deferred_v1"
P4_TARGETS = (1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8)
MPI_GROUPS = {
    "ginzburg_landau_mpi_consistency_cluster": ("gl_l6", 16, "01:00:00"),
    "hyperelasticity_mpi_consistency_cluster": ("he_l2_nonlinear", 32, "02:00:00"),
    "plasticity3d_mpi_consistency_cluster": ("p3d_p2_nonlinear", 32, "04:00:00"),
}
EXPECTED_DEFERRED_IDS = {
    *(f"p3d_p4_nonlinear_{local._float_id(target)}_cluster" for target in P4_TARGETS),
    *MPI_GROUPS,
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise reviewed.CampaignContractError(f"{path} must contain one JSON object")
    return value


def _local_inputs(analysis_path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    analysis_path = Path(analysis_path).resolve()
    analysis = _read(analysis_path)
    if (
        analysis.get("schema_id") != local.ANALYSIS_SCHEMA_ID
        or analysis.get("schema_version") != local.ANALYSIS_SCHEMA_VERSION
        or analysis.get("experiment_id") != "EXP-STOP-001"
        or analysis.get("terminal_decision")
        != "local_calibration_complete_cluster_computations_deferred"
        or analysis.get("complete_exp_stop_pass") is not False
    ):
        raise reviewed.CampaignContractError(
            "local analysis is not a complete cluster-deferred EXP-STOP-001 result"
        )
    counts = analysis.get("counts")
    if not isinstance(counts, dict) or any(
        int(counts.get(key, -1)) != 0
        for key in (
            "missing_local", "invalid_local", "runtime_censored_local", "reference_failures"
        )
    ):
        raise reviewed.CampaignContractError("local stopping analysis contains missing or invalid rows")
    deferred = analysis.get("deferred_cluster_computations")
    if not isinstance(deferred, list) or {str(row.get("row_id", "")) for row in deferred} != EXPECTED_DEFERRED_IDS:
        raise reviewed.CampaignContractError(
            "local stopping analysis must defer exactly the reviewed seven cluster rows"
        )
    plan_record = analysis.get("plan")
    if not isinstance(plan_record, dict) or not isinstance(plan_record.get("path"), str):
        raise reviewed.CampaignContractError("local analysis does not bind its plan")
    plan_path = Path(plan_record["path"]).resolve()
    if not plan_path.is_file() or reviewed.sha256_file(plan_path) != plan_record.get("sha256"):
        raise reviewed.CampaignContractError("local analysis plan is missing or stale")
    plan = _read(plan_path)
    if (
        plan.get("schema_id") != local.PLAN_SCHEMA_ID
        or plan.get("schema_version") != local.PLAN_SCHEMA_VERSION
        or plan.get("source", {}).get("commit") != plan_record.get("source_commit")
        or plan.get("run_kind") != "publication"
        or plan.get("source", {}).get("dirty") is not False
        or int(plan.get("row_counts", {}).get("required_local", -1)) != 45
        or int(plan.get("row_counts", {}).get("deferred_cluster_computation", -1)) != 7
    ):
        raise reviewed.CampaignContractError("local plan is not the clean 45+7 publication matrix")
    selected = analysis.get("selected_local_policies")
    if not isinstance(selected, dict):
        raise reviewed.CampaignContractError("local analysis has no selected policies")
    for group in (value[0] for value in MPI_GROUPS.values()):
        record = selected.get(group)
        if not isinstance(record, dict) or record.get("status") != "selected_loosest_accepted_same_discretization_policy":
            raise reviewed.CampaignContractError(f"local group {group} has no accepted policy")
    return analysis, plan_path, plan


def _row(plan: Mapping[str, Any], row_id: str) -> dict[str, Any]:
    matches = [row for row in plan["rows"] if row["row_id"] == row_id]
    if len(matches) != 1:
        raise reviewed.CampaignContractError(f"local row {row_id} is absent or duplicated")
    return deepcopy(matches[0])


def _set_option(argv: list[str], option: str, value: str) -> None:
    try:
        index = argv.index(option)
    except ValueError as exc:
        raise reviewed.CampaignContractError(f"frozen local command lacks {option}") from exc
    if index + 1 >= len(argv):
        raise reviewed.CampaignContractError(f"frozen local command has no value for {option}")
    argv[index + 1] = value


def _payload(row: Mapping[str, Any], *, result: str = "result.json", state: str = "state.npz") -> list[str]:
    command = [str(value) for value in row["command"]]
    if len(command) < 2:
        raise reviewed.CampaignContractError("frozen local command is malformed")
    runner = Path(command[1])
    if runner.is_absolute():
        try:
            runner = runner.resolve().relative_to(REPO_ROOT)
        except ValueError as exc:
            raise reviewed.CampaignContractError("local runner is outside the repository") from exc
    payload = ["{PYTHON}", "-u", f"{{REPO_ROOT}}/{runner.as_posix()}", *command[2:]]
    if "--out" in payload:
        _set_option(payload, "--out", f"{{JOB_ROOT}}/{result}")
    if "--output-json" in payload:
        _set_option(payload, "--output-json", f"{{JOB_ROOT}}/{result}")
    if "--state-out" in payload:
        _set_option(payload, "--state-out", f"{{JOB_ROOT}}/{state}")
    if "--out-dir" in payload:
        _set_option(payload, "--out-dir", "{JOB_ROOT}/work")
    return payload


def build_cases(analysis: Mapping[str, Any], plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    selected = analysis["selected_local_policies"]
    p2_reference = _row(plan, str(selected["p3d_p2_nonlinear"]["row_id"]))
    cases: list[dict[str, Any]] = []
    for target in P4_TARGETS:
        case_id = f"p3d_p4_nonlinear_{local._float_id(target)}_cluster"
        payload = _payload(p2_reference)
        _set_option(payload, "--elem-degree", "4")
        _set_option(payload, "--quadrature-rule", "tetra_24point")
        _set_option(payload, "--grad-stop-rtol", f"{target:.0e}")
        _set_option(payload, "--stop-tol", f"{target:.0e}")
        cases.append(
            {
                "case_id": case_id,
                "family": "plasticity3d_nonlinear_stopping",
                "nodes": 1,
                "total_ranks": 32,
                "ranks_per_node": 32,
                "partition": "qcpu_exp",
                "walltime": "04:00:00",
                "payload_argv": payload,
                "expected_outputs": ["result.json", "state.npz"],
                "scientific_contract": {
                    "kind": "p4_nonlinear_tolerance_sweep",
                    "template_row_id": p2_reference["row_id"],
                    "element_degree": 4,
                    "quadrature_rule_id": "tetra_24point",
                    "relative_dual_residual_target": target,
                    "reference_case_id": "p3d_p4_nonlinear_1em08_cluster",
                    "timing_claim_admissible": False,
                },
            }
        )
    for case_id, (group, ranks, walltime) in MPI_GROUPS.items():
        selected_row_id = str(selected[group]["row_id"])
        template = _row(plan, selected_row_id)
        cases.append(
            {
                "case_id": case_id,
                "family": template["family"],
                "nodes": 1,
                "total_ranks": ranks,
                "ranks_per_node": ranks,
                "partition": "qcpu_exp",
                "walltime": walltime,
                "payload_argv": _payload(template),
                "expected_outputs": ["result.json", "state.npz"],
                "scientific_contract": {
                    "kind": "mpi_consistency",
                    "local_group_id": group,
                    "local_reference_row_id": selected_row_id,
                    "publication_rank_count": ranks,
                    "same_accuracy_gate_as_local": True,
                    "timing_claim_admissible": False,
                },
            }
        )
    if {case["case_id"] for case in cases} != EXPECTED_DEFERRED_IDS or len(cases) != 7:
        raise reviewed.CampaignContractError("constructed STOP cluster matrix is not exactly seven rows")
    return cases


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    analysis, plan_path, plan = _local_inputs(args.local_analysis)
    cases = build_cases(analysis, plan)
    source_commit = str(plan["source"]["commit"])
    current = reviewed.git_metadata()
    if current != {"commit": source_commit, "dirty": False}:
        raise reviewed.CampaignContractError(
            "cluster preparation must use the same clean commit as the local campaign"
        )
    sources = {
        REPO_ROOT / "experiments/runners/prepare_exp_stop_001_karolina.py",
        REPO_ROOT / "experiments/runners/execute_reviewed_karolina_case.py",
        REPO_ROOT / "experiments/runners/submit_reviewed_karolina_campaign.py",
        REPO_ROOT / "experiments/analysis/finalize_reviewed_karolina_archive.py",
        REPO_ROOT / local.RUNNER_PATH,
        REPO_ROOT / local.TRUST_RUNNER_PATH,
        REPO_ROOT / local.P3D_BACKEND_PATH,
    }
    return reviewed.prepare_campaign(
        output_root=args.output_root,
        experiment_id="EXP-STOP-001",
        campaign_id=CAMPAIGN_ID,
        cases=cases,
        protocol=PROTOCOL,
        reviewed_sources=sources,
        env_setup=args.env_setup,
        env_lock=args.env_lock,
        git=current,
        external_bindings={
            "local_source_commit": source_commit,
            "local_terminal_decision": analysis["terminal_decision"],
            "required_local_rows": analysis["counts"]["required_local"],
            "deferred_cluster_rows": 7,
        },
        bound_inputs={"local_analysis": args.local_analysis, "local_plan": plan_path},
    )


def preflight(root: Path) -> dict[str, Any]:
    result = reviewed.offline_preflight(root)
    _manifest, plan = reviewed.load_plan(root)
    if (
        plan.get("experiment_id") != "EXP-STOP-001"
        or plan.get("campaign_id") != CAMPAIGN_ID
        or len(plan.get("cases", [])) != 7
        or {case["case_id"] for case in plan["cases"]} != EXPECTED_DEFERRED_IDS
    ):
        raise reviewed.CampaignContractError("prepared STOP campaign scope is stale")
    resources = {
        case["case_id"]: {
            "ranks": case["total_ranks"],
            "nodes": case["nodes"],
            "walltime": case["walltime"],
        }
        for case in plan["cases"]
    }
    return {**result, "reviewed_resources": resources, "node_hour_ceiling": 23.0}


def _job_roots(root: Path, plan: Mapping[str, Any]) -> dict[str, Path]:
    jobs = archive.submitted_jobs(root, dict(plan))
    return {
        case_id: root / "jobs" / case_id / f"job_{job_id}"
        for case_id, job_id in jobs.items()
    }


def _cluster_row(case: Mapping[str, Any], template: Mapping[str, Any], job_root: Path) -> dict[str, Any]:
    row = deepcopy(template)
    row["row_id"] = case["case_id"]
    row["group_id"] = (
        "p3d_p4_nonlinear_cluster"
        if case["scientific_contract"]["kind"] == "p4_nonlinear_tolerance_sweep"
        else template["group_id"]
    )
    row["expected_outputs"] = [str(job_root / name) for name in case["expected_outputs"]]
    if case["scientific_contract"]["kind"] == "p4_nonlinear_tolerance_sweep":
        row["parameters"] = deepcopy(template["parameters"])
        row["parameters"].update(
            {
                "element_degree": 4,
                "quadrature_rule_id": "tetra_24point",
                "relative_dual_residual_target": case["scientific_contract"][
                    "relative_dual_residual_target"
                ],
            }
        )
        row["reference_row"] = (
            case["case_id"] == case["scientific_contract"]["reference_case_id"]
        )
    return row


def _endpoint(row: Mapping[str, Any]) -> dict[str, Any]:
    if row["family"] == "ginzburg_landau":
        return local._gl_endpoint(row)
    if row["family"] == "hyperelasticity_nonlinear_stopping":
        return local._he_nonlinear_endpoint(row)
    if row["family"] == "plasticity3d_nonlinear_stopping":
        return local._p3d_nonlinear_endpoint(row)
    raise reviewed.CampaignContractError(f"unsupported cluster family {row['family']}")


def _compare(
    row: Mapping[str, Any], endpoint: Mapping[str, Any], reference_row: Mapping[str, Any],
    reference: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    if row["family"] == "ginzburg_landau":
        return local._compare_gl(row, endpoint, reference_row, reference, contract)
    if row["family"] == "hyperelasticity_nonlinear_stopping":
        return local._compare_he_nonlinear(row, endpoint, reference_row, reference, contract)
    return local._compare_p3d_nonlinear(row, endpoint, reference_row, reference, contract)


def adjudicate(root: Path, *, expected_checksum: str) -> dict[str, Any]:
    root = Path(root).resolve()
    verify_archive(root, expected_manifest_sha256=expected_checksum)
    manifest, cluster_plan = reviewed.load_plan(root)
    if manifest.get("status") != "submitted":
        raise reviewed.CampaignContractError("STOP adjudication requires a submitted archive")
    bindings = cluster_plan["external_bindings"]["archived_inputs"]
    analysis_path = root / bindings["local_analysis"]["path"]
    local_plan_path = root / bindings["local_plan"]["path"]
    if reviewed.sha256_file(analysis_path) != bindings["local_analysis"]["sha256"] or reviewed.sha256_file(local_plan_path) != bindings["local_plan"]["sha256"]:
        raise reviewed.CampaignContractError("archived local stopping inputs are stale")
    local_analysis = _read(analysis_path)
    local_plan = _read(local_plan_path)
    if local_plan["source"]["commit"] != cluster_plan["source_commit"]:
        raise reviewed.CampaignContractError("local and cluster evidence use different commits")
    templates = {row["row_id"]: row for row in local_plan["rows"]}
    roots = _job_roots(root, cluster_plan)
    rows: dict[str, dict[str, Any]] = {}
    endpoints: dict[str, dict[str, Any]] = {}
    for case in cluster_plan["cases"]:
        scientific = case["scientific_contract"]
        template_id = scientific.get("template_row_id") or scientific.get("local_reference_row_id")
        template = templates[str(template_id)]
        row = _cluster_row(case, template, roots[case["case_id"]])
        rows[case["case_id"]] = row
        endpoints[case["case_id"]] = _endpoint(row)
    analysis_contract = local_plan["policies"]["analysis_contract"]
    comparisons: dict[str, dict[str, Any]] = {}
    p4_reference_id = "p3d_p4_nonlinear_1em08_cluster"
    p4_reference_row = rows[p4_reference_id]
    p4_reference = endpoints[p4_reference_id]
    for case_id in sorted(case for case in rows if case.startswith("p3d_p4_")):
        comparisons[case_id] = _compare(
            rows[case_id], endpoints[case_id], p4_reference_row, p4_reference, analysis_contract
        )
    for case_id in MPI_GROUPS:
        scientific = next(
            case["scientific_contract"] for case in cluster_plan["cases"] if case["case_id"] == case_id
        )
        reference_id = str(scientific["local_reference_row_id"])
        reference_row = templates[reference_id]
        reference_endpoint = local_analysis["endpoints"].get(reference_id)
        if not isinstance(reference_endpoint, dict):
            raise reviewed.CampaignContractError(f"local endpoint {reference_id} is absent")
        comparisons[case_id] = _compare(
            rows[case_id], endpoints[case_id], reference_row, reference_endpoint, analysis_contract
        )
    rejected = sorted(case_id for case_id, value in comparisons.items() if value.get("status") != "accepted")
    selected = deepcopy(local_analysis["selected_local_policies"])
    p4_policy = local._selected_group_policy(
        [rows[key] for key in sorted(rows) if key.startswith("p3d_p4_")], comparisons
    )
    selected["p3d_p4_nonlinear_cluster"] = p4_policy
    nonlinear_targets = {
        float(record["tolerance"])
        for group, record in selected.items()
        if group in {
            "gl_l6", "he_l2_nonlinear", "p3d_p2_nonlinear", "p3d_p4_nonlinear_cluster"
        }
        and record.get("status") == "selected_loosest_accepted_same_discretization_policy"
    }
    if rejected:
        terminal = "CENSORED"
    elif len(nonlinear_targets) == 1:
        terminal = "PASS"
    else:
        terminal = "SCOPED_PASS"
    return {
        "schema_id": "fenics-nonlinear-energies.exp-stop-001.final-adjudication",
        "schema_version": 1,
        "experiment_id": "EXP-STOP-001",
        "terminal_decision": terminal,
        "complete_exp_stop_pass": terminal in {"PASS", "SCOPED_PASS"},
        "source_commit": cluster_plan["source_commit"],
        "local_analysis_sha256": reviewed.sha256_file(analysis_path),
        "cluster_archive_checksum_sha256": expected_checksum,
        "cluster_case_count": 7,
        "publication_timing_admissible": False,
        "comparisons": comparisons,
        "rejected_or_censored_cases": rejected,
        "selected_policies": selected,
        "policy_scope": (
            "One common nonlinear target across retained families"
            if terminal == "PASS"
            else "Family/degree-specific policies; cross-policy timing comparisons are prohibited"
        ),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--local-analysis", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--env-setup", type=Path)
    prepare_parser.add_argument("--env-lock", type=Path)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--campaign-root", type=Path, required=True)
    adjudicate_parser = subparsers.add_parser("adjudicate")
    adjudicate_parser.add_argument("--campaign-root", type=Path, required=True)
    adjudicate_parser.add_argument("--expected-checksum-manifest-sha256", required=True)
    adjudicate_parser.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args)
        elif args.command == "preflight":
            result = preflight(args.campaign_root)
        else:
            output = Path(args.output).resolve()
            campaign_root = Path(args.campaign_root).resolve()
            if output == campaign_root or campaign_root in output.parents:
                raise reviewed.CampaignContractError(
                    "final adjudication output must be detached from the sealed archive"
                )
            result = adjudicate(
                args.campaign_root,
                expected_checksum=args.expected_checksum_manifest_sha256,
            )
            reviewed.atomic_json(output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
