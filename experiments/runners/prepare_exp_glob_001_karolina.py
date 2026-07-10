#!/usr/bin/env python3
"""Prepare, analyze, and release the 60-launch full-rank EXP-GLOB-001 tranche.

All preparation, analysis, and release operations are scheduler-free.  The
only scheduler commands are inert text in the prepared command inventory until
an operator later uses the separately guarded submission utility.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tempfile
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.analysis import finalize_reviewed_karolina_archive as archive
from experiments.analysis.finalize_karolina_campaign_archive import verify_archive
from experiments.runners import karolina_reviewed_campaign as reviewed
from experiments.runners import run_globalization_method_compare as glob
from src.core.benchmark.run_record import (
    ExperimentPreflight,
    atomic_write_run_record,
    validate_run_record,
)


PROTOCOL = REPO_ROOT / "paper/protocols/EXP-GLOB-001.md"
CAMPAIGN_ID = "exp_glob_001_karolina_full_controlled_v1"
ANALYSIS_SCHEMA_ID = "fenics-nonlinear-energies.exp-glob-001.karolina-analysis"
FINAL_SCHEMA_ID = "fenics-nonlinear-energies.exp-glob-001.karolina-adjudication"
CONTROLLED_ENVIRONMENT = {
    "JAX_PLATFORMS": "cpu",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
}


def _cases() -> list[glob.CaseSpec]:
    cases = glob.build_case_matrix(
        "full", "controlled", timing_repetitions=glob.DEFAULT_TIMING_REPETITIONS
    )
    if (
        len(cases) != 60
        or {case.benchmark.problem for case in cases} != {"gl", "he"}
        or {case.benchmark.nprocs for case in cases if case.benchmark.problem == "gl"} != {16}
        or {case.benchmark.nprocs for case in cases if case.benchmark.problem == "he"} != {32}
        or {case.timing_repetition for case in cases} != {1, 2, 3, 4, 5}
        or len({case.key for case in cases}) != 60
    ):
        raise reviewed.CampaignContractError("full controlled globalization matrix is not 60 launches")
    return cases


def _label(start_key: str) -> str:
    value = "start_" + "_".join(
        part for part in start_key.replace("::", "_").replace("-", "_").split("_") if part
    )
    if not all(character.isalnum() or character in "_.-" for character in value):
        raise reviewed.CampaignContractError(f"unsafe common-start key {start_key!r}")
    return value


def _set_option(argv: list[str], option: str, value: str) -> None:
    try:
        index = argv.index(option)
    except ValueError as exc:
        raise reviewed.CampaignContractError(f"controlled command lacks {option}") from exc
    if index + 1 >= len(argv):
        raise reviewed.CampaignContractError(f"controlled command has no value for {option}")
    argv[index + 1] = value


def _payload(case: glob.CaseSpec, *, start_archive_path: str) -> list[str]:
    command = glob.build_command(
        case,
        Path("/tmp/exp-glob-output.json"),
        state_in=Path("/tmp/exp-glob-start.npz"),
        state_out=Path("/tmp/exp-glob-final.npz"),
    )
    if command[:3] != ["mpiexec", "-n", str(case.benchmark.nprocs)]:
        raise reviewed.CampaignContractError("globalization command has an unexpected MPI prefix")
    command = command[3:]
    if len(command) < 3 or command[1] != "-u":
        raise reviewed.CampaignContractError("globalization payload has an unexpected Python prefix")
    runner = Path(command[2]).resolve()
    try:
        runner = runner.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise reviewed.CampaignContractError("globalization runner is outside the repository") from exc
    payload = ["{PYTHON}", "-u", f"{{REPO_ROOT}}/{runner.as_posix()}", *command[3:]]
    _set_option(payload, "--out", "{JOB_ROOT}/output.json")
    _set_option(payload, "--state-in", f"{{CAMPAIGN_ROOT}}/{start_archive_path}")
    _set_option(payload, "--state-out", "{JOB_ROOT}/final_state.npz")
    return payload


def _generate_starts(
    cases: Sequence[glob.CaseSpec], temporary_root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    identities = glob.prepare_controlled_starts(list(cases), temporary_root)
    expected = {case.start_key for case in cases}
    if set(identities) != expected or len(identities) != 6:
        raise reviewed.CampaignContractError("controlled full matrix did not generate six starts")
    bindings: dict[str, Path] = {}
    archived: dict[str, dict[str, Any]] = {}
    for start_key, identity in sorted(identities.items()):
        label = _label(start_key)
        source = Path(str(identity["path"])).resolve()
        if not source.is_file() or reviewed.sha256_file(source) != identity.get("file_sha256"):
            raise reviewed.CampaignContractError(f"generated common start is stale: {start_key}")
        bindings[label] = source
        archived[start_key] = {
            **{key: value for key, value in identity.items() if key != "path"},
            "binding_label": label,
            "archive_path": f"bound_inputs/{label}{source.suffix}",
        }
    return archived, bindings


def _plan_cases(
    cases: Sequence[glob.CaseSpec], starts: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for case in cases:
        start = starts[case.start_key]
        walltime = "00:10:00" if case.benchmark.problem == "gl" else "00:15:00"
        result.append(
            {
                "case_id": case.key,
                "family": case.benchmark.problem,
                "nodes": 1,
                "total_ranks": case.benchmark.nprocs,
                "ranks_per_node": case.benchmark.nprocs,
                "partition": "qcpu_exp",
                "walltime": walltime,
                "payload_argv": _payload(
                    case, start_archive_path=str(start["archive_path"])
                ),
                "expected_outputs": ["output.json", "final_state.npz"],
                "scientific_contract": {
                    "kind": "controlled_globalization_full_rank",
                    "benchmark": case.benchmark.key,
                    "method": case.method.key,
                    "robustness_instance": case.robustness_instance.key,
                    "timing_repetition": case.timing_repetition,
                    "common_start_key": case.start_key,
                    "common_start_file_sha256": start["file_sha256"],
                    "common_start_state_sha256": start["state_sha256"],
                    "solver_wall_cap_s": case.benchmark.wall_cap_s,
                    "controlled_environment": dict(CONTROLLED_ENVIRONMENT),
                    "robustness_generalization_claim_admissible": False,
                },
            }
        )
    return result


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    current = reviewed.git_metadata()
    if bool(current.get("dirty")):
        raise reviewed.CampaignContractError("globalization preparation requires a clean worktree")
    cases = _cases()
    with tempfile.TemporaryDirectory(prefix="exp_glob_001_starts_") as temporary:
        starts, bindings = _generate_starts(cases, Path(temporary))
        sources = {
            REPO_ROOT / "experiments/runners/prepare_exp_glob_001_karolina.py",
            REPO_ROOT / "experiments/runners/run_globalization_method_compare.py",
            REPO_ROOT / "experiments/runners/run_trust_region_case.py",
            REPO_ROOT / "experiments/runners/execute_reviewed_karolina_case.py",
            REPO_ROOT / "experiments/runners/submit_reviewed_karolina_campaign.py",
            REPO_ROOT / "experiments/analysis/finalize_reviewed_karolina_archive.py",
            REPO_ROOT / "data/meshes/GinzburgLandau/GL_level10.h5",
        }
        return reviewed.prepare_campaign(
            output_root=args.output_root,
            experiment_id="EXP-GLOB-001",
            campaign_id=CAMPAIGN_ID,
            cases=_plan_cases(cases, starts),
            protocol=PROTOCOL,
            reviewed_sources=sources,
            env_setup=args.env_setup,
            env_lock=args.env_lock,
            git=current,
            external_bindings={
                "mode": "full",
                "comparison_tier": "controlled",
                "timing_repetitions": 5,
                "robustness_instances": [instance.key for instance in glob.ROBUSTNESS_INSTANCES],
                "controlled_environment": dict(CONTROLLED_ENVIRONMENT),
                "canonical_starts": starts,
            },
            bound_inputs=bindings,
        )


def preflight(root: Path) -> dict[str, Any]:
    receipt = reviewed.offline_preflight(root)
    _manifest, plan = reviewed.load_plan(root)
    cases = plan.get("cases", [])
    if (
        plan.get("experiment_id") != "EXP-GLOB-001"
        or plan.get("campaign_id") != CAMPAIGN_ID
        or len(cases) != 60
        or sum(case["family"] == "gl" for case in cases) != 30
        or sum(case["family"] == "he" for case in cases) != 30
    ):
        raise reviewed.CampaignContractError("prepared globalization scope is stale")
    metadata = plan["external_bindings"]["metadata"]
    starts = metadata.get("canonical_starts")
    if not isinstance(starts, dict) or len(starts) != 6:
        raise reviewed.CampaignContractError("prepared globalization starts are incomplete")
    for case in cases:
        scientific = case["scientific_contract"]
        start = starts.get(scientific["common_start_key"])
        if (
            not isinstance(start, dict)
            or start.get("file_sha256") != scientific["common_start_file_sha256"]
            or start.get("state_sha256") != scientific["common_start_state_sha256"]
        ):
            raise reviewed.CampaignContractError("case/common-start identity is stale")
    return {
        **receipt,
        "gl_launches_16_ranks": 30,
        "he_launches_32_ranks": 30,
        "canonical_start_count": 6,
        "node_hour_ceiling": 12.5,
    }


def _case_map() -> dict[str, glob.CaseSpec]:
    cases = _cases()
    return {case.key: case for case in cases}


def _job_roots(root: Path, plan: Mapping[str, Any]) -> dict[str, Path]:
    jobs = archive.submitted_jobs(root, dict(plan))
    return {
        case_id: root / "jobs" / case_id / f"job_{job_id}"
        for case_id, job_id in jobs.items()
    }


def _preflight_record(source_commit: str) -> ExperimentPreflight:
    return ExperimentPreflight(
        run_kind="publication",
        git_commit=source_commit,
        git_clean=True,
        git_status_porcelain=(),
        pilot_override=False,
        pilot_override_reason=None,
        checked_at_utc=reviewed.utc_now(),
    )


def _canonical_starts(root: Path, plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    metadata = plan["external_bindings"]["metadata"]
    starts: dict[str, dict[str, Any]] = {}
    for key, raw in metadata["canonical_starts"].items():
        identity = dict(raw)
        path = (root / identity["archive_path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise reviewed.CampaignContractError("canonical start escapes the campaign") from exc
        if not path.is_file() or path.is_symlink() or reviewed.sha256_file(path) != identity["file_sha256"]:
            raise reviewed.CampaignContractError(f"canonical start is missing or stale: {key}")
        identity["path"] = str(path)
        starts[key] = identity
    return starts


def analyze(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest, plan = reviewed.load_plan(root)
    if manifest.get("status") != "submitted" or manifest.get("scheduler_contact") is not True:
        raise reviewed.CampaignContractError("globalization analysis requires all 60 submitted jobs")
    cases = _case_map()
    planned = {case["case_id"]: case for case in plan["cases"]}
    if set(cases) != set(planned):
        raise reviewed.CampaignContractError("submitted globalization cases differ from the frozen matrix")
    starts = _canonical_starts(root, plan)
    job_roots = _job_roots(root, plan)
    configuration = glob.campaign_configuration(
        mode="full",
        comparison_tier="controlled",
        cases=list(cases.values()),
        child_environment=CONTROLLED_ENVIRONMENT,
    )
    configuration_sha256 = glob._json_sha256(configuration)
    freeze = reviewed.read_object(root / manifest["source_freeze"]["path"])
    source_hashes = {
        path: str(record["sha256"])
        for path, record in freeze["reviewed_sources"].items()
    }
    rows: list[dict[str, Any]] = []
    records: list[dict[str, str]] = []
    for case_id in [case["case_id"] for case in plan["cases"]]:
        case = cases[case_id]
        job_root = job_roots[case_id]
        execution = reviewed.read_object(job_root / "execution.json")
        metadata = reviewed.read_object(job_root / "job_metadata.json")
        if int(execution.get("returncode", 1)) != 0:
            raise reviewed.CampaignContractError(f"launcher failed for {case_id}")
        payload = reviewed.read_object(job_root / "output.json")
        command = [str(value) for value in metadata["payload_argv"]]
        row = glob.summarize_payload(
            mode="full",
            case=case,
            payload=payload,
            json_path=job_root / "output.json",
            log_path=job_root / "stdout.log",
            command=command,
            returncode=int(execution["returncode"]),
            wall_time_s=float(execution["wall_time_s"]),
            started_at_utc=str(execution["started_at_utc"]),
            finished_at_utc=str(execution["finished_at_utc"]),
        )
        start = starts[case.start_key]
        environment = reviewed.read_object(job_root / "environment.json")
        environment.update(
            {
                "scheduler": "Slurm",
                "cluster": "Karolina CPU",
                "job_id": str(execution["job_id"]),
                "account": reviewed.ACCOUNT,
                "qos": reviewed.QOS,
            }
        )
        record = glob.build_publication_run_record(
            case=case,
            mode="full",
            row=row,
            command=command,
            preflight=_preflight_record(str(plan["source_commit"])),
            environment=environment,
            source_hashes=source_hashes,
            campaign_configuration_sha256=configuration_sha256,
            canonical_start=start,
            raw_dir=root / "_record_layout_only",
        )
        record["identifiers"]["route"] = "controlled-karolina-common-start"
        record["resources"].update(
            {
                "nodes": 1,
                "ranks": case.benchmark.nprocs,
                "threads_per_rank": 1,
                "notes": "One reviewed Karolina CPU allocation; timing admission remains campaign-gated.",
            }
        )
        record["artifacts"] = {
            "raw_outputs": [str(job_root / "output.json")],
            "states": [str(Path(start["path"])), str(job_root / "final_state.npz")],
            "logs": [str(job_root / "stdout.log"), str(job_root / "stderr.log")],
            "tables": [],
            "figures": [],
            "reports": [],
        }
        record_path = job_root / "run_record.json"
        atomic_write_run_record(record_path, record, require_publication_ready=True)
        validate_run_record(record, require_publication_ready=True)
        row["run_record_path"] = str(record_path)
        row["run_record_sha256"] = reviewed.sha256_file(record_path)
        rows.append(row)
        records.append({"case_id": case_id, "path": str(record_path.relative_to(root)), "sha256": row["run_record_sha256"]})
    audit = glob.controlled_identity_audit(
        rows,
        starts,
        expected_repetitions=5,
        expected_instances=[instance.key for instance in glob.ROBUSTNESS_INSTANCES],
    )
    analysis_root = root / "analysis"
    analysis_root.mkdir(exist_ok=False)
    csv_path = analysis_root / "full_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=glob.CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "schema_id": ANALYSIS_SCHEMA_ID,
        "schema_version": 1,
        "experiment_id": "EXP-GLOB-001",
        "source_commit": plan["source_commit"],
        "case_count": 60,
        "rows": rows,
        "run_records": records,
        "identity_audit": audit,
        "claim_admission": {
            "status": "pending_offline_accounting_and_archive_seal",
            "timing_claim_admissible": False,
            "tested_instance_comparison_admissible": False,
            "robustness_generalization_claim_admissible": False,
        },
    }
    reviewed.atomic_json(analysis_root / "analysis.json", result)
    return result


def adjudicate(root: Path, *, expected_checksum: str) -> dict[str, Any]:
    root = Path(root).resolve()
    verified = verify_archive(root, expected_manifest_sha256=expected_checksum)
    settled = archive.verify_settled_archive(root)
    _manifest, plan = reviewed.load_plan(root)
    analysis_path = root / "analysis" / "analysis.json"
    analysis = reviewed.read_object(analysis_path)
    if (
        analysis.get("schema_id") != ANALYSIS_SCHEMA_ID
        or analysis.get("schema_version") != 1
        or analysis.get("source_commit") != plan.get("source_commit")
        or int(analysis.get("case_count", -1)) != 60
    ):
        raise reviewed.CampaignContractError("globalization analysis identity is stale")
    audit = analysis.get("identity_audit")
    if not isinstance(audit, dict):
        raise reviewed.CampaignContractError("globalization identity audit is absent")
    records = analysis.get("run_records")
    if not isinstance(records, list) or len(records) != 60:
        raise reviewed.CampaignContractError("globalization run-record inventory is incomplete")
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {"case_id", "path", "sha256"}:
            raise reviewed.CampaignContractError("globalization run-record entry is malformed")
        case_id = str(record["case_id"])
        path = (root / str(record["path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise reviewed.CampaignContractError("globalization run record escapes the archive") from exc
        if case_id in seen or not path.is_file() or path.is_symlink() or reviewed.sha256_file(path) != record["sha256"]:
            raise reviewed.CampaignContractError("globalization run record is missing, duplicated, or stale")
        seen.add(case_id)
    if seen != {case["case_id"] for case in plan["cases"]}:
        raise reviewed.CampaignContractError("globalization run records do not cover the plan")
    passed = audit.get("status") == "passed"
    return {
        "schema_id": FINAL_SCHEMA_ID,
        "schema_version": 1,
        "experiment_id": "EXP-GLOB-001",
        "source_commit": plan["source_commit"],
        "cluster_archive_checksum_sha256": expected_checksum,
        "archive_file_count": verified["file_count"],
        "settled_job_count": settled["job_count"],
        "analysis_sha256": reviewed.sha256_file(analysis_path),
        "identity_audit_status": audit.get("status"),
        "timing_claim_admissible": bool(passed and audit.get("timing_claim_admissible") is True),
        "tested_instance_comparison_admissible": bool(
            passed and audit.get("tested_instance_comparison_admissible") is True
        ),
        "robustness_generalization_claim_admissible": False,
        "release_status": "admitted_for_tested_instances" if passed else "identity_gate_failed",
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--env-setup", type=Path)
    prepare_parser.add_argument("--env-lock", type=Path)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--campaign-root", type=Path, required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--campaign-root", type=Path, required=True)
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
        elif args.command == "analyze":
            result = analyze(args.campaign_root)
        else:
            output = Path(args.output).resolve()
            root = Path(args.campaign_root).resolve()
            if output == root or root in output.parents:
                raise reviewed.CampaignContractError("adjudication output must be detached from the sealed archive")
            result = adjudicate(
                root, expected_checksum=args.expected_checksum_manifest_sha256
            )
            reviewed.atomic_json(output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
