from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from experiments.analysis import collect_slurm_accounting as accounting
from experiments.analysis import finalize_reviewed_karolina_archive as finalizer
from experiments.runners import karolina_reviewed_campaign as reviewed
from experiments.runners import prepare_exp_stop_001_karolina as stop
from experiments.runners import run_exp_stop_001_local_calibration as local
from experiments.runners import submit_reviewed_karolina_campaign as submitter


COMMIT = "a" * 40


def _mock_fresh_analysis(
    monkeypatch: pytest.MonkeyPatch, analysis_path: Path
) -> None:
    def fresh(_plan_path: Path) -> dict[str, object]:
        value = json.loads(analysis_path.read_text(encoding="utf-8"))
        value["created_utc"] = "independently-regenerated"
        return value

    monkeypatch.setattr(local, "analyze_plan", fresh)


def _local_campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(local, "_git_metadata", lambda: {"commit": COMMIT, "dirty": False})
    plan = local.build_plan(
        tmp_path / "local",
        run_kind="publication",
        allow_dirty=False,
        p4_policy="local",
        confirm_p4_local_feasible=True,
    )
    local_root = tmp_path / "local"
    local_root.mkdir(parents=True)
    (local_root / "receipts").mkdir()
    (local_root / "logs").mkdir()
    plan_path = local_root / "plan.json"
    plan_path.write_text(json.dumps(plan) + "\n", encoding="utf-8")
    rows = {row["row_id"]: row for row in plan["rows"]}
    cluster_selected_ids = {
        "gl_l6": next(
            row["row_id"] for row in plan["rows"]
            if row["group_id"] == "gl_l6" and row["reference_row"]
        ),
        "he_l2_nonlinear": next(
            row["row_id"] for row in plan["rows"]
            if row["group_id"] == "he_l2_nonlinear" and row["reference_row"]
        ),
        "p3d_p2_nonlinear": next(
            row["row_id"] for row in plan["rows"]
            if row["group_id"] == "p3d_p2_nonlinear" and row["reference_row"]
        ),
    }
    selection_parameter = {
        "ginzburg_landau": "relative_dual_residual_target",
        "hyperelasticity_reference_riesz": "riesz_ksp_rtol",
        "hyperelasticity_nonlinear_stopping": "relative_dual_residual_target",
        "plasticity3d_fixed_state_linear": "ksp_rtol",
        "plasticity3d_nonlinear_stopping": "relative_dual_residual_target",
    }
    selected: dict[str, dict[str, object]] = {}
    for group in sorted(local.COMPLETE_REQUIRED_LOCAL_GROUPS):
        reference = next(
            row
            for row in plan["rows"]
            if row["group_id"] == group and row["reference_row"]
        )
        parameter = selection_parameter[str(reference["family"])]
        selected[group] = {
            "status": local.ACCEPTED_POLICY_STATUS,
            "row_id": reference["row_id"],
            "parameter": parameter,
            "tolerance": reference["parameters"][parameter],
        }
    plan_sha256 = reviewed.sha256_file(plan_path)
    for row_id in cluster_selected_ids.values():
        row = rows[row_id]
        output_hashes: dict[str, str] = {}
        for raw in row["expected_outputs"]:
            output = Path(raw)
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.suffix == ".json":
                output.write_text("{}\n", encoding="utf-8")
            else:
                output.write_bytes(f"local-state-{row_id}".encode())
            output_hashes[str(output.resolve())] = reviewed.sha256_file(output)
        log_root = local_root / "logs" / row_id
        log_root.mkdir(parents=True)
        stdout = log_root / "stdout.log"
        stderr = log_root / "stderr.log"
        stdout.write_text("fixture\n", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        (local_root / "receipts" / f"{row_id}.json").write_text(
            json.dumps(
                {
                    "schema_id": local.RECEIPT_SCHEMA_ID,
                    "schema_version": local.RECEIPT_SCHEMA_VERSION,
                    "row_id": row_id,
                    "status": "completed",
                    "plan_sha256": plan_sha256,
                    "command": row["command"],
                    "output_hashes": output_hashes,
                    "logs": {"stdout": str(stdout), "stderr": str(stderr)},
                }
            )
            + "\n",
            encoding="utf-8",
        )
    deferred = [
        {
            "row_id": row["row_id"],
            "family": row["family"],
            "parameters": row["parameters"],
            "censor": row["censor"],
        }
        for row in plan["rows"]
        if row["execution_class"] == "deferred_cluster_computation"
    ]
    analysis = {
        "schema_id": local.ANALYSIS_SCHEMA_ID,
        "schema_version": local.ANALYSIS_SCHEMA_VERSION,
        "experiment_id": "EXP-STOP-001",
        "terminal_decision": "local_calibration_complete_cluster_computations_deferred",
        "complete_exp_stop_pass": False,
        "plan": {
            "path": str(plan_path.resolve()),
            "sha256": reviewed.sha256_file(plan_path),
            "source_commit": COMMIT,
        },
        "counts": {
            "required_local": 45,
            "missing_local": 0,
            "invalid_local": 0,
            "runtime_censored_local": 0,
            "reference_failures": 0,
            "policy_gate_failures": 0,
            "deferred_cluster_computations": 7,
        },
        "deferred_cluster_computations": deferred,
        "selected_local_policies": selected,
        "required_local_policy_grid": local._required_local_policy_grid(
            [
                row
                for row in plan["rows"]
                if row["execution_class"] == "required_local"
            ],
            selected,
        ),
        "endpoints": {
            row_id: {"status": "endpoint_admitted", "row_id": row_id}
            for row_id in cluster_selected_ids.values()
        },
    }
    path = tmp_path / "local_analysis.json"
    path.write_text(json.dumps(analysis) + "\n", encoding="utf-8")
    return path


def _prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, environment: bool = False
) -> Path:
    analysis = _local_campaign(tmp_path, monkeypatch)
    _mock_fresh_analysis(monkeypatch, analysis)
    monkeypatch.setattr(reviewed, "git_metadata", lambda: {"commit": COMMIT, "dirty": False})
    setup = lock = None
    if environment:
        setup = tmp_path / "setup.sh"
        lock = tmp_path / "lock.json"
        setup.write_text("export PYTHON=./.venv/bin/python\n", encoding="utf-8")
        lock.write_text("{}\n", encoding="utf-8")
    root = tmp_path / "cluster"
    stop.prepare(
        SimpleNamespace(
            local_analysis=analysis,
            output_root=root,
            env_setup=setup,
            env_lock=lock,
        )
    )
    return root


def _sacct(case: dict[str, object], job_id: str) -> str:
    values = {
        "JobIDRaw": job_id,
        "JobID": job_id,
        "JobName": str(case["case_id"]),
        "Cluster": "karolina",
        "Account": reviewed.ACCOUNT,
        "Partition": str(case["partition"]),
        "QOS": reviewed.QOS,
        "State": "COMPLETED",
        "ElapsedRaw": "10",
        "AllocNodes": str(case["nodes"]),
        "AllocCPUS": str(case["total_ranks"]),
        "TotalCPU": "00:00:10",
        "CPUTimeRAW": str(10 * int(case["total_ranks"])),
        "MaxRSS": "1K",
        "MaxVMSize": "2K",
        "ConsumedEnergyRaw": "0",
        "ExitCode": "0:0",
        "Start": "2026-07-10T10:00:00",
        "End": "2026-07-10T10:00:10",
        "NodeList": "cn001",
    }
    return "|".join(accounting.SACCT_FIELDS) + "\n" + "|".join(
        values[field] for field in accounting.SACCT_FIELDS
    ) + "\n"


def _fake_submitted_archive(root: Path) -> Path:
    manifest = reviewed.read_object(root / "prepared_manifest.json")
    plan = reviewed.read_object(root / manifest["plan"]["path"])
    manifest["status"] = "submitted"
    manifest["scheduler_contact"] = True
    manifest["accepted_jobs"] = len(plan["cases"])
    reviewed.atomic_json(root / "prepared_manifest.json", manifest)
    ledgers: list[str] = []
    journals: list[str] = []
    snapshots = root.parent / "snapshots"
    snapshots.mkdir()
    records: list[dict[str, str]] = []
    for index, case in enumerate(plan["cases"], 101):
        job_id = str(index)
        line = f"sbatch fixture {case['case_id']}"
        ledgers.append(json.dumps({
            "case_id": case["case_id"], "command": line, "returncode": 0,
            "stdout": f"Submitted batch job {job_id}", "stderr": "", "job_id": job_id,
        }))
        for event in ("intent", "result"):
            journals.append(json.dumps({
                "event": event, "attempt_id": f"fixture-{case['case_id']}",
                "case_id": case["case_id"], "command": line,
            }))
        job_root = root / "jobs" / case["case_id"] / f"job_{job_id}"
        job_root.mkdir(parents=True)
        hashes: dict[str, str] = {}
        for output in case["expected_outputs"]:
            path = job_root / output
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"fixture:{case['case_id']}:{output}".encode())
            hashes[output] = reviewed.sha256_file(path)
        reviewed.atomic_json(job_root / "job_metadata.json", {
            "case_id": case["case_id"], "job_id": job_id,
            "source_commit": COMMIT,
            "plan_sha256": manifest["plan"]["sha256"],
            "source_freeze_sha256": manifest["source_freeze"]["sha256"],
            "resources": {
                "account": reviewed.ACCOUNT, "qos": reviewed.QOS,
                "partition": case["partition"], "nodes": case["nodes"],
                "total_ranks": case["total_ranks"],
            },
        })
        reviewed.atomic_json(job_root / "execution.json", {
            "case_id": case["case_id"], "job_id": job_id,
            "returncode": 0, "output_hashes": hashes,
        })
        reviewed.atomic_json(job_root / "environment.json", {"fixture": True})
        (job_root / "stdout.log").write_text("fixture\n", encoding="utf-8")
        (job_root / "stderr.log").write_text("", encoding="utf-8")
        raw = snapshots / f"{job_id}.sacct"
        raw.write_text(_sacct(case, job_id), encoding="utf-8")
        records.append({
            "case_id": case["case_id"], "job_id": job_id,
            "path": raw.name, "sha256": reviewed.sha256_file(raw),
        })
    (root / "submitted_jobs.jsonl").write_text("\n".join(ledgers) + "\n", encoding="utf-8")
    (root / "submission_journal.jsonl").write_text("\n".join(journals) + "\n", encoding="utf-8")
    index = snapshots / "index.json"
    index.write_text(json.dumps({
        "schema_id": finalizer.OFFLINE_INDEX_SCHEMA_ID,
        "schema_version": finalizer.OFFLINE_INDEX_SCHEMA_VERSION,
        "campaign_manifest_sha256": reviewed.sha256_file(root / "prepared_manifest.json"),
        "records": records,
    }) + "\n", encoding="utf-8")
    return index


def test_preparation_freezes_exact_seven_rows_and_never_contacts_scheduler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _prepare(tmp_path, monkeypatch)
    receipt = stop.preflight(root)
    _manifest, plan = reviewed.load_plan(root)
    source_freeze = reviewed.read_object(root / "reviewed_source_freeze.json")

    assert receipt["status"] == "passed_without_scheduler_contact"
    assert receipt["submission_admissible"] is False
    assert receipt["node_hour_ceiling"] == 23.0
    assert len(plan["cases"]) == 7
    assert {case["case_id"] for case in plan["cases"]} == stop.EXPECTED_DEFERRED_IDS
    assert {
        case["case_id"]: case["total_ranks"] for case in plan["cases"]
        if case["scientific_contract"]["kind"] == "mpi_consistency"
    } == {
        "ginzburg_landau_mpi_consistency_cluster": 16,
        "hyperelasticity_mpi_consistency_cluster": 32,
        "plasticity3d_mpi_consistency_cluster": 32,
    }
    p4 = [case for case in plan["cases"] if case["case_id"].startswith("p3d_p4_")]
    assert len(p4) == 4 and {case["total_ranks"] for case in p4} == {32}
    assert all(case["expected_outputs"] == ["result.json", "state.npz"] for case in plan["cases"])
    references = plan["external_bindings"]["metadata"]["local_reference_artifacts"]
    assert set(references) == {"gl_l6", "he_l2_nonlinear", "p3d_p2_nonlinear"}
    assert all(
        set(reference["artifacts"]) == {"result", "state", "receipt", "stdout", "stderr"}
        for reference in references.values()
    )
    assert {
        "src/core/petsc/metrics.py",
        "src/problems/hyperelasticity/jax_petsc/solver.py",
        "src/problems/slope_stability_3d/jax_petsc/solver.py",
    } <= set(source_freeze["reviewed_sources"])
    assert submitter.submit(root, execute=False, confirmed=False)["status"] == "dry_run_no_scheduler_contact"


def test_preparation_rejects_a_required_local_group_without_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis_path = _local_campaign(tmp_path, monkeypatch)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["selected_local_policies"]["p3d_p4"] = {
        "status": "no_acceptable_policy",
        "row_id": None,
        "tolerance": None,
    }
    plan = json.loads((tmp_path / "local" / "plan.json").read_text(encoding="utf-8"))
    analysis["required_local_policy_grid"] = local._required_local_policy_grid(
        [
            row
            for row in plan["rows"]
            if row["execution_class"] == "required_local"
        ],
        analysis["selected_local_policies"],
    )
    assert all(
        analysis["counts"][key] == 0
        for key in (
            "missing_local",
            "invalid_local",
            "runtime_censored_local",
            "reference_failures",
            "policy_gate_failures",
        )
    )
    analysis_path.write_text(json.dumps(analysis) + "\n", encoding="utf-8")
    _mock_fresh_analysis(monkeypatch, analysis_path)

    with pytest.raises(
        reviewed.CampaignContractError,
        match="lack accepted policies",
    ):
        stop._local_inputs(analysis_path)


def test_preparation_rejects_fabricated_complete_summary_with_missing_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis_path = _local_campaign(tmp_path, monkeypatch)
    assert len(list((tmp_path / "local" / "receipts").glob("*.json"))) == 3

    with pytest.raises(
        reviewed.CampaignContractError,
        match="differs from a fresh 45-row reanalysis",
    ):
        stop._local_inputs(analysis_path)


def test_preparation_rejects_local_riesz_contract_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis_path = _local_campaign(tmp_path, monkeypatch)

    def fail_reanalysis(_plan_path: Path) -> dict[str, object]:
        raise local.CampaignError("Riesz solver provenance differs from frozen plan")

    monkeypatch.setattr(local, "analyze_plan", fail_reanalysis)
    with pytest.raises(
        reviewed.CampaignContractError,
        match="could not be independently reanalyzed",
    ):
        stop._local_inputs(analysis_path)


def test_preflight_fails_closed_on_command_or_plan_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _prepare(tmp_path, monkeypatch)
    (root / "sbatch_commands.txt").write_text("sbatch changed\n", encoding="utf-8")
    with pytest.raises(reviewed.CampaignContractError, match="commands is missing or stale"):
        stop.preflight(root)


def test_offline_archive_settlement_and_final_merge_are_hash_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _prepare(tmp_path, monkeypatch, environment=True)
    for external in (tmp_path / "local", tmp_path / "local_analysis.json"):
        if external.is_dir():
            shutil.rmtree(external)
        else:
            external.unlink()
    assert not (tmp_path / "local").exists()
    assert not (tmp_path / "local_analysis.json").exists()
    index = _fake_submitted_archive(root)
    settled = finalizer.finalize(root, offline_index=index)
    digest = settled["archive_checksums"]["sha256"]
    assert settled["settled_jobs"] == 7

    monkeypatch.setattr(
        stop,
        "_endpoint",
        lambda row: {"status": "endpoint_admitted", "row_id": row["row_id"]},
    )
    monkeypatch.setattr(
        stop,
        "_compare",
        lambda row, endpoint, reference_row, reference, contract: {
            "status": "accepted",
            "row_id": row["row_id"],
        },
    )
    result = stop.adjudicate(root, expected_checksum=digest)
    assert result["schema_version"] == 2
    assert (
        result["terminal_decision"]
        == "CALIBRATION_SCOPED_PASS_PENDING_DISCRETIZATION_GATE"
    )
    assert result["calibration_scope_passed"] is True
    assert result["complete_exp_stop_pass"] is False
    assert result["discretization_gate"]["status"] == "not_bound"
    assert result["remaining_completion_gates"] == [
        "hash-bound EXP-DISC-001 discretization-error adjudication"
    ]
    assert len(result["comparisons"]) == 7
    assert result["publication_timing_admissible"] is False

    (root / "unindexed.txt").write_text("tamper\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or unindexed"):
        stop.adjudicate(root, expected_checksum=digest)
