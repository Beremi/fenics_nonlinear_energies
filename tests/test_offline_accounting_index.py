from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.analysis import collect_slurm_accounting as accounting
from experiments.analysis import generate_offline_accounting_index as generator
from experiments.runners import karolina_reviewed_campaign as reviewed
from experiments.runners.paper_revision_karolina import prepare_campaign as legacy_preparer


COMMIT = "c" * 40


def _raw(
    *,
    case_id: str,
    job_id: str,
    partition: str,
    nodes: int,
    ranks: int,
    state: str = "COMPLETED",
    exit_code: str = "0:0",
) -> str:
    values = {
        "JobIDRaw": job_id,
        "JobID": job_id,
        "JobName": case_id,
        "Cluster": "karolina",
        "Account": reviewed.ACCOUNT,
        "Partition": partition,
        "QOS": reviewed.QOS,
        "State": state,
        "ElapsedRaw": "10",
        "AllocNodes": str(nodes),
        "AllocCPUS": str(ranks),
        "TotalCPU": "00:00:10",
        "CPUTimeRAW": str(10 * ranks),
        "MaxRSS": "1K",
        "MaxVMSize": "2K",
        "ConsumedEnergyRaw": "0",
        "ExitCode": exit_code,
        "Start": "2026-07-10T10:00:00",
        "End": "2026-07-10T10:00:10",
        "NodeList": "cn001",
    }
    return "|".join(accounting.SACCT_FIELDS) + "\n" + "|".join(
        values[field] for field in accounting.SACCT_FIELDS
    ) + "\n"


def _generic_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    monkeypatch.setattr(reviewed, "git_metadata", lambda: {"commit": COMMIT, "dirty": False})
    setup = tmp_path / "setup.sh"
    lock = tmp_path / "lock.json"
    setup.write_text("export PYTHON=./.venv/bin/python\n", encoding="utf-8")
    lock.write_text("{}\n", encoding="utf-8")
    root = tmp_path / "campaign"
    case = {
        "case_id": "fixture_case",
        "family": "fixture",
        "nodes": 1,
        "total_ranks": 8,
        "ranks_per_node": 8,
        "partition": "qcpu_exp",
        "walltime": "00:10:00",
        "payload_argv": [
            "{PYTHON}", "-u", "{REPO_ROOT}/experiments/runners/run_trust_region_case.py",
            "--out", "{JOB_ROOT}/result.json",
        ],
        "expected_outputs": ["result.json"],
        "scientific_contract": {"kind": "fixture"},
    }
    reviewed.prepare_campaign(
        output_root=root,
        experiment_id="EXP-FIXTURE",
        campaign_id="offline-accounting-fixture",
        cases=[case],
        protocol=reviewed.REPO_ROOT / "paper/protocols/EXP-STOP-001.md",
        reviewed_sources=[
            reviewed.REPO_ROOT / "experiments/analysis/generate_offline_accounting_index.py"
        ],
        env_setup=setup,
        env_lock=lock,
        git={"commit": COMMIT, "dirty": False},
    )
    manifest = reviewed.read_object(root / "prepared_manifest.json")
    command_line = (root / manifest["commands"]["path"]).read_text(encoding="utf-8")
    assert "--job-name=fixture_case" in command_line
    manifest["status"] = "submitted"
    manifest["scheduler_contact"] = True
    manifest["accepted_jobs"] = 1
    reviewed.atomic_json(root / "prepared_manifest.json", manifest)
    command = "sbatch fixture"
    (root / "submitted_jobs.jsonl").write_text(
        json.dumps(
            {
                "case_id": "fixture_case",
                "command": command,
                "returncode": 0,
                "stdout": "Submitted batch job 123",
                "stderr": "",
                "job_id": "123",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "submission_journal.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "event": event,
                    "attempt_id": "fixture-attempt",
                    "case_id": "fixture_case",
                    "command": command,
                }
            )
            for event in ("intent", "result")
        )
        + "\n",
        encoding="utf-8",
    )
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    (snapshots / "123.sacct").write_text(
        _raw(
            case_id="fixture_case",
            job_id="123",
            partition="qcpu_exp",
            nodes=1,
            ranks=8,
        ),
        encoding="utf-8",
    )
    return root, snapshots, snapshots / "accounting-index.json"


def test_generic_index_is_complete_canonical_and_byte_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, snapshots, output = _generic_fixture(tmp_path, monkeypatch)
    first = generator.generate(
        campaign_root=root, snapshot_root=snapshots, output=output
    )
    first_bytes = output.read_bytes()
    assert first == {
        "schema_id": generator.SCHEMA_ID,
        "schema_version": generator.SCHEMA_VERSION,
        "campaign_manifest_sha256": reviewed.sha256_file(
            root / "prepared_manifest.json"
        ),
        "records": [
            {
                "case_id": "fixture_case",
                "job_id": "123",
                "path": "123.sacct",
                "sha256": reviewed.sha256_file(snapshots / "123.sacct"),
            }
        ],
    }
    assert generator.verify(
        campaign_root=root, snapshot_root=snapshots, output=output
    )["status"] == "verified"
    output.unlink()
    generator.generate(campaign_root=root, snapshot_root=snapshots, output=output)
    assert output.read_bytes() == first_bytes


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "coverage is not exact"),
        ("additional", "coverage is not exact"),
        ("symlink", "symlink or non-file"),
        ("wrong_partition", "partition differs"),
        ("failed", "state differs"),
        ("wrong_job", "invalid for fixture_case/123"),
    ],
)
def test_generation_rejects_path_coverage_identity_and_resource_gaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    root, snapshots, output = _generic_fixture(tmp_path, monkeypatch)
    raw = snapshots / "123.sacct"
    if mutation == "missing":
        raw.unlink()
    elif mutation == "additional":
        (snapshots / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    elif mutation == "symlink":
        raw.rename(snapshots / "source.txt")
        raw.symlink_to(snapshots / "source.txt")
    elif mutation == "wrong_partition":
        raw.write_text(
            _raw(
                case_id="fixture_case", job_id="123", partition="qcpu",
                nodes=1, ranks=8,
            ),
            encoding="utf-8",
        )
    elif mutation == "failed":
        raw.write_text(
            _raw(
                case_id="fixture_case", job_id="123", partition="qcpu_exp",
                nodes=1, ranks=8, state="FAILED", exit_code="1:0",
            ),
            encoding="utf-8",
        )
    else:
        raw.write_text(
            _raw(
                case_id="fixture_case", job_id="999", partition="qcpu_exp",
                nodes=1, ranks=8,
            ),
            encoding="utf-8",
        )
    with pytest.raises(generator.IndexGenerationError, match=message):
        generator.generate(
            campaign_root=root, snapshot_root=snapshots, output=output
        )


def test_verification_rejects_changed_snapshot_manifest_and_index_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, snapshots, output = _generic_fixture(tmp_path, monkeypatch)
    generator.generate(campaign_root=root, snapshot_root=snapshots, output=output)
    raw = snapshots / "123.sacct"
    raw.write_text(raw.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(generator.IndexGenerationError, match="differs from"):
        generator.verify(campaign_root=root, snapshot_root=snapshots, output=output)

    with pytest.raises(generator.IndexGenerationError, match="directly in snapshot root"):
        generator.build_payload(
            campaign_root=root,
            snapshot_root=snapshots,
            output=tmp_path / "outside.json",
        )


def test_legacy_prepared_campaign_dispatch_uses_same_index_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = legacy_preparer.read_matrix(legacy_preparer.DEFAULT_MATRIX)[0]
    root = tmp_path / "legacy-campaign"
    root.mkdir()
    plan = root / "prepared_plan.csv"
    with plan.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (root / "prepared_manifest.json").write_text(
        json.dumps({"status": "submitted", "plan_file": plan.name}) + "\n",
        encoding="utf-8",
    )
    command = "sbatch fixture"
    (root / "submitted_jobs.jsonl").write_text(
        json.dumps(
            {
                "case_id": row["case_id"], "returncode": 0,
                "stdout": "Submitted batch job 456", "job_id": "456",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "submission_journal.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "event": event, "attempt_id": "fixture-attempt",
                        "case_id": row["case_id"], "command": command,
                        "recorded_at_utc": (
                            "2026-07-10T10:00:00+00:00"
                            if event == "intent"
                            else "2026-07-10T10:00:01+00:00"
                        ),
                        **(
                            {"returncode": 0, "stdout": "Submitted batch job 456", "job_id": "456"}
                            if event == "result" else {}
                    ),
                }
            )
            for event in ("intent", "result")
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        generator.legacy_finalizer.campaign,
        "offline_preflight",
        lambda *args, **kwargs: {"status": "passed"},
    )
    snapshots = tmp_path / "legacy-snapshots"
    snapshots.mkdir()
    (snapshots / "456.sacct").write_text(
        _raw(
            case_id=row["case_id"], job_id="456", partition=row["partition"],
            nodes=int(row["nodes"]), ranks=int(row["total_ranks"]),
        ),
        encoding="utf-8",
    )
    output = snapshots / "index.json"
    payload = generator.generate(
        campaign_root=root, snapshot_root=snapshots, output=output
    )
    assert payload["records"] == [
        {
            "case_id": row["case_id"],
            "job_id": "456",
            "path": "456.sacct",
            "sha256": reviewed.sha256_file(snapshots / "456.sacct"),
        }
    ]
    assert (
        legacy_preparer.REVIEWED_SOURCES["offline_accounting_index_generator"]
        == reviewed.REPO_ROOT
        / "experiments/analysis/generate_offline_accounting_index.py"
    )
