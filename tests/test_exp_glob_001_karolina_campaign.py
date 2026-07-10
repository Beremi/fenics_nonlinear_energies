from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.analysis.finalize_karolina_campaign_archive import write_archive_checksums
from experiments.runners import karolina_reviewed_campaign as reviewed
from experiments.runners import prepare_exp_glob_001_karolina as cluster
from experiments.runners import run_globalization_method_compare as glob
from experiments.runners import submit_reviewed_karolina_campaign as submitter


COMMIT = "b" * 40


def _fake_starts(cases: list[glob.CaseSpec], root: Path) -> dict[str, dict[str, object]]:
    identities: dict[str, dict[str, object]] = {}
    for index, key in enumerate(sorted({case.start_key for case in cases}), 1):
        path = root / "_canonical_starts" / f"start_{index}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"canonical-start-{key}".encode())
        identities[key] = {
            "path": str(path.resolve()),
            "file_sha256": reviewed.sha256_file(path),
            "state_sha256": hashlib.sha256(f"state-{key}".encode()).hexdigest(),
            "problem": key.split("_", 1)[0],
            "benchmark": key.split("::", 1)[0],
            "robustness_instance": key.split("::", 1)[1],
        }
    return identities


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(reviewed, "git_metadata", lambda: {"commit": COMMIT, "dirty": False})
    monkeypatch.setattr(glob, "prepare_controlled_starts", _fake_starts)
    root = tmp_path / "campaign"
    cluster.prepare(
        SimpleNamespace(
            output_root=root,
            env_setup=None,
            env_lock=None,
        )
    )
    return root


def _mark_submitted_with_jobs(root: Path) -> None:
    manifest = reviewed.read_object(root / "prepared_manifest.json")
    plan = reviewed.read_object(root / manifest["plan"]["path"])
    manifest["status"] = "submitted"
    manifest["scheduler_contact"] = True
    manifest["accepted_jobs"] = 60
    reviewed.atomic_json(root / "prepared_manifest.json", manifest)
    ledger: list[str] = []
    journal: list[str] = []
    for index, case in enumerate(plan["cases"], 1001):
        job_id = str(index)
        command = f"sbatch fixture {case['case_id']}"
        ledger.append(json.dumps({
            "case_id": case["case_id"], "job_id": job_id, "returncode": 0,
            "command": command, "stdout": f"Submitted batch job {job_id}", "stderr": "",
        }))
        attempt = f"fixture-{case['case_id']}"
        journal.extend(
            json.dumps({
                "event": event, "attempt_id": attempt, "case_id": case["case_id"],
                "command": command,
            })
            for event in ("intent", "result")
        )
        job_root = root / "jobs" / case["case_id"] / f"job_{job_id}"
        job_root.mkdir(parents=True)
        (job_root / "output.json").write_text("{}\n", encoding="utf-8")
        (job_root / "final_state.npz").write_bytes(b"final-state")
        (job_root / "stdout.log").write_text("fixture\n", encoding="utf-8")
        (job_root / "stderr.log").write_text("", encoding="utf-8")
        reviewed.atomic_json(job_root / "environment.json", {"node": "fixture"})
        reviewed.atomic_json(job_root / "job_metadata.json", {
            "case_id": case["case_id"], "job_id": job_id,
            "payload_argv": case["payload_argv"],
        })
        reviewed.atomic_json(job_root / "execution.json", {
            "case_id": case["case_id"], "job_id": job_id, "returncode": 0,
            "wall_time_s": 1.0,
            "started_at_utc": "2026-07-10T10:00:00+00:00",
            "finished_at_utc": "2026-07-10T10:00:01+00:00",
        })
    (root / "submitted_jobs.jsonl").write_text("\n".join(ledger) + "\n", encoding="utf-8")
    (root / "submission_journal.jsonl").write_text("\n".join(journal) + "\n", encoding="utf-8")


def test_preparation_freezes_sixty_full_rank_launches_and_six_common_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _prepare(tmp_path, monkeypatch)
    receipt = cluster.preflight(root)
    manifest, plan = reviewed.load_plan(root)

    assert receipt["status"] == "passed_without_scheduler_contact"
    assert receipt["submission_admissible"] is False
    assert receipt["gl_launches_16_ranks"] == 30
    assert receipt["he_launches_32_ranks"] == 30
    assert receipt["canonical_start_count"] == 6
    assert receipt["node_hour_ceiling"] == 12.5
    assert manifest["estimated_node_hours_ceiling"] == 12.5
    assert len(plan["cases"]) == 60
    assert {case["total_ranks"] for case in plan["cases"] if case["family"] == "gl"} == {16}
    assert {case["total_ranks"] for case in plan["cases"] if case["family"] == "he"} == {32}
    assert all("{CAMPAIGN_ROOT}/bound_inputs/" in " ".join(case["payload_argv"]) for case in plan["cases"])
    assert all(case["scientific_contract"]["controlled_environment"] == cluster.CONTROLLED_ENVIRONMENT for case in plan["cases"])
    assert submitter.submit(root, execute=False, confirmed=False)["case_count"] == 60


def test_case_pairs_share_start_identity_but_not_output_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _prepare(tmp_path, monkeypatch)
    _manifest, plan = reviewed.load_plan(root)
    groups: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for case in plan["cases"]:
        science = case["scientific_contract"]
        key = (
            str(science["benchmark"]),
            str(science["robustness_instance"]),
            int(science["timing_repetition"]),
        )
        groups.setdefault(key, []).append(case)
    assert len(groups) == 30
    for pair in groups.values():
        assert len(pair) == 2
        assert {case["scientific_contract"]["method"] for case in pair} == {
            "newton_armijo", "reduced_trust_armijo"
        }
        assert len({case["scientific_contract"]["common_start_file_sha256"] for case in pair}) == 1
        assert len({case["case_id"] for case in pair}) == 2


def test_preseal_analysis_writes_sixty_records_and_release_requires_sealed_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _prepare(tmp_path, monkeypatch)
    _mark_submitted_with_jobs(root)

    def summary(**kwargs: object) -> dict[str, object]:
        case = kwargs["case"]
        assert isinstance(case, glob.CaseSpec)
        row = {field: "" for field in glob.CSV_FIELDS}
        row.update(
            {
                "benchmark": case.benchmark.key,
                "robustness_instance": case.robustness_instance.key,
                "timing_repetition": case.timing_repetition,
                "method": case.method.key,
                "result": "completed",
                "wall_time_s": 1.0,
            }
        )
        return row

    monkeypatch.setattr(glob, "summarize_payload", summary)
    monkeypatch.setattr(
        glob,
        "build_publication_run_record",
        lambda **kwargs: {
            "identifiers": {"route": "fixture"},
            "resources": {},
            "artifacts": {},
        },
    )
    monkeypatch.setattr(cluster, "validate_run_record", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        cluster,
        "atomic_write_run_record",
        lambda path, record, **kwargs: Path(path).write_text(json.dumps(record) + "\n", encoding="utf-8"),
    )
    audit = {
        "status": "passed",
        "timing_claim_admissible": True,
        "tested_instance_comparison_admissible": True,
        "robustness_generalization_claim_admissible": False,
    }
    monkeypatch.setattr(glob, "controlled_identity_audit", lambda *args, **kwargs: audit)

    analysis = cluster.analyze(root)
    assert analysis["case_count"] == 60
    assert len(analysis["run_records"]) == 60
    assert analysis["claim_admission"]["timing_claim_admissible"] is False
    assert all((root / record["path"]).is_file() for record in analysis["run_records"])

    checksum = write_archive_checksums(root)
    monkeypatch.setattr(
        cluster.archive,
        "verify_settled_archive",
        lambda _root: {"status": "settled_archive_verified", "job_count": 60},
    )
    final = cluster.adjudicate(root, expected_checksum=checksum["sha256"])
    assert final["settled_job_count"] == 60
    assert final["timing_claim_admissible"] is True
    assert final["tested_instance_comparison_admissible"] is True
    assert final["robustness_generalization_claim_admissible"] is False

    (root / "analysis" / "analysis.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or changed"):
        cluster.adjudicate(root, expected_checksum=checksum["sha256"])

