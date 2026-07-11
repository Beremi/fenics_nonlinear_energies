from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from experiments.analysis import finalize_revision_publication_campaign as finalizer
from experiments.analysis import stage_route_publication_dependencies as staging


COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _stopping_binding(path: Path, *, declared_path: str | None = None) -> dict[str, object]:
    return {
        "schema_id": "fenics-nonlinear-energies.exp-stop-001.final-adjudication",
        "schema_version": 3,
        "path": declared_path if declared_path is not None else str(path.resolve()),
        "sha256": staging._sha256(path),
        "computation_source_commit": "1" * 40,
        "adjudicator_source_commit": "2" * 40,
        "adjudicator_sha256": "3" * 64,
        "local_analysis_sha256": "4" * 64,
        "cluster_archive_checksum_sha256": "5" * 64,
        "p4_reference_row_id": "p3d_p4_nonlinear_1em07_cluster",
        "p4_reference_status": "accepted",
    }


def _closure_fixture(root: Path) -> None:
    _write(root / "environment.json", "{}\n")
    _write(root / "nested/result.txt", "result\n")
    files = {
        path.relative_to(root).as_posix(): staging._sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "output_hash_closure": {
            "algorithm": "sha256",
            "scope": "all_regular_files_below_output_root_except_manifest",
            "excluded_paths": ["workstation_manifest.json"],
            "file_count": len(files),
            "files": files,
            "files_map_sha256": staging._inventory_fingerprint(files),
        }
    }
    _write(root / "workstation_manifest.json", json.dumps(manifest) + "\n")


def test_workstation_recursive_closure_rejects_tampering_and_links(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workstation"
    _closure_fixture(root)
    inventory = staging._validate_workstation_hash_closure(root)
    assert set(inventory) == {
        "environment.json",
        "nested/result.txt",
        "workstation_manifest.json",
    }

    _write(root / "nested/result.txt", "tampered\n")
    with pytest.raises(staging.RouteDependencyError, match="closure is stale"):
        staging._validate_workstation_hash_closure(root)

    link_root = tmp_path / "linked"
    link_root.mkdir()
    (link_root / "outside").symlink_to(root / "environment.json")
    with pytest.raises(staging.RouteDependencyError, match="symbolic link"):
        staging._tree_inventory(link_root, label="test archive")


def test_tree_staging_replaces_only_precreated_empty_skeleton(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write(source / "a.txt", "a\n")
    _write(source / "nested/b.txt", "b\n")
    inventory = staging._tree_inventory(source, label="source")
    fingerprint = staging._inventory_fingerprint(inventory)

    destination = tmp_path / "staging/archive"
    (destination / "nested/deeper").mkdir(parents=True)
    staged = staging._stage_tree(
        source,
        destination,
        expected_inventory_sha256=fingerprint,
        label="source",
    )
    assert staged == inventory
    assert (destination / "nested/b.txt").read_text(encoding="utf-8") == "b\n"

    occupied = tmp_path / "occupied"
    _write(occupied / "keep.txt", "do not replace\n")
    with pytest.raises(staging.RouteDependencyError, match="refusing to replace"):
        staging._stage_tree(
            source,
            occupied,
            expected_inventory_sha256=fingerprint,
            label="source",
        )
    assert (occupied / "keep.txt").read_text(encoding="utf-8") == "do not replace\n"


def test_tree_staging_rejects_source_change_bound_by_plan(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write(source / "result.json", "{}\n")
    fingerprint = staging._inventory_fingerprint(
        staging._tree_inventory(source, label="source")
    )
    _write(source / "result.json", '{"changed": true}\n')
    destination = tmp_path / "destination"
    with pytest.raises(staging.RouteDependencyError, match="changed after dependency-plan"):
        staging._stage_tree(
            source,
            destination,
            expected_inventory_sha256=fingerprint,
            label="source",
        )
    assert not destination.exists()


def test_endpoint_gate_cross_checks_actual_stop_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    karolina = tmp_path / "karolina"
    workstation = tmp_path / "workstation"
    workstation.mkdir()
    endpoint = karolina / "analysis/endpoint.json"
    stopping = karolina / "evidence/stopping.json"
    _write(endpoint, '{"schema": {"version": 2}}\n')
    _write(stopping, '{"schema_version": 3}\n')
    validated = _stopping_binding(stopping)
    gate_binding = dict(validated)
    gate_binding["sha256"] = "9" * 64
    monkeypatch.setattr(
        staging.route_analysis,
        "_endpoint_analysis_gate",
        lambda *_args, **_kwargs: {
            "publication_admissible": True,
            "stopping_binding_matches_manifest": True,
            "stopping_adjudication": gate_binding,
        },
    )
    monkeypatch.setattr(
        staging,
        "validate_stop_adjudication",
        lambda _path: validated,
    )
    with pytest.raises(staging.RouteDependencyError, match="identities differ"):
        staging._require_endpoint_gate(
            endpoint,
            stopping,
            sources=[
                ("workstation_local", workstation),
                ("karolina_cpu", karolina),
            ],
            contract={},
        )


def test_endpoint_declared_stop_must_be_an_actual_confined_json(tmp_path: Path) -> None:
    karolina = tmp_path / "karolina"
    endpoint = karolina / "analysis/endpoint.json"
    stopping = karolina / "evidence/stopping.json"
    _write(stopping, '{"schema_version": 3}\n')
    _write(
        endpoint,
        json.dumps(
            {
                "schema": {"version": 2},
                "stopping_adjudication": {"path": "evidence/stopping.json"},
            }
        )
        + "\n",
    )
    actual, relative = staging._declared_stopping_adjudication(
        endpoint,
        karolina_root=karolina,
    )
    assert actual == stopping.resolve()
    assert relative == Path("evidence/stopping.json")

    stopping.unlink()
    with pytest.raises(staging.RouteDependencyError, match="not a regular file"):
        staging._declared_stopping_adjudication(
            endpoint,
            karolina_root=karolina,
        )


def test_stop_staging_is_independently_hash_bound_and_rejects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging_root = tmp_path / "_publication_staging"
    workstation = staging_root / staging.WORKSTATION_TARGET
    karolina = staging_root / staging.KAROLINA_TARGET
    workstation.mkdir(parents=True)
    endpoint_relative = Path("analysis/endpoint.json")
    stopping_relative = Path("evidence/stopping.json")
    _write(karolina / endpoint_relative, '{"schema": {"version": 2}}\n')
    _write(karolina / stopping_relative, '{"schema_version": 3}\n')
    expected = staging._sha256(karolina / stopping_relative)
    binding = _stopping_binding(karolina / stopping_relative)
    validation = {
        "publication_admissible": True,
        "endpoint": {
            "stopping_binding_matches_manifest": True,
            "stopping_adjudication": {
                **staging._stopping_identity(binding),
                "path": stopping_relative.as_posix(),
            },
        },
    }
    monkeypatch.setattr(
        staging,
        "validate_complete_route_evidence",
        lambda **_kwargs: validation,
    )
    monkeypatch.setattr(
        staging,
        "_require_endpoint_gate",
        lambda *_args, **_kwargs: (
            {
                "publication_admissible": True,
                "stopping_binding_matches_manifest": True,
                "stopping_adjudication": binding,
            },
            binding,
        ),
    )
    destination = staging_root / staging.CANONICAL_STOPPING_ADJUDICATION
    result = staging.stage_stopping_adjudication(
        workstation_root=workstation,
        karolina_root=karolina,
        endpoint_relative=endpoint_relative,
        stopping_relative=stopping_relative,
        destination=destination,
        contract_path=staging.REPO_ROOT / staging.DEFAULT_CONTRACT,
        expected_commit=COMMIT,
        expected_sha256=expected,
    )
    assert result["schema_version"] == 3
    assert result["sha256"] == expected
    assert destination.read_bytes() == (karolina / stopping_relative).read_bytes()

    destination.unlink()
    _write(karolina / stopping_relative, '{"schema_version": 3, "tampered": true}\n')
    with pytest.raises(staging.RouteDependencyError, match="changed after dependency-plan"):
        staging.stage_stopping_adjudication(
            workstation_root=workstation,
            karolina_root=karolina,
            endpoint_relative=endpoint_relative,
            stopping_relative=stopping_relative,
            destination=destination,
            contract_path=staging.REPO_ROOT / staging.DEFAULT_CONTRACT,
            expected_commit=COMMIT,
            expected_sha256=expected,
        )
    assert not destination.exists()


def test_dependency_plan_binds_every_file_and_matches_finalizer_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workstation = tmp_path / "workstation"
    karolina = tmp_path / "karolina"
    _write(workstation / "workstation_manifest.json", "{}\n")
    _write(workstation / "cases/result.bin", "workstation\n")
    _write(karolina / "route_campaign_master_manifest.json", "{}\n")
    endpoint_relative = Path("analysis/tier_b_endpoint_analysis.json")
    _write(
        karolina / endpoint_relative,
        json.dumps(
            {
                "endpoint": True,
                "stopping_adjudication": {
                    "path": "/obsolete/cluster/archive/stopping_adjudication.json"
                },
            }
        )
        + "\n",
    )
    stopping_relative = Path("evidence/final_stopping_adjudication.json")
    stopping_path = karolina / stopping_relative
    _write(stopping_path, '{"schema_version": 3}\n')
    stopping_binding = _stopping_binding(
        stopping_path,
        declared_path=stopping_relative.as_posix(),
    )
    def validate_with_explicit_relocated_stop(**kwargs):
        assert Path(kwargs["stopping_adjudication_path"]).resolve() == (
            stopping_path.resolve()
        )
        return {
            "experiment_id": "EXP-ROUTE-001",
            "source_commit": COMMIT,
            "publication_admissible": True,
            "endpoint": {
                "stopping_binding_matches_manifest": True,
                "stopping_adjudication": stopping_binding,
            },
        }

    monkeypatch.setattr(
        staging,
        "validate_complete_route_evidence",
        validate_with_explicit_relocated_stop,
    )
    monkeypatch.setattr(
        staging,
        "validate_stop_adjudication",
        lambda path: _stopping_binding(Path(path)),
    )

    plan = staging.build_dependency_plan(
        expected_commit=COMMIT,
        workstation_source=workstation,
        karolina_source=karolina,
        endpoint_relative=endpoint_relative,
        stopping_relative=stopping_relative,
    )
    commands = finalizer._plan_command_map(plan)
    assert plan["plan_kind"] == "dependency_preparation"
    assert plan["execution_order"] == [
        "prepare_workstation_archive",
        "prepare_route_campaign_master",
        "prepare_route_stopping_adjudication",
        "prepare_tier_b_endpoint_analysis",
    ]
    workstation_expected = set(commands["prepare_workstation_archive"]["expected_artifacts"])
    assert workstation_expected == {
        (staging.WORKSTATION_TARGET / "workstation_manifest.json").as_posix(),
        (staging.WORKSTATION_TARGET / "cases/result.bin").as_posix(),
    }
    karolina_inputs = commands["prepare_route_campaign_master"]["input_files"]
    assert {row["path"] for row in karolina_inputs} == workstation_expected
    assert {
        row["attestation"]["path"] for row in karolina_inputs
    } == {"_publication_receipts/prepare_workstation_archive.json"}
    endpoint_command = commands["prepare_tier_b_endpoint_analysis"]
    assert endpoint_command["expected_artifacts"] == [
        staging.CANONICAL_ENDPOINT.as_posix()
    ]
    assert endpoint_command["input_files"] == [
        {
            "scope": "staging",
            "path": (staging.KAROLINA_TARGET / endpoint_relative).as_posix(),
            "attestation": {
                "path": "_publication_receipts/prepare_route_campaign_master.json"
            },
        },
        {
            "scope": "staging",
            "path": staging.CANONICAL_STOPPING_ADJUDICATION.as_posix(),
            "attestation": {
                "path": "_publication_receipts/prepare_route_stopping_adjudication.json"
            },
        },
    ]
    stopping_command = commands["prepare_route_stopping_adjudication"]
    assert stopping_command["expected_artifacts"] == [
        staging.CANONICAL_STOPPING_ADJUDICATION.as_posix()
    ]
    assert stopping_command["input_files"] == [
        {
            "scope": "staging",
            "path": (staging.KAROLINA_TARGET / stopping_relative).as_posix(),
            "attestation": {
                "path": "_publication_receipts/prepare_route_campaign_master.json"
            },
        }
    ]
    required_validators = {
        "paper/protocols/EXP-ROUTE-001-tier-b-stopping-policy.json",
        "experiments/runners/paper_revision_karolina/tier_b_stopping.py",
        "experiments/runners/prepare_exp_stop_001_karolina.py",
    }
    assert all(
        required_validators.issubset(set(command["configuration_files"]))
        for command in commands.values()
    )
    assert plan["tier_b_stopping_adjudication"] == {
        "source_relative": stopping_relative.as_posix(),
        "source_sha256": staging._sha256(stopping_path),
        "canonical_target": staging.CANONICAL_STOPPING_ADJUDICATION.as_posix(),
        "identity": staging._stopping_identity(stopping_binding),
    }
    assert plan["source_archives"]["workstation"]["files"] == {
        "cases/result.bin": hashlib.sha256(b"workstation\n").hexdigest(),
        "workstation_manifest.json": hashlib.sha256(b"{}\n").hexdigest(),
    }
    assert plan["source_archives"]["karolina"]["files"][
        stopping_relative.as_posix()
    ] == staging._sha256(stopping_path)

    plan_path = tmp_path / "canonical_dependency_plan.json"
    _write(plan_path, json.dumps(plan, indent=2) + "\n")
    finalizer._validate_route_dependency_plan_contract(
        plan_path,
        experiment_commit=COMMIT,
    )

    substituted = json.loads(json.dumps(plan))
    substituted["commands"][0]["producer"] = (
        "experiments/analysis/finalize_revision_publication_campaign.py"
    )
    _write(plan_path, json.dumps(substituted, indent=2) + "\n")
    with pytest.raises(finalizer.FinalizationError, match="canonical staging contract"):
        finalizer._validate_route_dependency_plan_contract(
            plan_path,
            experiment_commit=COMMIT,
        )

    substituted = json.loads(json.dumps(plan))
    substituted["commands"][2]["argv"][2] = "stage-endpoint"
    _write(plan_path, json.dumps(substituted, indent=2) + "\n")
    with pytest.raises(finalizer.FinalizationError, match="canonical staging contract"):
        finalizer._validate_route_dependency_plan_contract(
            plan_path,
            experiment_commit=COMMIT,
        )


def test_plan_writer_requires_the_exact_clean_experiment_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "route_dependency_plan.json"
    monkeypatch.setattr(finalizer, "_require_clean_head", lambda _root: "f" * 40)
    with pytest.raises(staging.RouteDependencyError, match="exact clean experiment"):
        staging.write_dependency_plan(
            output=output,
            expected_commit=COMMIT,
            workstation_source=tmp_path / "missing-workstation",
            karolina_source=tmp_path / "missing-karolina",
            endpoint_relative=Path("analysis/endpoint.json"),
            stopping_relative=Path("analysis/stopping.json"),
            contract_path=staging.REPO_ROOT / staging.DEFAULT_CONTRACT,
        )
    assert not output.exists()


def test_canonical_source_plan_passes_endpoint_inside_karolina_archive() -> None:
    plan = finalizer.build_execution_plan_template(experiment_commit=COMMIT)
    command = finalizer._plan_command_map(plan)["route_cost_analysis"]
    endpoint = staging.CANONICAL_ENDPOINT.as_posix()
    stopping = staging.CANONICAL_STOPPING_ADJUDICATION.as_posix()
    endpoint_index = command["argv"].index("--endpoint-analysis")
    assert command["argv"][endpoint_index + 1] == f"{{staging_root}}/{endpoint}"
    assert command["route_endpoint_analysis"] == endpoint
    stopping_index = command["argv"].index("--stopping-adjudication")
    assert command["argv"][stopping_index + 1] == f"{{staging_root}}/{stopping}"
    assert command["route_stopping_adjudication"] == stopping
    assert (
        "EXP-ROUTE-001/analysis_contract_v1/endpoint_analysis.json"
        in command["expected_artifacts"]
    )
    endpoint_input = next(
        row for row in command["input_files"] if row["path"] == endpoint
    )
    assert endpoint_input["attestation"]["path"].endswith(
        "prepare_tier_b_endpoint_analysis.json"
    )
    stopping_input = next(
        row for row in command["input_files"] if row["path"] == stopping
    )
    assert stopping_input["attestation"]["path"].endswith(
        "prepare_route_stopping_adjudication.json"
    )


def test_finalizer_revalidates_dependency_receipt_plan_and_full_output_closure(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    evidence = tmp_path / "evidence"
    producer_relative = Path("prepare_dependency.py")
    producer = repo / producer_relative
    configuration_relative = Path("protocol.json")
    configuration = repo / configuration_relative
    _write(
        producer,
        "from pathlib import Path\n"
        "import sys\n"
        "path = Path(sys.argv[1])\n"
        "path.parent.mkdir(parents=True, exist_ok=True)\n"
        "path.write_text('{}\\n', encoding='utf-8')\n",
    )
    _write(configuration, "{}\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test Author"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "fixture"],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    output_relative = Path("EXP-ROUTE-001/source_archives/workstation/result.json")
    output = evidence / finalizer.STAGING_DIRECTORY / output_relative
    command = {
        "id": "prepare_workstation_archive",
        "source_keys": [],
        "role": "preparation",
        "producer": producer_relative.as_posix(),
        "argv": [
            "{python}",
            f"{{repo_root}}/{producer_relative.as_posix()}",
            f"{{staging_root}}/{output_relative.as_posix()}",
        ],
        "environment": {},
        "configuration_files": [configuration_relative.as_posix()],
        "input_files": [],
        "expected_artifacts": [output_relative.as_posix()],
    }
    plan = {
        "schema_id": finalizer.PLAN_SCHEMA_ID,
        "schema_version": finalizer.PLAN_SCHEMA_VERSION,
        "campaign_id": staging.DEPENDENCY_CAMPAIGN_ID,
        "plan_kind": "dependency_preparation",
        "experiment_commit": commit,
        "commands": [command],
    }
    plan_path = evidence / "route_dependency_plan.json"
    _write(plan_path, json.dumps(plan) + "\n")
    receipt_path = finalizer.execute_plan_command(
        plan_path=plan_path,
        command_id=command["id"],
        evidence_root=evidence,
        repo_root=repo,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    consumer = {
        "configuration_files": [],
        "input_files": [
            {
                "scope": "staging",
                "path": output_relative.as_posix(),
                "attestation": {
                    "path": receipt_path.relative_to(evidence).as_posix()
                },
            }
        ],
    }
    _configuration_hashes, integrated_inputs = finalizer._input_hashes(
        consumer,
        repo_root=repo,
        evidence_root=evidence,
        staging_root=evidence / finalizer.STAGING_DIRECTORY,
        experiment_commit=commit,
    )
    assert integrated_inputs[
        (Path(finalizer.STAGING_DIRECTORY) / output_relative).as_posix()
    ] == finalizer.sha256_file(output)

    receipt["campaign_id"] = "tampered"
    _write(receipt_path, json.dumps(receipt) + "\n")
    with pytest.raises(finalizer.FinalizationError, match="fingerprint mismatch"):
        finalizer._input_hashes(
            consumer,
            repo_root=repo,
            evidence_root=evidence,
            staging_root=evidence / finalizer.STAGING_DIRECTORY,
            experiment_commit=commit,
        )

    receipt["campaign_id"] = staging.DEPENDENCY_CAMPAIGN_ID
    receipt["command"]["return_code"] = 1
    receipt["receipt_fingerprint_sha256"] = finalizer._json_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_fingerprint_sha256"
        }
    )
    _write(receipt_path, json.dumps(receipt) + "\n")
    with pytest.raises(finalizer.FinalizationError, match="differs from its plan or failed"):
        finalizer._input_hashes(
            consumer,
            repo_root=repo,
            evidence_root=evidence,
            staging_root=evidence / finalizer.STAGING_DIRECTORY,
            experiment_commit=commit,
        )

    receipt["command"]["return_code"] = 0
    external_plan = tmp_path / "outside-evidence-plan.json"
    external_plan.write_bytes(plan_path.read_bytes())
    receipt["plan"] = {
        "path": str(external_plan.resolve()),
        "sha256": finalizer.sha256_file(external_plan),
    }
    receipt["receipt_fingerprint_sha256"] = finalizer._json_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_fingerprint_sha256"
        }
    )
    _write(receipt_path, json.dumps(receipt) + "\n")
    with pytest.raises(finalizer.FinalizationError, match="plan escapes the evidence root"):
        finalizer._input_hashes(
            consumer,
            repo_root=repo,
            evidence_root=evidence,
            staging_root=evidence / finalizer.STAGING_DIRECTORY,
            experiment_commit=commit,
        )

    receipt["plan"] = {
        "path": str(plan_path.resolve()),
        "sha256": finalizer.sha256_file(plan_path),
    }
    receipt["receipt_fingerprint_sha256"] = finalizer._json_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_fingerprint_sha256"
        }
    )
    _write(receipt_path, json.dumps(receipt) + "\n")
    _write(output, '{"tampered": true}\n')
    with pytest.raises(finalizer.FinalizationError, match="output closure is stale"):
        finalizer._input_hashes(
            consumer,
            repo_root=repo,
            evidence_root=evidence,
            staging_root=evidence / finalizer.STAGING_DIRECTORY,
            experiment_commit=commit,
        )


def test_endpoint_staging_is_hash_and_semantic_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging_root = tmp_path / "_publication_staging"
    workstation = staging_root / staging.WORKSTATION_TARGET
    karolina = staging_root / staging.KAROLINA_TARGET
    workstation.mkdir(parents=True)
    endpoint_relative = Path("analysis/endpoint.json")
    _write(karolina / endpoint_relative, '{"endpoint": true}\n')
    stopping = karolina / "reviewed_inputs/stopping_adjudication.json"
    _write(stopping, '{"schema_version": 3}\n')
    expected = staging._sha256(karolina / endpoint_relative)
    monkeypatch.setattr(
        staging,
        "validate_complete_route_evidence",
        lambda **_kwargs: {"publication_admissible": True},
    )
    stopping_binding = _stopping_binding(stopping)
    monkeypatch.setattr(
        staging,
        "_require_endpoint_gate",
        lambda *_args, **_kwargs: (
            {
                "publication_admissible": True,
                "stopping_binding_matches_manifest": True,
                "stopping_adjudication": stopping_binding,
            },
            stopping_binding,
        ),
    )

    destination = staging_root / staging.CANONICAL_ENDPOINT
    result = staging.stage_endpoint(
        workstation_root=workstation,
        karolina_root=karolina,
        endpoint_relative=endpoint_relative,
        stopping_adjudication=stopping,
        destination=destination,
        contract_path=staging.REPO_ROOT / staging.DEFAULT_CONTRACT,
        expected_commit=COMMIT,
        expected_sha256=expected,
    )
    assert result["sha256"] == expected
    assert destination.read_bytes() == (karolina / endpoint_relative).read_bytes()

    destination.unlink()
    with pytest.raises(staging.RouteDependencyError, match="changed after dependency-plan"):
        staging.stage_endpoint(
            workstation_root=workstation,
            karolina_root=karolina,
            endpoint_relative=endpoint_relative,
            stopping_adjudication=stopping,
            destination=destination,
            contract_path=staging.REPO_ROOT / staging.DEFAULT_CONTRACT,
            expected_commit=COMMIT,
            expected_sha256="0" * 64,
        )
    assert not destination.exists()


def test_staging_module_contains_no_execution_or_remote_transport_api() -> None:
    source = Path(staging.__file__).read_text(encoding="utf-8")
    assert "import subprocess" not in source
    assert "paramiko" not in source
    assert "requests." not in source
    assert "os.system" not in source
