from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.runners.paper_revision_karolina import tier_b_stopping


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = REPO_ROOT / "experiments/runners/paper_revision_karolina"
MATRIX = CAMPAIGN_DIR / "campaign_matrix.csv"
SUBMITTER = CAMPAIGN_DIR / "submit_prepared_campaigns.sh"
EXECUTOR = CAMPAIGN_DIR / "execute_case.py"
PREPARER = CAMPAIGN_DIR / "prepare_campaign.py"
BATCH_RUNNER = CAMPAIGN_DIR / "run_revision_case.sbatch"
RELEASE_AUTHORIZATION_SCHEMA = (
    REPO_ROOT / "paper/protocols/human-release-authorization-v1.schema.json"
)
RELEASE_AUTHORIZATION_EXAMPLE = (
    REPO_ROOT / "paper/protocols/human-release-authorization-v1.example.json"
)


def _rows() -> list[dict[str, str]]:
    with MATRIX.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_executor():
    spec = importlib.util.spec_from_file_location("paper_revision_karolina_executor", EXECUTOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_preparer():
    spec = importlib.util.spec_from_file_location("paper_revision_karolina_preparer", PREPARER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _current_reviewed_hashes(module) -> dict[str, str]:
    return {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in module.REVIEWED_SOURCES.items()
    }


def test_matrix_freezes_required_and_optional_resource_ceiling() -> None:
    rows = _rows()
    assert len(rows) == 148
    assert len({row["case_id"] for row in rows}) == len(rows)
    required = [row for row in rows if row["optional"] == "0"]
    optional = [row for row in rows if row["optional"] == "1"]
    assert len(required) == 115
    assert len(optional) == 33
    assert sum(float(row["estimated_node_hours"]) for row in required) == pytest.approx(99.95)
    assert sum(float(row["estimated_node_hours"]) for row in optional) == 62.5
    assert sum(float(row["estimated_node_hours"]) for row in rows) == pytest.approx(162.45)

    for row in rows:
        nodes = int(row["nodes"])
        ranks_per_node = int(row["ranks_per_node"])
        assert 1 <= ranks_per_node <= 128
        assert int(row["total_ranks"]) == nodes * ranks_per_node
        assert int(row["repetitions"]) >= 1
        assert row["partition"] == ("qcpu_exp" if nodes <= 2 else "qcpu")


def test_release_authorization_archives_reviewed_artifacts_relocatably(
    tmp_path: Path,
) -> None:
    module = _load_preparer()
    source = tmp_path / "source"
    source.mkdir()
    reviewed = source / "analysis.json"
    reviewed.write_text(json.dumps({"status": "reviewed"}) + "\n", encoding="utf-8")
    reviewed_sha = hashlib.sha256(reviewed.read_bytes()).hexdigest()
    gate = source / "authorization.json"
    gate.write_text(
        json.dumps(
            {
                "schema_id": "fenics-nonlinear-energies.human-release-authorization",
                "schema_version": 1,
                "status": "approved",
                "decision": "explicit_human_release_after_review",
                "reviewer": "archive-test-reviewer",
                "reviewed_artifacts": [
                    {"path": reviewed.name, "sha256": reviewed_sha}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    archive = tmp_path / "campaign_archive"
    archive.mkdir()
    metadata = module._archive_release_authorization(
        {
            "schema_id": "fenics-nonlinear-energies.human-release-authorization",
            "path": str(gate),
            "sha256": hashlib.sha256(gate.read_bytes()).hexdigest(),
            "reviewer": "archive-test-reviewer",
        },
        out_root=archive,
    )
    relocated = tmp_path / "relocated_campaign_archive"
    archive.rename(relocated)
    archived_gate = relocated / metadata["path"]
    assert hashlib.sha256(archived_gate.read_bytes()).hexdigest() == metadata["sha256"]
    payload = json.loads(archived_gate.read_text(encoding="utf-8"))
    artifact_path = Path(payload["reviewed_artifacts"][0]["path"])
    assert artifact_path.is_absolute() is False
    relocated_artifact = relocated / artifact_path
    assert relocated_artifact.is_file()
    assert hashlib.sha256(relocated_artifact.read_bytes()).hexdigest() == reviewed_sha


def test_release_authorization_schema_and_example_are_maintained_and_strict() -> None:
    module = _load_preparer()
    schema = json.loads(RELEASE_AUTHORIZATION_SCHEMA.read_text(encoding="utf-8"))
    example = json.loads(RELEASE_AUTHORIZATION_EXAMPLE.read_text(encoding="utf-8"))
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(example)
    assert schema["properties"]["matrix_sha256"]["pattern"] == "^[0-9a-f]{64}$"
    assert schema["properties"]["source_commit"]["pattern"] == "^[0-9a-f]{40}$"
    module._validate_release_authorization_shape(example)
    assert example["matrix_sha256"] == "0" * 64
    assert example["source_commit"] == "0" * 40
    assert str(example["reviewer"]).startswith("EXAMPLE_ONLY")
    assert example["matrix_sha256"] != hashlib.sha256(MATRIX.read_bytes()).hexdigest()

    malformed = dict(example)
    malformed["unreviewed_extra"] = True
    with pytest.raises(RuntimeError, match="exactly the fields"):
        module._validate_release_authorization_shape(malformed)
    duplicated = dict(example)
    duplicated["authorizes_tiers"] = ["quadrature", "quadrature"]
    with pytest.raises(RuntimeError, match="unique nonempty tiers"):
        module._validate_release_authorization_shape(duplicated)


def test_release_authorization_binds_context_and_reviewed_artifact(
    tmp_path: Path,
) -> None:
    module = _load_preparer()
    reviewed = tmp_path / "reviewed_route_gate.json"
    reviewed.write_text('{"status":"passed"}\n', encoding="utf-8")
    source_commit = "0123456789abcdef0123456789abcdef01234567"
    gate = {
        "schema_id": "fenics-nonlinear-energies.human-release-authorization",
        "schema_version": 1,
        "status": "approved",
        "decision": "explicit_human_release_after_review",
        "matrix_sha256": hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
        "source_commit": source_commit,
        "authorizes_experiment": "EXP-ROUTE-001",
        "authorizes_tiers": ["factorized_microbenchmark"],
        "reviewer": "human-reviewer-test",
        "reviewed_artifacts": [
            {
                "path": reviewed.name,
                "sha256": hashlib.sha256(reviewed.read_bytes()).hexdigest(),
            }
        ],
    }
    gate_path = tmp_path / "authorization.json"
    gate_path.write_text(json.dumps(gate) + "\n", encoding="utf-8")
    args = SimpleNamespace(
        test_only=False,
        experiment=["EXP-ROUTE-001"],
        tier=["factorized_microbenchmark"],
        admission_gate=gate_path,
        route_phase="training",
    )
    selected = module.select_rows(
        module.read_matrix(MATRIX),
        experiments={"EXP-ROUTE-001"},
        include_optional=False,
        tiers={"factorized_microbenchmark"},
        route_phase="training",
    )
    result = module._require_staged_real_submission(
        args,
        selected=selected,
        matrix=MATRIX,
        git={"commit": source_commit, "dirty": False},
    )
    assert result is not None
    assert result["reviewer"] == "human-reviewer-test"
    gate["reviewed_artifacts"][0]["sha256"] = "0" * 64
    gate_path.write_text(json.dumps(gate) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing or has a stale hash"):
        module._require_staged_real_submission(
            args,
            selected=selected,
            matrix=MATRIX,
            git={"commit": source_commit, "dirty": False},
        )
    gate["reviewed_artifacts"][0]["sha256"] = hashlib.sha256(
        reviewed.read_bytes()
    ).hexdigest()
    gate["authorizes_tiers"] = ["nonexistent_tier"]
    gate_path.write_text(json.dumps(gate) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="tier unknown"):
        module._require_staged_real_submission(
            args,
            selected=selected,
            matrix=MATRIX,
            git={"commit": source_commit, "dirty": False},
        )


def test_route_matrix_has_exact_second_architecture_screen_and_confirmation() -> None:
    rows = [row for row in _rows() if row["experiment_id"] == "EXP-ROUTE-001"]
    screens = [row for row in rows if row["tier"] == "fixed_state_screen"]
    full = [row for row in rows if row["tier"] == "full_solve_confirmation"]
    low = [row for row in rows if row["tier"] == "low_order_confirmation"]
    quadrature = [row for row in rows if row["tier"] == "factorized_quadrature"]
    factors = [row for row in rows if row["tier"] == "factorized_microbenchmark"]
    assert len(screens) == 78
    assert {
        (
            row["mesh_name"],
            int(row["element_degree"]),
            row["quadrature_rule"],
            row["state_label"],
            int(row["total_ranks"]),
            int(row["block_repetition"]),
        )
        for row in screens
    } == {
        (mesh, degree, rule, state, ranks, block)
        for mesh, degree, rule, blocks in (
            ("hetero_ssr_L1", 1, "tetra_1point", 3),
            ("hetero_ssr_L1_2", 1, "tetra_1point", 3),
            ("hetero_ssr_L1", 2, "tetra_11point", 3),
            ("hetero_ssr_L1", 4, "tetra_24point", 4),
        )
        for state in ("elastic", "mixed")
        for ranks in (1, 8, 32)
        for block in range(1, blocks + 1)
    }
    assert all((int(row["warmups"]), int(row["repetitions"])) == (1, 5) for row in screens)
    assert all(row["runner"] == "p3d_fixed_state_block" for row in screens)
    assert all(int(row["probe_count"]) == 4 for row in screens)
    assert all(row["timing_reduction"] == "mpi_collective_max" for row in screens)
    assert len(quadrature) == 18
    assert {row["quadrature_rule"] for row in quadrature} == {
        "tetra_1point",
        "tetra_24point",
        "tetra_duffy_125point",
    }
    assert {
        (int(row["total_ranks"]), int(row["block_repetition"]), row["runner"])
        for row in factors
    } == {
        (ranks, block, "route_factor_microbench")
        for ranks in (1, 8, 32)
        for block in (1, 2, 3)
    }
    assert len(full) == 20
    assert len(low) == 10
    assert {(int(row["element_degree"]), int(row["total_ranks"])) for row in full + low} == {
        (4, 8),
        (4, 32),
        (1, 8),
    }
    assert all((int(row["warmups"]), int(row["repetitions"])) == (0, 1) for row in full + low)
    assert all(float(row["ksp_rtol"]) <= 1.0e-8 for row in full + low)
    assert all(int(row["ksp_max_it"]) == 1000 for row in full + low)
    assert all(float(row["grad_stop_tol"]) == 0.0 for row in full + low)
    assert all(float(row["stop_tol"]) == 1.0e-7 for row in full)
    assert all(float(row["stop_tol"]) == 1.0e-6 for row in low)
    assert all(
        row["convergence_metric"] == "reference_elastic_energy" for row in full + low
    )
    assert all(row["runner"] == "p3d_solve_block" for row in full + low)
    for comparison in {row["comparison_id"] for row in screens + quadrature + full + low}:
        group = [row for row in screens + quadrature + full + low if row["comparison_id"] == comparison]
        orders = [row["route_order"].split("|") for row in group]
        routes = set(orders[0])
        counts = [
            sum(order[position] == route for order in orders)
            for route in routes
            for position in range(len(routes))
        ]
        assert len(set(counts)) == 1
        base = (
            ["element_ad", "colored_sfd", "constitutive_ad"]
            if len(routes) == 3
            else ["element_ad", "constitutive_ad"]
        )
        random.Random(
            int(hashlib.sha256(comparison.encode()).hexdigest()[:16], 16)
        ).shuffle(base)
        for row in group:
            repetition = int(row["block_repetition"])
            if row["route_order_policy"] == "seeded_balanced_cyclic_v1":
                offset = (repetition - 1) % len(base)
                expected = base[offset:] + base[:offset]
            else:
                expected = base if repetition % 2 else list(reversed(base))
            assert row["route_order"].split("|") == expected


def test_discretization_matrix_separates_quadrature_mesh_and_tolerance() -> None:
    rows = [row for row in _rows() if row["experiment_id"] == "EXP-DISC-001"]
    assert [row["tier"] for row in rows] == [
        "smoke",
        "quadrature",
        "quadrature",
        "mesh",
        "mesh_quadrature",
        "tolerance",
    ]
    assert {row["quadrature_rule"] for row in rows} == {
        "tetra_24point",
        "tetra_duffy_125point",
    }
    assert {row["mesh_name"] for row in rows} == {"hetero_ssr_L1", "hetero_ssr_L1_2"}
    assert all(int(row["element_degree"]) == 4 for row in rows)
    assert max(float(row["estimated_node_hours"]) for row in rows) <= 4.0
    assert all(row["time_limit"] <= "02:00:00" for row in rows)


def test_scaling_matrix_is_fixed_policy_and_keeps_optional_p3d_separate() -> None:
    rows = [row for row in _rows() if row["experiment_id"] == "EXP-SCALE-001"]
    he = [row for row in rows if row["tier"] == "fixed_policy_he_l5"]
    p3d = [row for row in rows if row["tier"] == "optional_fixed_policy_p3d"]
    assert [int(row["nodes"]) for row in he] == [1, 2, 4, 8]
    assert all(int(row["ranks_per_node"]) == 128 for row in he)
    assert all(int(row["repetitions"]) == 5 for row in he)
    frozen_he_fields = {
        (
            row["runner"],
            row["mesh_name"],
            row["assembly_backend"],
            row["solver_backend"],
            row["pmg_strategy"],
            row["maxit"],
            row["ksp_rtol"],
            row["stop_tol"],
            row["grad_stop_tol"],
        )
        for row in he
    }
    assert len(frozen_he_fields) == 1
    assert [int(row["nodes"]) for row in p3d] == [1, 2, 4]
    assert all(row["optional"] == "1" for row in p3d)
    assert all(int(row["repetitions"]) == 5 for row in p3d)


def test_dry_run_prepares_exact_test_only_commands_without_sbatch(tmp_path: Path) -> None:
    out_root = tmp_path / "prepared"
    env = dict(os.environ)
    env.update(
        {
            "DRY_RUN": "1",
            "SBATCH_TEST_ONLY": "1",
            "OUT_ROOT": str(out_root),
        }
    )
    completed = subprocess.run(
        ["bash", str(SUBMITTER)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "no sbatch process was invoked" in completed.stdout
    manifest = json.loads((out_root / "prepared_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "prepared_not_submitted"
    assert manifest["case_count"] == 115
    assert manifest["estimated_node_hours"] == pytest.approx(99.95)
    assert manifest["matrix"] == str(MATRIX.relative_to(REPO_ROOT))
    assert manifest["out_root"] == "."
    assert manifest["offline_preflight"]["status"] == "passed"
    freeze_metadata = manifest["queued_source_freeze"]
    freeze_path = out_root / freeze_metadata["path"]
    assert freeze_path == out_root / "reviewed_source_freeze.json"
    assert freeze_metadata["sha256"] == hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    assert freeze["source_commit"] == manifest["source_commit"]
    assert freeze["matrix"]["sha256"] == manifest["matrix_sha256"]
    assert {
        key: record["sha256"] for key, record in freeze["reviewed_sources"].items()
    } == manifest["reviewed_source_sha256"]
    commands = (out_root / "sbatch_commands.txt").read_text(encoding="utf-8").splitlines()
    assert len(commands) == 115
    for line in commands:
        tokens = shlex.split(line)
        assert tokens[0] == "sbatch"
        assert "--test-only" in tokens
        assert tokens[tokens.index("--account") + 1] == "fta-26-40"
        assert tokens[tokens.index("--qos") + 1] == "3571_6328"
        assert tokens[tokens.index("--distribution") + 1] == "block:block"
        assert tokens[tokens.index("--cpus-per-task") + 1] == "1"
        assert "--exclusive" not in tokens
        assert "--mem" not in tokens
        assert "--mem-per-cpu" not in tokens
        batch_index = tokens.index(str(BATCH_RUNNER))
        batch_arguments = tokens[batch_index + 1 :]
        assert len(batch_arguments) == 11
        assert batch_arguments[0] == str(MATRIX)
        assert batch_arguments[2] == str(out_root.resolve())
        assert batch_arguments[3] == manifest["source_commit"]
        assert batch_arguments[4] == manifest["matrix_sha256"]
        assert batch_arguments[5] == str(freeze_path)
        assert batch_arguments[6] == freeze_metadata["sha256"]
        assert batch_arguments[7] == "UNBOUND"
        assert batch_arguments[8] == "0" * 64
        assert batch_arguments[9] == "UNBOUND"
        assert batch_arguments[10] == "0" * 64


def test_execute_mode_hard_stops_without_current_revalidation(tmp_path: Path) -> None:
    env = dict(os.environ)
    for key in (
        "ALLOCATION_REVALIDATED",
        "ACCOUNT_QOS_REVALIDATED",
        "ALLOCATION_VALID_UNTIL",
        "SUBMIT_CONFIRMED",
    ):
        env.pop(key, None)
    env.update({"DRY_RUN": "0", "OUT_ROOT": str(tmp_path / "blocked")})
    env_setup = tmp_path / "env.sh"
    env_lock = tmp_path / "env.lock"
    env_setup.write_text("export TEST_ENV=1\n", encoding="utf-8")
    env_lock.write_text("lock\n", encoding="utf-8")
    env.update({"ENV_SETUP": str(env_setup), "ENV_LOCK": str(env_lock)})
    completed = subprocess.run(
        ["bash", str(SUBMITTER)],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "submission disabled" in completed.stderr
    assert not (tmp_path / "blocked" / "submitted_jobs.jsonl").exists()


def test_batch_rejects_queued_commit_drift_before_creating_solver_output(
    tmp_path: Path,
) -> None:
    freeze = tmp_path / "reviewed_source_freeze.json"
    freeze.write_text("{}\n", encoding="utf-8")
    env = dict(os.environ)
    env.update(
        {
            "ALLOCATION_REVALIDATED": "YES",
            "ACCOUNT_QOS_REVALIDATED": "YES",
            "ALLOCATION_VALID_UNTIL": "2099-12-31",
            "SLURM_JOB_ACCOUNT": "fta-26-40",
            "SLURM_JOB_QOS": "3571_6328",
        }
    )
    out_root = tmp_path / "must_remain_empty"
    env_setup = tmp_path / "env.sh"
    env_lock = tmp_path / "env.lock"
    env_setup.write_text("export TEST_ENV=1\n", encoding="utf-8")
    env_lock.write_text("lock\n", encoding="utf-8")
    completed = subprocess.run(
        [
            "bash",
            str(BATCH_RUNNER),
            str(MATRIX),
            "route_factor_micro_np1_b01",
            str(out_root),
            "0" * 40,
            hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
            str(freeze),
            hashlib.sha256(freeze.read_bytes()).hexdigest(),
            str(env_setup),
            hashlib.sha256(env_setup.read_bytes()).hexdigest(),
            str(env_lock),
            hashlib.sha256(env_lock.read_bytes()).hexdigest(),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "queued source commit differs" in completed.stderr
    assert not out_root.exists()


def _prepare_args(out_root: Path, *, execute: bool) -> SimpleNamespace:
    env_setup = out_root.parent / "reviewed_env_setup.sh"
    env_lock = out_root.parent / "reviewed_env.lock"
    env_setup.write_text("export TEST_REVIEWED_ENV=1\n", encoding="utf-8")
    env_lock.write_text("synthetic-lock-v1\n", encoding="utf-8")
    return SimpleNamespace(
        matrix=MATRIX,
        out_root=out_root,
        experiment=["EXP-ROUTE-001"],
        include_optional=False,
        only_optional=False,
        tier=["factorized_microbenchmark"],
        max_node_hours=100.0,
        test_only=False,
        execute=execute,
        admission_gate=None,
        route_phase="training",
        model_freeze_receipt=None,
        env_setup=env_setup,
        env_lock=env_lock,
    )


def _tier_b_prepare_args(
    out_root: Path, *, execute: bool, stopping_adjudication: Path | None
) -> SimpleNamespace:
    args = _prepare_args(out_root, execute=execute)
    args.only_optional = True
    args.tier = ["full_solve_confirmation", "low_order_confirmation"]
    args.stopping_adjudication = stopping_adjudication
    return args


def _write_valid_stop_adjudication(path: Path) -> Path:
    local = dict(tier_b_stopping.load_policy()["local_calibration"])
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
    adjudicator = REPO_ROOT / "experiments/runners/prepare_exp_stop_001_karolina.py"
    payload = {
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
            "sha256": hashlib.sha256(adjudicator.read_bytes()).hexdigest(),
        },
        "local_analysis_sha256": local["analysis_sha256"],
        "cluster_archive_checksum_sha256": "a" * 64,
        "cluster_case_count": 7,
        "publication_timing_admissible": False,
        "comparisons": {
            case_id: {
                "status": "accepted",
                "reference_row_id": reference_id,
                "gates": {"passed": True},
            }
            for case_id in comparison_ids
        },
        "rejected_or_censored_cases": [],
        "required_gate_failures": [],
        "selected_policies": {
            "p3d_p4_nonlinear_cluster": {
                "status": "selected_loosest_accepted_same_discretization_policy",
                "row_id": "p3d_p4_nonlinear_1em06_cluster",
                "parameter": "relative_dual_residual_target",
                "tolerance": 1.0e-6,
            }
        },
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def test_preparation_requires_a_fresh_output_root(tmp_path: Path, monkeypatch) -> None:
    module = _load_preparer()
    reviewed_hashes = _current_reviewed_hashes(module)
    monkeypatch.setattr(
        module, "_validate_reviewed_sources", lambda: reviewed_hashes
    )
    monkeypatch.setattr(
        module,
        "_git_metadata",
        lambda: {"commit": "0" * 40, "dirty": True},
    )
    out_root = tmp_path / "fresh"
    module.prepare(_prepare_args(out_root, execute=False))
    with pytest.raises(RuntimeError, match="already exists"):
        module.prepare(_prepare_args(out_root, execute=False))


def test_tier_b_preparation_is_pending_without_final_stop_gate(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_preparer()
    monkeypatch.setattr(
        module, "_validate_reviewed_sources", lambda: _current_reviewed_hashes(module)
    )
    monkeypatch.setattr(
        module,
        "_git_metadata",
        lambda: {"commit": "0" * 40, "dirty": True},
    )
    root = tmp_path / "tier_b_pending"
    manifest = module.prepare(
        _tier_b_prepare_args(root, execute=False, stopping_adjudication=None)
    )
    assert manifest["case_count"] == 20
    assert manifest["tier_b_stopping_gate"] == {
        "status": "pending_required_before_scheduler_contact",
        "submission_admissible": False,
        "policy": {
            "path": "paper/protocols/EXP-ROUTE-001-tier-b-stopping-policy.json",
            "sha256": hashlib.sha256(
                tier_b_stopping.POLICY_PATH.read_bytes()
            ).hexdigest(),
        },
        "adjudication": None,
    }
    assert module.offline_preflight(root, matrix=MATRIX)["status"] == "passed"


def test_tier_b_scheduler_contact_fails_before_output_without_stop_gate(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_preparer()
    monkeypatch.setattr(
        module, "_validate_reviewed_sources", lambda: _current_reviewed_hashes(module)
    )
    root = tmp_path / "tier_b_blocked"
    with pytest.raises(RuntimeError, match="requires --stopping-adjudication"):
        module.prepare(
            _tier_b_prepare_args(root, execute=True, stopping_adjudication=None)
        )
    assert not root.exists()


def test_tier_b_stop_gate_is_archived_relocatably_and_tamper_evident(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_preparer()
    monkeypatch.setattr(
        module, "_validate_reviewed_sources", lambda: _current_reviewed_hashes(module)
    )
    monkeypatch.setattr(
        module,
        "_git_metadata",
        lambda: {
            "commit": "0123456789abcdef0123456789abcdef01234567",
            "dirty": False,
        },
    )
    stop = _write_valid_stop_adjudication(tmp_path / "final_stop.json")
    root = tmp_path / "tier_b_valid"
    manifest = module.prepare(
        _tier_b_prepare_args(root, execute=False, stopping_adjudication=stop)
    )
    record = manifest["tier_b_stopping_gate"]
    assert record["submission_admissible"] is True
    assert record["adjudication"]["path"] == "stopping_adjudication.json"

    relocated = tmp_path / "tier_b_relocated"
    root.rename(relocated)
    assert module.offline_preflight(relocated, matrix=MATRIX)["status"] == "passed"
    archived = relocated / "stopping_adjudication.json"
    archived.write_text(archived.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="binding is stale"):
        module.offline_preflight(relocated, matrix=MATRIX)


def test_partial_submission_is_persisted_fail_closed(tmp_path: Path, monkeypatch) -> None:
    module = _load_preparer()
    reviewed_hashes = _current_reviewed_hashes(module)
    monkeypatch.setattr(
        module, "_validate_reviewed_sources", lambda: reviewed_hashes
    )
    monkeypatch.setattr(
        module,
        "_git_metadata",
        lambda: {"commit": "0" * 40, "dirty": False},
    )
    monkeypatch.setattr(module, "_require_revalidation", lambda **_kwargs: None)
    monkeypatch.setattr(
        module,
        "_require_staged_real_submission",
        lambda *_args, **_kwargs: None,
    )
    calls = 0

    def fake_run(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            returncode=0 if calls == 1 else 1,
            stdout="Submitted batch job 123" if calls == 1 else "",
            stderr="rejected" if calls == 2 else "",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    out_root = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="sbatch failed"):
        module.prepare(_prepare_args(out_root, execute=True))
    manifest = json.loads((out_root / "prepared_manifest.json").read_text())
    assert manifest["status"] == "partial_submission"
    assert manifest["submission_progress"] == {
        "attempted": 2,
        "accepted": 1,
        "total": 6,
        "last_case_id": "route_factor_micro_np1_b02",
    }
    assert len((out_root / "submitted_jobs.jsonl").read_text().splitlines()) == 1
    assert len((out_root / "submission_journal.jsonl").read_text().splitlines()) == 4


def test_preparation_freezes_commit_matrix_and_every_reviewed_source(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_preparer()
    reviewed_hashes = _current_reviewed_hashes(module)
    source_commit = "0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setattr(
        module, "_validate_reviewed_sources", lambda: reviewed_hashes
    )
    monkeypatch.setattr(
        module,
        "_git_metadata",
        lambda: {"commit": source_commit, "dirty": True},
    )
    out_root = tmp_path / "source_frozen"
    manifest = module.prepare(_prepare_args(out_root, execute=False))
    freeze_path = out_root / manifest["queued_source_freeze"]["path"]
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    assert freeze["schema_id"] == "fenics-nonlinear-energies.queued-source-freeze"
    assert freeze["source_commit"] == source_commit
    assert freeze["matrix"] == {
        "path": str(MATRIX.relative_to(REPO_ROOT)),
        "sha256": hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
    }
    assert set(freeze["reviewed_sources"]) == set(module.REVIEWED_SOURCES)
    for key, source in module.REVIEWED_SOURCES.items():
        assert freeze["reviewed_sources"][key] == {
            "path": str(source.resolve().relative_to(REPO_ROOT)),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    module._validate_source_freeze_payload(
        freeze,
        matrix=MATRIX,
        source_commit=source_commit,
    )
    tampered = json.loads(json.dumps(freeze))
    first_key = sorted(tampered["reviewed_sources"])[0]
    tampered["reviewed_sources"][first_key]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match=f"hash for {first_key} is stale"):
        module._validate_source_freeze_payload(
            tampered,
            matrix=MATRIX,
            source_commit=source_commit,
        )

    command = shlex.split((out_root / "sbatch_commands.txt").read_text().splitlines()[0])
    batch_index = command.index(str(BATCH_RUNNER))
    assert command[batch_index + 4] == source_commit
    assert command[batch_index + 5] == freeze["matrix"]["sha256"]
    assert command[batch_index + 6] == str(freeze_path)
    assert command[batch_index + 7] == manifest["queued_source_freeze"]["sha256"]


def test_optional_route_and_scaling_tranches_are_separately_bounded(tmp_path: Path) -> None:
    preparer = _load_preparer()
    rows = preparer.read_matrix(MATRIX)
    route = preparer.select_rows(
        rows,
        experiments={"EXP-ROUTE-001"},
        include_optional=False,
        only_optional=True,
    )
    scaling = preparer.select_rows(
        rows,
        experiments={"EXP-SCALE-001"},
        include_optional=False,
        only_optional=True,
    )
    assert len(route) == 30
    assert sum(float(row["estimated_node_hours"]) for row in route) == 45.0
    assert len(scaling) == 3
    assert sum(float(row["estimated_node_hours"]) for row in scaling) == 17.5


def test_real_submission_scope_rejects_mixed_scaling_and_partial_tier_b() -> None:
    preparer = _load_preparer()
    rows = preparer.read_matrix(MATRIX)
    mixed_scaling = preparer.select_rows(
        rows,
        experiments={"EXP-SCALE-001"},
        include_optional=True,
        tiers={"fixed_policy_he_l5", "optional_fixed_policy_p3d"},
    )
    with pytest.raises(RuntimeError, match="exactly one"):
        preparer._validate_real_submission_scope(
            SimpleNamespace(
                experiment=["EXP-SCALE-001"],
                tier=["fixed_policy_he_l5", "optional_fixed_policy_p3d"],
                include_optional=True,
                only_optional=False,
            ),
            selected=mixed_scaling,
        )

    partial_tier_b = preparer.select_rows(
        rows,
        experiments={"EXP-ROUTE-001"},
        include_optional=False,
        only_optional=True,
        tiers={"full_solve_confirmation"},
        route_phase="training",
    )
    with pytest.raises(RuntimeError, match="exact prespecified Tier-B training phase"):
        preparer._validate_real_submission_scope(
            SimpleNamespace(
                experiment=["EXP-ROUTE-001"],
                tier=["full_solve_confirmation"],
                include_optional=False,
                only_optional=True,
                route_phase="training",
            ),
            selected=partial_tier_b,
        )


def test_offline_preflight_remains_valid_after_archive_copy_back(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_preparer()
    reviewed_hashes = _current_reviewed_hashes(module)
    monkeypatch.setattr(module, "_validate_reviewed_sources", lambda: reviewed_hashes)
    monkeypatch.setattr(
        module,
        "_git_metadata",
        lambda: {"commit": "0123456789abcdef0123456789abcdef01234567", "dirty": False},
    )
    source = tmp_path / "cluster_archive"
    module.prepare(_prepare_args(source, execute=False))
    copied = tmp_path / "copied_back_and_renamed"
    source.rename(copied)
    result = module.offline_preflight(copied, matrix=MATRIX)
    assert result["status"] == "passed"
    assert result["mode"] == "offline_no_scheduler_access"
    commands = copied / "sbatch_commands.txt"
    commands.write_text(commands.read_text(encoding="utf-8") + "sbatch tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale hash"):
        module.offline_preflight(copied, matrix=MATRIX)


def test_commands_use_explicit_karolina_binding_and_frozen_solver_policy(tmp_path: Path) -> None:
    module = _load_executor()
    rows = {row["case_id"]: row for row in _rows()}

    he = module.build_command(
        rows["scale_he_l5_n8_np1024"],
        python="python",
        run_dir=tmp_path / "he",
    )
    assert "--distribution=block:block" in he
    assert "--mem-bind=local" in he
    assert any(token.startswith("--cpu-bind=map_cpu:0,1,2") for token in he)
    assert he[he.index("--he-pmg-coarsest-level") + 1] == "3"
    assert he[he.index("--he-pmg-coarse-pc-type") + 1] == "hypre"
    assert he[he.index("--ksp-type") + 1] == "stcg"
    assert he.count("stcg") == 1
    assert "--state-out" in he

    disc = module.build_command(
        rows["disc_p4l2_q125_np128"],
        python="python",
        run_dir=tmp_path / "disc",
    )
    assert disc[disc.index("--quadrature-rule") + 1] == "tetra_duffy_125point"
    assert disc[disc.index("--mesh-name") + 1] == "hetero_ssr_L1_2"
    assert disc[disc.index("--solver-backend") + 1] == "local_pmg"
    assert "--state-out" in disc
    assert disc[disc.index("--convergence-metric") + 1] == "reference_elastic_energy"
    assert disc[disc.index("--riesz-ksp-type") + 1] == "gmres"
    assert disc[disc.index("--riesz-pc-type") + 1] == "hypre"
    assert disc[disc.index("--riesz-ksp-rtol") + 1] == "1e-10"
    assert disc[disc.index("--riesz-ksp-atol") + 1] == "1e-14"
    assert disc[disc.index("--riesz-ksp-max-it") + 1] == "1000"
    assert disc[disc.index("--riesz-true-residual-rtol") + 1] == "1e-8"
    assert disc[disc.index("--riesz-spd-factor-solver-type") + 1] == "mumps"
    assert disc[disc.index("--riesz-symmetry-tol") + 1] == "1e-12"
    assert "--convergence-state-scale" not in disc

    fixed_row = rows["route_block_p1l1_elastic_np1_b01"]
    fixed = module.p3d_fixed_block_commands(
        fixed_row, python="python", run_dir=tmp_path / "fixed"
    )
    assert [route for route, _command in fixed] == fixed_row["route_order"].split("|")
    assert all(command[command.index("--probe-count") + 1] == "4" for _, command in fixed)
    assert all("--save-direct-matrix" in command for _, command in fixed)
    assert all(
        command[command.index("--route-order-policy") + 1]
        == "seeded_balanced_cyclic_v1"
        for _, command in fixed
    )

    full_row = rows["route_full_block_p4l1_np8_b01"]
    full_commands = module.p3d_solve_block_commands(
        full_row, python="python", run_dir=tmp_path / "full"
    )
    assert [route for route, _command in full_commands] == [
        "element_ad",
        "constitutive_ad",
    ]
    assert all(
        command[command.index("--convergence-metric") + 1]
        == "reference_elastic_energy"
        for _, command in full_commands
    )

    factor = module.build_command(
        rows["route_factor_micro_np32_b01"],
        python="python",
        run_dir=tmp_path / "factor",
    )
    assert "run_route_factor_microbenchmarks.py" in " ".join(factor)
    assert factor[factor.index("--repetitions") + 1] == "5"


def _valid_riesz_output() -> dict[str, object]:
    return {
        "status": "completed",
        "solver_success": True,
        "convergence_metric_requested": "reference_elastic_energy",
        "convergence_metric": "reference_elastic_energy",
        "final_grad_norm": 2.5,
        "riesz_solver_requested": {
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "rtol": 1.0e-10,
            "atol": 1.0e-14,
            "max_it": 1000,
            "true_residual_rtol": 1.0e-8,
            "spd_factor_solver_type": "mumps",
            "symmetry_relative_tolerance": 1.0e-12,
        },
        "parallel_setup": {"owned_free_dofs_sum": 12},
        "nonlinear_convergence": {
            "configuration": {
                "selection": "reference_elastic_energy",
                "correction_normalization": "metric_current_state",
            },
            "metric": {
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "requested_rtol": 1.0e-10,
                "requested_atol": 1.0e-14,
                "requested_max_it": 1000,
                "effective_rtol": 1.0e-10,
                "effective_atol": 1.0e-14,
                "effective_max_it": 1000,
                "true_residual_rtol_gate": 1.0e-8,
                "provenance": {
                    "free_dofs": 12,
                    "spd_certificate": {
                        "certified_spd": True,
                        "factor_solver_type": "mumps",
                        "inertia": {"negative": 0, "zero": 0, "positive": 12},
                    },
                }
            },
            "initial_absolute_dual_residual": {"value": 4.0},
            "absolute_dual_residual": {"value": 1.0e-5},
            "state_norm": {"value": 8.0},
            "relative_correction": {"value": 1.0e-4},
            "coefficient_gradient_l2": 2.5,
            "last_riesz_solve": {
                "riesz_solve": "iterative",
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "requested_rtol": 1.0e-10,
                "requested_atol": 1.0e-14,
                "requested_max_it": 1000,
                "effective_rtol": 1.0e-10,
                "effective_atol": 1.0e-14,
                "effective_max_it": 1000,
                "reason": 2,
                "rhs_norm": 2.5,
                "relative_true_residual": 2.0e-10,
                "true_residual_rtol_gate": 1.0e-8,
            },
            "residual_gate": {"passed": True},
        },
    }


def _valid_tier_b_output(
    row: dict[str, str], *, relative_correction: float = 3.0e-3
) -> dict[str, object]:
    target = float(row["stop_tol"])
    initial = 4.0
    terminal = 0.5 * initial * target
    effective_ksp = {
        "ksp_type": "fgmres",
        "pc_type": "mg",
        "rtol": 1.0e-8,
        "max_it": 1000,
        "captured_after_set_from_options": True,
    }
    metric = {
        "ksp_type": "cg",
        "pc_type": "jacobi",
        "requested_rtol": 1.0e-10,
        "requested_atol": 0.0,
        "requested_max_it": 5000,
        "requested_norm_type": "unpreconditioned",
        "effective_rtol": 1.0e-10,
        "effective_atol": 0.0,
        "effective_max_it": 5000,
        "effective_norm_type": "unpreconditioned",
        "true_residual_rtol_gate": 1.0e-8,
        "set_from_petsc_options": False,
        "provenance": {
            "free_dofs": 12,
            "spd_certificate": {
                "certified_spd": True,
                "factor_solver_type": "mumps",
                "inertia": {"negative": 0, "zero": 0, "positive": 12},
            },
        },
    }
    last = {
        "riesz_solve": "iterative",
        "ksp_type": "cg",
        "pc_type": "jacobi",
        "requested_rtol": 1.0e-10,
        "requested_atol": 0.0,
        "requested_max_it": 5000,
        "requested_norm_type": "unpreconditioned",
        "effective_rtol": 1.0e-10,
        "effective_atol": 0.0,
        "effective_max_it": 5000,
        "effective_norm_type": "unpreconditioned",
        "reported_residual_norm_type": "unpreconditioned",
        "reason": 2,
        "rhs_norm": 2.5,
        "relative_true_residual": 2.0e-10,
        "true_residual_rtol_gate": 1.0e-8,
    }
    return {
        "status": "completed",
        "solver_success": True,
        "convergence_metric_requested": "reference_elastic_energy",
        "convergence_metric": "reference_elastic_energy",
        "stop_metric_name": "dual_residual_norm",
        "stop_tol": target,
        "grad_stop_tol": 0.0,
        "grad_stop_rtol": target,
        "ksp_rtol": 1.0e-8,
        "ksp_max_it": 1000,
        "final_grad_norm": 2.5,
        "riesz_solver_requested": {
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "rtol": 1.0e-10,
            "atol": 0.0,
            "max_it": 5000,
            "true_residual_rtol": 1.0e-8,
            "spd_factor_solver_type": "mumps",
            "symmetry_relative_tolerance": 1.0e-12,
        },
        "parallel_setup": {"owned_free_dofs_sum": 12},
        "initial_guess": {
            "success": True,
            "ksp_reason_code": 2,
            "effective_ksp": dict(effective_ksp),
        },
        "linear_history": [
            {
                "ksp_reason_code": 2,
                "effective_ksp": dict(effective_ksp),
            }
        ],
        "nonlinear_convergence": {
            "configuration": {
                "selection": "reference_elastic_energy",
                "correction_normalization": "metric_current_state",
            },
            "metric": metric,
            "initial_absolute_dual_residual": {"value": initial},
            "initial_relative_dual_residual": {"value": terminal / initial},
            "absolute_dual_residual": {"value": terminal},
            "state_norm": {"value": 8.0},
            "relative_correction": {"value": relative_correction},
            "coefficient_gradient_l2": 2.5,
            "last_riesz_solve": last,
            "residual_gate": {
                "absolute_tolerance": 0.0,
                "initial_relative_tolerance": target,
                "effective_absolute_target": initial * target,
                "passed": True,
            },
        },
    }


def test_executor_hard_validates_riesz_output_and_rejects_coefficient_stopping() -> None:
    module = _load_executor()
    row = next(row for row in _rows() if row["runner"] == "p3d_solve")
    result = module.validate_p3d_solve_output(_valid_riesz_output(), row)
    assert result["status"] == "passed"
    assert result["positive_inertia"] == 12

    coefficient = _valid_riesz_output()
    coefficient["nonlinear_convergence"]["configuration"]["selection"] = "coefficient_l2"
    try:
        module.validate_p3d_solve_output(coefficient, row)
    except ValueError as exc:
        assert "coefficient" in str(exc)
    else:
        raise AssertionError("coefficient stopping was incorrectly admitted")

    bad_inertia = _valid_riesz_output()
    bad_inertia["nonlinear_convergence"]["metric"]["provenance"]["spd_certificate"][
        "inertia"
    ]["zero"] = 1
    try:
        module.validate_p3d_solve_output(bad_inertia, row)
    except ValueError as exc:
        assert "inertia" in str(exc)
    else:
        raise AssertionError("non-SPD convergence metric was incorrectly admitted")

    stale = _valid_riesz_output()
    stale["nonlinear_convergence"]["last_riesz_solve"]["rhs_norm"] = 2.0
    try:
        module.validate_p3d_solve_output(stale, row)
    except ValueError as exc:
        assert "stale" in str(exc)
    else:
        raise AssertionError("stale endpoint Riesz evidence was incorrectly admitted")

    false_gate = _valid_riesz_output()
    false_gate["nonlinear_convergence"]["residual_gate"]["passed"] = False
    try:
        module.validate_p3d_solve_output(false_gate, row)
    except ValueError as exc:
        assert "residual gate" in str(exc)
    else:
        raise AssertionError("completed row with failed residual gate was admitted")


def test_tier_b_commands_and_outputs_use_stop_aligned_relative_policy(
    tmp_path: Path,
) -> None:
    module = _load_executor()
    rows = {row["case_id"]: row for row in _rows()}
    for case_id, target in (
        ("route_full_block_p1l1_np8_b01", 1.0e-6),
        ("route_full_block_p4l1_np8_b01", 1.0e-7),
    ):
        row = rows[case_id]
        command = module.p3d_solve_command(
            row, python="python", run_dir=tmp_path / case_id, route="element_ad"
        )
        assert command[command.index("--convergence-mode") + 1] == "gradient_only"
        assert float(command[command.index("--grad-stop-rtol") + 1]) == target
        assert float(command[command.index("--grad-stop-tol") + 1]) == 0.0
        assert float(command[command.index("--stop-tol") + 1]) == target
        assert command[command.index("--riesz-ksp-type") + 1] == "cg"
        assert command[command.index("--riesz-pc-type") + 1] == "jacobi"
        assert float(command[command.index("--riesz-ksp-atol") + 1]) == 0.0
        assert int(command[command.index("--riesz-ksp-max-it") + 1]) == 5000

        result = module.validate_p3d_solve_output(
            _valid_tier_b_output(row), row
        )
        assert result["status"] == "passed"
        assert result["terminal_initial_relative_dual_residual"] == pytest.approx(
            0.5 * target
        )
        assert result["correction_diagnostic_within_limit"] is False
        assert result["tier_b_stopping_policy"][
            "relative_dual_residual_target"
        ] == target


def test_tier_b_output_rejects_effective_ksp_norm_and_relative_gate_drift() -> None:
    module = _load_executor()
    row = next(
        row
        for row in _rows()
        if row["tier"] == "low_order_confirmation"
    )

    effective = _valid_tier_b_output(row)
    effective["linear_history"][0]["effective_ksp"]["max_it"] = 999
    with pytest.raises(ValueError, match="effective KSP policy"):
        module.validate_p3d_solve_output(effective, row)

    norm = _valid_tier_b_output(row)
    norm["nonlinear_convergence"]["last_riesz_solve"][
        "reported_residual_norm_type"
    ] = "preconditioned"
    with pytest.raises(ValueError, match="reported_residual_norm_type"):
        module.validate_p3d_solve_output(norm, row)

    residual = _valid_tier_b_output(row)
    residual["nonlinear_convergence"]["initial_relative_dual_residual"][
        "value"
    ] = 2.0e-6
    with pytest.raises(ValueError, match="internally stale|relative residual gate"):
        module.validate_p3d_solve_output(residual, row)


def test_matrix_validator_rejects_any_p3d_coefficient_stopping_row(tmp_path: Path) -> None:
    module = _load_preparer()
    rows = _rows()
    target = next(row for row in rows if row["runner"] == "p3d_solve")
    target["convergence_metric"] = "coefficient_l2"
    path = tmp_path / "mutated_matrix.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        module.read_matrix(path)
    except ValueError as exc:
        assert "reference_elastic_energy" in str(exc)
    else:
        raise AssertionError("coefficient-stopping P3D matrix row was admitted")


def _write_fixed_block_route(
    route_dir: Path,
    *,
    row: dict[str, str],
    route: str,
    position: int,
    action_shift: float = 0.0,
) -> None:
    route_dir.mkdir(parents=True, exist_ok=True)
    state = np.asarray([1.0, 2.0, 3.0])
    gradient = np.asarray([4.0, 5.0, 6.0])
    actions = np.asarray(
        [[7.0, 8.0, 9.0], [2.0, 3.0, 4.0], [5.0, 7.0, 11.0], [1.0, 4.0, 9.0]]
    )
    actions = actions + action_shift
    np.savez_compressed(
        route_dir / "tangent_action.npz",
        state=state,
        gradient=gradient,
        tangent_action=actions[0],
        tangent_actions=actions,
    )
    np.savez_compressed(
        route_dir / "tangent_matrix_csr.npz",
        indptr=np.asarray([0, 2, 3, 4]),
        indices=np.asarray([0, 1, 1, 2]),
        values=np.asarray([2.0, 0.5, 3.0, 4.0]) + action_shift,
        shape=np.asarray([3, 3]),
    )
    payload = {
        "experiment_id": row["experiment_id"],
        "tier": row["tier"],
        "mesh_name": row["mesh_name"],
        "element_degree": int(row["element_degree"]),
        "quadrature_rule_id": row["quadrature_rule"],
        "state_label": row["state_label"],
        "state_amplitude": float(row["state_amplitude"]),
        "mpi_ranks": int(row["total_ranks"]),
        "route": route,
        "constraint_variant": "glued_bottom",
        "lambda_target": 1.55,
        "warmup_repetitions": int(row["warmups"]),
        "measured_repetitions": int(row["repetitions"]),
        "probe_count": 4,
        "wall_time_reduction": "mpi_collective_max",
        "wall_times_s": [1.0, 1.1, 0.9, 1.05, 0.95],
        "wall_times_by_rank_s": [[1.0], [1.1], [0.9], [1.05], [0.95]],
        "branch_diagnostics": {"counts": {"elastic": 10}},
        "comparison_design": {
            "comparison_id": row["comparison_id"],
            "block_repetition": int(row["block_repetition"]),
            "route_order_position": position,
            "route_order_policy": row["route_order_policy"],
        },
    }
    (route_dir / "output.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_fixed_state_block_requires_exact_state_gradient_multiple_actions_and_matrix(
    tmp_path: Path,
) -> None:
    module = _load_executor()
    row = next(row for row in _rows() if row["case_id"] == "route_block_p1l1_elastic_np1_b01")
    route_dirs: dict[str, Path] = {}
    for position, route in enumerate(row["route_order"].split("|")):
        route_dir = tmp_path / route
        _write_fixed_block_route(
            route_dir, row=row, route=route, position=position
        )
        route_dirs[route] = route_dir
    result = module.validate_fixed_state_block(row, route_dirs)
    assert result["status"] == "admitted_correctness_block"
    assert result["probe_count"] == 4
    assert result["timing_claim_released"] is False
    assert result["routes"]["element_ad"]["timing_rank_count"] == 1

    tampered_path = route_dirs["element_ad"] / "output.json"
    tampered = json.loads(tampered_path.read_text(encoding="utf-8"))
    tampered["tier"] = "factorized_quadrature"
    tampered_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    try:
        module.validate_fixed_state_block(row, route_dirs)
    except ValueError as exc:
        assert "tier differs from the matrix row" in str(exc)
    else:
        raise AssertionError("fixed-state payload with a mismatched matrix tier was admitted")
    _write_fixed_block_route(
        route_dirs["element_ad"],
        row=row,
        route="element_ad",
        position=row["route_order"].split("|").index("element_ad"),
    )

    missing_matrix = route_dirs["colored_sfd"] / "tangent_matrix_csr.npz"
    missing_matrix.unlink()
    try:
        module.validate_fixed_state_block(row, route_dirs)
    except ValueError as exc:
        assert "require a direct CSR matrix" in str(exc)
    else:
        raise AssertionError("rank-one P1 block without direct CSR evidence was admitted")
    _write_fixed_block_route(
        route_dirs["colored_sfd"],
        row=row,
        route="colored_sfd",
        position=row["route_order"].split("|").index("colored_sfd"),
    )

    _write_fixed_block_route(
        route_dirs["constitutive_ad"],
        row=row,
        route="constitutive_ad",
        position=2,
        action_shift=1.0,
    )
    try:
        module.validate_fixed_state_block(row, route_dirs)
    except ValueError as exc:
        assert "tangent mismatch" in str(exc) or "matrix values mismatch" in str(exc)
    else:
        raise AssertionError("mismatched multi-probe block was admitted")


def test_batch_and_manifest_preserve_safety_and_card_links() -> None:
    batch_text = (CAMPAIGN_DIR / "run_revision_case.sbatch").read_text(encoding="utf-8")
    assert "ALLOCATION_REVALIDATED" in batch_text
    assert "ACCOUNT_QOS_REVALIDATED" in batch_text
    assert "ALLOCATION_VALID_UNTIL" in batch_text
    assert "--distribution=block:block" not in batch_text  # execution builds it per row
    assert "#SBATCH --exclusive" not in batch_text
    assert "#SBATCH --mem=" not in batch_text
    assert "#SBATCH --mem-per-cpu" not in batch_text
    assert "OMP_NUM_THREADS=1" in batch_text
    assert "env | sort" not in batch_text
    assert "pending_post_job_collection" in batch_text
    assert "EXPECTED_SOURCE_COMMIT" in batch_text
    assert "EXPECTED_MATRIX_SHA256" in batch_text
    assert "SOURCE_FREEZE_SHA256" in batch_text
    assert "fenics-nonlinear-energies.queued-source-freeze" in batch_text
    assert "reviewed source freeze changed after command preparation" in batch_text
    assert batch_text.index("RUNTIME_SOURCE_COMMIT=") < batch_text.index(
        '"$PYTHON" -u "$SCRIPT_DIR/execute_case.py"'
    )
    assert batch_text.index("verified queued source freeze for") < batch_text.index(
        '"$PYTHON" -u "$SCRIPT_DIR/execute_case.py"'
    )
    assert batch_text.index('source "$ENV_SETUP"') < batch_text.index(
        "export JAX_PLATFORMS=cpu"
    )

    digest = hashlib.sha256(MATRIX.read_bytes()).hexdigest()
    manifest_text = (CAMPAIGN_DIR / "campaign_manifest.yaml").read_text(encoding="utf-8")
    handoff_text = (CAMPAIGN_DIR / "handoff.yaml").read_text(encoding="utf-8")
    assert digest in manifest_text
    assert digest in handoff_text
    for path in (
        CAMPAIGN_DIR / "execute_case.py",
        CAMPAIGN_DIR / "prepare_campaign.py",
        CAMPAIGN_DIR / "run_revision_case.sbatch",
        CAMPAIGN_DIR / "submit_prepared_campaigns.sh",
        REPO_ROOT / "paper/protocols/EXP-ROUTE-001-analysis-contract.json",
        RELEASE_AUTHORIZATION_SCHEMA,
        RELEASE_AUTHORIZATION_EXAMPLE,
        REPO_ROOT / "experiments/analysis/analyze_plasticity3d_route_cost_model.py",
    ):
        source_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert source_digest in manifest_text
    contract_digest = hashlib.sha256(
        (REPO_ROOT / "paper/protocols/EXP-ROUTE-001-analysis-contract.json").read_bytes()
    ).hexdigest()
    assert contract_digest in handoff_text
    for experiment in ("EXP-ROUTE-001", "EXP-DISC-001", "EXP-SCALE-001"):
        assert (REPO_ROOT / "paper/protocols" / f"{experiment}.md").is_file()
        assert experiment in manifest_text
