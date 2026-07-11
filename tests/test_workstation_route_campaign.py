from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments.runners import run_workstation_route_campaign as campaign


COMMIT = "0123456789abcdef0123456789abcdef01234567"
TREE = "89abcdef0123456789abcdef0123456789abcdef"
LIMIT_64_GIB = campaign.MAX_WORKSTATION_ADDRESS_SPACE_BYTES


def _address_space_limit(
    soft: int | None = LIMIT_64_GIB,
    hard: int | None = LIMIT_64_GIB,
) -> dict[str, object]:
    return {
        "resource": "RLIMIT_AS",
        "soft_limit_bytes": soft,
        "hard_limit_bytes": hard,
    }


@pytest.fixture(autouse=True)
def _controlled_workstation_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in campaign.REQUIRED_WORKSTATION_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)


def _git(*, dirty: bool = False) -> dict[str, object]:
    return {
        "commit": COMMIT,
        "tree": TREE,
        "branch": "paper/test",
        "dirty": dirty,
        "status_porcelain": [" M tracked.py"] if dirty else [],
    }


def _args(root: Path, **overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "plan": campaign.DEFAULT_PLAN,
        "out_root": root,
        "python": sys.executable,
        "run_id": "publication-test",
        "expected_commit": COMMIT,
        "row_wall_s": 60.0,
        "campaign_wall_s": 600.0,
        "execute": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _one_real_row(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    plan, rows = campaign._load_plan(campaign.DEFAULT_PLAN)
    row = rows[0]
    monkeypatch.setattr(campaign, "_load_plan", lambda _path: (plan, [row]))
    return row


def _lightweight_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    limit = _address_space_limit()
    monkeypatch.setattr(campaign, "_git_metadata", lambda: _git())
    monkeypatch.setattr(
        campaign,
        "_capture_address_space_limit",
        lambda: dict(limit),
    )
    monkeypatch.setattr(
        campaign,
        "_collect_code_hashes",
        lambda: {"experiments/runners/runner.py": "a" * 64},
    )
    monkeypatch.setattr(
        campaign,
        "_collect_configuration_hashes",
        lambda _path: {"paper/protocols/plan.json": "b" * 64},
    )
    monkeypatch.setattr(
        campaign,
        "_collect_input_hashes",
        lambda _rows: {"data/meshes/input.h5": "c" * 64},
    )
    monkeypatch.setattr(
        campaign,
        "_capture_environment",
        lambda python, *, address_space_limit=None: {
            "python_executable": str(python),
            "python_executable_sha256": "d" * 64,
            "packages": {"numpy": "test"},
            "process_resource_limits": {
                "address_space": dict(address_space_limit or limit),
            },
        },
    )
    monkeypatch.setattr(campaign, "_verify_hash_inventory", lambda _inventory: [])


def test_python_launcher_preserves_virtual_environment_symlink() -> None:
    launcher = Path(sys.executable)
    resolved = campaign._resolve_python(launcher)
    assert resolved == Path(campaign.os.path.abspath(launcher))
    completed = subprocess.run(
        [str(resolved), "-c", "import numpy; print(numpy.__version__)"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("overrides", "environment", "dirty", "message"),
    [
        (
            {"execute": True},
            {},
            False,
            "WORKSTATION_RUN_CONFIRMED=YES",
        ),
        (
            {"execute": True},
            {"WORKSTATION_RUN_CONFIRMED": "YES"},
            True,
            "clean worktree",
        ),
        (
            {"execute": True, "expected_commit": "f" * 40},
            {"WORKSTATION_RUN_CONFIRMED": "YES"},
            False,
            "differs from --expected-commit",
        ),
        (
            {"execute": True, "expected_commit": ""},
            {"WORKSTATION_RUN_CONFIRMED": "YES"},
            False,
            "requires --expected-commit",
        ),
        (
            {"execute": True},
            {"WORKSTATION_RUN_CONFIRMED": "YES", "OMP_NUM_THREADS": "2"},
            False,
            "controlled workstation environment differs",
        ),
    ],
)
def test_publication_preflight_fails_before_output_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    environment: dict[str, str],
    dirty: bool,
    message: str,
) -> None:
    root = tmp_path / "must-not-exist"
    monkeypatch.delenv("WORKSTATION_RUN_CONFIRMED", raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(campaign, "_git_metadata", lambda: _git(dirty=dirty))

    with pytest.raises(campaign.WorkstationCampaignError, match=message):
        campaign.prepare_or_execute(_args(root, **overrides))

    assert not root.exists()


def test_environment_capture_records_the_exact_frozen_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit = _address_space_limit()
    captured = campaign._capture_environment(
        Path(sys.executable),
        address_space_limit=limit,
    )
    assert captured["controlled_environment"] == {
        "status": "passed",
        "values": campaign.REQUIRED_WORKSTATION_ENVIRONMENT,
    }
    assert captured["process_resource_limits"] == {
        "address_space": limit,
    }

    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE")
    with pytest.raises(
        campaign.WorkstationCampaignError,
        match="XLA_PYTHON_CLIENT_PREALLOCATE",
    ):
        campaign._capture_environment(
            Path(sys.executable),
            address_space_limit=limit,
        )


def test_address_space_limit_normalization_and_execution_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        campaign.resource,
        "getrlimit",
        lambda _resource: (LIMIT_64_GIB, campaign.resource.RLIM_INFINITY),
    )
    assert campaign._capture_address_space_limit() == _address_space_limit(
        LIMIT_64_GIB,
        None,
    )
    campaign._require_execution_address_space_limit(
        _address_space_limit(LIMIT_64_GIB, None)
    )

    for invalid in (None, -2, 0, LIMIT_64_GIB + 1, True):
        with pytest.raises(
            campaign.WorkstationCampaignError,
            match="finite positive RLIMIT_AS",
        ):
            campaign._require_execution_address_space_limit(
                _address_space_limit(invalid, None)  # type: ignore[arg-type]
            )


def test_execute_rejects_unsafe_address_space_limit_before_root_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORKSTATION_RUN_CONFIRMED", "YES")
    monkeypatch.setattr(campaign, "_git_metadata", lambda: _git())
    monkeypatch.setattr(
        campaign,
        "_capture_address_space_limit",
        lambda: _address_space_limit(None, None),
    )
    root = tmp_path / "unsafe-limit"
    with pytest.raises(
        campaign.WorkstationCampaignError,
        match="finite positive RLIMIT_AS",
    ):
        campaign.prepare_or_execute(_args(root, execute=True))
    assert not root.exists()


def test_prepare_requires_fresh_root_and_freezes_commands_hashes_and_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _one_real_row(monkeypatch)
    _lightweight_provenance(monkeypatch)
    root = tmp_path / "prepared"

    manifest = campaign.prepare_or_execute(_args(root))
    manifest_path = root / campaign.MANIFEST_NAME
    parsed = json.loads(
        manifest_path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )

    assert parsed == manifest
    assert manifest["status"] == "prepared_not_executed"
    assert manifest["source_commit"] == COMMIT
    assert manifest["source_dirty"] is False
    assert manifest["code_hashes"]
    assert manifest["configuration_hashes"]
    assert manifest["input_hashes"]
    assert manifest["environment_sha256"] == campaign._sha256(root / "environment.json")
    environment = json.loads((root / "environment.json").read_text(encoding="utf-8"))
    expected_limit = _address_space_limit()
    assert environment["process_resource_limits"]["address_space"] == expected_limit
    assert manifest["memory_safety"] == {
        "policy": "finite_posix_address_space_limit_inherited_by_direct_workers",
        "maximum_soft_limit_bytes": LIMIT_64_GIB,
        "driver_start_address_space_limit": expected_limit,
        "worker_inheritance": "direct_subprocess_exec_inherits_RLIMIT_AS",
        "terminal_verification": {
            "status": "not_applicable_preparation_only"
        },
    }
    assert len(manifest["normalized_commands"]) == 3
    for command in manifest["normalized_commands"]:
        argv = command["normalized_argv"]
        assert argv[0] == "${PYTHON}"
        assert argv[2].startswith("${REPO_ROOT}/")
        assert any(value.startswith("${OUTPUT_ROOT}/") for value in argv)
        assert command["normalized_argv_sha256"] == campaign._json_sha256(argv)
    closure = manifest["output_hash_closure"]
    assert closure["excluded_paths"] == [campaign.MANIFEST_NAME]
    assert set(closure["files"]) == {"environment.json", "workstation_plan.json"}
    assert closure["files_map_sha256"] == campaign._json_sha256(closure["files"])

    before = manifest_path.read_bytes()
    with pytest.raises(campaign.WorkstationCampaignError, match="fresh and nonexisting"):
        campaign.prepare_or_execute(_args(root))
    assert manifest_path.read_bytes() == before


def test_preparation_records_unlimited_address_space_without_claiming_enforcement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _one_real_row(monkeypatch)
    _lightweight_provenance(monkeypatch)
    unlimited = _address_space_limit(None, None)
    monkeypatch.setattr(
        campaign,
        "_capture_address_space_limit",
        lambda: dict(unlimited),
    )
    monkeypatch.setattr(
        campaign,
        "_capture_environment",
        lambda python, *, address_space_limit=None: {
            "python_executable": str(python),
            "python_executable_sha256": "d" * 64,
            "packages": {"numpy": "test"},
            "process_resource_limits": {
                "address_space": dict(address_space_limit or unlimited),
            },
        },
    )
    root = tmp_path / "prepared-unlimited"
    manifest = campaign.prepare_or_execute(_args(root))
    assert manifest["memory_safety"]["driver_start_address_space_limit"] == unlimited
    assert manifest["memory_safety"]["terminal_verification"] == {
        "status": "not_applicable_preparation_only"
    }
    assert "Infinity" not in (root / "environment.json").read_text(encoding="utf-8")


def test_timeout_is_a_visible_censor_and_completed_process_provenance_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _one_real_row(monkeypatch)
    _lightweight_provenance(monkeypatch)
    monkeypatch.setenv("WORKSTATION_RUN_CONFIRMED", "YES")
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], *, stdout: Path, stderr: Path, timeout_s: float
    ) -> dict[str, object]:
        calls.append(command)
        stdout.write_text("worker stdout\n", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        if len(calls) == 1:
            route_dir = stdout.parent
            (route_dir / "output.json").write_text(
                json.dumps({"status": "completed"}) + "\n", encoding="utf-8"
            )
            (route_dir / "tangent_action.npz").write_bytes(b"test-npz")
            return {"returncode": 0, "timed_out": False, "wall_time_s": 1.0}
        return {"returncode": 124, "timed_out": True, "wall_time_s": timeout_s}

    monkeypatch.setattr(campaign, "_run", fake_run)
    monkeypatch.setattr(
        campaign,
        "validate_fixed_state_block",
        lambda *_args, **_kwargs: pytest.fail("censored block must not be validated"),
    )
    root = tmp_path / "executed"
    manifest = campaign.prepare_or_execute(
        _args(root, execute=True, row_wall_s=30.0, campaign_wall_s=120.0)
    )

    assert manifest["status"] == "completed_with_censors"
    assert manifest["route_terminal_counts"] == {
        "completed": 1,
        "censored": 2,
        "failed": 0,
    }
    assert manifest["route_processes_launched"] == 2
    assert manifest["case_statuses"] == {row["case_id"]: "censored"}
    assert len(calls) == 2

    job = root / "cases" / row["case_id"] / "job_publication-test"
    records = json.loads((job / "run_records.json").read_text(encoding="utf-8"))
    assert [record["status"] for record in records] == [
        "completed",
        "censored",
        "censored",
    ]
    assert records[0]["artifact_hash_closure"]["files"]["output.json"]
    assert records[0]["process_record_sha256"] == campaign._sha256(
        job / "measure_01" / records[0]["route"] / campaign.PROCESS_RECORD_NAME
    )
    assert records[1]["launched"] is True
    assert records[1]["timed_out"] is True
    assert records[0]["worker_address_space_limit"] == _address_space_limit()
    assert records[0]["driver_address_space_limit"] == _address_space_limit()
    assert records[0]["worker_limit_provenance"] == (
        "inherited_across_direct_posix_subprocess_exec"
    )
    assert records[1]["censor_reason"] in {
        "campaign_wall_cap_timeout",
        "row_wall_cap_timeout",
    }
    assert records[2]["launched"] is False
    assert records[2]["censor_reason"] == "not_launched_after_route_timeout"
    assert records[2]["worker_address_space_limit"] is None
    assert records[2]["worker_limit_provenance"] == "not_launched"
    assert manifest["memory_safety"]["terminal_verification"] == {
        "status": "passed",
        "terminal_address_space_limit": _address_space_limit(),
        "errors": [],
    }
    assert not (job / "measure_01" / "block_result.json").exists()
    assert (job / "measure_01" / "block_terminal.json").is_file()

    closure = manifest["output_hash_closure"]
    current = campaign._tree_hashes(
        root, exclude={root / campaign.MANIFEST_NAME}
    )
    assert closure["files"] == current
    assert closure["file_count"] == len(current)
    assert closure["files_map_sha256"] == campaign._json_sha256(current)


def test_changed_address_space_limit_blocks_later_workers_and_rejects_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _one_real_row(monkeypatch)
    _lightweight_provenance(monkeypatch)
    monkeypatch.setenv("WORKSTATION_RUN_CONFIRMED", "YES")
    current = _address_space_limit()
    monkeypatch.setattr(
        campaign,
        "_capture_address_space_limit",
        lambda: dict(current),
    )
    launches = 0

    def fake_run(
        command: list[str], *, stdout: Path, stderr: Path, timeout_s: float
    ) -> dict[str, object]:
        nonlocal launches, current
        launches += 1
        route_dir = stdout.parent
        stdout.write_text("worker stdout\n", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        (route_dir / "output.json").write_text(
            json.dumps({"status": "completed"}) + "\n",
            encoding="utf-8",
        )
        (route_dir / "tangent_action.npz").write_bytes(b"test-npz")
        current = _address_space_limit(32 * 1024**3, 32 * 1024**3)
        return {"returncode": 0, "timed_out": False, "wall_time_s": 1.0}

    monkeypatch.setattr(campaign, "_run", fake_run)
    monkeypatch.setattr(
        campaign,
        "validate_fixed_state_block",
        lambda *_args, **_kwargs: pytest.fail("changed-limit block must not validate"),
    )
    root = tmp_path / "changed-limit"
    with pytest.raises(
        campaign.WorkstationCampaignError,
        match="resource limit changed",
    ):
        campaign.prepare_or_execute(_args(root, execute=True))

    assert launches == 1
    manifest = json.loads((root / campaign.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["status"] == "failed_resource_limit_changed_during_execution"
    assert manifest["route_processes_launched"] == 1
    assert manifest["memory_safety"]["terminal_verification"]["status"] == "failed"
    assert manifest["memory_safety"]["terminal_verification"]["errors"]
    job = root / "cases" / row["case_id"] / "job_publication-test"
    records = json.loads((job / "run_records.json").read_text(encoding="utf-8"))
    assert [record["launched"] for record in records] == [True, False, False]
    assert records[1]["censor_reason"] == "address_space_limit_changed_before_launch"
    assert records[1]["worker_address_space_limit"] is None


def test_nonpositive_caps_and_unsafe_run_id_fail_before_root_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WORKSTATION_RUN_CONFIRMED", "YES")
    monkeypatch.setattr(campaign, "_git_metadata", lambda: _git())
    for suffix, overrides, message in (
        ("cap", {"execute": True, "row_wall_s": 0.0}, "row-wall-s"),
        ("id", {"run_id": "../escape"}, "run-id"),
    ):
        root = tmp_path / suffix
        with pytest.raises((campaign.WorkstationCampaignError, ValueError), match=message):
            campaign.prepare_or_execute(_args(root, **overrides))
        assert not root.exists()
