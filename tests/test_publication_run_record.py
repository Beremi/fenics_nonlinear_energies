from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from src.core.benchmark import run_record as rr


def _record(*, status: str = "success", run_kind: str = "publication") -> dict:
    limited = status in {"capped", "timeout"}
    successful = status == "success"
    return {
        "schema": {"id": rr.RUN_RECORD_SCHEMA_ID, "version": rr.RUN_RECORD_SCHEMA_VERSION},
        "record_id": "revision-2026-exp-deriv-case-a-element-r01",
        "run_kind": run_kind,
        "identifiers": {
            "campaign": "paper_revision_2026",
            "experiment": "EXP-DERIV-001",
            "case": "case-a",
            "method": "newton",
            "route": "element-ad",
            "repetition": 1,
        },
        "problem": {
            "name": "p-laplace",
            "mesh": "P4(L1)",
            "degree": 4,
            "quadrature": "24-point tetrahedral",
            "total_degrees_of_freedom": 100,
            "free_degrees_of_freedom": 80,
            "notes": "Constrained free-space dimensions are reported.",
        },
        "solver": {
            "algorithm": "damped Newton",
            "implementation": "example runner",
            "parameters": {"ksp_rtol": 0.01},
            "preconditioner": {"type": "gamg"},
            "stopping_contract": "accuracy-contract-v1",
        },
        "termination": {
            "status": status,
            "reason": "all preregistered gates passed" if successful else f"terminal {status}",
            "exit_code": 0 if successful else 2,
            "started_at_utc": "2026-07-10T08:00:00Z",
            "finished_at_utc": "2026-07-10T08:00:02Z",
            "limit_kind": "wall-time" if limited else None,
            "limit_value": 2.0 if limited else None,
            "censored": limited,
        },
        "accuracy": {
            "contract_id": "accuracy-contract-v1",
            "gate_passed": successful,
            "absolute_residual": 1.0e-9 if successful else None,
            "relative_residual": 2.0e-8 if successful else None,
            "scaled_residual": 3.0e-8 if successful else None,
            "relative_correction": 4.0e-9 if successful else None,
            "energy_change": -1.0e-10 if successful else None,
            "custom_metrics": {"symmetry_defect": 1.0e-14},
            "notes": "Null values are unavailable for unsuccessful terminal records.",
        },
        "counts": {
            "nonlinear_iterations": 3,
            "krylov_iterations": 12,
            "function_evaluations": 4,
            "gradient_evaluations": 4,
            "hessian_evaluations": 3,
            "hvp_evaluations": 0,
            "preconditioner_setups": 3,
            "notes": "Counters cover the declared timing region.",
        },
        "timing": {
            "aggregation": "rank-maximum",
            "cold_process": False,
            "barrier_policy": "MPI barriers before and after total region",
            "synchronization_policy": "JAX block-until-ready before each stop",
            "phases_overlap": False,
            "relation_to_total": "phases are subsets; unattributed time is total minus their sum",
            "process_startup_s": 0.1,
            "jit_compilation_s": 0.2,
            "coloring_s": 0.0,
            "derivative_evaluation_s": 0.3,
            "constitutive_contraction_s": 0.0,
            "assembly_s": 0.4,
            "communication_s": 0.1,
            "preconditioner_setup_s": 0.2,
            "krylov_solve_s": 0.5,
            "globalization_s": 0.1,
            "state_output_s": 0.1,
            "total_s": 2.0,
            "notes": "Launcher wall time is recorded separately in the raw log.",
        },
        "resources": {
            "nodes": 1,
            "ranks": 2,
            "threads_per_rank": 1,
            "peak_memory_per_rank_bytes": 1_000_000,
            "peak_memory_per_node_bytes": 2_000_000,
            "tracked_allocations_bytes": None,
            "measurement_method": "scheduler high-water mark and per-rank RSS",
            "notes": "Tracked allocations are not available in this runner.",
        },
        "diagnostics": {
            "state": {"weighted_difference": 1.0e-12},
            "branch": {},
            "feasibility": {},
            "kkt": {},
        },
        "environment": {
            "python": "3.12.9",
            "packages": {"jax": "0.4.38", "petsc4py": "3.22.2"},
            "platform": "Linux",
            "jax": "0.4.38",
            "xla": "0.4.38",
            "jax_enable_x64": True,
            "petsc": "3.22.2",
            "mpi": "Open MPI 4.1",
            "compiler": "GCC 13",
            "blas": "OpenBLAS",
            "cpu_model": "test CPU",
            "node_model": "test node",
            "memory_model": "128 GiB/node",
            "scheduler": "local",
            "scheduler_job_id": None,
            "affinity": "one thread per rank",
        },
        "provenance": {
            "git_commit": "a" * 40,
            "git_clean": True,
            "git_status_porcelain": [],
            "pilot_override": False,
            "pilot_override_reason": None,
            "command_argv": ["python", "runner.py", "--case", "case-a"],
            "working_directory": "/repo",
            "code_hashes": {"runner.py": "b" * 64},
            "configuration_hashes": {"card.json": "c" * 64},
            "input_hashes": {"mesh.h5": "d" * 64},
            "dirty_patch_sha256": None,
            "seed": 1729,
            "deterministic_policy": "fixed seed and deterministic route order block",
            "recorded_at_utc": "2026-07-10T08:00:03Z",
        },
        "artifacts": {
            "raw_outputs": ["raw/result.json"],
            "states": ["states/final.npz"],
            "logs": ["logs/run.log"],
            "tables": [],
            "figures": [],
            "reports": [],
        },
    }


@pytest.mark.parametrize("status", ["success", "failure", "capped", "timeout"])
def test_validator_accepts_all_terminal_statuses(status: str) -> None:
    rr.validate_run_record(_record(status=status))


def test_exported_schema_versions_every_required_section() -> None:
    schema = rr.RUN_RECORD_JSON_SCHEMA
    assert schema["$id"].endswith(f":v{rr.RUN_RECORD_SCHEMA_VERSION}")
    assert set(schema["required"]) == set(rr.TOP_LEVEL_FIELDS)
    for section, required in rr.SECTION_FIELDS.items():
        assert set(schema["properties"][section]["required"]) == set(required)
    assert set(schema["properties"]["termination"]["properties"]["status"]["enum"]) == {
        "success",
        "failure",
        "capped",
        "timeout",
    }


def test_validator_reports_required_and_status_specific_errors() -> None:
    record = _record(status="timeout")
    del record["resources"]["peak_memory_per_node_bytes"]
    record["termination"]["limit_value"] = None
    record["accuracy"]["gate_passed"] = True

    with pytest.raises(rr.RunRecordValidationError) as exc_info:
        rr.validate_run_record(record)

    message = str(exc_info.value)
    assert "record.resources.peak_memory_per_node_bytes is required" in message
    assert "limit_value is required" in message
    assert "cannot be true" in message


def test_publication_boundary_rejects_pilot_record() -> None:
    record = _record(run_kind="pilot")
    record["provenance"].update(
        {
            "git_clean": False,
            "git_status_porcelain": [" M runner.py"],
            "pilot_override": True,
            "pilot_override_reason": "quick local schema smoke test",
            "dirty_patch_sha256": "e" * 64,
        }
    )

    rr.validate_run_record(record)
    with pytest.raises(rr.RunRecordValidationError, match="publication ingestion boundary"):
        rr.validate_run_record(record, require_publication_ready=True)


def test_publication_label_cannot_disable_clean_tree_validation() -> None:
    record = _record()
    record["provenance"].update(
        {
            "git_clean": False,
            "git_status_porcelain": [" M runner.py"],
            "pilot_override": True,
            "pilot_override_reason": "incorrectly labeled dirty run",
            "dirty_patch_sha256": "e" * 64,
        }
    )

    with pytest.raises(rr.RunRecordValidationError, match="clean worktree"):
        rr.validate_run_record(record, require_publication_ready=False)


def test_atomic_write_json_does_not_clobber_previous_checkpoint_on_serialization_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.json"
    rr.atomic_write_json(path, {"sequence": 1})

    with pytest.raises(ValueError):
        rr.atomic_write_json(path, {"sequence": 2, "bad": float("nan")})

    assert json.loads(path.read_text(encoding="utf-8")) == {"sequence": 1}
    assert list(tmp_path.glob(".checkpoint.json.*.tmp")) == []


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_raw_solver_history_writer_maps_optional_nonfinite_values_to_json_null(
    tmp_path: Path,
    status: str,
) -> None:
    path = tmp_path / f"{status}.json"
    payload = {
        "status": status,
        "history": [
            {
                "it": 0,
                "energy": -2.5,
                "step_rel": float("nan"),
                "trial_ratio": float("inf") if status == "failed" else 0.75,
            },
            {
                "it": 1,
                "energy": -3.0,
                "step_rel": 1.25e-4,
                "trial_ratio": float("-inf") if status == "failed" else 0.9,
            },
        ],
    }

    rr.atomic_write_json(path, payload, nonfinite_as_null=True)

    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw
    written = json.loads(raw)
    assert written["history"][0]["step_rel"] is None
    assert written["history"][0]["energy"] == -2.5
    assert written["history"][1]["step_rel"] == 1.25e-4
    if status == "failed":
        assert written["history"][0]["trial_ratio"] is None
        assert written["history"][1]["trial_ratio"] is None
    else:
        assert written["history"][0]["trial_ratio"] == 0.75
        assert written["history"][1]["trial_ratio"] == 0.9


def test_atomic_write_json_cleans_temporary_file_if_replace_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.json"
    path.write_text('{"sequence": 1}\n', encoding="utf-8")

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(rr.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        rr.atomic_write_json(path, {"sequence": 2})

    assert json.loads(path.read_text(encoding="utf-8")) == {"sequence": 1}
    assert list(tmp_path.glob(".checkpoint.json.*.tmp")) == []


def test_checkpoint_writer_adds_version_and_sequence(tmp_path: Path) -> None:
    path = tmp_path / "progress_latest.json"
    payload = rr.atomic_write_checkpoint(
        path,
        record_id="run-1",
        sequence=7,
        progress={"nonlinear_iteration": 12, "last_scaled_residual": 0.02},
        written_at_utc="2026-07-10T08:01:00Z",
    )

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert payload["schema"] == {"id": rr.CHECKPOINT_SCHEMA_ID, "version": 1}
    assert payload["sequence"] == 7


def _init_repository(path: Path) -> str:
    subprocess.run(["git", "init", str(path)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test User"], check=True)
    (path / "runner.py").write_text("print('baseline')\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "runner.py"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "baseline"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def test_preflight_requires_clean_publication_commit_and_labels_dirty_pilot(tmp_path: Path) -> None:
    commit = _init_repository(tmp_path)
    clean = rr.check_experiment_preflight(tmp_path)
    assert clean.run_kind == "publication"
    assert clean.git_commit == commit
    assert clean.git_clean is True
    assert clean.git_status_porcelain == ()

    (tmp_path / "runner.py").write_text("print('pilot')\n", encoding="utf-8")
    with pytest.raises(rr.ExperimentPreflightError, match="empty git status"):
        rr.check_experiment_preflight(tmp_path)
    with pytest.raises(rr.ExperimentPreflightError, match="pilot_dirty_override"):
        rr.check_experiment_preflight(tmp_path, run_kind="pilot")
    with pytest.raises(rr.ExperimentPreflightError, match="non-empty reason"):
        rr.check_experiment_preflight(
            tmp_path,
            run_kind="pilot",
            pilot_dirty_override=True,
        )

    pilot = rr.check_experiment_preflight(
        tmp_path,
        run_kind="pilot",
        pilot_dirty_override=True,
        pilot_override_reason="local diagnostic before committing the runner",
    )
    assert pilot.git_clean is False
    assert pilot.pilot_override is True
    assert pilot.pilot_override_reason == "local diagnostic before committing the runner"
    assert pilot.git_status_porcelain == (" M runner.py",)


def test_atomic_run_record_writer_validates_before_replacing(tmp_path: Path) -> None:
    path = tmp_path / "run_record.json"
    valid = _record()
    rr.atomic_write_run_record(path, valid)
    previous = path.read_text(encoding="utf-8")

    invalid = deepcopy(valid)
    invalid["timing"]["total_s"] = float("inf")
    with pytest.raises(rr.RunRecordValidationError):
        rr.atomic_write_run_record(path, invalid)

    assert path.read_text(encoding="utf-8") == previous
