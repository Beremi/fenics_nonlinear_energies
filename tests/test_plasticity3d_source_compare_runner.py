from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from experiments.runners import run_plasticity3d_p4_l1_lambda1p5_source_compare as runner


def test_normalize_local_payload_contract(tmp_path: Path) -> None:
    result_path = tmp_path / "local_output.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "failed",
                "message": "Maximum number of iterations reached",
                "total_time": 12.5,
                "solve_time": 10.0,
                "nit": 20,
                "linear_iterations_total": 123,
                "energy": -4.5,
                "omega": 2.0,
                "u_max": 0.25,
                "history": [
                    {"it": 1, "grad_norm": 10.0},
                    {"it": 20, "grad_norm": 0.5},
                ],
                "initial_guess": {"enabled": True, "success": True, "ksp_iterations": 14},
            }
        ),
        encoding="utf-8",
    )

    row = runner._normalize_local_payload(
        case_id="fixed_work:maintained_local:np4",
        mode="fixed_work",
        ranks=4,
        exit_code=0,
        fixed_maxit=20,
        case_dir=tmp_path,
        stdout_path=tmp_path / "stdout.txt",
        stderr_path=tmp_path / "stderr.txt",
        result_path=result_path,
        command=["python", "solve.py"],
    )

    assert set(row) == set(runner.NORMALIZED_ROW_KEYS)
    assert row["status"] == "completed_fixed_work"
    assert row["history_metric_name"] == "relative_grad_norm"
    assert row["history_iterations"] == [1, 20]
    assert row["history_metric"] == [1.0, 0.05]


def test_normalize_source_payload_contract(tmp_path: Path) -> None:
    result_path = tmp_path / "source_output.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "message": "Converged",
                "solver_success": True,
                "total_time": 9.0,
                "solve_time": 8.0,
                "nit": 7,
                "linear_iterations_total": 77,
                "final_metric": 1.0e-2,
                "final_metric_name": "relative_residual",
                "energy": -5.0,
                "omega": 2.1,
                "u_max": 0.3,
                "history_metric_name": "relative_residual",
                "history": [
                    {"iteration": 1, "metric": 1.0},
                    {"iteration": 7, "metric": 1.0e-2},
                ],
                "initial_guess": {"enabled": True, "success": True, "ksp_iterations": 11},
                "native_run_info": "run_info.json",
                "native_npz": "petsc_run.npz",
                "native_history_json": "history.json",
                "native_debug_bundle": "debug_bundle.h5",
                "native_vtu": "solution.vtu",
            }
        ),
        encoding="utf-8",
    )

    row = runner._normalize_source_payload(
        case_id="reference:source_petsc4py:np16",
        mode="reference",
        ranks=16,
        exit_code=0,
        case_dir=tmp_path,
        stdout_path=tmp_path / "stdout.txt",
        stderr_path=tmp_path / "stderr.txt",
        result_path=result_path,
        command=["python", "source.py"],
    )

    assert set(row) == set(runner.NORMALIZED_ROW_KEYS)
    assert row["status"] == "completed"
    assert row["history_iterations"] == [1, 7]
    assert row["history_metric"] == [1.0, 1.0e-2]


def test_normalize_local_reference_step_metric(tmp_path: Path) -> None:
    result_path = tmp_path / "local_reference_output.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "message": "Converged (energy, step, gradient)",
                "total_time": 11.0,
                "solve_time": 9.0,
                "nit": 3,
                "linear_iterations_total": 18,
                "energy": -4.5,
                "omega": 2.2,
                "u_max": 0.24,
                "history": [
                    {"it": 1, "step_rel": 1.2e-1},
                    {"it": 2, "step_rel": 2.4e-2},
                    {"it": 3, "step_rel": 9.0e-3},
                ],
                "initial_guess": {"enabled": True, "success": True, "ksp_iterations": 6},
            }
        ),
        encoding="utf-8",
    )

    row = runner._normalize_local_payload(
        case_id="reference:maintained_local:np8",
        mode="reference",
        ranks=8,
        exit_code=0,
        fixed_maxit=20,
        reference_metric_name="relative_correction",
        case_dir=tmp_path,
        stdout_path=tmp_path / "stdout.txt",
        stderr_path=tmp_path / "stderr.txt",
        result_path=result_path,
        command=["python", "solve.py"],
    )

    assert row["status"] == "completed"
    assert row["final_metric_name"] == "relative_correction"
    assert row["history_metric_name"] == "relative_correction"
    assert row["history_iterations"] == [1, 2, 3]
    assert row["history_metric"] == [1.2e-1, 2.4e-2, 9.0e-3]


def test_ensure_source_helper_writes_wrapper(tmp_path: Path) -> None:
    source_root = tmp_path / "slope_stability_petsc4py"
    helper = runner.ensure_source_helper(source_root)
    text = helper.read_text(encoding="utf-8")
    assert helper.exists()
    assert "source_fixed_lambda_3d_impl" in text


def test_build_source_command_enforces_single_thread_and_light_exports(tmp_path: Path) -> None:
    source_root = tmp_path / "slope_stability_petsc4py"
    helper = source_root / "scripts_local" / "run_fixed_lambda_3d.py"
    command = runner._build_source_command(
        source_root=source_root,
        source_python=runner.PYTHON,
        helper_path=helper,
        case_dir=tmp_path / "case",
        result_path=tmp_path / "case" / "output.json",
        ranks=4,
        mode="fixed_work",
        fixed_maxit=20,
    )

    assert "--threads" in command
    assert command[command.index("--threads") + 1] == "1"
    assert "--no-write-debug-bundle" in command
    assert "--write-history-json" in command
    assert "--no-write-solution-vtu" in command
    assert "--no-write-plots" in command


def test_build_source_command_supports_pmg_reference_mode(tmp_path: Path) -> None:
    source_root = tmp_path / "slope_stability_petsc4py"
    helper = source_root / "scripts_local" / "run_fixed_lambda_3d.py"
    command = runner._build_source_command(
        source_root=source_root,
        source_python=runner.PYTHON,
        helper_path=helper,
        case_dir=tmp_path / "case",
        result_path=tmp_path / "case" / "output.json",
        ranks=8,
        mode="reference",
        fixed_maxit=20,
        reference_stop_policy="matched_relative_correction",
        reference_stop_tol=2.0e-3,
        reference_maxit=80,
        source_pc_backend="pmg",
    )

    assert command[command.index("--pc-backend") + 1] == "pmg"
    assert command[command.index("--stopping-criterion") + 1] == "relative_correction"
    assert command[command.index("--stopping-tol") + 1] == "0.002"
    assert "mg_levels_ksp_type=chebyshev" in command
    assert "mg_levels_pc_type=jacobi" in command


def test_resolve_source_python_builds_missing_extension(
    monkeypatch, tmp_path: Path
) -> None:
    source_root = tmp_path / "slope_stability_petsc4py"
    state = {"built": False}
    calls: list[list[str]] = []

    def fake_run(cmd, cwd, env, check, capture_output, text):
        assert check is False
        assert capture_output is True
        assert text is True
        calls.append(list(cmd))
        payload = cmd[-1] if cmd else ""
        if "run_case_from_config" in payload:
            return SimpleNamespace(returncode=0, stdout="import_ok\n", stderr="")
        if "assemble_overlap_strain_3d" in payload:
            if state["built"]:
                return SimpleNamespace(returncode=0, stdout="compiled\n", stderr="")
            return SimpleNamespace(returncode=1, stdout="", stderr="missing compiled kernels")
        if len(cmd) >= 4 and cmd[1:] == ["setup.py", "build_ext", "--inplace"]:
            state["built"] = True
            return SimpleNamespace(returncode=0, stdout="built\n", stderr="")
        raise AssertionError(f"Unexpected subprocess invocation: {cmd}")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    python_bin, mode = runner.resolve_source_python(source_root)

    assert python_bin == runner.PYTHON
    assert mode == "shared_env_built_ext"
    assert any(call[1:] == ["setup.py", "build_ext", "--inplace"] for call in calls)
