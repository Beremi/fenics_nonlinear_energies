from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX = REPO_ROOT / "experiments/runners/paper_revision_karolina/campaign_matrix.csv"
EXECUTOR = REPO_ROOT / "experiments/runners/paper_revision_karolina/execute_case.py"
ANALYZER = REPO_ROOT / "experiments/analysis/analyze_plasticity3d_discretization.py"
ACCOUNTING_COLLECTOR = REPO_ROOT / "experiments/analysis/collect_slurm_accounting.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


accounting_collector = _load(ACCOUNTING_COLLECTOR, "disc_accounting_collector")


def _disc_rows() -> list[dict[str, str]]:
    with MATRIX.open(newline="", encoding="utf-8") as handle:
        return [
            dict(row)
            for row in csv.DictReader(handle)
            if row["experiment_id"] == "EXP-DISC-001"
        ]


def test_executor_propagates_mandatory_quadrature_evaluator_failure(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load(EXECUTOR, "disc_executor_failure_test")
    row = _disc_rows()[0]
    calls = 0

    def fake_run(command, *, stdout, stderr, timeout_s):
        nonlocal calls
        calls += 1
        if calls == 1:
            run_dir = stdout.parent
            (run_dir / "state.npz").write_bytes(b"state")
            (run_dir / "output.json").write_text("{}\n", encoding="utf-8")
            return {"returncode": 0, "timed_out": False, "wall_time_s": 1.0}
        return {"returncode": 17, "timed_out": False, "wall_time_s": 2.0}

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(
        module,
        "validate_p3d_solve_output",
        lambda payload, planned: {"status": "passed"},
    )
    records = module.execute(row, out_root=tmp_path, python="python")
    assert calls == 2
    assert records[0]["process_returncode"] == 0
    assert records[0]["returncode"] == 86
    assert records[0]["quadrature_reference"]["returncode"] == 17
    assert records[0]["quadrature_reference_validation"]["status"] == "failed"


def _content_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(run_dir: Path, name: str, values: np.ndarray) -> dict[str, object]:
    path = run_dir / "quadrature_vectors" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values, allow_pickle=False)
    return {
        "path": str(path.relative_to(run_dir)),
        "sha256": _file_sha256(path),
        "content_sha256": _content_sha256(values),
        "dtype": str(values.dtype),
        "shape": list(values.shape),
        "content": "synthetic unit-test vector",
    }


def _evaluation(
    run_dir: Path,
    rule: str,
    *,
    residual_norm: float,
    scalar_shift: float,
) -> dict[str, object]:
    points = 24 if rule == "tetra_24point" else 125
    residual = np.full(6, residual_norm / np.sqrt(6.0), dtype=np.float64)
    action = np.arange(1.0, 7.0, dtype=np.float64)
    branch_map = np.zeros(points, dtype=np.int8)
    residual_artifact = _artifact(run_dir, f"{rule}_residual.npy", residual)
    action_artifact = _artifact(run_dir, f"{rule}_hessian_action.npy", action)
    branch_artifact = _artifact(run_dir, f"{rule}_branch_map.npy", branch_map)
    return {
        "quadrature_rule_id": rule,
        "quadrature_points": points,
        "quadrature_points_per_element": points,
        "element_degree": 4,
        "elements": 1,
        "degrees_of_freedom": 6,
        "free_degrees_of_freedom": 6,
        "lambda_target": 1.55,
        "internal_energy": 2.0 + scalar_shift,
        "external_work": 1.0,
        "total_potential_energy": 1.0 + scalar_shift,
        "u_max": 0.1,
        "full_residual_l2_norm": float(np.linalg.norm(residual)),
        "full_residual_linf_norm": float(np.linalg.norm(residual, ord=np.inf)),
        "free_residual_l2_norm": float(np.linalg.norm(residual)),
        "free_residual_linf_norm": float(np.linalg.norm(residual, ord=np.inf)),
        "full_hessian_action_l2_norm": float(np.linalg.norm(action)),
        "full_hessian_action_linf_norm": float(np.linalg.norm(action, ord=np.inf)),
        "free_hessian_action_l2_norm": float(np.linalg.norm(action)),
        "free_hessian_action_linf_norm": float(np.linalg.norm(action, ord=np.inf)),
        "branch_point_counts": {
            "elastic": points,
            "shear": 0,
            "left_edge": 0,
            "right_edge": 0,
            "apex": 0,
        },
        "branch_point_fractions": {
            "elastic": 1.0,
            "shear": 0.0,
            "left_edge": 0.0,
            "right_edge": 0.0,
            "apex": 0.0,
        },
        "branch_sample_points": points,
        "branch_margin_gate": 1.0e-8,
        "quadrature_points_at_or_below_margin_gate": 0,
        "minimum_normalized_active_branch_margin": 1.0e-4,
        "minimum_raw_principal_value_gap": 1.0e-4,
        "minimum_normalized_principal_value_gap": 1.0e-4,
        "minimum_normalized_constitutive_denominator": 0.5,
        "quadrature_weights_are_strictly_positive": True,
        "residual_content_sha256": residual_artifact["content_sha256"],
        "hessian_action_content_sha256": action_artifact["content_sha256"],
        "branch_map_content_sha256": branch_artifact["content_sha256"],
        "residual_artifact": residual_artifact,
        "hessian_action_artifact": action_artifact,
        "branch_map_artifact": branch_artifact,
    }


def _sacct_text(row: dict[str, str], job_id: str) -> str:
    fields = list(accounting_collector.SACCT_FIELDS)
    allocation = {
        "JobIDRaw": job_id,
        "JobID": job_id,
        "JobName": row["case_id"],
        "Cluster": "karolina",
        "Account": "fta-26-40",
        "Partition": row["partition"],
        "QOS": "3571_6328",
        "State": "COMPLETED",
        "ElapsedRaw": "120",
        "AllocNodes": row["nodes"],
        "AllocCPUS": row["total_ranks"],
        "TotalCPU": "00:01:00",
        "CPUTimeRAW": str(120 * int(row["total_ranks"])),
        "MaxRSS": "",
        "MaxVMSize": "",
        "ConsumedEnergyRaw": "100",
        "ExitCode": "0:0",
        "Start": "2026-07-10T10:00:00",
        "End": "2026-07-10T10:02:00",
        "NodeList": "cn001",
    }
    batch = {
        **allocation,
        "JobIDRaw": f"{job_id}.batch",
        "JobID": f"{job_id}.batch",
        "JobName": "batch",
        "ElapsedRaw": "119",
        "MaxRSS": "2G",
        "MaxVMSize": "3G",
    }
    lines = ["|".join(fields) + "|"]
    for item in (allocation, batch):
        lines.append("|".join(str(item[field]) for field in fields) + "|")
    return "\n".join(lines) + "\n"


def _write_case(
    root: Path,
    row: dict[str, str],
    *,
    commit: str,
    residual_norm: float,
) -> None:
    job_id = str(1000 + _disc_rows().index(row))
    job = root / "cases" / row["case_id"] / f"job_{job_id}"
    batch_job = root / "jobs" / row["case_id"] / f"job_{job_id}"
    run_dir = job / "measure_01"
    run_dir.mkdir(parents=True)
    batch_job.mkdir(parents=True)
    (root / "slurm").mkdir(exist_ok=True)
    (job / "matrix_row.json").write_text(json.dumps(row) + "\n", encoding="utf-8")
    metadata = {
        "case_id": row["case_id"],
        "job_id": job_id,
        "job_name": row["case_id"],
        "account": "fta-26-40",
        "nodes": row["nodes"],
        "ntasks": row["total_ranks"],
        "cpus_per_task": "1",
        "nodelist": "cn001",
        "submit_dir": str(REPO_ROOT),
        "matrix": str(MATRIX.resolve()),
        "matrix_sha256": _file_sha256(MATRIX),
        "git_commit": commit,
        "git_dirty": "false",
        "env_setup": "",
        "env_setup_sha256": "not_applicable",
        "allocation_revalidated": "YES",
        "account_qos_revalidated": "YES",
        "allocation_valid_until": "2026-12-31",
        "started_at": "2026-07-10T10:00:00+00:00",
        "finished_at": "2026-07-10T10:02:00+00:00",
        "accounting_status": "pending_post_job_collection",
    }
    (batch_job / "job_metadata.env").write_text(
        "".join(f"{key}={value}\n" for key, value in metadata.items()),
        encoding="utf-8",
    )
    (batch_job / "environment.txt").write_text(
        "PETSc 3.24 fixture\n", encoding="utf-8"
    )
    (batch_job / "execute.log").write_text("", encoding="utf-8")
    (root / "slurm" / f"{row['case_id']}-{job_id}.out").write_text(
        "completed\n", encoding="utf-8"
    )
    (root / "slurm" / f"{row['case_id']}-{job_id}.err").write_text(
        "", encoding="utf-8"
    )
    raw_accounting = batch_job / "sacct_raw.txt"
    raw_accounting.write_text(_sacct_text(row, job_id), encoding="utf-8")
    accounting = accounting_collector.collect_accounting(
        job_id=job_id,
        sacct_file=raw_accounting,
        collected_at_utc="2026-07-10T10:03:00+00:00",
    )
    (batch_job / "sacct_final.json").write_text(
        json.dumps(accounting) + "\n", encoding="utf-8"
    )
    state_path = run_dir / "state.npz"
    coords = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float64)
    final = coords + np.asarray(
        [[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float64
    )
    displacement = final - coords
    free = displacement.reshape(-1)
    np.savez(
        state_path,
        coords_ref=coords,
        coords_final=final,
        displacement=displacement,
        tetrahedra=np.asarray([[0, 1]], dtype=np.int32),
        surface_faces=np.empty((0, 2), dtype=np.int32),
        boundary_label=np.empty(0, dtype=np.int32),
        free_displacement_reordered=free,
        reference_elastic_action=free,
        reference_elastic_state_quadratic=float(np.dot(free, free)),
        mesh_name=np.asarray(row["mesh_name"]),
        element_degree=np.asarray(4),
        quadrature_rule_id=np.asarray(row["quadrature_rule"]),
        constraint_variant=np.asarray("glued_bottom"),
        assembly_backend=np.asarray("local_constitutiveAD"),
        mpi_ranks=np.asarray(int(row["total_ranks"])),
        lambda_target=np.asarray(1.55),
        energy=np.asarray(1.0),
    )
    payload = {
        "status": "completed",
        "solver_success": True,
        "assembly_backend": "local_constitutiveAD",
        "solver_backend": "local_pmg",
        "mesh_name": row["mesh_name"],
        "elem_degree": 4,
        "quadrature_rule_id": row["quadrature_rule"],
        "quadrature_points": 24 if row["quadrature_rule"] == "tetra_24point" else 125,
        "constraint_variant": "glued_bottom",
        "pmg_strategy": "same_mesh_p4_p2_p1",
        "ranks": int(row["total_ranks"]),
        "maxit": int(row["maxit"]),
        "ksp_rtol": float(row["ksp_rtol"]),
        "ksp_max_it": int(row["ksp_max_it"]),
        "stop_tol": float(row["stop_tol"]),
        "grad_stop_tol": float(row["grad_stop_tol"]),
        "line_search": "armijo",
        "linesearch_tol": 1.0e-3,
        "use_trust_region": True,
        "trust_subproblem_line_search": True,
        "lambda_target": 1.55,
        "energy": 1.0,
        "final_grad_norm": residual_norm,
        "state_out": state_path.name,
        "git": {"commit": commit, "dirty": False},
        "job_metadata": {"slurm_job_id": job.name.removeprefix("job_")},
        "nonlinear_convergence": {
            "absolute_dual_residual": {"value": 0.1 * float(row["grad_stop_tol"])},
            "relative_correction": {"value": 0.1 * float(row["stop_tol"])},
        },
    }
    (run_dir / "output.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    q24 = _evaluation(run_dir, "tetra_24point", residual_norm=residual_norm, scalar_shift=0.0)
    q125 = _evaluation(run_dir, "tetra_duffy_125point", residual_norm=residual_norm, scalar_shift=0.0)
    reference = {
        "experiment_id": "EXP-DISC-001-P3D-FIXED-STATE-QUADRATURE",
        "status": "completed",
        "state_path": state_path.name,
        "mesh_name": row["mesh_name"],
        "element_degree": 4,
        "constraint_variant": "glued_bottom",
        "solve_quadrature_rule_id": row["quadrature_rule"],
        "lambda_target": 1.55,
        "reference_rule_id": "tetra_duffy_125point",
        "common_direction_content_sha256": "a" * 64,
        "common_free_dof_set": True,
        "evaluations": [q24, q125],
    }
    (run_dir / "quadrature_reference.json").write_text(
        json.dumps(reference) + "\n", encoding="utf-8"
    )
    record = {
        "kind": "measure",
        "index": 1,
        "returncode": 0,
        "scientific_validation": {"status": "passed"},
        "quadrature_reference": {"returncode": 0, "timed_out": False},
        "quadrature_reference_validation": {"status": "passed"},
    }
    (job / "run_records.json").write_text(json.dumps([record]) + "\n", encoding="utf-8")


def _campaigns(tmp_path: Path) -> list[Path]:
    rows = _disc_rows()
    commit = "1" * 40
    by_tier: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_tier.setdefault(row["tier"], []).append(row)
    roots = []
    for tier, tier_rows in by_tier.items():
        root = (tmp_path / tier).resolve()
        root.mkdir()
        plan = root / "prepared_plan.csv"
        commands = root / "sbatch_commands.txt"
        freeze = root / "reviewed_source_freeze.json"
        plan.write_text("case_id\n" + "\n".join(row["case_id"] for row in tier_rows) + "\n", encoding="utf-8")
        commands.write_text("\n".join(f"sbatch {row['case_id']}" for row in tier_rows) + "\n", encoding="utf-8")
        freeze.write_text('{"synthetic_test_freeze":true}\n', encoding="utf-8")
        stage_order = ("smoke", "quadrature", "mesh", "mesh_quadrature", "tolerance")
        position = stage_order.index(tier) + 1
        manifest = {
            "manifest_version": 1,
            "status": "submitted",
            "cluster": "Karolina CPU",
            "account": "fta-26-40",
            "qos": "3571_6328",
            "matrix": str(MATRIX.resolve().relative_to(REPO_ROOT)),
            "matrix_sha256": hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
            "selected_experiments": ["EXP-DISC-001"],
            "selected_tiers": [tier],
            "include_optional": False,
            "only_optional": False,
            "case_count": len(tier_rows),
            "test_only_commands": False,
            "source_commit": commit,
            "source_dirty": False,
            "out_root": ".",
            "plan_file": plan.name,
            "plan_sha256": _file_sha256(plan),
            "commands_file": commands.name,
            "commands_sha256": _file_sha256(commands),
            "queued_source_freeze": {
                "path": freeze.name,
                "sha256": _file_sha256(freeze),
            },
            "disc_release_stage": {
                "unit": "protocol_stage",
                "stage": tier,
                "position": position,
                "stage_count": len(stage_order),
                "case_count": len(tier_rows),
                "prerequisite_stage": None if position == 1 else stage_order[position - 2],
                "later_stage_release_requires_separate_human_authorization": True,
            },
        }
        (root / "prepared_manifest.json").write_text(
            json.dumps(manifest) + "\n", encoding="utf-8"
        )
        for row in tier_rows:
            residual = 5.0e-6 if row["tier"] == "tolerance" else 5.0e-5
            _write_case(root, row, commit=commit, residual_norm=residual)
        roots.append(root)
    return roots


def test_discretization_adjudicator_admits_complete_clean_six_row_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load(ANALYZER, "disc_analysis_pass_test")
    monkeypatch.setattr(
        module,
        "validate_p3d_solve_output",
        lambda payload, row: {"status": "passed"},
    )
    result = module.analyze(MATRIX, _campaigns(tmp_path))
    assert result["all_six_rows_admitted"] is True
    assert result["publication_evidence_valid"] is True
    assert result["terminal_decision"] == "VERIFIED_POLICY"
    assert len(result["cases"]) == 6
    first_evidence = result["cases"][0]["job_evidence"]
    assert set(first_evidence["artifacts"]) == {
        "job_metadata",
        "environment",
        "execute_log",
        "accounting",
        "slurm_stdout",
        "slurm_stderr",
    }
    for artifact in first_evidence["artifacts"].values():
        path = Path(artifact["path"])
        assert artifact["sha256"] == _file_sha256(path)
        assert artifact["bytes"] == path.stat().st_size
    assert first_evidence["accounting"]["cluster"] == "karolina"
    assert first_evidence["accounting"]["state"] == "COMPLETED"
    assert first_evidence["accounting"]["exit_code"] == "0:0"
    assert result["comparisons"]["l1_quadrature"]["common_scalar_gate_passed"] is True
    assert result["comparisons"]["l1_quadrature"]["branch_map_changed_samples"] == 0


def test_discretization_manifests_and_internal_records_survive_copy_back(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load(ANALYZER, "disc_analysis_relocated_test")
    monkeypatch.setattr(
        module,
        "validate_p3d_solve_output",
        lambda payload, row: {"status": "passed"},
    )
    cluster_parent = tmp_path / "cluster"
    cluster_parent.mkdir()
    roots = _campaigns(cluster_parent)
    copied_parent = tmp_path / "copied_back"
    copied_parent.mkdir()
    copied = []
    for root in roots:
        destination = copied_parent / f"renamed_{root.name}"
        root.rename(destination)
        copied.append(destination)
    result = module.analyze(MATRIX, copied)
    assert result["all_six_rows_admitted"] is True
    assert result["publication_evidence_valid"] is True


def test_discretization_adjudicator_rejects_dirty_or_failed_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load(ANALYZER, "disc_analysis_reject_test")
    monkeypatch.setattr(
        module,
        "validate_p3d_solve_output",
        lambda payload, row: {"status": "passed"},
    )
    roots = _campaigns(tmp_path)
    target = roots[0] / "prepared_manifest.json"
    manifest = json.loads(target.read_text(encoding="utf-8"))
    manifest["source_dirty"] = True
    target.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    try:
        module.analyze(MATRIX, roots)
    except module.AdmissionError as exc:
        assert "dirty" in str(exc)
    else:
        raise AssertionError("dirty submitted manifest was admitted")


@pytest.mark.parametrize(
    ("mutation", "reason_fragment"),
    (
        ("missing_reference", "missing quadrature_reference.json"),
        ("failed_wrapper", "wrapper or mandatory postprocessor failed"),
        ("dirty_output", "clean submitted commit"),
        ("stale_matrix_row", "executed matrix row is stale"),
    ),
)
def test_discretization_adjudicator_rejects_incomplete_or_stale_case_records(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
    reason_fragment: str,
) -> None:
    module = _load(ANALYZER, f"disc_analysis_{mutation}_test")
    monkeypatch.setattr(
        module,
        "validate_p3d_solve_output",
        lambda payload, row: {"status": "passed"},
    )
    roots = _campaigns(tmp_path)
    root = next(path for path in roots if path.name == "quadrature")
    job = next((root / "cases" / "disc_p4l1_q24_np64").glob("job_*"))
    if mutation == "missing_reference":
        (job / "measure_01" / "quadrature_reference.json").unlink()
    elif mutation == "failed_wrapper":
        path = job / "run_records.json"
        records = json.loads(path.read_text(encoding="utf-8"))
        records[0]["returncode"] = 86
        path.write_text(json.dumps(records) + "\n", encoding="utf-8")
    elif mutation == "dirty_output":
        path = job / "measure_01" / "output.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["git"]["dirty"] = True
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    else:
        path = job / "matrix_row.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        row["grad_stop_tol"] = "9e-9"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = module.analyze(MATRIX, roots)
    assert result["all_six_rows_admitted"] is False
    assert result["terminal_decision"] == "INVALID"
    failure = next(
        item for item in result["case_failures"] if item["case_id"] == "disc_p4l1_q24_np64"
    )
    assert reason_fragment in failure["reason"]


@pytest.mark.parametrize(
    ("mutation", "reason_fragment"),
    (
        ("missing_metadata", "required job metadata is missing"),
        ("missing_environment", "required environment is missing"),
        ("missing_execute_log", "required execute log is missing"),
        ("missing_accounting", "required accounting is missing"),
        ("missing_slurm_stderr", "required slurm stderr is missing"),
        ("stale_accounting_hash", "raw accounting hash is stale"),
        ("wrong_account", "allocation account differs"),
        ("failed_job", "allocation state differs"),
    ),
)
def test_discretization_adjudicator_requires_complete_hash_bound_job_evidence(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
    reason_fragment: str,
) -> None:
    module = _load(ANALYZER, f"disc_job_evidence_{mutation}_test")
    monkeypatch.setattr(
        module,
        "validate_p3d_solve_output",
        lambda payload, row: {"status": "passed"},
    )
    roots = _campaigns(tmp_path)
    root = next(path for path in roots if path.name == "quadrature")
    row = next(item for item in _disc_rows() if item["case_id"] == "disc_p4l1_q24_np64")
    batch_job = next((root / "jobs" / row["case_id"]).glob("job_*"))
    job_id = batch_job.name.removeprefix("job_")
    if mutation == "missing_metadata":
        (batch_job / "job_metadata.env").unlink()
    elif mutation == "missing_environment":
        (batch_job / "environment.txt").unlink()
    elif mutation == "missing_execute_log":
        (batch_job / "execute.log").unlink()
    elif mutation == "missing_accounting":
        (batch_job / "sacct_final.json").unlink()
    elif mutation == "missing_slurm_stderr":
        (root / "slurm" / f"{row['case_id']}-{job_id}.err").unlink()
    elif mutation == "stale_accounting_hash":
        path = batch_job / "sacct_final.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["source"]["raw_parsable2"] += "\n"
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    else:
        raw_path = batch_job / "sacct_raw_mutated.txt"
        raw = _sacct_text(row, job_id)
        if mutation == "wrong_account":
            raw = raw.replace("fta-26-40", "wrong-account")
        else:
            raw = raw.replace("COMPLETED", "FAILED").replace("0:0", "1:0")
        raw_path.write_text(raw, encoding="utf-8")
        accounting = accounting_collector.collect_accounting(
            job_id=job_id,
            sacct_file=raw_path,
            collected_at_utc="2026-07-10T10:03:00+00:00",
        )
        (batch_job / "sacct_final.json").write_text(
            json.dumps(accounting) + "\n", encoding="utf-8"
        )
    result = module.analyze(MATRIX, roots)
    assert result["all_six_rows_admitted"] is False
    failure = next(
        item for item in result["case_failures"] if item["case_id"] == row["case_id"]
    )
    assert reason_fragment in failure["reason"]
