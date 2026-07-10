from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYZER_PATH = REPO_ROOT / "experiments/analysis/analyze_exp_scale_001.py"
COLLECTOR_PATH = REPO_ROOT / "experiments/analysis/collect_slurm_accounting.py"
CONTRACT_PATH = REPO_ROOT / "paper/protocols/EXP-SCALE-001-analysis-contract.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyzer = _load(ANALYZER_PATH, "exp_scale_analyzer")
collector = _load(COLLECTOR_PATH, "slurm_accounting_collector")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sacct_text(job_id: str, *, nodes: int, cpus: int, partition: str) -> str:
    fields = list(collector.SACCT_FIELDS)
    allocation = {
        "JobIDRaw": job_id,
        "JobID": job_id,
        "JobName": f"scale-{job_id}",
        "Cluster": "karolina",
        "Account": "fta-26-40",
        "Partition": partition,
        "QOS": "3571_6328",
        "State": "COMPLETED",
        "ElapsedRaw": "120",
        "AllocNodes": str(nodes),
        "AllocCPUS": str(cpus),
        "TotalCPU": "00:03:00",
        "CPUTimeRAW": str(120 * cpus),
        "MaxRSS": "",
        "MaxVMSize": "",
        "ConsumedEnergyRaw": "1200",
        "ExitCode": "0:0",
        "Start": "2026-07-10T10:00:00",
        "End": "2026-07-10T10:02:00",
        "NodeList": "cn[001-008]",
    }
    batch = {
        **allocation,
        "JobIDRaw": f"{job_id}.batch",
        "JobID": f"{job_id}.batch",
        "JobName": "batch",
        "ElapsedRaw": "119",
        "MaxRSS": "1.5G",
        "MaxVMSize": "2G",
    }
    lines = ["|".join(fields) + "|"]
    for row in (allocation, batch):
        lines.append("|".join(str(row[field]) for field in fields) + "|")
    return "\n".join(lines) + "\n"


def test_accounting_collector_parses_offline_snapshot_and_memory(tmp_path: Path) -> None:
    raw = tmp_path / "sacct.txt"
    raw.write_text(_sacct_text("123", nodes=2, cpus=256, partition="qcpu_exp"))
    payload = collector.collect_accounting(
        job_id="123",
        sacct_file=raw,
        collected_at_utc="2026-07-10T12:00:00+00:00",
    )
    assert payload["schema_id"] == collector.SCHEMA_ID
    assert payload["source"]["mode"] == "offline_file"
    assert payload["source"]["sha256"] == _sha256(raw)
    assert payload["allocation"]["alloc_nodes"] == 2
    assert payload["derived"]["allocated_node_seconds"] == 240
    assert payload["derived"]["allocated_cpu_seconds"] == 30720
    assert payload["derived"]["maximum_step_rss_bytes"] == round(1.5 * 1024**3)
    assert payload["derived"]["maximum_step_vm_size_bytes"] == 2 * 1024**3


def test_accounting_live_query_is_explicit_and_never_uses_a_shell() -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_runner(command, **kwargs):
        calls.append((list(command), dict(kwargs)))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=_sacct_text("456", nodes=1, cpus=128, partition="qcpu_exp"),
            stderr="",
        )

    payload = collector.collect_accounting(
        job_id="456",
        query_live=True,
        executable="/opt/slurm/bin/sacct",
        collected_at_utc="2026-07-10T12:00:00Z",
        runner=fake_runner,
    )
    assert payload["source"]["mode"] == "explicit_live_query"
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[0] == "/opt/slurm/bin/sacct"
    assert command[1:3] == ["--jobs", "456"]
    assert all(token not in command for token in ("sbatch", "srun", "scontrol"))
    assert "shell" not in kwargs
    assert kwargs == {"check": False, "capture_output": True, "text": True}


def test_accounting_collector_refuses_implicit_or_ambiguous_source(tmp_path: Path) -> None:
    raw = tmp_path / "sacct.txt"
    raw.write_text(_sacct_text("123", nodes=1, cpus=128, partition="qcpu_exp"))
    with pytest.raises(collector.AccountingError, match="exactly one"):
        collector.collect_accounting(job_id="123")
    with pytest.raises(collector.AccountingError, match="exactly one"):
        collector.collect_accounting(job_id="123", sacct_file=raw, query_live=True)


def _timing(ranks: int, first_step: float) -> dict[str, object]:
    phase_values = {
        "setup": first_step * 0.5,
        "first_step": first_step,
        "solve": first_step * 1.05,
        "total": first_step * 1.8,
    }
    return {
        "schema_id": "fenics-nonlinear-energies.mpi-phase-timing",
        "schema_version": 1,
        "reduction": "mpi_collective_max",
        "rank_count": ranks,
        "measured_region_excludes_reporting_collective": True,
        "phases": {
            phase: {
                "collective_max_s": value,
                "per_rank_s": [value] * ranks,
            }
            for phase, value in phase_values.items()
        },
    }


def _write_matrix(path: Path, series: dict[str, object]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for point in series["rank_node_points"]:
        nodes = int(point["nodes"])
        ranks = int(point["ranks"])
        row = {
            "experiment_id": "EXP-SCALE-001",
            "tier": str(series["tier"]),
            "case_id": f"scale_he_n{nodes}_np{ranks}",
            "optional": "0",
            "nodes": str(nodes),
            "ranks_per_node": "128",
            "total_ranks": str(ranks),
            "repetitions": str(series["repetitions_per_point"]),
            "warmups": "0",
            "partition": str(point["partition"]),
            "runner": str(series["runner"]),
            **{key: str(value) for key, value in series["matrix_scientific_fields"].items()},
        }
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _write_scale_campaign(tmp_path: Path) -> tuple[Path, Path]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    series = contract["series"]["required_he"]
    matrix = tmp_path / "matrix.csv"
    rows = _write_matrix(matrix, series)
    root = tmp_path / "campaign"
    (root / "slurm").mkdir(parents=True)
    commit = "a" * 40
    manifest = {
        "status": "submitted",
        "test_only_commands": False,
        "selected_experiments": ["EXP-SCALE-001"],
        "selected_tiers": [series["tier"]],
        "case_count": len(rows),
        "matrix_sha256": _sha256(matrix),
        "source_commit": commit,
        "source_dirty": False,
        "cluster": "Karolina CPU",
        "account": "fta-26-40",
        "qos": "3571_6328",
    }
    (root / "prepared_manifest.json").write_text(json.dumps(manifest) + "\n")
    ledger: list[str] = []
    case_fields = dict(series["output_case_fields"])
    coords = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    displacement = np.asarray(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]
    )
    cells = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
    for case_number, row in enumerate(rows, start=1):
        job_id = str(100 + case_number)
        case_id = row["case_id"]
        ledger.append(
            json.dumps(
                {
                    "case_id": case_id,
                    "returncode": 0,
                    "stdout": f"Submitted batch job {job_id}",
                    "stderr": "",
                }
            )
        )
        case_job = root / "cases" / case_id / f"job_{job_id}"
        batch_job = root / "jobs" / case_id / f"job_{job_id}"
        case_job.mkdir(parents=True)
        batch_job.mkdir(parents=True)
        (case_job / "matrix_row.json").write_text(json.dumps(row) + "\n")
        metadata = {
            "case_id": case_id,
            "job_id": job_id,
            "account": "fta-26-40",
            "qos": "3571_6328",
            "partition": row["partition"],
            "cluster": "karolina",
            "nodes": row["nodes"],
            "ntasks": row["total_ranks"],
            "matrix_sha256": _sha256(matrix),
            "git_commit": commit,
            "git_dirty": "false",
            "allocation_revalidated": "YES",
            "account_qos_revalidated": "YES",
            "allocation_valid_until": "2026-12-31",
        }
        (batch_job / "job_metadata.env").write_text(
            "".join(f"{key}={value}\n" for key, value in metadata.items())
        )
        (batch_job / "environment.txt").write_text("PETSc 3.24 fixture\n")
        raw = batch_job / "sacct_raw.txt"
        raw.write_text(
            _sacct_text(
                job_id,
                nodes=int(row["nodes"]),
                cpus=int(row["total_ranks"]),
                partition=row["partition"],
            )
        )
        accounting = collector.collect_accounting(
            job_id=job_id,
            sacct_file=raw,
            collected_at_utc="2026-07-10T12:00:00Z",
        )
        (batch_job / "sacct_final.json").write_text(json.dumps(accounting) + "\n")
        (root / "slurm" / f"{case_id}-{job_id}.out").write_text("completed\n")
        (root / "slurm" / f"{case_id}-{job_id}.err").write_text("")
        records: list[dict[str, object]] = []
        nodes = int(row["nodes"])
        ranks = int(row["total_ranks"])
        for repetition in range(1, 6):
            run_dir = case_job / f"measure_{repetition:02d}"
            run_dir.mkdir()
            command = f"srun -n {ranks} solver --out {run_dir}/output.json"
            (run_dir / "command.txt").write_text(command + "\n")
            records.append(
                {
                    "kind": "measure",
                    "index": repetition,
                    "command": command,
                    "returncode": 0,
                    "timed_out": False,
                }
            )
            first_step = (100.0 / nodes) * (1.0 + 0.01 * (repetition - 3))
            output = {
                "case": case_fields,
                "result": {
                    "metadata": {"nprocs": ranks},
                    "publication_timing": _timing(ranks, first_step),
                    "steps": [
                        {
                            "success": True,
                            "attempt": "primary",
                            "energy": 1.25,
                            "nit": 3,
                            "convergence": {
                                "dual_residual_gate_pass": True,
                                "dual_residual_norm": 1.0e-5,
                                "relative_correction": 1.0e-5,
                            },
                        }
                    ],
                },
            }
            (run_dir / "output.json").write_text(json.dumps(output) + "\n")
            np.savez(
                run_dir / "state.npz",
                coords_ref=coords,
                displacement=displacement,
                tetrahedra=cells,
                energy=1.25,
            )
        (case_job / "run_records.json").write_text(json.dumps(records) + "\n")
    (root / "submitted_jobs.jsonl").write_text("\n".join(ledger) + "\n")
    return root, matrix


def test_complete_he_series_admits_five_repetitions_and_scaling(tmp_path: Path) -> None:
    root, matrix = _write_scale_campaign(tmp_path)
    result = analyzer.analyze(campaign_root=root, matrix=matrix)
    assert result["status"] == "admitted_fixed_policy_viability"
    assert result["timing_claim_released"] is True
    assert result["series"] == "required_he"
    assert result["must_not_merge_with"] == "optional_p3d"
    assert result["admitted_repetitions"] == 20
    assert result["violations"] == []
    assert [row["nodes"] for row in result["scaling_statistics"]] == [1, 2, 4, 8]
    for row in result["scaling_statistics"]:
        assert row["repetitions"] == 5
        assert row["speedup"] == pytest.approx(float(row["nodes"]))
        assert row["efficiency"] == pytest.approx(1.0)
        assert row["efficiency_basis"] == "nodes_relative_to_one_node"


def test_timing_is_withheld_when_rank_vector_does_not_prove_maximum(tmp_path: Path) -> None:
    root, matrix = _write_scale_campaign(tmp_path)
    target = next((root / "cases").glob("*/job_*/measure_03/output.json"))
    payload = json.loads(target.read_text())
    payload["result"]["publication_timing"]["phases"]["first_step"][
        "collective_max_s"
    ] *= 1.1
    target.write_text(json.dumps(payload) + "\n")
    result = analyzer.analyze(campaign_root=root, matrix=matrix)
    assert result["status"] == "invalid_no_timing_claim"
    assert result["timing_claim_released"] is False
    assert result["admitted_repetitions"] == 0
    assert result["scaling_statistics"] == []
    assert any("not proved by rank values" in row["reason"] for row in result["violations"])


def test_optional_p3d_total_timing_adapter_stays_strict_and_separate() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    gates = contract["admission_gates"]
    maxima, vectors = analyzer._validate_timing(
        {
            "total_time_reduction": "mpi_collective_max",
            "total_time": 3.0,
            "total_time_by_rank_s": [2.5, 3.0],
        },
        ranks=2,
        gates=gates,
        required_phases=["total"],
    )
    assert maxima == {"total": 3.0}
    assert vectors == {"total": [2.5, 3.0]}
    with pytest.raises(analyzer.AdmissionError, match="not proved"):
        analyzer._validate_timing(
            {
                "total_time_reduction": "mpi_collective_max",
                "total_time": 2.5,
                "total_time_by_rank_s": [2.5, 3.0],
            },
            ranks=2,
            gates=gates,
            required_phases=["total"],
        )


def test_combined_required_and_optional_manifest_is_rejected(tmp_path: Path) -> None:
    root, matrix = _write_scale_campaign(tmp_path)
    manifest_path = root / "prepared_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["selected_tiers"].append("optional_fixed_policy_p3d")
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(analyzer.AdmissionError, match="separate tranches"):
        analyzer.analyze(campaign_root=root, matrix=matrix)


def test_he_timing_helper_retains_rank_values_and_exact_maxima() -> None:
    from src.problems.hyperelasticity.jax_petsc.solver import _collect_publication_timing

    class FakeComm:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 3

        def gather(self, _local, root=0):
            assert root == 0
            return [
                {"rank": 0, "setup_s": 1.0, "first_step_s": 4.0, "solve_s": 5.0, "total_s": 6.0},
                {"rank": 1, "setup_s": 2.0, "first_step_s": 3.0, "solve_s": 6.0, "total_s": 8.0},
                {"rank": 2, "setup_s": 1.5, "first_step_s": 5.0, "solve_s": 4.0, "total_s": 7.0},
            ]

    result = _collect_publication_timing(
        FakeComm(), setup_s=1.0, first_step_s=4.0, solve_s=5.0, total_s=6.0
    )
    assert result["reduction"] == "mpi_collective_max"
    assert result["rank_count"] == 3
    assert result["phases"]["first_step"] == {
        "collective_max_s": 5.0,
        "per_rank_s": [4.0, 3.0, 5.0],
    }
    assert result["phases"]["total"]["collective_max_s"] == 8.0
