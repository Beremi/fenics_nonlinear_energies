from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from experiments.analysis import analyze_plasticity3d_route_cost_model as analysis
from experiments.runners import run_route_factor_microbenchmarks as runner


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv/bin/python"
SCRIPT = REPO_ROOT / "experiments/runners/run_route_factor_microbenchmarks.py"
CONTRACT = REPO_ROOT / "paper/protocols/EXP-ROUTE-001-analysis-contract.json"


def test_factor_design_varies_one_declared_factor_at_a_time() -> None:
    rows = runner._design()
    baseline = rows[0]
    factor_names = {
        "element_dofs",
        "quadrature_points",
        "constitutive_dimension",
        "color_count",
        "nonzeros_per_row",
        "message_bytes",
        "imbalance_ratio",
    }
    assert baseline["case_id"] == "baseline"
    assert len(rows) == 16
    for row in rows[1:]:
        changed = {
            name for name in factor_names if row[name] != baseline[name]
        }
        assert len(changed) == 1


def test_factor_runner_emits_strict_collective_max_calibration(tmp_path: Path) -> None:
    output = tmp_path / "factor.json"
    subprocess.run(
        [
            str(PYTHON),
            "-u",
            str(SCRIPT),
            "--output",
            str(output),
            "--repetitions",
            "5",
            "--block-repetition",
            "1",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(
        output.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    assert payload["status"] == "completed"
    assert payload["scope"] == "synthetic_calibration_not_production_route_timing"
    assert payload["timing_reduction"] == "mpi_collective_max"
    assert len(payload["results"]) == 16
    for row in payload["results"]:
        assert row["timing_reduction"] == "mpi_collective_max"
        assert row["mpi_ranks"] == 1
        assert row["tracked_allocation_bytes_max"] > 0
        assert row["peak_rss_bytes_max"] > 0
        assert len(row["input_sha256_by_rank"]) == 1
        assert len(row["allocation_times_by_rank_s"]) == 1
        assert row["allocation_collective_max_s"] == max(
            row["allocation_times_by_rank_s"]
        )
        assert row["local_batch_by_rank"] == [8]
        assert row["insertion_rows_by_rank"] == [512]
        assert row["realized_batch_max_over_mean"] == pytest.approx(1.0)
        assert row["realized_insertion_max_over_mean"] == pytest.approx(1.0)
        assert row["imbalance_factor_applicable"] is (
            int(row["imbalance_ratio"]) == 1
        )
        for stage, values in row["stage_collective_max_times_s"].items():
            assert len(values) == 5
            assert all(value > 0.0 for value in values)
            raw = row["stage_times_by_rank_s"][stage]
            assert len(raw) == 5
            assert all(len(rank_values) == 1 for rank_values in raw)
            assert values == [max(rank_values) for rank_values in raw]
    assert payload["command"]
    assert payload["numerical_runtime"]["numpy"]
    analysis._validate_factor_payload_design_and_timings(payload)
    payload["results"][0]["case_id"] = "tampered"
    with pytest.raises(ValueError, match="reviewed field case_id"):
        analysis._validate_factor_payload_design_and_timings(payload)


def test_balanced_rank_work_conserves_total_and_realizes_declared_ratio() -> None:
    for size in (8, 32):
        for target in (1, 2, 4):
            work = [
                runner._balanced_rank_work(
                    rank=rank,
                    size=size,
                    baseline=8,
                    target_max_over_mean=target,
                )[0]
                for rank in range(size)
            ]
            assert sum(work) == 8 * size
            assert max(work) / (sum(work) / size) == pytest.approx(target)
    work, applicable = runner._balanced_rank_work(
        rank=0, size=1, baseline=8, target_max_over_mean=4
    )
    assert work == 8
    assert applicable is False


def test_factor_calibration_is_fitted_on_1_8_and_validated_on_32() -> None:
    stage_scale = {
        "contraction": 1.0e-4,
        "color_hvp": 2.0e-4,
        "insertion": 3.0e-5,
        "communication": 4.0e-5,
    }
    exponents = {
        "element_dofs": 0.7,
        "quadrature_points": 0.4,
        "constitutive_dimension": 0.3,
        "color_count": 0.2,
        "nonzeros_per_row": 0.15,
        "message_bytes": 0.1,
        "imbalance_ratio": 0.25,
    }

    payloads: dict[int, dict[str, object]] = {}
    for ranks in (1, 8, 32):
        rows = []
        for design in runner._design():
            multiplier = float(ranks) ** 0.2
            for name, exponent in exponents.items():
                multiplier *= (
                    float(design[name]) / float(analysis._FACTOR_BASELINE[name])
                ) ** exponent
            times = {
                stage: [
                    1.2 * scale * multiplier,
                    scale * multiplier,
                    1.01 * scale * multiplier,
                    0.99 * scale * multiplier,
                    scale * multiplier,
                ]
                for stage, scale in stage_scale.items()
            }
            rows.append({**design, "stage_collective_max_times_s": times})
        payloads[ranks] = {"results": rows}

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    calibration = analysis._fit_factorized_calibration(payloads, contract)
    assert calibration["status"] == "passed"
    assert calibration["training_ranks"] == [1, 8]
    assert calibration["validation_rank"] == 32
    assert calibration["validation_p90_absolute_percentage_error"] < 1.0e-10

    gate = {"calibration_model": calibration}
    predicted = analysis._calibrated_shared_stage_seconds(
        {
            "route": "constitutive_ad",
            "element_dofs": 30.0,
            "quadrature_points_per_element": 11.0,
            "constitutive_dimension": 6.0,
            "maximum_local_color_count": 0.0,
            "owned_matrix_nonzeros": 4800.0,
            "maximum_rank_owned_dofs": 100.0,
            "maximum_rank_overlap_dofs": 180.0,
            "owned_element_imbalance": 1.1,
            "overlap_dof_imbalance": 1.2,
            "rank_count": 32.0,
        },
        gate,
    )
    assert predicted > 0.0
