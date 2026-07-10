from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from experiments.runners import run_local_distributed_route_verification as runner


def _write_synthetic_campaign(root: Path, *, perturb: bool = False) -> None:
    state = np.linspace(0.0, 0.4, 9)
    gradient = np.linspace(-1.0, 1.0, 9)
    actions = np.vstack([gradient + float(index) for index in range(4)])
    indptr = np.asarray([0, 2, 4, 6], dtype=np.int64)
    indices = np.asarray([0, 1, 0, 1, 1, 2], dtype=np.int64)
    values = np.linspace(1.0, 2.0, 6)
    for block in runner.build_blocks():
        for route in runner.ROUTES:
            route_dir = root / "blocks" / block.block_id / route
            route_dir.mkdir(parents=True)
            route_actions = actions.copy()
            if perturb and block.block_id == "p2_mixed_np4" and route == "colored_sfd":
                route_actions[3, 4] += 1.0e-3
            payload = {
                "status": "completed",
                "route": route,
                "mpi_ranks": block.ranks,
                "probe_count": 4,
                "branch_diagnostics": {
                    "counts": {"elastic": 2, "shear": 1},
                    "normalized_boundary_margin_min": 0.1,
                },
                "model_covariates": {"global_free_dofs": 9},
                "rank_summaries": [
                    {
                        "rank": rank,
                        "owned_dofs": (9 // block.ranks) + (1 if rank < 9 % block.ranks else 0),
                    }
                    for rank in range(block.ranks)
                ],
            }
            (route_dir / "output.json").write_text(
                json.dumps(payload) + "\n",
                encoding="utf-8",
            )
            np.savez_compressed(
                route_dir / "tangent_action.npz",
                state=state,
                gradient=gradient,
                tangent_actions=route_actions,
            )
            if block.ranks == 1:
                np.savez_compressed(
                    route_dir / "tangent_matrix_csr.npz",
                    indptr=indptr,
                    indices=indices,
                    values=values,
                    shape=np.asarray([3, 3], dtype=np.int64),
                )


def test_local_distributed_matrix_is_complete_and_balanced() -> None:
    blocks = runner.build_blocks()
    assert len(blocks) == 12
    assert {(block.degree, block.state_label, block.ranks) for block in blocks} == {
        (degree, state, ranks)
        for degree in (1, 2)
        for state in ("elastic", "mixed")
        for ranks in (1, 2, 4)
    }
    assert all(set(block.route_order) == set(runner.ROUTES) for block in blocks)
    first_positions = [block.route_order[0] for block in blocks]
    assert {route: first_positions.count(route) for route in runner.ROUTES} == {
        route: 4 for route in runner.ROUTES
    }


def test_local_distributed_adjudicator_accepts_equal_routes_and_ranks(
    tmp_path: Path,
) -> None:
    _write_synthetic_campaign(tmp_path)
    result = runner.validate_campaign(tmp_path)
    assert result["status"] == "passed"
    assert result["errors"] == []
    assert result["planned_blocks"] == 12
    assert result["planned_route_processes"] == 36
    assert result["timing_claim_admissible"] is False


def test_local_distributed_adjudicator_rejects_colored_action_drift(
    tmp_path: Path,
) -> None:
    _write_synthetic_campaign(tmp_path, perturb=True)
    result = runner.validate_campaign(tmp_path)
    assert result["status"] == "failed"
    assert any("p2_mixed_np4" in error and "actions" in error for error in result["errors"])


def test_frozen_command_is_single_valued_and_normalizes_executables_and_outputs(
    tmp_path: Path,
) -> None:
    block = runner.build_blocks()[0]
    route = block.route_order[0]
    output = tmp_path / "campaign"
    command = runner._command(
        block,
        route,
        0,
        output / "blocks" / block.block_id / route,
        python=Path(sys.executable),
        mpiexec=Path(sys.executable),
    )
    assert command.count("--route-order-policy") == 1
    assert command[command.index("--route-order-policy") + 1] == (
        "local_distributed_correctness_v2"
    )
    normalized = runner._normalize_command(
        command,
        out_root=output,
        python=Path(sys.executable),
        mpiexec=Path(sys.executable),
    )
    assert normalized[0] == "${MPIEXEC}"
    assert normalized[4] == "${PYTHON}"
    assert "${OUTPUT_ROOT}/blocks/p1_elastic_np1" in " ".join(normalized)


def test_strict_child_identity_rejects_commit_and_saved_array_hash_drift() -> None:
    block = runner.build_blocks()[0]
    route = block.route_order[0]
    state = np.linspace(0.0, 1.0, 4)
    actions = np.vstack([state + index for index in range(4)])
    arrays = {"state": state, "gradient": state.copy(), "tangent_actions": actions}
    child_argv = ["child.py", "--example"]
    payload = {
        "experiment_id": "EXP-ROUTE-001",
        "tier": "fixed_state_screen",
        "status": "completed",
        "route": route,
        "mesh_name": "hetero_ssr_L1",
        "element_degree": block.degree,
        "quadrature_rule_id": runner.RULE_BY_DEGREE[block.degree],
        "constraint_variant": "glued_bottom",
        "lambda_target": 1.55,
        "state_family": "analytic_mesh_field_v1",
        "state_label": block.state_label,
        "state_amplitude": block.state_amplitude,
        "probe_count": 4,
        "mpi_ranks": block.ranks,
        "warmup_repetitions": 1,
        "measured_repetitions": 5,
        "comparison_design": {
            "comparison_id": block.block_id,
            "block_repetition": 1,
            "route_order_position": 0,
            "route_order_policy": "local_distributed_correctness_v2",
            "timing_reduction": "mpi_collective_max",
            "independent_process_block": True,
        },
        "git": {"commit": "b" * 40, "dirty": False},
        "command": "python child.py --example",
        "state_sha256": runner._array_sha256(state),
        "gradient_sha256": runner._array_sha256(state),
        "action_sha256": runner._array_sha256(actions[0]),
        "action_sha256_by_probe": [runner._array_sha256(row) for row in actions],
    }
    errors: list[str] = []
    runner._validate_child_identity(
        errors,
        block=block,
        route=route,
        route_position=0,
        payload=payload,
        arrays=arrays,
        source_commit="a" * 40,
        run_id="fixture",
        expected_child_argv=child_argv,
    )
    assert any("Git identity" in error for error in errors)

    payload["git"] = {"commit": "a" * 40, "dirty": False}
    payload["action_sha256_by_probe"] = ["0" * 64] * 4
    errors = []
    runner._validate_child_identity(
        errors,
        block=block,
        route=route,
        route_position=0,
        payload=payload,
        arrays=arrays,
        source_commit="a" * 40,
        run_id="fixture",
        expected_child_argv=child_argv,
    )
    assert any("all probes" in error for error in errors)


def test_route_hash_closure_detects_unrecorded_or_changed_files(tmp_path: Path) -> None:
    route = tmp_path / "route"
    route.mkdir()
    (route / "output.json").write_text("{}\n", encoding="utf-8")
    record = runner._write_process_record(route, {"status": "completed"})
    assert record["artifact_hash_closure"]["files"] == {
        "output.json": runner.sha256_file(route / "output.json")
    }
    (route / "output.json").write_text('{"tampered": true}\n', encoding="utf-8")
    assert runner._route_hash_closure(route) != record["artifact_hash_closure"]
