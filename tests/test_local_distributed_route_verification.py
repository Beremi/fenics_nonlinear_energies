from __future__ import annotations

import json
from pathlib import Path

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
