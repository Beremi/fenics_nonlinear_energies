#!/usr/bin/env python3
"""Run the local 1/2/4-rank colored-recovery correctness matrix.

This campaign is a correctness experiment, not a timing benchmark.  It runs
element AD, colored sparse finite differences, and constitutive AD at the same
prescribed P1/P2 states and compares canonical states, gradients, four tangent
actions, branch diagnostics, ownership summaries, and feasible direct CSR
matrices across both routes and MPI decompositions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import uuid

import numpy as np

from src.core.benchmark.run_record import (
    atomic_write_json,
    check_experiment_preflight,
    sha256_file,
    utc_now_iso,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ROUTES = ("element_ad", "colored_sfd", "constitutive_ad")
DEGREES = (1, 2)
RANKS = (1, 2, 4)
STATES = (("elastic", 2.0e-4), ("mixed", 2.0e-2))
RULE_BY_DEGREE = {1: "tetra_1point", 2: "tetra_11point"}
RELATIVE_TOLERANCE = 1.0e-8
ABSOLUTE_TOLERANCE = 1.0e-10


@dataclass(frozen=True, slots=True)
class Block:
    degree: int
    state_label: str
    state_amplitude: float
    ranks: int
    route_order: tuple[str, ...]

    @property
    def block_id(self) -> str:
        return f"p{self.degree}_{self.state_label}_np{self.ranks}"


def build_blocks() -> tuple[Block, ...]:
    blocks: list[Block] = []
    index = 0
    for degree in DEGREES:
        for state_label, amplitude in STATES:
            for ranks in RANKS:
                shift = index % len(ROUTES)
                route_order = ROUTES[shift:] + ROUTES[:shift]
                blocks.append(
                    Block(
                        degree=degree,
                        state_label=state_label,
                        state_amplitude=float(amplitude),
                        ranks=ranks,
                        route_order=route_order,
                    )
                )
                index += 1
    return tuple(blocks)


def _relative_error(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return float("inf"), float("inf")
    absolute = float(np.linalg.norm(left - right))
    scale = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), np.finfo(float).tiny)
    return absolute, float(absolute / scale)


def _load_route(route_dir: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    payload = json.loads((route_dir / "output.json").read_text(encoding="utf-8"))
    with np.load(route_dir / "tangent_action.npz", allow_pickle=False) as archive:
        arrays = {
            "state": np.asarray(archive["state"], dtype=np.float64),
            "tangent_actions": np.asarray(archive["tangent_actions"], dtype=np.float64),
            "gradient": np.asarray(archive["gradient"], dtype=np.float64),
        }
    return payload, arrays


def _compare(
    errors: list[str],
    maxima: dict[str, float],
    *,
    label: str,
    left: np.ndarray,
    right: np.ndarray,
    exact: bool = False,
) -> None:
    if exact:
        if not np.array_equal(left, right):
            errors.append(f"{label}: exact equality failed")
        return
    absolute, relative = _relative_error(left, right)
    maxima[f"{label}.absolute"] = max(maxima.get(f"{label}.absolute", 0.0), absolute)
    maxima[f"{label}.relative"] = max(maxima.get(f"{label}.relative", 0.0), relative)
    if relative > RELATIVE_TOLERANCE and absolute > ABSOLUTE_TOLERANCE:
        errors.append(
            f"{label}: absolute error {absolute:.6e}, relative error {relative:.6e}"
        )


def validate_campaign(out_root: Path) -> dict[str, object]:
    errors: list[str] = []
    maxima: dict[str, float] = {}
    loaded: dict[tuple[int, str, int, str], tuple[dict[str, object], dict[str, np.ndarray]]] = {}
    for block in build_blocks():
        route_data: dict[str, tuple[dict[str, object], dict[str, np.ndarray]]] = {}
        for route in ROUTES:
            route_dir = out_root / "blocks" / block.block_id / route
            try:
                payload, arrays = _load_route(route_dir)
            except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{block.block_id}/{route}: unreadable output ({exc})")
                continue
            route_data[route] = (payload, arrays)
            loaded[(block.degree, block.state_label, block.ranks, route)] = (payload, arrays)
            if payload.get("status") != "completed" or payload.get("route") != route:
                errors.append(f"{block.block_id}/{route}: invalid terminal identity")
            if int(payload.get("mpi_ranks", -1)) != block.ranks:
                errors.append(f"{block.block_id}/{route}: MPI rank count mismatch")
            if int(payload.get("probe_count", -1)) != 4 or arrays["tangent_actions"].shape[0] != 4:
                errors.append(f"{block.block_id}/{route}: four tangent probes are required")
            rank_rows = payload.get("rank_summaries")
            if not isinstance(rank_rows, list) or len(rank_rows) != block.ranks:
                errors.append(f"{block.block_id}/{route}: incomplete rank summaries")
            else:
                rank_ids = sorted(int(row.get("rank", -1)) for row in rank_rows if isinstance(row, dict))
                owned = sum(int(row.get("owned_dofs", 0)) for row in rank_rows if isinstance(row, dict))
                expected_owned = int(payload.get("model_covariates", {}).get("global_free_dofs", -1))
                if rank_ids != list(range(block.ranks)) or owned != expected_owned:
                    errors.append(f"{block.block_id}/{route}: ownership partition is inconsistent")

        if set(route_data) != set(ROUTES):
            continue
        reference_payload, reference_arrays = route_data["element_ad"]
        for route in ("colored_sfd", "constitutive_ad"):
            payload, arrays = route_data[route]
            prefix = f"{block.block_id}/{route}-vs-element"
            _compare(errors, maxima, label=f"{prefix}/state", left=arrays["state"], right=reference_arrays["state"], exact=True)
            _compare(errors, maxima, label=f"{prefix}/gradient", left=arrays["gradient"], right=reference_arrays["gradient"])
            _compare(errors, maxima, label=f"{prefix}/actions", left=arrays["tangent_actions"], right=reference_arrays["tangent_actions"])
            if payload.get("branch_diagnostics") != reference_payload.get("branch_diagnostics"):
                errors.append(f"{prefix}: branch diagnostics differ")
        if block.ranks == 1:
            matrices: dict[str, dict[str, np.ndarray]] = {}
            for route in ROUTES:
                matrix_path = out_root / "blocks" / block.block_id / route / "tangent_matrix_csr.npz"
                try:
                    with np.load(matrix_path, allow_pickle=False) as matrix:
                        matrices[route] = {
                            "indptr": np.asarray(matrix["indptr"], dtype=np.int64),
                            "indices": np.asarray(matrix["indices"], dtype=np.int64),
                            "values": np.asarray(matrix["values"], dtype=np.float64),
                            "shape": np.asarray(matrix["shape"], dtype=np.int64),
                        }
                except (OSError, KeyError, ValueError) as exc:
                    errors.append(f"{block.block_id}/{route}: missing direct CSR ({exc})")
            if set(matrices) == set(ROUTES):
                reference = matrices["element_ad"]
                for route in ("colored_sfd", "constitutive_ad"):
                    candidate = matrices[route]
                    prefix = f"{block.block_id}/{route}-vs-element/csr"
                    for field in ("indptr", "indices", "shape"):
                        _compare(errors, maxima, label=f"{prefix}/{field}", left=candidate[field], right=reference[field], exact=True)
                    _compare(errors, maxima, label=f"{prefix}/values", left=candidate["values"], right=reference["values"])

    for degree in DEGREES:
        for state_label, _amplitude in STATES:
            for route in ROUTES:
                reference = loaded.get((degree, state_label, 1, route))
                if reference is None:
                    continue
                reference_payload, reference_arrays = reference
                for ranks in (2, 4):
                    candidate = loaded.get((degree, state_label, ranks, route))
                    if candidate is None:
                        continue
                    payload, arrays = candidate
                    prefix = f"p{degree}_{state_label}/{route}/np{ranks}-vs-np1"
                    _compare(errors, maxima, label=f"{prefix}/state", left=arrays["state"], right=reference_arrays["state"], exact=True)
                    _compare(errors, maxima, label=f"{prefix}/gradient", left=arrays["gradient"], right=reference_arrays["gradient"])
                    _compare(errors, maxima, label=f"{prefix}/actions", left=arrays["tangent_actions"], right=reference_arrays["tangent_actions"])
                    if payload.get("branch_diagnostics") != reference_payload.get("branch_diagnostics"):
                        errors.append(f"{prefix}: branch diagnostics differ across ranks")

    return {
        "schema_id": "fenics-nonlinear-energies.exp-dist-001-colored-recovery",
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "comparison_tolerances": {
            "relative": RELATIVE_TOLERANCE,
            "absolute": ABSOLUTE_TOLERANCE,
            "integer_topology_and_state": "exact",
        },
        "planned_blocks": len(build_blocks()),
        "planned_route_processes": len(build_blocks()) * len(ROUTES),
        "maximum_observed_errors": dict(sorted(maxima.items())),
        "errors": errors,
        "timing_claim_admissible": False,
    }


def _command(block: Block, route: str, route_position: int, route_dir: Path, args: argparse.Namespace) -> list[str]:
    command = [
        str(args.mpiexec),
        "--oversubscribe",
        "-n",
        str(block.ranks),
        str(args.python),
        "experiments/runners/run_plasticity3d_fixed_state_route_screen.py",
        "--route",
        route,
        "--tier",
        "fixed_state_screen",
        "--mesh-name",
        "hetero_ssr_L1",
        "--element-degree",
        str(block.degree),
        "--quadrature-rule",
        RULE_BY_DEGREE[block.degree],
        "--constraint-variant",
        "glued_bottom",
        "--lambda-target",
        "1.55",
        "--state-label",
        block.state_label,
        "--state-amplitude",
        str(block.state_amplitude),
        "--warmup-repetitions",
        "1",
        "--measured-repetitions",
        "5",
        "--probe-count",
        "4",
        "--comparison-id",
        block.block_id,
        "--block-repetition",
        "1",
        "--route-order-position",
        str(route_position),
        "--route-order-policy",
        "local_distributed_correctness_v1",
        "--ksp-rtol",
        "1e-8",
        "--ksp-max-it",
        "500",
        "--output",
        str(route_dir / "output.json"),
        "--action-out",
        str(route_dir / "tangent_action.npz"),
    ]
    if block.ranks == 1:
        command.append("--save-direct-matrix")
    return command


def run_campaign(args: argparse.Namespace) -> dict[str, object]:
    out_root = Path(args.out_root).resolve()
    if out_root.exists() or out_root.is_symlink():
        raise FileExistsError(f"refusing to reuse output root {out_root}")
    preflight = check_experiment_preflight(
        REPO_ROOT,
        run_kind=str(args.run_kind),
        pilot_dirty_override=bool(args.pilot_dirty_override),
        pilot_override_reason=args.pilot_override_reason,
    )
    out_root.mkdir(parents=True)
    plan = {
        "schema_id": "fenics-nonlinear-energies.exp-dist-001-colored-recovery-plan",
        "schema_version": 1,
        "experiment_id": "EXP-DIST-001",
        "run_id": str(args.run_id or uuid.uuid4()),
        "created_at_utc": utc_now_iso(),
        "source_commit": preflight.git_commit,
        "source_clean": preflight.git_clean,
        "run_kind": str(args.run_kind),
        "blocks": [
            {
                "block_id": block.block_id,
                "degree": block.degree,
                "state_label": block.state_label,
                "state_amplitude": block.state_amplitude,
                "ranks": block.ranks,
                "route_order": list(block.route_order),
            }
            for block in build_blocks()
        ],
        "timing_claim_admissible": False,
    }
    atomic_write_json(out_root / "plan.json", plan)
    if not args.execute:
        plan["status"] = "prepared_not_executed"
        atomic_write_json(out_root / "manifest.json", plan)
        return plan
    if os.environ.get("LOCAL_DISTRIBUTED_RUN_CONFIRMED") != "YES":
        raise RuntimeError("execution requires LOCAL_DISTRIBUTED_RUN_CONFIRMED=YES")
    if shutil.which(str(args.mpiexec)) is None:
        raise RuntimeError(f"MPI launcher not found: {args.mpiexec}")

    environment = os.environ.copy()
    environment.update(
        {
            "JAX_PLATFORMS": "cpu",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
        }
    )
    try:
        for block in build_blocks():
            for position, route in enumerate(block.route_order, start=1):
                route_dir = out_root / "blocks" / block.block_id / route
                route_dir.mkdir(parents=True)
                command = _command(block, route, position, route_dir, args)
                (route_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
                with (route_dir / "stdout.txt").open("xb") as stdout, (route_dir / "stderr.txt").open("xb") as stderr:
                    completed = subprocess.run(
                        command,
                        cwd=REPO_ROOT,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout,
                        stderr=stderr,
                        check=False,
                    )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"route process failed: {block.block_id}/{route}; see stderr.txt"
                    )
        verification = validate_campaign(out_root)
        atomic_write_json(out_root / "verification_summary.json", verification)
        manifest = {
            **plan,
            "status": "completed" if verification["status"] == "passed" else "failed",
            "verification_summary": "verification_summary.json",
            "verification_sha256": sha256_file(out_root / "verification_summary.json"),
        }
        atomic_write_json(out_root / "manifest.json", manifest)
        if verification["status"] != "passed":
            raise RuntimeError("distributed route verification failed")
        return manifest
    except BaseException:
        failure = {**plan, "status": "failed"}
        atomic_write_json(out_root / "manifest.json", failure)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--python", default="./.venv/bin/python")
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--run-kind", choices=("publication", "pilot"), default="publication")
    parser.add_argument("--pilot-dirty-override", action="store_true")
    parser.add_argument("--pilot-override-reason")
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        result = run_campaign(args)
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
