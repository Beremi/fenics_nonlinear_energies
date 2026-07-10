#!/usr/bin/env python3
"""Prepare or execute the frozen clean-workstation EXP-ROUTE paired blocks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import uuid

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.runners.paper_revision_karolina.execute_case import (
    _run,
    _write_json,
    p3d_fixed_block_commands,
    validate_fixed_state_block,
)
from src.core.benchmark.run_record import atomic_write_json


DEFAULT_PLAN = REPO_ROOT / "paper/protocols/EXP-ROUTE-001-workstation-plan.json"
MATRIX = REPO_ROOT / "experiments/runners/paper_revision_karolina/campaign_matrix.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_metadata() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def _load_plan(path: Path) -> tuple[dict[str, object], list[dict[str, str]]]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.get("schema_id") != (
        "fenics-nonlinear-energies.exp-route-001-workstation-plan"
    ):
        raise ValueError("workstation plan has the wrong schema")
    if int(plan.get("schema_version", -1)) != 1:
        raise ValueError("workstation plan has the wrong schema version")
    if plan.get("source_matrix_sha256") != _sha256(MATRIX):
        raise ValueError("workstation plan source-matrix hash is stale")
    with MATRIX.open(newline="", encoding="utf-8") as handle:
        by_case = {row["case_id"]: dict(row) for row in csv.DictReader(handle)}
    case_ids = [str(value) for value in plan.get("case_ids", [])]
    if len(case_ids) != 12 or len(set(case_ids)) != 12:
        raise ValueError("workstation plan must contain 12 unique paired blocks")
    rows = [by_case[case_id] for case_id in case_ids]
    if any(
        row["experiment_id"] != "EXP-ROUTE-001"
        or row["tier"] != "fixed_state_screen"
        or row["runner"] != "p3d_fixed_state_block"
        or int(row["total_ranks"]) != 1
        or int(row["element_degree"]) not in {1, 2}
        or row["mesh_name"] != "hetero_ssr_L1"
        for row in rows
    ):
        raise ValueError("workstation plan selects an out-of-scope matrix row")
    expected = {
        (degree, state, block)
        for degree in (1, 2)
        for state in ("elastic", "mixed")
        for block in (1, 2, 3)
    }
    actual = {
        (int(row["element_degree"]), row["state_label"], int(row["block_repetition"]))
        for row in rows
    }
    if actual != expected:
        raise ValueError("workstation plan does not cover the frozen P1/P2 state blocks")
    return plan, rows


def prepare_or_execute(args: argparse.Namespace) -> dict[str, object]:
    plan_path = Path(args.plan).resolve()
    plan, rows = _load_plan(plan_path)
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    archived_plan = out_root / "workstation_plan.json"
    shutil.copy2(plan_path, archived_plan)
    git = _git_metadata()
    run_id = str(args.run_id or uuid.uuid4())
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "cpu_affinity": (
            sorted(int(value) for value in os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else []
        ),
        "thread_environment": {
            name: os.environ.get(name, "")
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "XLA_FLAGS",
            )
        },
    }
    atomic_write_json(out_root / "environment.json", environment)
    manifest: dict[str, object] = {
        "schema_id": "fenics-nonlinear-energies.exp-route-001-workstation-manifest",
        "schema_version": 1,
        "status": "prepared_not_executed",
        "experiment_id": "EXP-ROUTE-001",
        "hardware_id": "workstation_local",
        "plan_path": archived_plan.name,
        "plan_sha256": _sha256(archived_plan),
        "matrix_path": str(MATRIX.relative_to(REPO_ROOT)),
        "matrix_sha256": _sha256(MATRIX),
        "source_commit": git["commit"],
        "source_dirty": git["dirty"],
        "run_id": run_id,
        "case_count": len(rows),
        "route_process_executions": sum(len(row["route_order"].split("|")) for row in rows),
        "case_ids": [row["case_id"] for row in rows],
        "environment_path": "environment.json",
        "environment_sha256": _sha256(out_root / "environment.json"),
    }
    manifest_path = out_root / "workstation_manifest.json"
    atomic_write_json(manifest_path, manifest)
    if not args.execute:
        return manifest
    if os.environ.get("WORKSTATION_RUN_CONFIRMED") != "YES":
        raise RuntimeError("execution requires WORKSTATION_RUN_CONFIRMED=YES")
    if git["dirty"] is not False:
        raise RuntimeError("publication workstation campaign requires a clean commit")
    if int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")) != 1:
        raise RuntimeError("workstation driver itself must run as one process")
    previous_run_id = os.environ.get("WORKSTATION_RUN_ID")
    os.environ["WORKSTATION_RUN_ID"] = run_id
    try:
        for row in rows:
            job_dir = out_root / "cases" / row["case_id"] / f"job_{run_id}"
            measure_dir = job_dir / "measure_01"
            measure_dir.mkdir(parents=True, exist_ok=True)
            _write_json(job_dir / "matrix_row.json", row)
            route_dirs: dict[str, Path] = {}
            records: list[dict[str, object]] = []
            for route, command in p3d_fixed_block_commands(
                row,
                python=str(args.python),
                run_dir=measure_dir,
                use_srun=False,
            ):
                route_dir = measure_dir / route
                route_dir.mkdir(parents=True, exist_ok=True)
                route_dirs[route] = route_dir
                (route_dir / "command.txt").write_text(
                    shlex.join(command) + "\n", encoding="utf-8"
                )
                record = {
                    "route": route,
                    "command": shlex.join(command),
                    **_run(
                        command,
                        stdout=route_dir / "stdout.txt",
                        stderr=route_dir / "stderr.txt",
                        timeout_s=None,
                    ),
                }
                records.append(record)
                _write_json(job_dir / "run_records.json", records)
                if int(record["returncode"]) != 0:
                    raise RuntimeError(f"workstation route failed: {row['case_id']}:{route}")
            block = validate_fixed_state_block(row, route_dirs)
            block["job_metadata"] = {"workstation_run_id": run_id}
            _write_json(measure_dir / "block_result.json", block)
        manifest["status"] = "completed"
        atomic_write_json(manifest_path, manifest)
        return manifest
    except BaseException:
        manifest["status"] = "failed"
        atomic_write_json(manifest_path, manifest)
        raise
    finally:
        if previous_run_id is None:
            os.environ.pop("WORKSTATION_RUN_ID", None)
        else:
            os.environ["WORKSTATION_RUN_ID"] = previous_run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--python", default="./.venv/bin/python")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        result = prepare_or_execute(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
