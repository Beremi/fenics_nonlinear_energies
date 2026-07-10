#!/usr/bin/env python3
"""Compute-node executor for one reviewed Karolina publication case."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.runners import karolina_reviewed_campaign as contract


def _expand(token: str, *, job_root: Path, python: str) -> str:
    return (
        token.replace("{PYTHON}", python)
        .replace("{REPO_ROOT}", str(REPO_ROOT))
        .replace("{JOB_ROOT}", str(job_root))
    )


def _integer_env(name: str) -> int:
    raw = os.environ.get(name, "")
    if not raw.isdigit() or int(raw) < 1:
        raise contract.CampaignContractError(f"required scheduler value {name} is invalid")
    return int(raw)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.campaign_root).resolve()
    manifest, plan = contract.load_plan(root)
    plan_path = root / str(manifest["plan"]["path"])
    freeze_path = root / str(manifest["source_freeze"]["path"])
    if contract.sha256_file(plan_path) != args.expected_plan_sha256:
        raise contract.CampaignContractError("prepared plan changed after queueing")
    if contract.sha256_file(freeze_path) != args.expected_source_freeze_sha256:
        raise contract.CampaignContractError("source freeze changed after queueing")
    current = contract.git_metadata()
    if current != {"commit": plan["source_commit"], "dirty": False}:
        raise contract.CampaignContractError("compute checkout differs from the clean frozen commit")
    environment = manifest.get("environment_contract")
    if not isinstance(environment, dict) or environment.get("status") != "hash_bound":
        raise contract.CampaignContractError("campaign has no hash-bound environment contract")
    lock = Path(args.environment_lock).resolve()
    if contract.sha256_file(lock) != environment["lock"]["sha256"]:
        raise contract.CampaignContractError("compute environment lock changed after preparation")
    matches = [case for case in plan["cases"] if case["case_id"] == args.case_id]
    if len(matches) != 1:
        raise contract.CampaignContractError("case ID is absent or duplicated in the plan")
    case = matches[0]
    exact = {
        "SLURM_JOB_ACCOUNT": contract.ACCOUNT,
        "SLURM_JOB_QOS": contract.QOS,
        "SLURM_JOB_PARTITION": str(case["partition"]),
    }
    for name, expected in exact.items():
        if os.environ.get(name) != expected:
            raise contract.CampaignContractError(f"{name} differs from the reviewed plan")
    if (
        _integer_env("SLURM_JOB_NUM_NODES") != int(case["nodes"])
        or _integer_env("SLURM_NTASKS") != int(case["total_ranks"])
        or _integer_env("SLURM_CPUS_PER_TASK") != 1
    ):
        raise contract.CampaignContractError("Slurm allocation shape differs from the reviewed plan")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdigit():
        raise contract.CampaignContractError("SLURM_JOB_ID is missing or malformed")
    job_root = root / "jobs" / str(case["case_id"]) / f"job_{job_id}"
    job_root.mkdir(parents=True, exist_ok=False)
    python = str(Path(sys.executable).absolute())
    payload = [_expand(token, job_root=job_root, python=python) for token in case["payload_argv"]]
    srun = [
        "srun",
        "--kill-on-bad-exit=1",
        f"--nodes={case['nodes']}",
        f"--ntasks={case['total_ranks']}",
        f"--ntasks-per-node={case['ranks_per_node']}",
        "--cpus-per-task=1",
        "--distribution=block:block",
        "--cpu-bind=cores",
        "--mem-bind=local",
    ]
    metadata = {
        "schema_id": "fenics-nonlinear-energies.reviewed-karolina-job",
        "schema_version": 1,
        "experiment_id": plan["experiment_id"],
        "case_id": case["case_id"],
        "job_id": job_id,
        "source_commit": plan["source_commit"],
        "plan_sha256": args.expected_plan_sha256,
        "source_freeze_sha256": args.expected_source_freeze_sha256,
        "resources": {
            "account": contract.ACCOUNT,
            "qos": contract.QOS,
            "partition": case["partition"],
            "nodes": case["nodes"],
            "total_ranks": case["total_ranks"],
            "ranks_per_node": case["ranks_per_node"],
            "cpus_per_task": 1,
        },
        "payload_argv": payload,
        "expected_outputs": case["expected_outputs"],
        "accounting_status": "pending_post_job_collection",
    }
    contract.atomic_json(job_root / "job_metadata.json", metadata)
    contract.atomic_json(
        job_root / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "node": platform.node(),
            "controlled_environment": {
                key: os.environ.get(key)
                for key in (
                    "JAX_ENABLE_X64", "JAX_PLATFORMS", "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "XLA_FLAGS",
                )
            },
        },
    )
    completed = subprocess.run(
        [*srun, *payload], check=False, capture_output=True, text=True, cwd=REPO_ROOT
    )
    (job_root / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (job_root / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    outputs: dict[str, str] = {}
    if completed.returncode == 0:
        for raw in case["expected_outputs"]:
            output = (job_root / raw).resolve()
            try:
                output.relative_to(job_root)
            except ValueError as exc:
                raise contract.CampaignContractError("expected output escapes job root") from exc
            if not output.is_file() or output.is_symlink():
                raise contract.CampaignContractError(f"required output is missing: {raw}")
            outputs[str(raw)] = contract.sha256_file(output)
    receipt = {
        "schema_id": "fenics-nonlinear-energies.reviewed-karolina-execution",
        "schema_version": 1,
        "case_id": case["case_id"],
        "job_id": job_id,
        "returncode": int(completed.returncode),
        "output_hashes": outputs,
    }
    contract.atomic_json(job_root / "execution.json", receipt)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--campaign-root", type=Path, required=True)
    result.add_argument("--case-id", required=True)
    result.add_argument("--expected-plan-sha256", required=True)
    result.add_argument("--expected-source-freeze-sha256", required=True)
    result.add_argument("--environment-lock", type=Path, required=True)
    return result


def main() -> None:
    try:
        result = execute(parser().parse_args())
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()

