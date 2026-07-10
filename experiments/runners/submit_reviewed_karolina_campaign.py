#!/usr/bin/env python3
"""Journal and submit an already reviewed Karolina command inventory.

The default mode prints the frozen commands and never contacts a scheduler.
Actual submission is guarded by explicit flags and is intentionally not part
of local paper preparation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.runners import karolina_reviewed_campaign as contract


_SUBMITTED = re.compile(r"Submitted batch job ([1-9][0-9]*)")


def _append(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def submit(
    root: Path,
    *,
    execute: bool,
    confirmed: bool,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest, plan = contract.load_plan(root)
    lines = [
        line for line in (root / manifest["commands"]["path"]).read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not execute:
        return {
            "status": "dry_run_no_scheduler_contact",
            "case_count": len(lines),
            "commands": lines,
        }
    if not confirmed:
        raise contract.CampaignContractError("real submission requires --confirm-submit")
    if manifest.get("environment_contract", {}).get("status") != "hash_bound":
        raise contract.CampaignContractError("real submission requires a hash-bound environment")
    if os.environ.get("ALLOCATION_REVALIDATED") != "YES" or os.environ.get("ACCOUNT_QOS_REVALIDATED") != "YES":
        raise contract.CampaignContractError("allocation and account/QoS must be revalidated")
    if contract.git_metadata() != {"commit": plan["source_commit"], "dirty": False}:
        raise contract.CampaignContractError("real submission requires the frozen clean commit")
    if manifest.get("status") != "prepared_not_submitted":
        raise contract.CampaignContractError("campaign is not in a fresh prepared state")
    manifest["status"] = "submitting"
    manifest["scheduler_contact"] = True
    contract.atomic_json(root / "prepared_manifest.json", manifest)
    ledger = root / "submitted_jobs.jsonl"
    journal = root / "submission_journal.jsonl"
    ledger.open("x", encoding="utf-8").close()
    journal.open("x", encoding="utf-8").close()
    accepted = 0
    try:
        for index, (case, line) in enumerate(zip(plan["cases"], lines, strict=True), 1):
            attempt = f"initial-{index:04d}-{case['case_id']}"
            intent = {
                "event": "intent",
                "attempt_id": attempt,
                "case_id": case["case_id"],
                "command": line,
                "recorded_at_utc": contract.utc_now(),
            }
            _append(journal, intent)
            completed = runner(shlex.split(line), check=False, capture_output=True, text=True)
            match = _SUBMITTED.fullmatch(completed.stdout.strip())
            result = {
                "event": "result",
                "attempt_id": attempt,
                "case_id": case["case_id"],
                "command": line,
                "recorded_at_utc": contract.utc_now(),
                "returncode": int(completed.returncode),
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
                "job_id": None if match is None else match.group(1),
            }
            _append(journal, result)
            if completed.returncode or match is None:
                raise contract.CampaignContractError(f"scheduler rejected {case['case_id']}")
            _append(ledger, {key: result[key] for key in ("case_id", "command", "returncode", "stdout", "stderr", "job_id")})
            accepted += 1
    except BaseException:
        manifest["status"] = "partial_submission" if accepted else "submission_failed"
        manifest["accepted_jobs"] = accepted
        contract.atomic_json(root / "prepared_manifest.json", manifest)
        raise
    manifest["status"] = "submitted"
    manifest["accepted_jobs"] = accepted
    contract.atomic_json(root / "prepared_manifest.json", manifest)
    return {"status": "submitted", "accepted_jobs": accepted}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--campaign-root", type=Path, required=True)
    result.add_argument("--execute", action="store_true")
    result.add_argument("--confirm-submit", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        result = submit(
            args.campaign_root,
            execute=bool(args.execute),
            confirmed=bool(args.confirm_submit),
        )
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
