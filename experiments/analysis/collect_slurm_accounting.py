#!/usr/bin/env python3
"""Collect one immutable Slurm accounting snapshot without submitting work.

The default/offline mode parses an already captured ``sacct --parsable2``
file.  Live accounting is possible only with the explicit ``--query-live``
flag; the command is executed as an argument vector with no shell.  This tool
never invokes ``sbatch``, ``srun``, ``scontrol``, or any mutating scheduler
operation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.benchmark.run_record import atomic_write_json


SCHEMA_ID = "fenics-nonlinear-energies.slurm-accounting-snapshot"
SCHEMA_VERSION = 1
SACCT_FIELDS = (
    "JobIDRaw",
    "JobID",
    "JobName",
    "Cluster",
    "Account",
    "Partition",
    "QOS",
    "State",
    "ElapsedRaw",
    "AllocNodes",
    "AllocCPUS",
    "TotalCPU",
    "CPUTimeRAW",
    "MaxRSS",
    "MaxVMSize",
    "ConsumedEnergyRaw",
    "ExitCode",
    "Start",
    "End",
    "NodeList",
)
_JOB_ID = re.compile(r"^[0-9]+(?:_[0-9]+)?$")
_SIZE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)([KMGTPE]?)$", re.IGNORECASE)


class AccountingError(ValueError):
    """The accounting evidence is missing, ambiguous, or malformed."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _integer(value: object, name: str, *, allow_blank: bool = False) -> int | None:
    text = str(value or "").strip()
    if allow_blank and text in {"", "Unknown", "N/A", "None"}:
        return None
    try:
        parsed = int(text)
    except ValueError as exc:
        raise AccountingError(f"{name} must be an integer") from exc
    if parsed < 0:
        raise AccountingError(f"{name} must be nonnegative")
    return parsed


def _size_bytes(value: object, name: str) -> int | None:
    text = str(value or "").strip()
    if text in {"", "Unknown", "N/A", "None"}:
        return None
    match = _SIZE.fullmatch(text)
    if match is None:
        raise AccountingError(f"{name} has unsupported Slurm size {text!r}")
    magnitude = float(match.group(1))
    exponent = "KMGTPE".find(match.group(2).upper()) + 1 if match.group(2) else 0
    converted = magnitude * float(1024**exponent)
    if not math.isfinite(converted) or converted < 0.0:
        raise AccountingError(f"{name} must be finite and nonnegative")
    return int(round(converted))


def _normalize_state(value: object) -> str:
    text = str(value or "").strip()
    # Slurm may append qualifiers such as ``CANCELLED by 123`` or ``+``.
    return text.split()[0].rstrip("+") if text else ""


def parse_sacct(text: str, *, job_id: str) -> dict[str, Any]:
    """Parse strict parsable2 output and identify the allocation record."""

    if not _JOB_ID.fullmatch(str(job_id)):
        raise AccountingError("job_id must be a numeric Slurm job or array-task ID")
    reader = csv.DictReader(io.StringIO(text), delimiter="|")
    if reader.fieldnames is None:
        raise AccountingError("sacct output has no header")
    fields = {str(field) for field in reader.fieldnames if field}
    missing = set(SACCT_FIELDS) - fields
    if missing:
        raise AccountingError(f"sacct output is missing fields: {sorted(missing)}")

    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(reader):
        if not any(str(value or "").strip() for key, value in raw.items() if key):
            continue
        row_id = str(raw.get("JobIDRaw") or raw.get("JobID") or "").strip()
        if not row_id:
            raise AccountingError(f"sacct row {index} has no job identity")
        normalized.append(
            {
                "job_id_raw": row_id,
                "job_id_display": str(raw.get("JobID") or "").strip(),
                "job_name": str(raw.get("JobName") or "").strip(),
                "cluster": str(raw.get("Cluster") or "").strip(),
                "account": str(raw.get("Account") or "").strip(),
                "partition": str(raw.get("Partition") or "").strip(),
                "qos": str(raw.get("QOS") or "").strip(),
                "state": _normalize_state(raw.get("State")),
                "elapsed_raw_s": _integer(
                    raw.get("ElapsedRaw"), "ElapsedRaw", allow_blank=True
                ),
                "alloc_nodes": _integer(
                    raw.get("AllocNodes"), "AllocNodes", allow_blank=True
                ),
                "alloc_cpus": _integer(
                    raw.get("AllocCPUS"), "AllocCPUS", allow_blank=True
                ),
                "total_cpu": str(raw.get("TotalCPU") or "").strip(),
                "cpu_time_raw_s": _integer(
                    raw.get("CPUTimeRAW"), "CPUTimeRAW", allow_blank=True
                ),
                "max_rss_bytes": _size_bytes(raw.get("MaxRSS"), "MaxRSS"),
                "max_vm_size_bytes": _size_bytes(
                    raw.get("MaxVMSize"), "MaxVMSize"
                ),
                "consumed_energy_raw_j": _integer(
                    raw.get("ConsumedEnergyRaw"),
                    "ConsumedEnergyRaw",
                    allow_blank=True,
                ),
                "exit_code": str(raw.get("ExitCode") or "").strip(),
                "start": str(raw.get("Start") or "").strip(),
                "end": str(raw.get("End") or "").strip(),
                "node_list": str(raw.get("NodeList") or "").strip(),
            }
        )
    if not normalized:
        raise AccountingError("sacct output contains no records")

    exact = [row for row in normalized if row["job_id_raw"] == str(job_id)]
    if len(exact) != 1:
        raise AccountingError(
            f"expected one allocation row for job {job_id}, found {len(exact)}"
        )
    allocation = dict(exact[0])
    if allocation["elapsed_raw_s"] in {None, 0}:
        raise AccountingError("allocation ElapsedRaw must be positive")
    if allocation["alloc_nodes"] in {None, 0} or allocation["alloc_cpus"] in {
        None,
        0,
    }:
        raise AccountingError("allocation nodes and CPUs must be positive")

    return {
        "job_id": str(job_id),
        "allocation": allocation,
        "rows": normalized,
        "derived": {
            "allocated_node_seconds": int(allocation["elapsed_raw_s"])
            * int(allocation["alloc_nodes"]),
            "allocated_cpu_seconds": int(allocation["elapsed_raw_s"])
            * int(allocation["alloc_cpus"]),
            "maximum_step_rss_bytes": max(
                (
                    int(row["max_rss_bytes"])
                    for row in normalized
                    if row["max_rss_bytes"] is not None
                ),
                default=None,
            ),
            "maximum_step_vm_size_bytes": max(
                (
                    int(row["max_vm_size_bytes"])
                    for row in normalized
                    if row["max_vm_size_bytes"] is not None
                ),
                default=None,
            ),
        },
    }


def sacct_command(job_id: str, *, executable: str = "sacct") -> list[str]:
    if not _JOB_ID.fullmatch(str(job_id)):
        raise AccountingError("job_id must be a numeric Slurm job or array-task ID")
    return [
        str(executable),
        "--jobs",
        str(job_id),
        "--parsable2",
        "--units=K",
        "--format=" + ",".join(SACCT_FIELDS),
    ]


def collect_accounting(
    *,
    job_id: str,
    sacct_file: Path | None = None,
    query_live: bool = False,
    executable: str = "sacct",
    collected_at_utc: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Build a hash-bound snapshot from exactly one explicit source mode."""

    if (sacct_file is None) == (not query_live):
        raise AccountingError("select exactly one of sacct_file or query_live")
    command: list[str] | None = None
    if sacct_file is not None:
        path = Path(sacct_file).resolve()
        raw = path.read_bytes()
        source = {"mode": "offline_file", "path": str(path)}
    else:
        command = sacct_command(job_id, executable=executable)
        completed = runner(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if int(completed.returncode) != 0:
            raise AccountingError(
                f"sacct query failed with return code {completed.returncode}: "
                f"{str(completed.stderr).strip()}"
            )
        raw = str(completed.stdout).encode("utf-8")
        source = {"mode": "explicit_live_query", "command": command}
    parsed = parse_sacct(raw.decode("utf-8"), job_id=str(job_id))
    timestamp = collected_at_utc or datetime.now(timezone.utc).isoformat()
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AccountingError("collected_at_utc must be an ISO-8601 timestamp") from exc
    if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() is None:
        raise AccountingError("collected_at_utc must include a UTC offset")
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "collected_at_utc": parsed_timestamp.astimezone(timezone.utc).isoformat(),
        "source": {
            **source,
            "sha256": _sha256_bytes(raw),
            "byte_count": len(raw),
            "raw_parsable2": raw.decode("utf-8"),
            "requested_fields": list(SACCT_FIELDS),
        },
        **parsed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sacct-file", type=Path)
    source.add_argument("--query-live", action="store_true")
    parser.add_argument("--sacct-executable", default="sacct")
    parser.add_argument("--collected-at-utc")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    payload = collect_accounting(
        job_id=str(args.job_id),
        sacct_file=args.sacct_file,
        query_live=bool(args.query_live),
        executable=str(args.sacct_executable),
        collected_at_utc=args.collected_at_utc,
    )
    destination = Path(args.output).resolve()
    atomic_write_json(destination, payload)
    print(destination)


if __name__ == "__main__":
    main()
