#!/usr/bin/env python3
"""Monitor one replication campaign and finalize summaries when complete."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from experiments.runners import run_replications
from src.core.benchmark.replication import now_iso


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "replications" / "2026-03-16_maintained_refresh"


def _log_path(out_dir: Path) -> Path:
    path = out_dir / "_tasks" / "watch_campaign.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _log(out_dir: Path, message: str) -> None:
    with _log_path(out_dir).open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_iso()}] {message}\n")


def _load_manifest_entries(out_dir: Path) -> dict[str, dict]:
    manifest = out_dir / "manifest.json"
    if not manifest.exists():
        return {}
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return {entry["id"]: entry for entry in payload.get("entries", [])}


def _suite_task_ids(out_dir: Path) -> list[str]:
    return [task.id for task in run_replications.build_task_specs(out_dir) if task.category == "suites"]


def _reports_complete(entries: dict[str, dict]) -> bool:
    return entries.get("generate_reports", {}).get("status") == "completed"


def _suites_complete(entries: dict[str, dict], suite_ids: list[str]) -> bool:
    return all(entries.get(task_id, {}).get("status") == "completed" for task_id in suite_ids)


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    result = subprocess.run(["ps", "-p", str(pid)], capture_output=True, text=True, check=False)
    return result.returncode == 0


def _read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _write_pid(path: Path, pid: int) -> None:
    path.write_text(f"{pid}\n", encoding="utf-8")


def _continue_pid_path(out_dir: Path) -> Path:
    return out_dir / "_tasks" / "continue_campaign.pid"


def _watch_pid_path(out_dir: Path) -> Path:
    return out_dir / "_tasks" / "watch_campaign.pid"


def _continue_running(out_dir: Path) -> bool:
    pid = _read_pid(_continue_pid_path(out_dir))
    return _pid_running(pid or -1)


def _run_python(*args: str) -> None:
    command = [str(REPO_ROOT / ".venv" / "bin" / "python"), *args]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _start_detached_continue(out_dir: Path) -> None:
    continue_script = out_dir / "continue_campaign.sh"
    log_path = out_dir / "_tasks" / "continue_campaign.log"
    subprocess.run(
        [
            "bash",
            "-lc",
            (
                f"setsid -f bash {continue_script} > {log_path} 2>&1 < /dev/null"
            ),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    result = subprocess.run(
        ["pgrep", "-f", str(continue_script.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        pid = int(result.stdout.strip().splitlines()[0])
        _write_pid(_continue_pid_path(out_dir), pid)


def _run_reports(out_dir: Path) -> None:
    _run_python(
        "experiments/runners/run_replications.py",
        "--out-dir",
        str(out_dir.relative_to(REPO_ROOT)),
        "--only",
        "reports",
        "--resume",
    )


def _refresh_reports(out_dir: Path) -> None:
    _run_python(
        "experiments/analysis/generate_replication_reports.py",
        "--out-dir",
        str(out_dir),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--interval-seconds", type=int, default=1800)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_pid(_watch_pid_path(out_dir), os.getpid())
    suite_ids = _suite_task_ids(out_dir)
    _log(out_dir, f"watcher started for {out_dir}")

    while True:
        entries = _load_manifest_entries(out_dir)
        suites_complete = _suites_complete(entries, suite_ids)
        reports_complete = _reports_complete(entries)

        if suites_complete and reports_complete:
            _refresh_reports(out_dir)
            _log(out_dir, "campaign complete; refreshed final summaries")
            break

        if suites_complete and not reports_complete:
            _log(out_dir, "suites complete; generating final report set")
            _run_reports(out_dir)
        elif not suites_complete and not _continue_running(out_dir):
            _log(out_dir, "continuation not running; relaunching")
            _start_detached_continue(out_dir)
        else:
            _log(out_dir, "campaign still running; next check scheduled")

        time.sleep(int(args.interval_seconds))


if __name__ == "__main__":
    main()
