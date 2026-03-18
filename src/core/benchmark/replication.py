"""Helpers for running and recording reproducible replication tasks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any, Mapping


def now_iso() -> str:
    """Return a UTC ISO-8601 timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def command_text(command: list[str]) -> str:
    """Return a shell-safe one-line representation of a command."""
    return shlex.join(command)


def output_paths_exist(paths: list[Path]) -> bool:
    """Return True when every expected output path exists."""
    return all(path.exists() for path in paths)


def write_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    """Write one JSON payload with stable formatting."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    """Load JSON from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_command_files(
    leaf_dir: Path,
    *,
    command: list[str],
    cwd: Path,
    env: Mapping[str, str] | None = None,
) -> None:
    """Write command capture files for one replication leaf directory."""
    leaf_dir.mkdir(parents=True, exist_ok=True)
    env = dict(env or {})
    cmd_txt = command_text(command)
    (leaf_dir / "command.txt").write_text(cmd_txt + "\n", encoding="utf-8")

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(cwd))}",
    ]
    for key, value in sorted(env.items()):
        lines.append(f"export {key}={shlex.quote(str(value))}")
    lines.append(cmd_txt)

    command_sh = leaf_dir / "command.sh"
    command_sh.write_text("\n".join(lines) + "\n", encoding="utf-8")
    command_sh.chmod(0o755)


@dataclass(slots=True)
class CommandRunResult:
    """The persisted result of one external command invocation."""

    success: bool
    skipped: bool
    exit_code: int
    duration_s: float
    started_at: str
    finished_at: str
    outputs: list[str]
    command: str
    command_argv: list[str]
    cwd: str
    notes: str
    log_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "skipped": self.skipped,
            "exit_code": self.exit_code,
            "duration_s": self.duration_s,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "outputs": self.outputs,
            "command": self.command,
            "command_argv": self.command_argv,
            "cwd": self.cwd,
            "notes": self.notes,
            "log_path": self.log_path,
        }


def run_logged_command(
    *,
    command: list[str],
    cwd: Path,
    leaf_dir: Path,
    expected_outputs: list[Path],
    env: Mapping[str, str] | None = None,
    resume: bool = False,
    notes: str = "",
) -> CommandRunResult:
    """Run one command, tee output into a leaf directory, and persist status."""
    env = dict(env or {})
    leaf_dir.mkdir(parents=True, exist_ok=True)
    write_command_files(leaf_dir, command=command, cwd=cwd, env=env)

    status_path = leaf_dir / "status.json"
    log_path = leaf_dir / "run.log"
    cmd_text = command_text(command)

    if resume and status_path.exists():
        previous = read_json(status_path)
        if previous.get("success") and output_paths_exist(expected_outputs):
            previous["skipped"] = True
            write_json(status_path, previous)
            return CommandRunResult(
                success=True,
                skipped=True,
                exit_code=int(previous.get("exit_code", 0)),
                duration_s=float(previous.get("duration_s", 0.0)),
                started_at=str(previous.get("started_at", "")),
                finished_at=str(previous.get("finished_at", "")),
                outputs=[str(path) for path in expected_outputs],
                command=cmd_text,
                command_argv=list(command),
                cwd=str(cwd),
                notes=str(previous.get("notes", notes)),
                log_path=str(log_path),
            )

    started_at = now_iso()
    started = time.perf_counter()
    merged_env = None
    if env:
        merged_env = dict(os.environ)
        merged_env.update(env)

    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"# cwd: {cwd}\n")
        if env:
            for key, value in sorted(env.items()):
                log_handle.write(f"# env {key}={value}\n")
        log_handle.write(f"# command: {cmd_text}\n\n")
        log_handle.flush()

        proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_handle.write(line)
            log_handle.flush()
        exit_code = proc.wait()

    duration_s = time.perf_counter() - started
    finished_at = now_iso()
    success = exit_code == 0 and output_paths_exist(expected_outputs)
    result = CommandRunResult(
        success=success,
        skipped=False,
        exit_code=int(exit_code),
        duration_s=float(duration_s),
        started_at=started_at,
        finished_at=finished_at,
        outputs=[str(path) for path in expected_outputs],
        command=cmd_text,
        command_argv=list(command),
        cwd=str(cwd),
        notes=notes,
        log_path=str(log_path),
    )
    write_json(status_path, result.to_dict())

    if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, command)
    if not output_paths_exist(expected_outputs):
        missing = [str(path) for path in expected_outputs if not path.exists()]
        raise FileNotFoundError(f"Missing expected outputs for {cmd_text}: {missing}")
    return result
