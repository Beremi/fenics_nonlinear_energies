#!/usr/bin/env python3
"""Run the full local canonical retest, sync tracked docs data, and rebuild docs assets."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def _run(argv: list[str]) -> dict:
    proc = subprocess.run(argv, cwd=REPO_ROOT, check=True)
    return {"command": " ".join(argv), "exit_code": int(proc.returncode)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=str, required=True)
    args = parser.parse_args()

    campaign_root = REPO_ROOT / "artifacts" / "reproduction" / args.campaign
    runs_root = campaign_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    steps = [
        {
            "name": "smoke",
            "argv": [str(PYTHON), "-u", "experiments/runners/run_readme_docs_smoke.py", "--out-dir", str(runs_root / "readme_docs_smoke")],
        },
        {
            "name": "plaplace_final_suite",
            "argv": [str(PYTHON), "-u", "experiments/runners/run_plaplace_final_suite.py", "--out-dir", str(runs_root / "plaplace" / "final_suite")],
        },
        {
            "name": "gl_final_suite",
            "argv": [str(PYTHON), "-u", "experiments/runners/run_gl_final_suite.py", "--out-dir", str(runs_root / "ginzburg_landau" / "final_suite")],
        },
        {
            "name": "topology_docs_suite",
            "argv": [str(PYTHON), "-u", "experiments/runners/run_topology_docs_suite.py", "--out-dir", str(runs_root / "topology")],
        },
        {
            "name": "he_final_suite_best",
            "argv": [
                str(PYTHON),
                "-u",
                "experiments/runners/run_he_final_suite_best.py",
                "--out-dir",
                str(runs_root / "hyperelasticity" / "final_suite_best"),
                "--no-seed-known-results",
            ],
        },
        {
            "name": "he_pure_jax_suite_best",
            "argv": [
                str(PYTHON),
                "-u",
                "experiments/runners/run_he_pure_jax_suite_best.py",
                "--out-dir",
                str(runs_root / "hyperelasticity" / "pure_jax_suite_best"),
            ],
        },
        {
            "name": "sync_docs_assets_data",
            "argv": [str(PYTHON), "-u", "experiments/analysis/docs_assets/sync_tracked_docs_data.py", "--campaign-root", str(campaign_root)],
        },
        {
            "name": "build_all_docs_assets",
            "argv": [str(PYTHON), "-u", "experiments/analysis/docs_assets/build_all.py"],
        },
    ]

    summary = {"campaign_root": str(campaign_root), "steps": []}
    for step in steps:
        print(f"[retest] {step['name']}", flush=True)
        summary["steps"].append({"name": step["name"], **_run(step["argv"])})

    (campaign_root / "runs" / "pipeline_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
