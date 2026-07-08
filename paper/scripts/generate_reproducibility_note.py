#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from common import BUILD_ROOT, REPO_ROOT, ensure_paper_dirs, read_json, write_text


DEFAULT_OUTPUT = BUILD_ROOT / "reproducibility_note.md"
P3D_VALIDATION_MANIFEST = REPO_ROOT / "artifacts/raw_results/plasticity3d_validation/validation_manifest.json"
P3D_ABLATION_SUMMARY = REPO_ROOT / "artifacts/raw_results/plasticity3d_derivative_ablation/comparison_summary.json"
JAX_FEM_BASELINE_MANIFEST = REPO_ROOT / "artifacts/raw_results/jax_fem_hyperelastic_baseline/run_manifest.json"
SUBMISSION_BUNDLE_MANIFEST = REPO_ROOT / "artifacts/reproduction/paper_submission_2026_07_08/manifest.json"


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _python_version(python_bin: Path) -> str:
    return subprocess.check_output([str(python_bin), "--version"], cwd=REPO_ROOT, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a compact reproducibility note for the paper artifact bundle.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-python", type=Path, default=REPO_ROOT / ".venv" / "bin" / "python")
    args = parser.parse_args()

    ensure_paper_dirs()
    validation_manifest = read_json(P3D_VALIDATION_MANIFEST)
    ablation_summary = read_json(P3D_ABLATION_SUMMARY)
    baseline_manifest = read_json(JAX_FEM_BASELINE_MANIFEST)

    lines = [
        "# Reproducibility Note",
        "",
        f"- git commit: `{_git_head()}`",
        f"- Python used to generate this note: `{_python_version(args.repo_python)}`",
        "",
        "## Archive Bundle",
        "",
        f"- Submission bundle manifest: `{SUBMISSION_BUNDLE_MANIFEST.relative_to(REPO_ROOT)}`",
        "- Build or refresh bundle: `python paper/scripts/build_submission_bundle.py`",
        "- Validate paper assets: `python paper/scripts/validate_paper_assets.py`",
        "- Validate archive-neutral provenance: `python paper/scripts/validate_paper_assets.py --archive-neutral`",
        "",
        "## Source Inputs",
        "",
        f"- Plasticity3D validation manifest source: `{P3D_VALIDATION_MANIFEST.relative_to(REPO_ROOT)}`",
        f"- Plasticity3D derivative-route summary source: `{P3D_ABLATION_SUMMARY.relative_to(REPO_ROOT)}`",
        f"- JAX-FEM baseline manifest source: `{JAX_FEM_BASELINE_MANIFEST.relative_to(REPO_ROOT)}`",
        "",
        "## Reported Cases",
        "",
        f"- Plasticity3D validation schedule: `{validation_manifest['validation_contract']['schedule']}`",
        f"- Plasticity3D derivative routes: `{[row['route'] for row in ablation_summary['rows']]}`",
        f"- JAX-FEM baseline schedule: `{baseline_manifest['schedule']}`",
        "",
    ]
    write_text(args.out, "\n".join(lines))
    print(args.out)


if __name__ == "__main__":
    main()
