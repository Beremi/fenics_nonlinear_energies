#!/usr/bin/env python3
"""Sweep HE trust-region STCG settings on level 3, 32 MPI ranks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiment_results_cache" / "he_stcg_tuning_l3_np32"

COMMON_ARGS = [
    "--problem",
    "he",
    "--level",
    "3",
    "--steps",
    "24",
    "--start-step",
    "1",
    "--total-steps",
    "24",
    "--ksp-type",
    "stcg",
    "--pc-type",
    "gamg",
    "--ksp-rtol",
    "1e-1",
    "--ksp-max-it",
    "100",
    "--gamg-threshold",
    "0.05",
    "--gamg-agg-nsmooths",
    "1",
    "--gamg-set-coordinates",
    "--use-near-nullspace",
    "--tolf",
    "1e-4",
    "--tolg",
    "1e-3",
    "--tolg-rel",
    "1e-3",
    "--tolx-rel",
    "1e-3",
    "--tolx-abs",
    "1e-10",
    "--maxit",
    "100",
    "--linesearch-a",
    "-0.5",
    "--linesearch-b",
    "2.0",
    "--use-trust-region",
    "--trust-radius-init",
    "2.0",
    "--trust-radius-min",
    "1e-8",
    "--trust-radius-max",
    "1e6",
    "--trust-shrink",
    "0.5",
    "--trust-expand",
    "1.5",
    "--trust-eta-shrink",
    "0.05",
    "--trust-eta-expand",
    "0.75",
    "--trust-max-reject",
    "6",
    "--save-history",
    "--save-linear-timing",
    "--quiet",
    "--no-pc-setup-on-ksp-cap",
]

BACKENDS = [
    {
        "name": "fenics_custom",
        "backend": "fenics",
        "extra_args": [],
    },
    {
        "name": "jax_petsc_element",
        "backend": "element",
        "extra_args": [
            "--profile",
            "performance",
            "--local-coloring",
            "--nproc-threads",
            "1",
            "--element-reorder-mode",
            "block_xyz",
            "--local-hessian-mode",
            "element",
        ],
    },
]

VARIANTS = [
    {
        "name": "stcg_only",
        "post_ls": False,
        "ls_tol": None,
    },
    {
        "name": "stcg_postls_tol1e-1",
        "post_ls": True,
        "ls_tol": 1e-1,
    },
    {
        "name": "stcg_postls_tol1e-3",
        "post_ls": True,
        "ls_tol": 1e-3,
    },
    {
        "name": "stcg_postls_tol1e-6",
        "post_ls": True,
        "ls_tol": 1e-6,
    },
]


def _build_command(backend: dict, variant: dict, out_path: Path) -> list[str]:
    cmd = [
        "mpirun",
        "-n",
        "32",
        sys.executable,
        "-u",
        str(ROOT / "experiment_scripts" / "run_trust_region_case.py"),
        "--backend",
        backend["backend"],
        "--out",
        str(out_path),
    ]
    cmd.extend(COMMON_ARGS)
    cmd.extend(["--linesearch-tol", str(variant["ls_tol"] if variant["ls_tol"] is not None else 1e-3)])
    if variant["post_ls"]:
        cmd.append("--trust-subproblem-line-search")
    else:
        cmd.append("--no-trust-subproblem-line-search")
    cmd.extend(backend["extra_args"])
    return cmd


def _summarize(case_json: Path, backend: dict, variant: dict) -> dict:
    payload = json.loads(case_json.read_text())
    result = payload["result"]
    steps = result.get("steps", [])
    histories = [h for step in steps for h in step.get("history", [])]
    linear_records = [r for step in steps for r in step.get("linear_timing", [])]
    ksp_max_it = int(payload["case"]["ksp_max_it"])

    max_step = max((step["time"] for step in steps), default=0.0)
    max_ksp_its = max((int(rec.get("ksp_its", 0)) for rec in linear_records), default=0)
    ksp_cap_hits = sum(1 for rec in linear_records if int(rec.get("ksp_its", 0)) >= ksp_max_it)
    newton_maxit_steps = [
        int(step["step"])
        for step in steps
        if "maximum number of iterations reached" in str(step.get("message", "")).lower()
    ]
    converged_steps = sum(
        1 for step in steps if "converged" in str(step.get("message", "")).lower()
    )
    total_newton = sum(int(step.get("nit", step.get("iters", 0))) for step in steps)
    total_linear = sum(
        int(
            step.get(
                "linear_iters",
                sum(int(rec.get("ksp_its", 0)) for rec in step.get("linear_timing", [])),
            )
        )
        for step in steps
    )
    ls_repaired = sum(1 for hist in histories if hist.get("ls_repaired"))
    ls_evals = sum(int(hist.get("ls_evals", 0)) for hist in histories)
    used_alpha_not_one = sum(
        1
        for hist in histories
        if abs(float(hist.get("alpha", 1.0)) - 1.0) > 1e-12
    )

    return {
        "backend": backend["name"],
        "variant": variant["name"],
        "post_ls": bool(variant["post_ls"]),
        "linesearch_tol": variant["ls_tol"],
        "completed_steps": len(steps),
        "converged_steps": converged_steps,
        "all_steps_converged": len(steps) == 24 and converged_steps == 24,
        "total_time": float(result.get("total_time", 0.0)),
        "solve_time_total": float(result.get("solve_time_total", 0.0)),
        "total_newton": total_newton,
        "total_linear": total_linear,
        "final_energy": float(steps[-1]["energy"]) if steps else None,
        "final_message": steps[-1]["message"] if steps else "no steps",
        "max_step_time": float(max_step),
        "max_ksp_its": int(max_ksp_its),
        "ksp_cap_hits": int(ksp_cap_hits),
        "newton_maxit_steps": newton_maxit_steps,
        "used_max_it": bool(ksp_cap_hits or newton_maxit_steps),
        "ls_repaired": int(ls_repaired),
        "ls_evals": int(ls_evals),
        "used_alpha_not_one": int(used_alpha_not_one),
        "json": str(case_json.relative_to(ROOT)),
    }


def _write_summary(rows: list[dict]) -> None:
    summary_json = OUT_DIR / "summary.json"
    summary_md = OUT_DIR / "summary.md"
    summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "# HE STCG Trust-Region Sweep",
        "",
        "Level `3`, `32` MPI ranks, full `24/24` trajectory.",
        "",
        "Shared settings:",
        "- `ksp_type=stcg`, `pc_type=gamg`",
        "- `ksp_rtol=1e-1`, `ksp_max_it=100`",
        "- rebuild PC every Newton iteration (`pc_setup_on_ksp_cap=False`)",
        "- trust region on, `trust_radius_init=2.0`",
        "- line search interval `[-0.5, 2.0]`",
        "- `maxit=100`",
        "",
        "| Backend | Variant | Post LS | LS tol | All 24 converged | Total [s] | Newton | Linear | Final energy | Max step [s] | Max KSP it | KSP cap hits | Newton maxit steps | Used max it | Final message | JSON |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in rows:
        ls_tol = "-" if row["linesearch_tol"] is None else f"{row['linesearch_tol']:.0e}"
        newton_steps = ",".join(str(s) for s in row["newton_maxit_steps"]) or "-"
        final_message = row["final_message"].replace("|", "/")
        lines.append(
            f"| {row['backend']} | {row['variant']} | "
            f"{'yes' if row['post_ls'] else 'no'} | {ls_tol} | "
            f"{'yes' if row['all_steps_converged'] else 'no'} | "
            f"{row['total_time']:.3f} | {row['total_newton']} | {row['total_linear']} | "
            f"{row['final_energy']:.6f} | {row['max_step_time']:.3f} | "
            f"{row['max_ksp_its']} | {row['ksp_cap_hits']} | {newton_steps} | "
            f"{'yes' if row['used_max_it'] else 'no'} | {final_message} | `{row['json']}` |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for backend in BACKENDS:
        for variant in VARIANTS:
            out_path = OUT_DIR / f"{backend['name']}_{variant['name']}.json"
            cmd = _build_command(backend, variant, out_path)
            print(f"=== Running {backend['name']} / {variant['name']} ===", flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)
            rows.append(_summarize(out_path, backend, variant))
            _write_summary(rows)

    _write_summary(rows)
    print((OUT_DIR / "summary.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
