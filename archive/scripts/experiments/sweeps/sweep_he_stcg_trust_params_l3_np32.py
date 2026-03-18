#!/usr/bin/env python3
"""Tune HE STCG trust-region parameters on level 3, 32 MPI ranks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiment_results_cache" / "he_stcg_trust_params_l3_np32"

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
    "--linesearch-tol",
    "1e-1",
    "--use-trust-region",
    "--trust-subproblem-line-search",
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

RADIUS_STAGE = [
    {"name": "radius_0_5", "trust_radius_init": 0.5},
    {"name": "radius_1_0", "trust_radius_init": 1.0},
    {"name": "radius_2_0", "trust_radius_init": 2.0},
    {"name": "radius_4_0", "trust_radius_init": 4.0},
]

UPDATE_PROFILES = [
    {
        "name": "base",
        "trust_shrink": 0.5,
        "trust_expand": 1.5,
        "trust_eta_shrink": 0.05,
        "trust_eta_expand": 0.75,
    },
    {
        "name": "expand2",
        "trust_shrink": 0.5,
        "trust_expand": 2.0,
        "trust_eta_shrink": 0.05,
        "trust_eta_expand": 0.75,
    },
    {
        "name": "stricter",
        "trust_shrink": 0.5,
        "trust_expand": 1.5,
        "trust_eta_shrink": 0.1,
        "trust_eta_expand": 0.9,
    },
    {
        "name": "strong_shrink",
        "trust_shrink": 0.25,
        "trust_expand": 1.5,
        "trust_eta_shrink": 0.1,
        "trust_eta_expand": 0.75,
    },
]


def _build_command(backend: dict, params: dict, out_path: Path) -> list[str]:
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
    cmd.extend(
        [
            "--trust-radius-init",
            str(params["trust_radius_init"]),
            "--trust-radius-min",
            "1e-8",
            "--trust-radius-max",
            "1e6",
            "--trust-shrink",
            str(params["trust_shrink"]),
            "--trust-expand",
            str(params["trust_expand"]),
            "--trust-eta-shrink",
            str(params["trust_eta_shrink"]),
            "--trust-eta-expand",
            str(params["trust_eta_expand"]),
            "--trust-max-reject",
            "6",
        ]
    )
    cmd.extend(backend["extra_args"])
    return cmd


def _summarize(case_json: Path, backend: dict, label: str, stage: str, params: dict) -> dict:
    payload = json.loads(case_json.read_text())
    result = payload["result"]
    steps = result.get("steps", [])
    histories = [h for step in steps for h in step.get("history", [])]
    linear_records = [r for step in steps for r in step.get("linear_timing", [])]
    ksp_max_it = int(payload["case"]["ksp_max_it"])
    converged_steps = sum(
        1 for step in steps if "converged" in str(step.get("message", "")).lower()
    )
    all_steps_converged = len(steps) == 24 and converged_steps == 24
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
    max_ksp = max((int(rec.get("ksp_its", 0)) for rec in linear_records), default=0)
    ksp_cap_hits = sum(1 for rec in linear_records if int(rec.get("ksp_its", 0)) >= ksp_max_it)
    newton_maxit_steps = [
        int(step["step"])
        for step in steps
        if "maximum number of iterations reached" in str(step.get("message", "")).lower()
    ]
    ls_repaired = sum(1 for hist in histories if hist.get("ls_repaired"))
    ls_evals = sum(int(hist.get("ls_evals", 0)) for hist in histories)
    used_alpha_not_one = sum(
        1 for hist in histories if abs(float(hist.get("alpha", 1.0)) - 1.0) > 1e-12
    )
    max_step = max((float(step["time"]) for step in steps), default=0.0)
    final_message = steps[-1]["message"] if steps else "no steps"
    final_energy = float(steps[-1]["energy"]) if steps else None
    return {
        "backend": backend["name"],
        "stage": stage,
        "label": label,
        "trust_radius_init": float(params["trust_radius_init"]),
        "trust_shrink": float(params["trust_shrink"]),
        "trust_expand": float(params["trust_expand"]),
        "trust_eta_shrink": float(params["trust_eta_shrink"]),
        "trust_eta_expand": float(params["trust_eta_expand"]),
        "all_steps_converged": bool(all_steps_converged),
        "completed_steps": len(steps),
        "converged_steps": converged_steps,
        "total_time": float(result.get("total_time", 0.0)),
        "total_newton": total_newton,
        "total_linear": total_linear,
        "final_energy": final_energy,
        "final_message": final_message,
        "max_step_time": max_step,
        "max_ksp_its": int(max_ksp),
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
        "# HE STCG Trust-Parameter Sweep",
        "",
        "Level `3`, `32` MPI ranks, full `24/24` trajectory.",
        "",
        "Shared settings:",
        "- `ksp_type=stcg`, `pc_type=gamg`",
        "- `ksp_rtol=1e-1`, `ksp_max_it=100`",
        "- rebuild PC every Newton iteration (`pc_setup_on_ksp_cap=False`)",
        "- trust-region post line search on, `linesearch_tol=1e-1`",
        "- line-search interval `[-0.5, 2.0]`",
        "- `maxit=100`",
        "",
        "| Backend | Stage | Label | All 24 converged | Total [s] | Newton | Linear | Radius | Shrink | Expand | Eta shrink | Eta expand | Max KSP it | KSP cap hits | Used max it | Final message | JSON |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        final_message = row["final_message"].replace("|", "/")
        lines.append(
            f"| {row['backend']} | {row['stage']} | {row['label']} | "
            f"{'yes' if row['all_steps_converged'] else 'no'} | "
            f"{row['total_time']:.3f} | {row['total_newton']} | {row['total_linear']} | "
            f"{row['trust_radius_init']:.3g} | {row['trust_shrink']:.3g} | "
            f"{row['trust_expand']:.3g} | {row['trust_eta_shrink']:.3g} | "
            f"{row['trust_eta_expand']:.3g} | {row['max_ksp_its']} | "
            f"{row['ksp_cap_hits']} | {'yes' if row['used_max_it'] else 'no'} | "
            f"{final_message} | `{row['json']}` |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _best_converged_radius(rows: list[dict], backend_name: str) -> dict:
    radius_rows = [
        row
        for row in rows
        if row["backend"] == backend_name and row["stage"] == "radius" and row["all_steps_converged"]
    ]
    if radius_rows:
        return min(radius_rows, key=lambda row: row["total_time"])
    fallback_rows = [
        row for row in rows if row["backend"] == backend_name and row["stage"] == "radius"
    ]
    return min(fallback_rows, key=lambda row: row["total_time"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    base_update = {
        "trust_shrink": 0.5,
        "trust_expand": 1.5,
        "trust_eta_shrink": 0.05,
        "trust_eta_expand": 0.75,
    }

    # Stage 1: tune trust_radius_init with the current update parameters.
    for backend in BACKENDS:
        for radius_case in RADIUS_STAGE:
            params = dict(base_update)
            params["trust_radius_init"] = radius_case["trust_radius_init"]
            out_path = OUT_DIR / f"{backend['name']}_{radius_case['name']}.json"
            print(f"=== Radius stage: {backend['name']} / {radius_case['name']} ===", flush=True)
            subprocess.run(_build_command(backend, params, out_path), cwd=ROOT, check=True)
            rows.append(
                _summarize(out_path, backend, radius_case["name"], "radius", params)
            )
            _write_summary(rows)

    # Stage 2: tune shrink/expand/eta around the best radius per backend.
    for backend in BACKENDS:
        best_radius_row = _best_converged_radius(rows, backend["name"])
        best_radius = best_radius_row["trust_radius_init"]
        for profile in UPDATE_PROFILES:
            params = dict(profile)
            params["trust_radius_init"] = best_radius
            label = f"update_{profile['name']}_r{str(best_radius).replace('.', '_')}"
            out_path = OUT_DIR / f"{backend['name']}_{label}.json"
            print(f"=== Update stage: {backend['name']} / {label} ===", flush=True)
            subprocess.run(_build_command(backend, params, out_path), cwd=ROOT, check=True)
            rows.append(_summarize(out_path, backend, label, "update", params))
            _write_summary(rows)

    _write_summary(rows)
    print((OUT_DIR / "summary.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
