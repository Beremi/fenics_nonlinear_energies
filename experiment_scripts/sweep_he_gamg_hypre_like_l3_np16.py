#!/usr/bin/env python3
"""Sweep GAMG settings to approach HYPRE iteration counts (level 3, np=16)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path


def parse_metrics(path: Path) -> dict:
    data = json.loads(path.read_text())
    steps = data.get("steps", [])
    total_time = sum(float(s.get("time", 0.0)) for s in steps)
    total_newton = sum(int(s.get("iters", 0)) for s in steps)
    total_ksp = 0
    converged_steps = 0
    for s in steps:
        msg = str(s.get("message", "")).lower()
        if "converged" in msg:
            converged_steps += 1
        total_ksp += sum(int(h.get("ksp_its", 0)) for h in s.get("history", []))
    final_energy = steps[-1].get("energy") if steps else None
    avg_ksp_newton = (total_ksp / total_newton) if total_newton > 0 else math.nan
    return {
        "steps": len(steps),
        "converged_steps": converged_steps,
        "all_steps_converged": converged_steps == len(steps) if steps else False,
        "total_dofs": data.get("total_dofs"),
        "total_time_s": total_time,
        "total_newton_iters": total_newton,
        "total_ksp_iters": total_ksp,
        "avg_ksp_per_newton": avg_ksp_newton,
        "final_energy": final_energy,
    }


def default_cases() -> list[dict]:
    # Focused sweep around the "loose" GAMG settings from Annex F.
    raw = [
        ("r1e-1_k20_reuse", 1e-1, 20, True),
        ("r5e-2_k20_reuse", 5e-2, 20, True),
        ("r2e-2_k20_reuse", 2e-2, 20, True),
        ("r1e-2_k20_reuse", 1e-2, 20, True),
        ("r5e-2_k20_fresh", 5e-2, 20, False),
        ("r2e-2_k20_fresh", 2e-2, 20, False),
        ("r1e-1_k30_reuse", 1e-1, 30, True),
        ("r5e-2_k30_reuse", 5e-2, 30, True),
        ("r2e-2_k30_reuse", 2e-2, 30, True),
        ("r1e-2_k30_reuse", 1e-2, 30, True),
        ("r5e-3_k30_reuse", 5e-3, 30, True),
        ("r2e-2_k50_reuse", 2e-2, 50, True),
        ("r1e-1_k50_fresh", 1e-1, 50, False),
        ("r1e-2_k50_fresh", 1e-2, 50, False),
        ("r1e-3_k50_fresh", 1e-3, 50, False),
        ("r1e-2_k50_reuse", 1e-2, 50, True),
        ("r5e-3_k50_reuse", 5e-3, 50, True),
        ("r2e-2_k30_fresh", 2e-2, 30, False),
        ("r1e-2_k30_fresh", 1e-2, 30, False),
    ]
    return [
        {
            "id": cid,
            "ksp_rtol": rtol,
            "ksp_max_it": kmax,
            "pc_setup_on_ksp_cap": reuse,
        }
        for cid, rtol, kmax, reuse in raw
    ]


def quick_cases() -> list[dict]:
    raw = [
        ("r1e-1_k30_reuse", 1e-1, 30, True),
        ("r5e-2_k30_reuse", 5e-2, 30, True),
        ("r2e-2_k30_reuse", 2e-2, 30, True),
        ("r1e-2_k30_reuse", 1e-2, 30, True),
        ("r2e-2_k30_fresh", 2e-2, 30, False),
        ("r1e-2_k30_fresh", 1e-2, 30, False),
    ]
    return [
        {
            "id": cid,
            "ksp_rtol": rtol,
            "ksp_max_it": kmax,
            "pc_setup_on_ksp_cap": reuse,
        }
        for cid, rtol, kmax, reuse in raw
    ]


def build_case_set(case_set: str) -> list[dict]:
    if case_set == "quick":
        return quick_cases()
    return default_cases()


def run_case(
    repo_dir: Path,
    out_dir: Path,
    container: str,
    nprocs: int,
    level: int,
    steps: int,
    total_steps: int,
    maxit: int,
    gamg_threshold: float,
    gamg_agg_nsmooths: int,
    no_gamg_coordinates: bool,
    case: dict,
    timeout_s: int,
) -> dict:
    out_json = out_dir / f"{case['id']}.json"
    out_rel = out_json.relative_to(repo_dir)

    cmd = [
        "docker",
        "exec",
        container,
        "mpirun",
        "-n",
        str(nprocs),
        "python3",
        "/workdir/HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py",
        "--level",
        str(level),
        "--steps",
        str(steps),
        "--start_step",
        "1",
        "--total_steps",
        str(total_steps),
        "--maxit",
        str(maxit),
        "--ksp_type",
        "gmres",
        "--pc_type",
        "gamg",
        "--ksp_rtol",
        f"{case['ksp_rtol']:.0e}",
        "--ksp_max_it",
        str(case["ksp_max_it"]),
        "--gamg_threshold",
        str(gamg_threshold),
        "--gamg_agg_nsmooths",
        str(gamg_agg_nsmooths),
        "--save_history",
        "--quiet",
        "--out",
        f"/workdir/{out_rel}",
    ]
    if case["pc_setup_on_ksp_cap"]:
        cmd.append("--pc_setup_on_ksp_cap")
    if no_gamg_coordinates:
        cmd.append("--no_gamg_coordinates")

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        rc = proc.returncode
        stdout_tail = "\n".join(proc.stdout.strip().splitlines()[-20:])
        stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        rc = 124
        stdout_tail = "\n".join((exc.stdout or "").splitlines()[-20:])
        stderr_tail = "\n".join((exc.stderr or "").splitlines()[-20:])
        timed_out = True
    elapsed = time.perf_counter() - t0

    row = {
        "id": case["id"],
        "ksp_rtol": case["ksp_rtol"],
        "ksp_max_it": case["ksp_max_it"],
        "pc_setup_on_ksp_cap": case["pc_setup_on_ksp_cap"],
        "return_code": rc,
        "timed_out": timed_out,
        "wall_elapsed_s": elapsed,
        "output_json": str(out_rel),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }

    if rc == 0 and out_json.exists():
        row.update(parse_metrics(out_json))
    return row


def write_outputs(out_dir: Path, summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))

    rows = summary["runs"]
    csv_path = out_dir / "summary.csv"
    fieldnames = [
        "id",
        "ksp_rtol",
        "ksp_max_it",
        "pc_setup_on_ksp_cap",
        "return_code",
        "steps",
        "converged_steps",
        "all_steps_converged",
        "total_time_s",
        "total_newton_iters",
        "total_ksp_iters",
        "avg_ksp_per_newton",
        "final_energy",
        "score_to_hypre",
        "time_speedup_vs_hypre",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})

    hyp = summary["hypre_baseline"]
    md_lines = [
        "# GAMG sweep for HYPRE-like iteration profile (L3, np=16)",
        "",
        "## Baseline (HYPRE reference)",
        "",
        f"- Source: `{summary['hypre_baseline_json']}`",
        f"- Steps: {hyp['steps']} ({hyp['converged_steps']} converged)",
        f"- Total time: {hyp['total_time_s']:.4f} s",
        f"- Total Newton iters: {hyp['total_newton_iters']}",
        f"- Total KSP iters: {hyp['total_ksp_iters']}",
        f"- Avg KSP/Newton: {hyp['avg_ksp_per_newton']:.4f}",
        "",
        "## Sweep results",
        "",
        "| Case | rtol | ksp_max_it | PC reuse on cap | RC | Conv steps | Time [s] | Newton | KSP | Avg KSP/Newton | Score to HYPRE | Speedup vs HYPRE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['id']} | {r['ksp_rtol']:.0e} | {r['ksp_max_it']} | "
            f"{'Yes' if r['pc_setup_on_ksp_cap'] else 'No'} | {r['return_code']} | "
            f"{r.get('converged_steps', '')}/{r.get('steps', '')} | "
            f"{r.get('total_time_s', float('nan')):.4f} | "
            f"{r.get('total_newton_iters', '')} | {r.get('total_ksp_iters', '')} | "
            f"{r.get('avg_ksp_per_newton', float('nan')):.4f} | "
            f"{r.get('score_to_hypre', float('nan')):.4f} | "
            f"{r.get('time_speedup_vs_hypre', float('nan')):.3f}x |"
        )

    top = summary.get("top3_by_score", [])
    if top:
        md_lines.extend(
            [
                "",
                "## Top 3 closest to HYPRE iteration totals",
                "",
                "| Rank | Case | Newton | KSP | Time [s] | Score |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for i, r in enumerate(top, 1):
            md_lines.append(
                f"| {i} | {r['id']} | {r['total_newton_iters']} | {r['total_ksp_iters']} | "
                f"{r['total_time_s']:.4f} | {r['score_to_hypre']:.4f} |"
            )

    md_lines.extend(
        [
            "",
            f"Summary JSON: `{summary_json}`",
            f"Summary CSV: `{csv_path}`",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default=".")
    parser.add_argument("--container", type=str, default="bench_container")
    parser.add_argument("--nprocs", type=int, default=16)
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--total_steps", type=int, default=24)
    parser.add_argument("--maxit", type=int, default=100)
    parser.add_argument("--gamg_threshold", type=float, default=0.05)
    parser.add_argument("--gamg_agg_nsmooths", type=int, default=1)
    parser.add_argument("--no_gamg_coordinates", action="store_true")
    parser.add_argument("--timeout_s", type=int, default=5400)
    parser.add_argument(
        "--case_set",
        type=str,
        default="default",
        choices=["quick", "default"],
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="experiment_scripts/he_gamg_hypre_like_sweep_l3_np16",
    )
    parser.add_argument(
        "--hypre_baseline_json",
        type=str,
        default="experiment_scripts/he_custom_l3_np16_bench.json",
    )
    args = parser.parse_args()

    repo_dir = Path(args.repo).resolve()
    out_dir = (repo_dir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    hypre_path = (repo_dir / args.hypre_baseline_json).resolve()
    hypre = parse_metrics(hypre_path)
    hypre_newton = hypre["total_newton_iters"]
    hypre_ksp = hypre["total_ksp_iters"]
    hypre_time = hypre["total_time_s"]

    runs = []
    cases = build_case_set(args.case_set)
    total = len(cases)
    for i, case in enumerate(cases, 1):
        print(
            f"[{i}/{total}] running {case['id']} "
            f"(rtol={case['ksp_rtol']:.0e}, kmax={case['ksp_max_it']}, "
            f"reuse={'on' if case['pc_setup_on_ksp_cap'] else 'off'})",
            flush=True,
        )
        row = run_case(
            repo_dir=repo_dir,
            out_dir=out_dir,
            container=args.container,
            nprocs=args.nprocs,
            level=args.level,
            steps=args.steps,
            total_steps=args.total_steps,
            maxit=args.maxit,
            gamg_threshold=args.gamg_threshold,
            gamg_agg_nsmooths=args.gamg_agg_nsmooths,
            no_gamg_coordinates=args.no_gamg_coordinates,
            case=case,
            timeout_s=args.timeout_s,
        )

        if row["return_code"] == 0 and "total_newton_iters" in row:
            score = (
                abs(row["total_newton_iters"] - hypre_newton) / max(hypre_newton, 1)
                + abs(row["total_ksp_iters"] - hypre_ksp) / max(hypre_ksp, 1)
            )
            row["score_to_hypre"] = score
            row["time_speedup_vs_hypre"] = hypre_time / row["total_time_s"] if row["total_time_s"] > 0 else math.nan
            print(
                f"    done: time={row['total_time_s']:.2f}s newton={row['total_newton_iters']} "
                f"ksp={row['total_ksp_iters']} score={score:.4f}",
                flush=True,
            )
        else:
            row["score_to_hypre"] = math.inf
            row["time_speedup_vs_hypre"] = math.nan
            print(f"    failed: rc={row['return_code']}", flush=True)
        runs.append(row)

    ok = [r for r in runs if math.isfinite(r["score_to_hypre"])]
    ok_sorted = sorted(ok, key=lambda r: (r["score_to_hypre"], r["total_time_s"]))
    all_sorted = ok_sorted + [r for r in runs if not math.isfinite(r["score_to_hypre"])]

    summary = {
        "problem": {
            "level": args.level,
            "steps": args.steps,
            "total_steps": args.total_steps,
            "nprocs": args.nprocs,
            "maxit": args.maxit,
            "pc_type": "gamg",
            "gamg_threshold": args.gamg_threshold,
            "gamg_agg_nsmooths": args.gamg_agg_nsmooths,
            "gamg_coordinates": not args.no_gamg_coordinates,
        },
        "hypre_baseline_json": str(hypre_path.relative_to(repo_dir)),
        "hypre_baseline": hypre,
        "runs": all_sorted,
        "top3_by_score": ok_sorted[:3],
    }

    write_outputs(out_dir, summary)
    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"Wrote {out_dir / 'summary.csv'}")
    print(f"Wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
