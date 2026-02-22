#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
from pathlib import Path


def run_one(repo_dir: Path, image: str, level: int, start_step: int, init_step: int,
            init_npz: str, maxit: int, ksp_type: str, pc_type: str, ksp_rtol: float,
            out_json: Path):
    out_json_rel = out_json.relative_to(repo_dir)
    cmd = [
        "docker", "run", "--rm", "--entrypoint", "python3",
        "-v", f"{repo_dir}:/workspace", "-w", "/workspace", image,
        "HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py",
        "--level", str(level),
        "--start_step", str(start_step),
        "--steps", "1",
        "--maxit", str(maxit),
        "--init_npz", init_npz,
        "--init_step", str(init_step),
        "--ksp_type", ksp_type,
        "--pc_type", pc_type,
        "--ksp_rtol", f"{ksp_rtol:.0e}",
        "--save_history",
        "--quiet",
        "--out", str(out_json_rel),
    ]
    proc = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default=".")
    parser.add_argument("--image", type=str, default="fenics_test")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--start_step", type=int, default=24)
    parser.add_argument("--init_step", type=int, default=23)
    parser.add_argument("--init_npz", type=str, default="experiment_scripts/he_jax_testdata_l1.npz")
    parser.add_argument("--maxit", type=int, default=300)
    parser.add_argument("--pc_type", type=str, default="hypre")
    parser.add_argument("--ksp_types", type=str, default="cg,gmres")
    parser.add_argument("--out_dir", type=str, default="experiment_scripts/he_step24_precision_sweep")
    args = parser.parse_args()

    repo_dir = Path(args.repo).resolve()
    out_dir = (repo_dir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tolerances = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    ksp_types = [k.strip() for k in args.ksp_types.split(",") if k.strip()]

    summary = {
        "level": args.level,
        "start_step": args.start_step,
        "init_step": args.init_step,
        "init_npz": args.init_npz,
        "maxit": args.maxit,
        "pc_type": args.pc_type,
        "ksp_types": ksp_types,
        "tolerances": tolerances,
        "runs": [],
    }

    profile_csv = out_dir / "step24_convergence_profiles.csv"
    with profile_csv.open("w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "ksp_type", "ksp_rtol", "it", "energy", "dE", "grad_norm", "alpha", "ksp_its", "ls_evals"
        ])

        for ksp_type in ksp_types:
            for tol in tolerances:
                out_json = out_dir / f"step24_{ksp_type}_rtol_{tol:.0e}.json"
                code, stdout, stderr = run_one(
                    repo_dir=repo_dir,
                    image=args.image,
                    level=args.level,
                    start_step=args.start_step,
                    init_step=args.init_step,
                    init_npz=args.init_npz,
                    maxit=args.maxit,
                    ksp_type=ksp_type,
                    pc_type=args.pc_type,
                    ksp_rtol=tol,
                    out_json=out_json,
                )

                run_entry = {
                    "ksp_type": ksp_type,
                    "ksp_rtol": tol,
                    "return_code": code,
                    "output_json": str(out_json.relative_to(repo_dir)),
                }

                if code == 0 and out_json.exists():
                    data = json.loads(out_json.read_text())
                    step = data["steps"][0]
                    run_entry.update({
                        "iters": step.get("iters"),
                        "energy": step.get("energy"),
                        "message": step.get("message", ""),
                    })
                    hist = step.get("history", [])
                    for rec in hist:
                        writer.writerow([
                            ksp_type,
                            f"{tol:.0e}",
                            rec.get("it"),
                            rec.get("energy"),
                            rec.get("dE"),
                            rec.get("grad_norm"),
                            rec.get("alpha"),
                            rec.get("ksp_its"),
                            rec.get("ls_evals"),
                        ])
                else:
                    run_entry["stderr_tail"] = "\n".join(stderr.strip().splitlines()[-10:])

                summary["runs"].append(run_entry)

    summary_json = out_dir / "step24_precision_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))

    # Markdown table for quick inspection
    md_lines = [
        "# Step 24 precision sweep (custom HE solver)",
        "",
        f"- Level: {args.level}",
        f"- Step: {args.start_step} (restart from step {args.init_step})",
        f"- PC: {args.pc_type}",
        f"- Max Newton iterations: {args.maxit}",
        "",
        "| KSP | ksp_rtol | Return | Newton iters | Final energy | Message |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in summary["runs"]:
        md_lines.append(
            f"| {r.get('ksp_type', '')} | {r.get('ksp_rtol', '')} | {r.get('return_code', '')} | "
            f"{r.get('iters', '')} | {r.get('energy', '')} | {r.get('message', '')} |"
        )

    md_lines.extend([
        "",
        f"Profile CSV: `{profile_csv.relative_to(repo_dir)}`",
        f"Summary JSON: `{summary_json.relative_to(repo_dir)}`",
    ])

    summary_md = out_dir / "step24_precision_summary.md"
    summary_md.write_text("\n".join(md_lines) + "\n")

    print(f"Wrote {summary_json.relative_to(repo_dir)}")
    print(f"Wrote {summary_md.relative_to(repo_dir)}")
    print(f"Wrote {profile_csv.relative_to(repo_dir)}")


if __name__ == "__main__":
    main()
