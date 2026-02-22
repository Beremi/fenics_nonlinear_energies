#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


def run_case(repo: Path, image: str, out_json_rel: str, extra_args: list[str]):
    cmd = [
        "docker", "run", "--rm", "--entrypoint", "python3",
        "-v", f"{repo}:/workspace", "-w", "/workspace", image,
        "HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py",
        "--level", "1",
        "--start_step", "24",
        "--steps", "1",
        "--maxit", "100",
        "--init_npz", "experiment_scripts/he_jax_testdata_l1.npz",
        "--init_step", "23",
        "--ksp_type", "gmres",
        "--pc_type", "hypre",
        "--ksp_rtol", "1e-1",
        "--ksp_max_it", "500",
        "--save_history",
        "--quiet",
        "--out", out_json_rel,
        *extra_args,
    ]
    proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
    return proc


def summarize(path: Path):
    d = json.loads(path.read_text())["steps"][0]
    h = d.get("history", [])
    return {
        "time": d["time"],
        "nit": d["iters"],
        "energy": d["energy"],
        "message": d["message"],
        "sum_ksp": int(sum(r["ksp_its"] for r in h)) if h else 0,
        "max_ksp": int(max(r["ksp_its"] for r in h)) if h else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--image", default="fenics_test")
    parser.add_argument("--out_dir", default="experiment_scripts/he_step24_hypre_options")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    out_dir = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ("baseline_n6_v3", []),
        ("strong_05", ["--hypre_strong_threshold", "0.5"]),
        ("strong_07", ["--hypre_strong_threshold", "0.7"]),
        ("coarsen_HMIS", ["--hypre_coarsen_type", "HMIS"]),
        ("coarsen_PMIS", ["--hypre_coarsen_type", "PMIS"]),
        ("nodal1_vec2", ["--hypre_nodal_coarsen", "1", "--hypre_vec_interp_variant", "2"]),
        ("nodal4_vec2", ["--hypre_nodal_coarsen", "4", "--hypre_vec_interp_variant", "2"]),
        ("skip_set_nodal_vec", ["--hypre_nodal_coarsen", "-1", "--hypre_vec_interp_variant", "-1"]),
    ]

    rows = []
    for name, extra in cases:
        out_json_rel = f"{args.out_dir}/{name}.json"
        proc = run_case(repo, args.image, out_json_rel, extra)
        row = {
            "case": name,
            "args": extra,
            "returncode": proc.returncode,
            "json": out_json_rel,
        }
        if proc.returncode == 0 and (repo / out_json_rel).exists():
            row.update(summarize(repo / out_json_rel))
        else:
            row["stderr_tail"] = "\n".join(proc.stderr.splitlines()[-15:])
        rows.append(row)

    summary = {
        "base": {
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "ksp_rtol": 1e-1,
            "ksp_max_it": 500,
            "maxit": 100,
            "step": 24,
            "init_step": 23,
            "near_nullspace": True,
        },
        "rows": rows,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# Step-24 HYPRE option sweep (near-nullspace ON)",
        "",
        "| Case | Time [s] | Newton iters | Energy | Sum inner iters | Max inner iters | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if r.get("returncode") == 0:
            lines.append(
                f"| {
                    r['case']} | {
                    r['time']} | {
                    r['nit']} | {
                    r['energy']} | {
                    r['sum_ksp']} | {
                        r['max_ksp']} | {
                            r['message']} |"
            )
        else:
            lines.append(
                f"| {r['case']} | - | - | - | - | - | FAILED (code {r['returncode']}) |"
            )

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {(out_dir / 'summary.json').relative_to(repo)}")
    print(f"Wrote {(out_dir / 'summary.md').relative_to(repo)}")


if __name__ == "__main__":
    main()
