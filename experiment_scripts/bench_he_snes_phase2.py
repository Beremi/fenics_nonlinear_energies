#!/usr/bin/env python3
import argparse
import json
import math
import subprocess
import time
from pathlib import Path


REFERENCE_ENERGY = {1: 0.3464113961964319, 2: 0.20270785577653685}


def build_cases():
    cases = []

    for ksp_rtol in [1e-1, 1e-2, 1e-3, 1e-6]:
        for snes_atol in [1e-5, 1e-8, 1e-10]:
            cases.append(
                {
                    "name": f"ls_basic_gmres_hypre_rtol{ksp_rtol:g}_atol{snes_atol:g}",
                    "snes_type": "newtonls",
                    "linesearch": "basic",
                    "ksp_type": "gmres",
                    "pc_type": "hypre",
                    "ksp_rtol": ksp_rtol,
                    "snes_atol": snes_atol,
                    "use_objective": False,
                }
            )

    cases.extend(
        [
            {
                "name": "ls_basic_cg_hypre",
                "snes_type": "newtonls",
                "linesearch": "basic",
                "ksp_type": "cg",
                "pc_type": "hypre",
                "ksp_rtol": 1e-3,
                "snes_atol": 1e-5,
                "use_objective": False,
            },
            {
                "name": "ls_bt_cg_hypre_obj",
                "snes_type": "newtonls",
                "linesearch": "bt",
                "ksp_type": "cg",
                "pc_type": "hypre",
                "ksp_rtol": 1e-3,
                "snes_atol": 1e-5,
                "use_objective": True,
            },
            {
                "name": "ls_bt_gmres_hypre_obj",
                "snes_type": "newtonls",
                "linesearch": "bt",
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "ksp_rtol": 1e-3,
                "snes_atol": 1e-5,
                "use_objective": True,
            },
            {
                "name": "ls_l2_gmres_hypre",
                "snes_type": "newtonls",
                "linesearch": "l2",
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "ksp_rtol": 1e-3,
                "snes_atol": 1e-5,
                "use_objective": False,
            },
            {
                "name": "ls_l2_fgmres_hypre",
                "snes_type": "newtonls",
                "linesearch": "l2",
                "ksp_type": "fgmres",
                "pc_type": "hypre",
                "ksp_rtol": 1e-3,
                "snes_atol": 1e-5,
                "use_objective": False,
            },
            {
                "name": "tr_cg_hypre_obj",
                "snes_type": "newtontr",
                "linesearch": "basic",
                "ksp_type": "cg",
                "pc_type": "hypre",
                "ksp_rtol": 1e-3,
                "snes_atol": 1e-5,
                "use_objective": True,
            },
            {
                "name": "tr_gmres_hypre_obj",
                "snes_type": "newtontr",
                "linesearch": "basic",
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "ksp_rtol": 1e-3,
                "snes_atol": 1e-5,
                "use_objective": True,
            },
            {
                "name": "ls_basic_fgmres_asm",
                "snes_type": "newtonls",
                "linesearch": "basic",
                "ksp_type": "fgmres",
                "pc_type": "asm",
                "ksp_rtol": 1e-3,
                "snes_atol": 1e-5,
                "use_objective": False,
            },
            {
                "name": "ls_basic_fgmres_ilu",
                "snes_type": "newtonls",
                "linesearch": "basic",
                "ksp_type": "fgmres",
                "pc_type": "ilu",
                "ksp_rtol": 1e-3,
                "snes_atol": 1e-5,
                "use_objective": False,
            },
            {
                "name": "ls_basic_gmres_asm",
                "snes_type": "newtonls",
                "linesearch": "basic",
                "ksp_type": "gmres",
                "pc_type": "asm",
                "ksp_rtol": 1e-3,
                "snes_atol": 1e-5,
                "use_objective": False,
            },
            {
                "name": "ls_basic_gmres_bjacobi",
                "snes_type": "newtonls",
                "linesearch": "basic",
                "ksp_type": "gmres",
                "pc_type": "bjacobi",
                "ksp_rtol": 1e-3,
                "snes_atol": 1e-5,
                "use_objective": False,
            },
        ]
    )

    return cases


def parse_json(stdout: str):
    i = stdout.find("{")
    j = stdout.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return None
    try:
        return json.loads(stdout[i:j + 1])
    except json.JSONDecodeError:
        return None


def classify(level, run_data):
    if run_data.get("returncode", 1) != 0:
        return "error"

    result = run_data.get("result")
    if not result:
        return "error"

    step = result["steps"][0]
    reason = int(step["reason"])
    energy = float(step["energy"])

    if reason <= 0:
        return "diverged"
    if not math.isfinite(energy):
        return "diverged"

    ref = REFERENCE_ENERGY[level]
    rel = abs(energy - ref) / max(abs(ref), 1e-16)
    if rel <= 5e-5:
        return "ok"
    return "wrong-energy"


def run_case(level, case, timeout):
    mount_dir = str(Path.cwd())
    cmd = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "",
        "-v",
        f"{mount_dir}:/work",
        "-w",
        "/work",
        "fenics_test",
        "python3",
        "HyperElasticity3D_fenics/solve_HE_snes_newton.py",
        "--level",
        str(level),
        "--steps",
        "1",
        "--quiet",
        "--snes_type",
        case["snes_type"],
        "--linesearch",
        case["linesearch"],
        "--ksp_type",
        case["ksp_type"],
        "--pc_type",
        case["pc_type"],
        "--ksp_rtol",
        str(case["ksp_rtol"]),
        "--snes_atol",
        str(case["snes_atol"]),
    ]
    if case["use_objective"]:
        cmd.append("--use_objective")

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        shell=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - t0

    out = {
        "case": case,
        "wall_time": round(elapsed, 4),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    out["result"] = parse_json(proc.stdout)
    out["status"] = classify(level, out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--out", type=str, default="experiment_scripts/he_snes_phase2_l1.json")
    args = parser.parse_args()

    cases = build_cases()
    runs = []

    print(f"Running {len(cases)} SNES configurations on level {args.level}...")
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx:02d}/{len(cases)}] {case['name']}", flush=True)
        try:
            run_data = run_case(args.level, case, args.timeout)
        except subprocess.TimeoutExpired as exc:
            run_data = {
                "case": case,
                "wall_time": args.timeout,
                "returncode": -999,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "result": None,
                "status": "timeout",
            }
        runs.append(run_data)
        print(f"      -> status={run_data['status']} returncode={run_data['returncode']}")

    summary = {
        "level": args.level,
        "reference_energy": REFERENCE_ENERGY[args.level],
        "n_cases": len(cases),
        "counts": {
            "ok": sum(1 for r in runs if r["status"] == "ok"),
            "wrong-energy": sum(1 for r in runs if r["status"] == "wrong-energy"),
            "diverged": sum(1 for r in runs if r["status"] == "diverged"),
            "error": sum(1 for r in runs if r["status"] == "error"),
            "timeout": sum(1 for r in runs if r["status"] == "timeout"),
        },
    }

    payload = {"summary": summary, "runs": runs}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
