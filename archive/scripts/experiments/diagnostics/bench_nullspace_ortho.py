#!/usr/bin/env python3
"""
Compare HYPRE vec_interp_variant options with the orthonormalized near-nullspace.

Variants tested (all use --level 1, --steps 48, --ksp_rtol 1e-1, --ksp_max_it 500):
  baseline_nonull_hypredefault  -- no nullspace, HYPRE defaults (ref=14521)
  null_ortho_hypredefault       -- ortho nullspace, HYPRE defaults (should ignore it)
  null_ortho_n6_v1              -- ortho nullspace, nodal_coarsen=6, vec_interp=1
  null_ortho_n6_v2              -- ortho nullspace, nodal_coarsen=6, vec_interp=2
  null_ortho_n6_v3              -- ortho nullspace, nodal_coarsen=6, vec_interp=3
  null_ortho_n1_v1              -- ortho nullspace, nodal_coarsen=1, vec_interp=1
  null_ortho_n1_v2              -- ortho nullspace, nodal_coarsen=1, vec_interp=2
"""
import json
import subprocess
import time
from pathlib import Path

OUTDIR = Path(__file__).parent / "he_nullspace_ortho"
OUTDIR.mkdir(exist_ok=True)

SOLVER = "HyperElasticity3D_fenics/solve_HE_snes_newton.py"
LEVEL = 1
STEPS = 48
KSP_RTOL = 1e-1
KSP_MAX_IT = 500

CASES = [
    {"name": "baseline_nonull_hypredefault",
     "no_near_nullspace": True, "nodal": -1, "vec": -1},
    {"name": "null_ortho_hypredefault",
     "no_near_nullspace": False, "nodal": -1, "vec": -1},
    {"name": "null_ortho_n6_v1",
     "no_near_nullspace": False, "nodal": 6, "vec": 1},
    {"name": "null_ortho_n6_v2",
     "no_near_nullspace": False, "nodal": 6, "vec": 2},
    {"name": "null_ortho_n6_v3",
     "no_near_nullspace": False, "nodal": 6, "vec": 3},
    {"name": "null_ortho_n1_v1",
     "no_near_nullspace": False, "nodal": 1, "vec": 1},
    {"name": "null_ortho_n1_v2",
     "no_near_nullspace": False, "nodal": 1, "vec": 2},
]


def run_case(case):
    mount = str(Path.cwd())
    cmd = [
        "docker", "run", "--rm", "--entrypoint", "",
        "-e", "PYTHONUNBUFFERED=1",
        "-v", f"{mount}:/work", "-w", "/work",
        "fenics_test", "python3", SOLVER,
        "--level", str(LEVEL),
        "--steps", str(STEPS),
        "--ksp_type", "gmres",
        "--pc_type", "hypre",
        "--ksp_rtol", str(KSP_RTOL),
        "--ksp_max_it", str(KSP_MAX_IT),
        "--snes_atol", "1e-5",
        "--hypre_nodal_coarsen", str(case["nodal"]),
        "--hypre_vec_interp_variant", str(case["vec"]),
    ]
    if case["no_near_nullspace"]:
        cmd.append("--no_near_nullspace")

    print(f"\n=== {case['name']} ===")
    print("CMD:", " ".join(cmd[-15:]))
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.perf_counter() - t0

    # Parse JSON result
    result = None
    stdout = proc.stdout
    i = stdout.find("{")
    j = stdout.rfind("}")
    if i != -1 and j > i:
        try:
            result = json.loads(stdout[i:j + 1])
        except json.JSONDecodeError:
            pass

    total_lin = None
    steps_ok = 0
    if result:
        total_lin = sum(s["linear_iters"] for s in result["steps"])
        steps_ok = sum(1 for s in result["steps"] if s["reason"] > 0)

    out = {
        "case": case,
        "wall_time": round(elapsed, 2),
        "returncode": proc.returncode,
        "total_linear_iters": total_lin,
        "steps_converged": steps_ok,
        "result": result,
    }

    print(f"  Wall time: {elapsed:.1f}s")
    print(f"  Return code: {proc.returncode}")
    print(f"  Steps converged: {steps_ok}/{STEPS}")
    print(f"  Total linear iters: {total_lin}")
    if proc.returncode != 0:
        print("  STDERR:", proc.stderr[-500:])

    outfile = OUTDIR / f"{case['name']}.json"
    with open(outfile, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {outfile}")
    return out


if __name__ == "__main__":
    print(f"Running {len(CASES)} cases, level={LEVEL}, steps={STEPS}, ksp_rtol={KSP_RTOL}")
    summary = []
    for case in CASES:
        try:
            r = run_case(case)
            summary.append({
                "name": case["name"],
                "total_linear_iters": r["total_linear_iters"],
                "steps_converged": r["steps_converged"],
                "wall_time": r["wall_time"],
            })
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT for {case['name']}")
            summary.append({"name": case["name"], "total_linear_iters": "TIMEOUT",
                            "steps_converged": 0, "wall_time": 600})
        except Exception as e:
            print(f"  ERROR: {e}")
            summary.append({"name": case["name"], "total_linear_iters": "ERROR",
                            "steps_converged": 0, "wall_time": 0})

    print("\n\n=== SUMMARY ===")
    print(f"{'Case':<40} {'Total lin iters':>18} {'Steps OK':>10} {'Time(s)':>10}")
    print("-" * 82)
    for s in summary:
        print(f"{s['name']:<40} {str(s['total_linear_iters']):>18} {str(s['steps_converged']):>10} {str(s['wall_time']):>10}")

    with open(OUTDIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {OUTDIR}/summary.json")
