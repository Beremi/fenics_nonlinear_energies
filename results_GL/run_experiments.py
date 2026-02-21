#!/usr/bin/env python3
"""
Run all Ginzburg-Landau benchmarks and store results in results_GL/ directory.

This is meant to be run INSIDE the devcontainer (it calls mpirun internally).

Usage (from the repo root):
  python3 results_GL/run_experiments.py [--nprocs N] [--levels L1 L2 ...] [--tag TAG]
"""
import argparse
import datetime
import json
import os
import platform
import subprocess
import sys


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def get_git_dirty():
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        )
        return len(result.stdout.strip()) > 0
    except Exception:
        return None


def get_cpu_info():
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def get_dolfinx_version():
    try:
        import dolfinx
        return dolfinx.__version__
    except Exception:
        return "unknown"


def run_benchmark(script, nprocs, levels, json_path, extra_args=None):
    cmd = []
    if nprocs > 1:
        cmd = ["mpirun", "--allow-run-as-root", "-n", str(nprocs)]
    cmd += ["python3", script, "--levels"] + [str(l) for l in levels]
    cmd += ["--json", json_path]
    if extra_args:
        cmd += extra_args

    print(f"  Running: {' '.join(cmd)}")
    sys.stdout.flush()

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.stdout:
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.stderr:
        # Filter out common PETSc/MPI warnings
        stderr_lines = [l for l in result.stderr.split("\n")
                       if l and "WARNING" not in l and "unused" not in l.lower()]
        if stderr_lines:
            print("\n".join(stderr_lines[-20:]), file=sys.stderr)
    sys.stdout.flush()
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run all Ginzburg-Landau benchmarks")
    parser.add_argument("--nprocs", type=int, nargs="+", default=[1, 4, 8, 16],
                        help="MPI process counts (default: 1 4 8 16)")
    parser.add_argument("--levels", type=int, nargs="+", default=[5, 6, 7, 8, 9],
                        help="Mesh levels (default: 5 6 7 8 9)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional tag for the experiment directory name")
    parser.add_argument("--repeat", type=int, default=3,
                        help="Number of repetitions per configuration (default: 3)")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"{timestamp}_{args.tag}" if args.tag else timestamp

    results_dir = os.path.join("results_GL", exp_id)
    os.makedirs(results_dir, exist_ok=True)

    metadata = {
        "experiment_id": exp_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
        "hostname": platform.node(),
        "cpu": get_cpu_info(),
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "dolfinx_version": get_dolfinx_version(),
        "mesh_levels": args.levels,
        "nprocs_tested": args.nprocs,
        "repetitions": args.repeat,
    }

    meta_path = os.path.join(results_dir, "metadata.json")
    with open(meta_path, "w") as fp:
        json.dump(metadata, fp, indent=2)
    print(f"Experiment: {exp_id}")
    print(f"Metadata saved to {meta_path}")
    print(f"DOLFINx version: {metadata['dolfinx_version']}")
    print(f"CPU: {metadata['cpu']}")
    print("=" * 80)
    sys.stdout.flush()

    solvers = [
        ("snes_newton", "GinzburgLandau2D_fenics/solve_GL_snes_newton.py", []),
        ("custom_jaxversion", "GinzburgLandau2D_fenics/solve_GL_custom_jaxversion.py", ["--quiet"]),
    ]

    for solver_name, script, extra_args in solvers:
        for np_count in args.nprocs:
            for rep in range(1, args.repeat + 1):
                json_name = f"{solver_name}_np{np_count}_run{rep}.json"
                json_path = os.path.join(results_dir, json_name)

                print(f"\n--- {solver_name} | np={np_count} | run {rep}/{args.repeat} ---")
                sys.stdout.flush()

                rc = run_benchmark(script, np_count, args.levels, json_path, extra_args)
                if rc != 0:
                    print(f"  WARNING: benchmark exited with code {rc}")
                    sys.stdout.flush()

    print("\n" + "=" * 80)
    print(f"All experiments complete. Results in: {results_dir}/")
    print(f"Files:")
    for f in sorted(os.listdir(results_dir)):
        print(f"  {f}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
