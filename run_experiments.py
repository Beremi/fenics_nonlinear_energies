#!/usr/bin/env python3
"""
Run all p-Laplace benchmarks and store results in the results/ directory.

This script:
1. Collects system/environment metadata (git commit, date, DOLFINx version, CPU info)
2. Runs both solver variants (SNES Newton + Custom Newton) in serial and parallel
3. Saves results as JSON files in results/<experiment_id>/

Usage (inside the devcontainer):
  python3 run_experiments.py [--nprocs N] [--levels L1 L2 ...] [--tag TAG]

The script must be run OUTSIDE of mpirun (it calls mpirun internally).
"""
import argparse
import datetime
import json
import os
import platform
import subprocess
import sys


def get_git_commit():
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def get_git_dirty():
    """Check if working tree is dirty."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        )
        return len(result.stdout.strip()) > 0
    except Exception:
        return None


def get_cpu_info():
    """Get CPU model name."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def get_dolfinx_version():
    """Get DOLFINx version from inside container."""
    try:
        result = subprocess.run(
            ["python3", "-c", "import dolfinx; print(dolfinx.__version__)"],
            capture_output=True, text=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def run_benchmark(script, nprocs, levels, json_path, extra_args=None):
    """Run a benchmark script with optional MPI."""
    cmd = []
    if nprocs > 1:
        cmd = ["mpirun", "--allow-run-as-root", "-n", str(nprocs)]
    cmd += ["python3", script, "--levels"] + [str(l) for l in levels]
    cmd += ["--json", json_path]
    if extra_args:
        cmd += extra_args

    print(f"  Running: {' '.join(cmd)}")
    sys.stdout.flush()

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    sys.stdout.flush()
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run all p-Laplace benchmarks")
    parser.add_argument("--nprocs", type=int, nargs="+", default=[1, 4, 8],
                        help="MPI process counts to test (default: 1 4 8)")
    parser.add_argument("--levels", type=int, nargs="+", default=[5, 6, 7, 8, 9],
                        help="Mesh levels (default: 5 6 7 8 9)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional tag for the experiment directory name")
    parser.add_argument("--repeat", type=int, default=3,
                        help="Number of repetitions per configuration (default: 3)")
    args = parser.parse_args()

    # Generate experiment ID
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"{timestamp}"
    if args.tag:
        exp_id = f"{timestamp}_{args.tag}"

    # Create results directory
    results_dir = os.path.join("results", exp_id)
    os.makedirs(results_dir, exist_ok=True)

    # Collect metadata
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

    # Save metadata
    meta_path = os.path.join(results_dir, "metadata.json")
    with open(meta_path, "w") as fp:
        json.dump(metadata, fp, indent=2)
    print(f"Experiment: {exp_id}")
    print(f"Metadata saved to {meta_path}")
    print(f"Git commit: {metadata['git_commit']}")
    print(f"DOLFINx version: {metadata['dolfinx_version']}")
    print(f"CPU: {metadata['cpu']}")
    print("=" * 80)
    sys.stdout.flush()

    solvers = [
        ("snes_newton", "solve_pLaplace_snes_newton.py", []),
        ("custom_newton", "solve_pLaplace_custom_newton.py", ["--quiet"]),
    ]

    # Run all combinations
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

    # Create summary
    print("\n" + "=" * 80)
    print(f"All experiments complete. Results in: {results_dir}/")
    print(f"Files:")
    for f in sorted(os.listdir(results_dir)):
        print(f"  {f}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
