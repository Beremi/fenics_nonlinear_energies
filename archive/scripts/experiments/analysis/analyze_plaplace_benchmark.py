#!/usr/bin/env python3
"""
Analyse pLaplace benchmark JSON outputs and print a comparison table.

Usage:
    python3 experiment_scripts/analyze_plaplace_benchmark.py \\
        experiment_results_cache/bench_plaplace_sfd_32p.json \\
        experiment_results_cache/bench_plaplace_element_32p.json \\
        experiment_results_cache/bench_plaplace_fenics_32p.json

Each JSON file is either the output of solve_pLaplace_dof.py (--json flag)
or solve_pLaplace_custom_jaxversion.py (--json flag).
"""

import sys
import json


def _load(path):
    with open(path) as f:
        return json.load(f)


def _extract(data):
    """Normalise a result dict from either JAX-PETSc or FEniCS solver."""
    # JAX-PETSc: top-level has "metadata" + "results" list
    if "results" in data:
        results = data["results"]
        meta = data.get("metadata", {})
        # find level-9 result (or last)
        r = results[-1]
    else:
        # FEniCS solver may write list of results or single dict
        if isinstance(data, list):
            r = data[-1]
            meta = {}
        else:
            r = data
            meta = {}

    assembly_mode = r.get("assembly_mode", meta.get("assembly_mode", "?"))
    pc_type = r.get("pc_type", meta.get("linear_solver", {}).get("pc_type", "?"))
    nprocs = r.get("nprocs", meta.get("nprocs", "?"))
    dofs = r.get("dofs", r.get("total_dofs", "?"))
    mesh_level = r.get("mesh_level", "?")

    setup_time = r.get("setup_time", float("nan"))
    solve_time = r.get("solve_time", float("nan"))
    total_time = r.get("total_time", float("nan"))

    iters = r.get("iters", "?")
    total_ksp_its = r.get("total_ksp_its", "?")
    energy = r.get("energy", float("nan"))
    message = r.get("message", "?")

    asm_time = r.get("asm_time_cumulative", float("nan"))
    ksp_time = r.get("ksp_time_cumulative", float("nan"))
    n_colors = r.get("n_colors", "?")

    return {
        "mode": assembly_mode,
        "pc": pc_type,
        "np": nprocs,
        "level": mesh_level,
        "dofs": dofs,
        "n_colors": n_colors,
        "setup_s": setup_time,
        "asm_s": asm_time,
        "ksp_s": ksp_time,
        "solve_s": solve_time,
        "total_s": total_time,
        "newton_iters": iters,
        "ksp_iters": total_ksp_its,
        "energy": energy,
        "msg": message,
    }


def main():
    paths = sys.argv[1:]
    if not paths:
        print("Usage: analyze_plaplace_benchmark.py <json1> [json2 ...]")
        sys.exit(1)

    rows = []
    for path in paths:
        try:
            data = _load(path)
            row = _extract(data)
            row["file"] = path
            rows.append(row)
        except Exception as e:
            print(f"  WARNING: could not load {path}: {e}", file=sys.stderr)

    if not rows:
        print("No results loaded.")
        sys.exit(1)

    # ---- Print comparison table ----
    col_w = {
        "mode": 12, "np": 4, "level": 5, "dofs": 9, "n_colors": 8,
        "setup_s": 8, "asm_s": 8, "ksp_s": 8, "solve_s": 8, "total_s": 8,
        "newton_iters": 6, "ksp_iters": 7, "energy": 14,
    }
    headers = {
        "mode": "mode", "np": "np", "level": "lvl", "dofs": "dofs",
        "n_colors": "n_colors", "setup_s": "setup(s)", "asm_s": "asm(s)",
        "ksp_s": "ksp(s)", "solve_s": "solve(s)", "total_s": "total(s)",
        "newton_iters": "N_it", "ksp_iters": "KSP_it", "energy": "energy",
    }

    keys = list(col_w.keys())
    header_line = " | ".join(f"{headers[k]:{col_w[k]}s}" for k in keys)
    sep = "-+-".join("-" * col_w[k] for k in keys)

    print()
    print("pLaplace 2D benchmark results")
    print("=" * len(header_line))
    print(header_line)
    print(sep)

    for r in rows:
        def _fmt(k, v):
            w = col_w[k]
            if isinstance(v, float):
                return f"{v:{w}.4f}"
            return f"{str(v):{w}s}"
        line = " | ".join(_fmt(k, r[k]) for k in keys)
        print(line)

    print()

    # ---- Per-row detail ----
    for r in rows:
        print(f"  [{r['mode']}]  message: {r['msg']}")

    print()

    # ---- Speedup vs FEniCS ----
    fenics = next((r for r in rows if r["mode"] == "fenics"), None)
    if fenics and not any(f != f for f in [fenics["total_s"]]):
        print("Speedup vs FEniCS (total_time):")
        for r in rows:
            if r["mode"] != "fenics" and isinstance(r["total_s"], float):
                ratio = fenics["total_s"] / r["total_s"] if r["total_s"] > 0 else float("nan")
                print(f"  {r['mode']:12s}: {ratio:.2f}x  ({r['total_s']:.3f}s vs {fenics['total_s']:.3f}s)")
        print()

    # ---- Assembly speedup ----
    fenics_asm = fenics["asm_s"] if fenics else None
    if fenics_asm and fenics_asm == fenics_asm:
        print("Assembly speedup vs FEniCS (asm_time_cumulative):")
        for r in rows:
            if r["mode"] != "fenics" and isinstance(r["asm_s"], float):
                ratio = fenics_asm / r["asm_s"] if r["asm_s"] > 0 else float("nan")
                print(f"  {r['mode']:12s}: {ratio:.2f}x  ({r['asm_s']:.3f}s vs {fenics_asm:.3f}s)")
        print()


if __name__ == "__main__":
    main()
