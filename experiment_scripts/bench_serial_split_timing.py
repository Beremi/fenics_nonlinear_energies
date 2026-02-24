#!/usr/bin/env python3
"""
Measure A² setup and greedy coloring times separately for the Custom C backend.

Usage:
    python bench_serial_split_timing.py
"""
from graph_coloring.mesh_loader import PROBLEMS, load_adjacency
from graph_coloring.coloring_custom import _get_lib, _i32, _ptr
import ctypes
import scipy.sparse as sp
import numpy as np
import sys
import os
import time
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


lib = _get_lib()

BENCHMARKS = {
    "pLaplace2D": list(range(5, 10)),
    "GinzburgLandau2D": list(range(5, 10)),
    "HyperElasticity3D": list(range(1, 5)),
}

results = {}

for prob_name, levels in BENCHMARKS.items():
    print(f"\n{'=' * 70}")
    print(f"  {prob_name}")
    print(f"{'=' * 70}")
    print(f"  {'Level':>5}  {'N':>10}  {'A² (s)':>10}  {'Greedy (s)':>10}  {'Total (s)':>10}  {'Colors':>6}")

    prob_results = []
    for lvl in levels:
        h5 = PROBLEMS[prob_name]["path"](lvl)
        if not os.path.isfile(h5):
            print(f"  {lvl:>5}  (file not found)")
            continue

        A = load_adjacency(h5)

        # Time A² computation (CSR — A² is symmetric, skip CSC conversion)
        t0 = time.perf_counter()
        A2 = sp.csr_matrix(A @ A)
        t_a2 = time.perf_counter() - t0

        n = A2.shape[0]
        indptr = _i32(A2.indptr)
        indices = _i32(A2.indices)
        colors = np.zeros(n, dtype=np.int32)

        # Time greedy coloring only
        t0 = time.perf_counter()
        nc = lib.custom_greedy_color(ctypes.c_int(n), _ptr(indptr), _ptr(indices), _ptr(colors))
        t_greedy = time.perf_counter() - t0

        total = t_a2 + t_greedy
        print(f"  {lvl:>5}  {n:>10,}  {t_a2:>10.4f}  {t_greedy:>10.4f}  {total:>10.4f}  {nc:>6}")

        prob_results.append({
            "level": lvl, "n": n, "t_a2": round(t_a2, 4),
            "t_greedy": round(t_greedy, 4), "total": round(total, 4), "colors": nc
        })

    results[prob_name] = prob_results

# Also dump JSON for easy parsing
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp_work", "bench_serial_split.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nJSON saved to {out_path}")
