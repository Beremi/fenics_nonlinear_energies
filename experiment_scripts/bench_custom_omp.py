#!/usr/bin/env python3
"""
Benchmark Custom OMP coloring (with and without RCM reordering).

Reports: n_colors, total time, C-only time for each configuration.
Run inside Docker: python bench_custom_omp.py
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_coloring.coloring_custom import (
    _get_lib, _i32, _ptr, color_custom, color_custom_omp,
)
from graph_coloring.mesh_loader import PROBLEMS, load_adjacency
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee
import numpy as np
import ctypes

lib = _get_lib()
omp_threads = lib.custom_has_openmp()
print(f"OpenMP max threads: {omp_threads}")

THREAD_COUNTS = [2, 4, 8, 16]

for prob_name, info in PROBLEMS.items():
    print(f"\n{'='*70}")
    print(f"  {prob_name}")
    print(f"{'='*70}")

    for lvl in info["levels"]:
        h5 = info["path"](lvl)
        if not os.path.exists(h5):
            continue
        A = load_adjacency(h5)
        n = A.shape[0]

        # Compute A² once
        t0 = time.perf_counter()
        A2 = sp.csc_matrix(A @ A)
        t_a2 = time.perf_counter() - t0

        nnz = A2.nnz
        print(f"\n  Level {lvl}: N={n:>10,}  nnz(A²)={nnz:>14,}  A²={t_a2:.4f}s")

        # --- Serial baseline ---
        ip = _i32(A2.indptr); ix = _i32(A2.indices)
        colors = np.zeros(n, dtype=np.int32)
        t0 = time.perf_counter()
        nc = lib.custom_greedy_color(ctypes.c_int(n), _ptr(ip), _ptr(ix), _ptr(colors))
        t_ser = time.perf_counter() - t0
        print(f"    Serial:          {nc:3d} colors  C={t_ser:.4f}s  total={t_a2+t_ser:.4f}s")

        # --- OMP without reorder ---
        for nt in THREAD_COUNTS:
            colors2 = np.zeros(n, dtype=np.int32)
            t0 = time.perf_counter()
            nc2 = lib.custom_greedy_color_omp(
                ctypes.c_int(n), _ptr(ip), _ptr(ix), _ptr(ip), _ptr(ix),
                _ptr(colors2), ctypes.c_int(nt))
            t_c = time.perf_counter() - t0
            # Validate
            ok = True
            A2_csr = sp.csr_matrix(A2)
            for i in range(min(n, 5000)):  # spot-check
                for j in A2_csr.indices[A2_csr.indptr[i]:A2_csr.indptr[i+1]]:
                    if i != j and colors2[i] == colors2[j]:
                        ok = False; break
                if not ok: break
            status = "OK" if ok else "CONFLICT"
            print(f"    OMP nt={nt:2d}:        {nc2:3d} colors  C={t_c:.4f}s  total={t_a2+t_c:.4f}s  [{status}]")

        # --- OMP with RCM reorder ---
        t0 = time.perf_counter()
        perm = reverse_cuthill_mckee(A2)
        t_rcm = time.perf_counter() - t0

        t0 = time.perf_counter()
        A2_perm = sp.csc_matrix(A2[perm][:, perm])
        t_reorder = time.perf_counter() - t0

        ip_p = _i32(A2_perm.indptr); ix_p = _i32(A2_perm.indices)
        inv_perm = np.empty(n, dtype=np.int32)
        inv_perm[perm] = np.arange(n, dtype=np.int32)

        t_overhead = t_rcm + t_reorder
        print(f"    RCM overhead:    rcm={t_rcm:.4f}s  reorder={t_reorder:.4f}s  total_oh={t_overhead:.4f}s")

        for nt in THREAD_COUNTS:
            colors3 = np.zeros(n, dtype=np.int32)
            t0 = time.perf_counter()
            nc3 = lib.custom_greedy_color_omp(
                ctypes.c_int(n), _ptr(ip_p), _ptr(ix_p), _ptr(ip_p), _ptr(ix_p),
                _ptr(colors3), ctypes.c_int(nt))
            t_c_r = time.perf_counter() - t0
            final = colors3[inv_perm]
            t_total = t_a2 + t_overhead + t_c_r

            # Spot-check validation
            ok = True
            for i in range(min(n, 5000)):
                for j in A2_csr.indices[A2_csr.indptr[i]:A2_csr.indptr[i+1]]:
                    if i != j and final[i] == final[j]:
                        ok = False; break
                if not ok: break
            status = "OK" if ok else "CONFLICT"
            print(f"    RCM+OMP nt={nt:2d}:    {nc3:3d} colors  C={t_c_r:.4f}s  total={t_total:.4f}s  [{status}]")
