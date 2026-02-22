#!/usr/bin/env python3
"""Parse all GL benchmark JSONs and print summary tables."""
import json
import os
import sys
from collections import defaultdict
import statistics

RESULTS_DIR = "results_GL/experiment_001"

# Collect data: data[solver][nprocs][mesh_level] = [list of (time, iters, energy, converged)]
data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for fname in sorted(os.listdir(RESULTS_DIR)):
    if not fname.endswith(".json"):
        continue
    fpath = os.path.join(RESULTS_DIR, fname)
    with open(fpath) as f:
        d = json.load(f)
    solver = d["metadata"]["solver"]
    nprocs = d["metadata"]["nprocs"]
    for r in d["results"]:
        lvl = r["mesh_level"]
        t = r["time"]
        iters = r["iters"]
        E = r["energy"]
        # Check convergence: correct energy should be ~0.345-0.347
        converged = abs(E - 0.3456) < 0.01
        reason = r.get("converged_reason", None)
        data[solver][nprocs][lvl].append({
            "time": t, "iters": iters, "energy": E, 
            "converged": converged, "reason": reason,
        })

def median_of(runs, key):
    vals = [r[key] for r in runs if r["converged"]]
    if vals:
        return statistics.median(vals)
    return None

def any_converged(runs):
    return any(r["converged"] for r in runs)

print("=" * 80)
print("Custom Newton (JAX-version) — all configs")
print("=" * 80)
solver = "custom_jaxversion"
for nprocs in sorted(data[solver].keys()):
    print(f"\n  np={nprocs}:")
    for lvl in sorted(data[solver][nprocs].keys()):
        runs = data[solver][nprocs][lvl]
        t_med = median_of(runs, "time")
        iters = runs[0]["iters"] if runs else "?"
        E = runs[0]["energy"] if runs else "?"
        conv_count = sum(1 for r in runs if r["converged"])
        print(f"    L{lvl}: dofs={runs[0].get('dofs', '?')} time={t_med:.4f}s iters={iters} J={E:.4f} ({conv_count}/{len(runs)} converged)")

print("\n" + "=" * 80)
print("SNES Newton — all configs")
print("=" * 80)
solver = "snes_newton"
for nprocs in sorted(data[solver].keys()):
    print(f"\n  np={nprocs}:")
    for lvl in sorted(data[solver][nprocs].keys()):
        runs = data[solver][nprocs][lvl]
        t_med = median_of(runs, "time")
        conv_count = sum(1 for r in runs if r["converged"])
        if conv_count > 0:
            conv_runs = [r for r in runs if r["converged"]]
            iters = conv_runs[0]["iters"]
            E = conv_runs[0]["energy"]
            print(f"    L{lvl}: time={t_med:.4f}s iters={iters} J={E:.4f} ({conv_count}/{len(runs)} converged)")
        else:
            E_vals = [r["energy"] for r in runs]
            reasons = [r["reason"] for r in runs]
            print(f"    L{lvl}: FAILED ({conv_count}/{len(runs)}) energies={E_vals} reasons={reasons}")

# Now print markdown tables
print("\n\n" + "=" * 80)
print("MARKDOWN TABLES")
print("=" * 80)

# Table 1: Custom Newton, all procs
solver = "custom_jaxversion"
nprocs_list = sorted(data[solver].keys())
levels = sorted(data[solver][1].keys())

# Get DOF counts
dof_map = {}
for lvl in levels:
    runs = data[solver][1][lvl]
    if runs:
        # Read from json
        pass

# Read a json to get dofs
with open(os.path.join(RESULTS_DIR, "custom_jaxversion_np1_run1.json")) as f:
    d = json.load(f)
for r in d["results"]:
    dof_map[r["mesh_level"]] = r["total_dofs"]

print("\n### Custom Newton (JAX-version algorithm) — FEniCS + PETSc\n")
header = "| lvl | dofs |"
sep = "| --- | --- |"
for np in nprocs_list:
    header += f" time ({np} proc) | iters |"
    sep += " --- | --- |"
header += " J(u) |"
sep += " --- |"
print(header)
print(sep)

for lvl in levels:
    dofs = dof_map.get(lvl, "?")
    row = f"| {lvl} | {dofs} |"
    E_final = None
    for np in nprocs_list:
        runs = data[solver][np][lvl]
        t_med = median_of(runs, "time")
        if t_med is not None:
            iters = [r["iters"] for r in runs if r["converged"]][0]
            E = [r["energy"] for r in runs if r["converged"]][0]
            E_final = E
            row += f" {t_med:.3f} | {iters} |"
        else:
            row += " FAIL | — |"
    row += f" {E_final:.4f} |" if E_final else " — |"
    print(row)

# Table 2: SNES Newton
solver = "snes_newton"
print("\n### SNES Newton (serial only — unreliable in parallel)\n")

# Read dofs from snes json  
with open(os.path.join(RESULTS_DIR, "snes_newton_np1_run1.json")) as f:
    d = json.load(f)
snes_dof_map = {}
for r in d["results"]:
    snes_dof_map[r["mesh_level"]] = r["total_dofs"]

nprocs_list = sorted(data[solver].keys())

header = "| lvl | dofs |"
sep = "| --- | --- |"
for np in nprocs_list:
    header += f" time ({np} proc) | iters | J(u) |"
    sep += " --- | --- | --- |"
print(header)
print(sep)

for lvl in levels:
    dofs = snes_dof_map.get(lvl, dof_map.get(lvl, "?"))
    row = f"| {lvl} | {dofs} |"
    for np in nprocs_list:
        runs = data[solver][np][lvl]
        conv_runs = [r for r in runs if r["converged"]]
        if conv_runs:
            t_med = statistics.median([r["time"] for r in conv_runs])
            iters = conv_runs[0]["iters"]
            E = conv_runs[0]["energy"]
            row += f" {t_med:.3f} | {iters} | {E:.4f} |"
        else:
            # Show as failed
            reasons = set(r.get("reason", "?") for r in runs)
            row += f" **FAIL** | — | — |"
    print(row)

# Table 3: All solvers comparison
print("\n### All Solver Configurations Comparison\n")
print("| lvl | dofs | Custom 1-proc | iters | Custom 4-proc | iters | Custom 8-proc | iters | Custom 16-proc | iters | SNES 1-proc | iters | J(u) |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")

for lvl in levels:
    dofs = dof_map.get(lvl, "?")
    row = f"| {lvl} | {dofs} |"
    E_final = None
    for np in [1, 4, 8, 16]:
        runs = data["custom_jaxversion"][np][lvl]
        t_med = median_of(runs, "time")
        if t_med is not None:
            iters = [r["iters"] for r in runs if r["converged"]][0]
            E = [r["energy"] for r in runs if r["converged"]][0]
            E_final = E
            row += f" {t_med:.3f} | {iters} |"
        else:
            row += " FAIL | — |"
    # SNES serial
    runs = data["snes_newton"][1][lvl]
    conv_runs = [r for r in runs if r["converged"]]
    if conv_runs:
        t_med = statistics.median([r["time"] for r in conv_runs])
        iters = conv_runs[0]["iters"]
        row += f" {t_med:.3f} | {iters} |"
    else:
        row += " FAIL | — |"
    row += f" {E_final:.4f} |" if E_final else " — |"
    print(row)

print("\n\nDone.")
