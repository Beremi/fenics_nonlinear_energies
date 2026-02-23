#!/usr/bin/env python3
"""Generate markdown table from custom solver JSON. Usage: gen_table.py <json_path>"""
import json
import math
import sys

path = sys.argv[1]
d = json.load(open(path))

steps = d['steps']
total_time = sum(s['time'] for s in steps)
total_ksp = sum(sum(h['ksp_its'] for h in s.get('history', [])) for s in steps)
total_newton = sum(s['iters'] for s in steps)

print(f'dofs={d["total_dofs"]} steps={len(steps)} newton={total_newton} ksp={total_ksp} time={total_time:.2f}')
print()
print('| Step | Angle [°] | Time [s] | Newton its | Sum KSP its | Message |')
print('|-----:|----------:|---------:|-----------:|------------:|---------|')
for s in steps:
    angle_deg = math.degrees(s['angle'])
    ksp = sum(h['ksp_its'] for h in s.get('history', []))
    print(f'| {s["step"]:4d} | {angle_deg:9.2f} | {s["time"]:8.4f} | {s["iters"]:10d} | {ksp:11d} | {s["message"]} |')
print()
print(f'| **Total** | | **{total_time:.2f}** | **{total_newton}** | **{total_ksp:,}** | |')
