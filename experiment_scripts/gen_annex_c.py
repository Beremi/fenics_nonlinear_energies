#!/usr/bin/env python3
"""Generate Annex C table from custom solver quarter-step JSON."""
import json
import math

d = json.load(open('/work/experiment_scripts/he_custom_quarter_steps_l1.json'))
print(f'mesh_level={d["mesh_level"]} total_dofs={d["total_dofs"]}')

steps = d['steps']
total_time = sum(s['time'] for s in steps)
total_ksp = sum(sum(h['ksp_its'] for h in s.get('history', [])) for s in steps)
total_newton = sum(s['iters'] for s in steps)

print(f'steps_completed={len(steps)}')
print(f'total_newton={total_newton}')
print(f'total_ksp={total_ksp}')
print(f'total_time={total_time:.2f}')
print()

print('| Step | Angle [°] | Time [s] | Newton its | Sum KSP its | Message |')
print('|-----:|----------:|---------:|-----------:|------------:|---------|')
for s in steps:
    angle_deg = math.degrees(s['angle'])
    ksp = sum(h['ksp_its'] for h in s.get('history', []))
    msg = s['message']
    print(f'| {s["step"]:4d} | {angle_deg:9.2f} | {s["time"]:8.4f} | {s["iters"]:10d} | {ksp:11d} | {msg} |')

print()
print(f'| **Total** | | **{total_time:.2f}** | **{total_newton}** | **{total_ksp}** | |')

with open('/work/experiment_scripts/annex_c_table.txt', 'w') as f:
    f.write(f'mesh_level={d["mesh_level"]} total_dofs={d["total_dofs"]} steps={len(steps)}\n')
    f.write(f'total_newton={total_newton} total_ksp={total_ksp} total_time={total_time:.2f}\n\n')
    f.write('| Step | Angle [°] | Time [s] | Newton its | Sum KSP its | Message |\n')
    f.write('|-----:|----------:|---------:|-----------:|------------:|---------|\n')
    for s in steps:
        angle_deg = math.degrees(s['angle'])
        ksp = sum(h['ksp_its'] for h in s.get('history', []))
        msg = s['message']
        f.write(f'| {s["step"]:4d} | {angle_deg:9.2f} | {s["time"]:8.4f} | {s["iters"]:10d} | {ksp:11d} | {msg} |\n')
    f.write(f'\n| **Total** | | **{total_time:.2f}** | **{total_newton}** | **{total_ksp}** | |\n')
print('Saved annex_c_table.txt')
