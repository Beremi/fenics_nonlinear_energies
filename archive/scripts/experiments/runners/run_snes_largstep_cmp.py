#!/usr/bin/env python3
"""Quick comparison: SNES with 24 large steps, varying HYPRE options with nullspace."""
import subprocess
import json
import os

cases = [
    ('nonull_defaults_24s', True, -1, -1),
    ('null_defaults_24s', False, -1, -1),
    ('null_n4_v2_24s', False, 4, 2),
    ('null_n6_v2_24s', False, 6, 2),
    ('null_n4_v2_48s', False, 4, 2),   # also 48 steps for comparison
    ('nonull_defaults_48s', True, -1, -1),
]

results = []
for name, no_null, nodal, vec in cases:
    nsteps = 48 if name.endswith('_48s') else 24
    cmd = ['docker', 'run', '--rm', '--entrypoint', '', '-e', 'PYTHONUNBUFFERED=1',
           '-v', f'{os.getcwd()}:/work', '-w', '/work', 'fenics_test',
           'python3', 'HyperElasticity3D_fenics/solve_HE_snes_newton.py',
           '--level', '1', '--steps', str(nsteps),
           '--ksp_type', 'gmres', '--pc_type', 'hypre',
           '--ksp_rtol', '1e-1', '--ksp_max_it', '500', '--snes_atol', '1e-5',
           '--hypre_nodal_coarsen', str(nodal), '--hypre_vec_interp_variant', str(vec)]
    if no_null:
        cmd.append('--no_near_nullspace')

    print(f"\n=== {name} (n={nodal} v={vec} steps={nsteps}) ===", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    out = proc.stdout
    i, j = out.find('{'), out.rfind('}')
    total, conv = 'FAIL', 0
    if i != -1 and j > i:
        try:
            d = json.loads(out[i:j + 1])
            total = sum(s['linear_iters'] for s in d['steps'])
            conv = sum(1 for s in d['steps'] if s['reason'] > 0)
        except BaseException:
            pass
    print(f"  lin={total}  conv={conv}/{nsteps}", flush=True)
    results.append((name, nodal, vec, nsteps, total, conv))

print('\n\n=== SUMMARY ===')
print(f'{"Case":<28} {"n":>3} {"v":>3} {"steps":>6} {"lin iters":>12} {"OK steps":>10}')
print('-' * 68)
for name, nodal, vec, nsteps, total, conv in results:
    print(f'{name:<28} {nodal:>3} {vec:>3} {nsteps:>6} {str(total):>12} {str(conv):>10}/{nsteps}')

# write summary
with open('experiment_scripts/snes_largstep_cmp.txt', 'w') as f:
    for name, nodal, vec, nsteps, total, conv in results:
        f.write(f'{name} n={nodal} v={vec} steps={nsteps}: lin={total} conv={conv}/{nsteps}\n')
print('\nSaved to experiment_scripts/snes_largstep_cmp.txt')
