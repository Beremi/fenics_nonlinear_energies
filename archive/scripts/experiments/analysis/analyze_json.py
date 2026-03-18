import json
import math

with open('experiment_scripts/he_custom_l3_np16_gamg.json') as f:
    data = json.load(f)

total_dofs = data['total_dofs']
print(f'Total DOFs: {total_dofs}')
print(f'Number of steps: {len(data["steps"])}')

total_time = 0.0
total_newton = 0
total_ksp = 0

rows = []
for s in data['steps']:
    step = s['step']
    time_s = s['time']
    iters = s['iters']
    message = s['message']
    ksp_its = sum(h['ksp_its'] for h in s['history'])
    if s['history']:
        last_energy = s['history'][-1]['energy']
    else:
        last_energy = float('nan')
    total_time += time_s
    total_newton += iters
    total_ksp += ksp_its
    if 'Maximum' in message:
        status = 'MAX_ITER'
    elif 'converge' in message.lower():
        status = 'CONVERGED'
    else:
        status = 'OTHER'
    rows.append((step, last_energy, time_s, iters, ksp_its, message, status))

print()
hdr = f'{"Step":>4} | {"Energy (last hist)":>22} | {"Time (s)":>10} | {"Newton":>6} | {"KSP":>6} | Message'
print(hdr)
print('-' * len(hdr) + '-' * 40)
for r in rows:
    step, eng, t, ni, ki, msg, st = r
    if math.isnan(eng):
        eng_str = f'{"NaN":>22}'
    else:
        eng_str = f'{eng:>22.10f}'
    print(f'{step:>4} | {eng_str} | {t:>10.4f} | {ni:>6} | {ki:>6} | {msg} [{st}]')

print('-' * len(hdr) + '-' * 40)
print(f'Total time:            {total_time:.4f} s')
print(f'Total Newton iters:    {total_newton}')
print(f'Total KSP iters:       {total_ksp}')
