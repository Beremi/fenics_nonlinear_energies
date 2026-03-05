import json

with open("cleaned_out.json", "r") as f:
    data = json.load(f)

for step in data['steps']:
    s = step['step']
    total_time = step['time']
    nits = step['nit']
    print(f"--- Step {s} --- (Time: {total_time:.3f}s | Newton: {nits})")
    
    if 'linear_timings' in step and len(step['linear_timings']) > 0:
        
        lin = step['linear_timings']
        print(f"Avgs per Newton step ({len(lin)} recorded solvers):")
        keys_to_avg = [
            'wait_imbalance',
            'assemble_total_time',
            'assemble_hvp_compute',
            'pc_setup_time',
            'solve_time',
            'linear_total_time'
        ]
        
        for k in keys_to_avg:
            avg = sum(l[k] for l in lin) / len(lin)
            print(f"  {k}: {avg:.4f}s")
            
