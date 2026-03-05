import json
import sys
import re

fname = sys.argv[1]

with open(fname, "r") as f:
    text = f.read()

m = re.search(r'(\{.*\})', text, flags=re.DOTALL)
if m:
    data = json.loads(m.group(1))
else:
    data = json.loads(text)

step0 = data["steps"][0]
print(f"Total time step 1: {step0['time']:.3f} s")
print(f"Total Newton iters: {step0['nit']}")
print(f"Total KSP iters: {step0['linear_iters']}")

if 'linear_timing' in step0:
    lin = step0['linear_timing']
    print(f"\nAverages across {len(lin)} Newton iterations:")
    keys = ['assemble_total_time', 'assemble_hvp_compute', 'assemble_p2p_exchange', 'wait_imbalance', 'pc_setup_time', 'solve_time']
    for k in keys:
        avg = sum(l.get(k, 0) for l in lin) / len(lin)
        print(f"  {k}: {avg:.4f} s")
        if k == 'wait_imbalance' or k == 'assemble_hvp_compute':
            pass
