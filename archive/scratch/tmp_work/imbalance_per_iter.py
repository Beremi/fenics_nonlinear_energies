import json, sys
data = sys.stdin.read()
idx = data.find('"mesh_level":')
if idx != -1:
    idx = data.rfind('{', 0, idx)
    j = json.loads(data[idx:])
    h = j['steps'][0]['linear_timing']
    for i, x in enumerate(h):
        print(f"Iter {i}: HVP={x.get('assemble_hvp_compute',0):.4f} Wait={x.get('wait_imbalance',0):.4f} COO={x.get('assemble_coo_assembly',0):.4f}")
