import json, re

with open("tmp_work/jax_sfd_n32.log") as f:
    text = f.read()

m = re.search(r'\{.*"level".*\}', text, re.DOTALL)
if m:
    j = json.loads(m.group(0))
    s = j['steps'][0]
    hist = s.get('linear_timing', [])
    ta = sum(h.get('assemble_total_time', 0) for h in hist)
    th = sum(h.get('assemble_hvp_compute', 0) for h in hist)
    tc = sum(h.get('assemble_coo_assembly', 0) for h in hist)
    tp = sum(h.get('pc_setup_time', 0) for h in hist)
    ts = sum(h.get('solve_time', 0) for h in hist)
    print("JAX+PETSc SFD 32")
    print(f"Total time:", s['time'])
    print(f"Newton:", s['nit'], "KSP:", s['linear_iters'])
    print(f"Assembly:", ta)
    print(f"HVP:", th, "COO:", tc)
    print(f"PC:", tp, "Solve:", ts)
else:
    print("No valid JSON found")
