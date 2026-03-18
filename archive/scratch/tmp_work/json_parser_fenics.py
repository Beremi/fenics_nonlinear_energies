import json, re

with open("tmp_work/fenics_gamg_n32.log") as f:
    text = f.read()

m = re.search(r'\{.*"level".*\}', text, re.DOTALL)
if m:
    j = json.loads(m.group(0))
    s = j['steps'][0]
    hist = s.get('linear_solve_history', [])
    ta = sum(h.get('assemble_time', 0) for h in hist)
    tp = sum(h.get('pc_setup_time', 0) for h in hist)
    ts = sum(h.get('solve_time', 0) for h in hist)
    print("FEniCS 32")
    nit = len(s.get('newton_iters', []))
    ttk = sum(x.get('ksp_its',0) for x in s.get('newton_iters', []))
    print(f"Total time:", s.get('total_time_s',0))
    print(f"Newton:", nit, "KSP:", ttk)
    print(f"Assembly:", ta)
    print(f"PC:", tp, "Solve:", ts)
