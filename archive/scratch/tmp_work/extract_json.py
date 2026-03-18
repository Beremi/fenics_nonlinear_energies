import json, sys

data = sys.stdin.read()
# Start looking from the end to find the valid JSON block
for i in range(len(data)):
    if data[i] == '{':
        try:
            j = json.loads(data[i:])
            step = j['steps'][0]
            tt = step.get('total_time_s', 0)
            iters = step.get('newton', [])
            if not iters:
                iters = step.get('newton_iters', [])
            tk = sum(it.get('ksp_its', 0) for it in iters)
            hist = step.get('linear_solve_history', [])
            
            ta = sum(h.get('assemble_time', h.get('assemble_total_time', h.get('total', 0))) for h in hist)
            th = sum(h.get('assemble_hvp_compute', h.get('elem_hessian_compute', 0)) for h in hist)
            t_ext = sum(h.get('assemble_extraction', 0) for h in hist)
            tc = sum(h.get('assemble_coo_assembly', h.get('coo_assembly', 0)) for h in hist)
            tp = sum(h.get('pc_setup_time', 0) for h in hist)
            ts = sum(h.get('solve_time', 0) for h in hist)
            
            print(f"JAX ELEMENT L3 np=16")
            print(f"Total Time: {tt:.3f}")
            print(f"Newton: {len(iters)}")
            print(f"KSP: {tk}")
            print(f"Assembly: {ta:.3f}")
            print(f"HVP/Elem: {th:.3f}")
            print(f"Extraction / Scatter: {t_ext:.3f}")
            print(f"COO: {tc:.3f}")
            print(f"PC: {tp:.3f}")
            print(f"Solve: {ts:.3f}")
            sys.exit(0)
        except json.JSONDecodeError:
            pass
