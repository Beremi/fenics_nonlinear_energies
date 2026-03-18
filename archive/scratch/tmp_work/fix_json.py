import json, sys

data = sys.stdin.read()
found = False
for i in range(len(data)):
    if data[i] == '{':
        try:
            j = json.loads(data[i:])
            if 'steps' in j:
                step = j['steps'][0]
                tt = step.get('time', 0)
                nit = step.get('nit', 0)
                tk = step.get('linear_iters', 0)
                
                # Check where timings are stored
                hist = step.get('linear_timing', [])
                if not hist:
                    hist = step.get('linear_solve_history', [])
                    
                ta = sum(h.get('assemble_total_time', 0) for h in hist)
                th = sum(h.get('assemble_hvp_compute', 0) for h in hist)
                tc = sum(h.get('assemble_coo_assembly', 0) for h in hist)
                tp = sum(h.get('pc_setup_time', 0) for h in hist)
                ts = sum(h.get('solve_time', 0) for h in hist)
                print(f"JAX+PETSc L3 np=32:\nTotal Time: {tt:.3f}\nNewton: {nit}\nKSP: {tk}\nAssembly: {ta:.3f}\n  HVP: {th:.3f}\n  COO: {tc:.3f}\nPC Setup: {tp:.3f}\nSolve: {ts:.3f}")
                found = True
                break
        except Exception:
            pass

if not found:
    print("Could not parse JSON")
