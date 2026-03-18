import json, sys
data = sys.stdin.read()
found = False
for i in range(len(data)):
    if data[i] == '{':
        try:
            j = json.loads(data[i:])
            step = j['steps'][0]
            tt = step.get('total_time_s', 0)
            iters = step.get('newton_iters', [])
            tk = sum(it.get('ksp_its', 0) for it in iters)
            hist = step.get('linear_solve_history', [])
            ta = sum(h.get('assemble_time', 0) for h in hist)
            tp = sum(h.get('pc_setup_time', 0) for h in hist)
            ts = sum(h.get('solve_time', 0) for h in hist)
            print(f"Total Time: {tt:.3f}\nNewton: {len(iters)}\nKSP: {tk}\nAssembly: {ta:.3f}\nPC: {tp:.3f}\nSolve: {ts:.3f}")
            found = True
            break
        except Exception:
            pass
if not found:
    print("Could not parse JSON")
