import json
import sys

def parse(filename):
    with open(filename) as f:
        data = f.read()

    idx = data.find('"mesh_level":')
    if idx != -1:
        idx = data.rfind('{', 0, idx)
        
        j = json.loads(data[idx:])
        s = j['steps'][0]
        
        tt = s.get('total_time_s', s.get('time', 0))
        nw = s.get('nit', 0)
        
        h = s.get('linear_timing', [])
        tk = sum(x.get('ksp_its',0) for x in h)
        
        ta = sum(x.get('assemble_total_time', 0) for x in h)
        th = sum(x.get('assemble_hvp_compute', 0) for x in h)
        ts = sum(x.get('assemble_extraction', 0) for x in h)
        tc = sum(x.get('assemble_coo_assembly', 0) for x in h)
        tp2p = sum(x.get('assemble_p2p_exchange', 0) for x in h)
        tp = sum(x.get('pc_setup_time',0) for x in h)
        tks = sum(x.get('solve_time',0) for x in h)
        
        print(f"File: {filename}")
        print(f"Total Time: {tt:.3f}")
        print(f"Newton: {nw}, KSP: {tk}")
        print(f"Linear solves: {len(h)}")
        print(f"Assembly: {ta:.3f} s (HVP: {th:.3f} s, Scatter/Ext: {ts:.3f} s, P2P: {tp2p:.3f} s, COO: {tc:.3f} s)")
        print(f"PC Setup: {tp:.3f} s")
        print(f"Solve: {tks:.3f} s")
        print("-" * 40)
    else:
        print(f'{filename} - JSON not found')

if __name__ == "__main__":
    parse(sys.argv[1])
