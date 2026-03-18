#!/usr/bin/env python3
import json
import os
import sys

files = sorted([f for f in os.listdir('tmp_work') if f.startswith('gl_') and f.endswith('.json')])
for fname in files:
    path = os.path.join('tmp_work', fname)
    d = json.load(open(path))
    sys.stdout.write("=== {} (solver={}, nprocs={}) ===\n".format(fname, d['metadata']['solver'], d['metadata']['nprocs']))
    for r in d['results']:
        sys.stdout.write("  lvl={} dofs={} time={:.3f} iters={} J={:.4f}\n".format(
            r['mesh_level'], r['total_dofs'], r['time'], r['iters'], r['energy']))
    sys.stdout.flush()
