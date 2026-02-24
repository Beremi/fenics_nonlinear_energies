#!/usr/bin/env python3
"""Quick validation: verify custom coloring produces valid distance-2 colorings."""
from graph_coloring.mesh_loader import PROBLEMS, load_adjacency
from graph_coloring.coloring_custom import color_custom
import sys
import os
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


for pname, pinfo in PROBLEMS.items():
    for level in pinfo["levels"][:4]:
        h5 = pinfo["path"](level)
        if not os.path.isfile(h5):
            continue
        adj = load_adjacency(h5)
        nc, colors = color_custom(adj)
        # Validate: no two A²-neighbours share a colour
        A2 = sp.csr_matrix(adj @ adj)
        n = A2.shape[0]
        conflicts = 0
        for i in range(n):
            nbrs = A2.indices[A2.indptr[i]:A2.indptr[i + 1]]
            for j in nbrs:
                if j != i and colors[i] == colors[j]:
                    conflicts += 1
        status = "OK" if conflicts == 0 else f"FAIL ({conflicts} conflicts)"
        print(f"{pname} lvl{level}: n={n:>8,}  colors={nc:>3}  {status}")
