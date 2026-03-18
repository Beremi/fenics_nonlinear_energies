#!/usr/bin/env python3
from graph_coloring.mesh_loader import PROBLEMS, load_adjacency
from graph_coloring.coloring_custom import color_custom
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


for pname, pinfo in PROBLEMS.items():
    for level in pinfo["levels"][:3]:
        h5 = pinfo["path"](level)
        if not os.path.isfile(h5):
            continue
        adj = load_adjacency(h5)
        nc, colors = color_custom(adj)
        print(f"{pname} lvl{level}: n={adj.shape[0]:>8,}  colors={nc}")
