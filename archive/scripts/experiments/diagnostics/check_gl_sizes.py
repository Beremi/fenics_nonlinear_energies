#!/usr/bin/env python3
"""Check GL mesh sizes across levels to determine the grid-size mapping."""
import sys
import h5py

for lvl in range(2, 10):
    f = h5py.File('mesh_data/GinzburgLandau/GL_level{}.h5'.format(lvl), 'r')
    n_nodes = f['nodes'].shape[0]
    n_elems = f['elems'].shape[0]
    n_free = f['freedofs'].shape[0]
    import math
    n_side = int(round(math.sqrt(n_nodes)))
    sys.stdout.write("level {}: nodes={} ({}x{}), elems={}, freedofs={}\n".format(
        lvl, n_nodes, n_side, n_side, n_elems, n_free))
    f.close()
