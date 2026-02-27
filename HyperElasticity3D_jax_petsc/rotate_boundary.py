import numpy as np


def rotate_right_face_from_reference(u0_reference, nodes2coord, angle, right_nodes=None):
    """Rotate right-face boundary coordinates from the original geometry."""
    u0_ref = np.asarray(u0_reference, dtype=np.float64).reshape(-1)
    coords = np.asarray(nodes2coord, dtype=np.float64)

    if right_nodes is None:
        right_x = np.max(coords[:, 0])
        right_nodes = np.where(np.isclose(coords[:, 0], right_x))[0]
    else:
        right_nodes = np.asarray(right_nodes, dtype=np.int64)

    y0 = coords[right_nodes, 1]
    z0 = coords[right_nodes, 2]

    out = u0_ref.copy()
    out[3 * right_nodes + 0] = coords[right_nodes, 0]
    out[3 * right_nodes + 1] = np.cos(angle) * y0 + np.sin(angle) * z0
    out[3 * right_nodes + 2] = -np.sin(angle) * y0 + np.cos(angle) * z0
    return out
