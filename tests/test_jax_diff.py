from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from src.core.serial.jax_diff import EnergyDerivator


def test_energy_derivator_ignores_extra_params():
    def energy(u, scale):
        return scale * jnp.dot(u, u)

    adjacency = sp.eye(2, format="coo")
    drv = EnergyDerivator(
        energy,
        {"scale": 2.0, "extra_metadata": np.array([1.0, 2.0])},
        adjacency,
        jnp.asarray([1.0, 2.0]),
    )

    assert set(drv.params) == {"scale"}
    f, df, _ = drv.get_derivatives()
    np.testing.assert_allclose(f(np.array([1.0, 2.0])), 10.0)
    np.testing.assert_allclose(df(np.array([1.0, 2.0])), np.array([4.0, 8.0]))
