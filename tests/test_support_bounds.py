"""Checks for support_dofs → simsopt DOF bound generation."""

import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from coil_fem.simsopt.optimizables import CoilSupport


def test_make_bounds_fixed_phis():
    tree = {'phis': jnp.arange(6.0).reshape(2, 3)}
    lb, ub = CoilSupport._make_bounds(
        None, tree, unit_interval_keys=('phis',),
    )
    flat, _ = ravel_pytree(tree)
    assert lb.shape == ub.shape == flat.shape
    assert np.all(lb == 0.0)
    assert np.all(ub == 1.0)


def test_make_bounds_beams_keys():
    tree = {
        'phis_end_cc': [jnp.array([0.1, 0.2]), jnp.array([0.3])],
        'phis_start_cc': [jnp.array([0.4]), jnp.array([0.5])],
        'phis_start_cf': [jnp.array([0.6]), jnp.array([0.7])],
        'thetas_orientation_cc': [jnp.zeros(2), jnp.zeros(1)],
        'thetas_orientation_cf': [jnp.zeros(1), jnp.zeros(1)],
        'r_beam': [jnp.array([0.01, 0.02]), jnp.array([0.03])],
        'x_foundation': [jnp.zeros((1, 3)), jnp.zeros((1, 3))],
    }
    unit_keys = (
        'phis', 'phis_start_cc', 'phis_end_cc', 'phis_start_cf',
        'thetas_orientation_cc', 'thetas_orientation_cf',
    )
    lb, ub = CoilSupport._make_bounds(
        None, tree,
        unit_interval_keys=unit_keys,
        nonnegative_keys=('r_beam',),
    )
    names = CoilSupport._make_names(None, tree)
    flat, _ = ravel_pytree(tree)
    assert len(lb) == len(ub) == len(flat) == len(names)

    for name, lo, hi in zip(names, lb, ub):
        key = name.split('(', 1)[0]
        if key in unit_keys:
            assert lo == 0.0 and hi == 1.0, name
        elif key == 'r_beam':
            assert lo == 0.0 and np.isposinf(hi), name
        elif key == 'x_foundation':
            assert np.isneginf(lo) and np.isposinf(hi), name
        else:
            raise AssertionError(f'unexpected key {key}')
