"""Checks for support_dofs → simsopt DOF name generation."""

import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from coil_fem.simsopt.optimizables import CoilSupport


def test_make_names_fixed_phis():
    tree = {'phis': jnp.arange(6.0).reshape(2, 3)}
    names = CoilSupport._make_names(None, tree)
    flat, _ = ravel_pytree(tree)
    assert len(names) == len(flat)
    assert names == [
        'phis(0,0)', 'phis(0,1)', 'phis(0,2)',
        'phis(1,0)', 'phis(1,1)', 'phis(1,2)',
    ]


def test_make_names_beams_ragged():
    tree = {
        'phis_end_cc': [jnp.array([0.1, 0.2]), jnp.array([0.3])],
        'phis_start_cc': [jnp.array([0.4]), jnp.array([0.5])],
        'x_foundation': [jnp.zeros((1, 3)), jnp.zeros((2, 3))],
    }
    names = CoilSupport._make_names(None, tree)
    flat, _ = ravel_pytree(tree)
    assert len(names) == len(flat)
    assert names[0] == 'phis_end_cc(0,0)'
    assert 'phis_end_cc(1,0)' in names
    assert names.index('phis_end_cc(1,0)') == int(
        np.where(np.isclose(np.asarray(flat), 0.3))[0][0]
    )
    assert 'x_foundation(0,0,0)' in names
    assert 'x_foundation(0,0,1)' in names
    assert 'x_foundation(0,0,2)' in names
    assert 'x_foundation(1,1,2)' in names
