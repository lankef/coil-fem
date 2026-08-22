"""Checks for support_dofs → simsopt DOF bound generation."""

import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from coil_fem.simsopt.coil_support import CoilSupport
from coil_fem.simsopt.sorted_dphis import (
    _apply_sorted_dphi_bounds,
    _fold_first_dphis,
    _fold_into_interval,
    _sector_width,
)


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


def test_fold_into_interval_half_turn_and_sector():
    np.testing.assert_allclose(_fold_into_interval(0.85, -0.5, 0.5), -0.15)
    np.testing.assert_allclose(_fold_into_interval(0.25, -0.5, 0.5), 0.25)
    s = 0.2
    np.testing.assert_allclose(_fold_into_interval(0.3, -0.5 * s, 0.5 * s), -0.1)
    np.testing.assert_allclose(_sector_width(5, True), 0.1)
    np.testing.assert_allclose(_sector_width(5, False), 0.2)


def test_fold_first_dphis_only_first_entry():
    nfp, stellsym = 5, True
    s = _sector_width(nfp, stellsym)
    tree = {
        'dphis_start_cc': [jnp.array([0.85, 0.1])],
        'dphis_end_cr': jnp.array([[0.3, 0.02], [0.01, 0.02]]),
        'dphis': jnp.array([0.85, 0.1]),
        'r_beam': [jnp.array([0.01])],
    }
    out = _fold_first_dphis(tree, nfp, stellsym)
    np.testing.assert_allclose(out['dphis_start_cc'][0], [-0.15, 0.1])
    # CR-end folds coil 0 (row 0), all beams; later coils unchanged.
    np.testing.assert_allclose(
        out['dphis_end_cr'][0],
        [
            _fold_into_interval(0.3, -0.5 * s, 0.5 * s),
            _fold_into_interval(0.02, -0.5 * s, 0.5 * s),
        ],
    )
    np.testing.assert_allclose(out['dphis_end_cr'][1], [0.01, 0.02])
    # Clamp dphis is not folded.
    np.testing.assert_allclose(out['dphis'], [0.85, 0.1])
    np.testing.assert_allclose(out['r_beam'][0], [0.01])


def test_apply_sorted_dphi_bounds_first_and_rest():
    nfp, stellsym = 5, True
    s = _sector_width(nfp, stellsym)
    tree = {
        'dphis_start_cc': [jnp.array([0.1, 0.2])],
        'dphis_end_cc': [jnp.array([0.1, 0.2, 0.3])],
        'dphis_start_cf': [jnp.array([0.05])],
        'dphis_start_cr': [jnp.array([0.4, 0.05])],
        'dphis_end_cr': jnp.array([[0.01, 0.02], [0.03, 0.04]]),
        'r_beam': [jnp.array([0.01])],
    }
    unit_keys = tuple(k for k in tree if k.startswith('dphis'))
    lb, ub = CoilSupport._make_bounds(
        None, tree,
        unit_interval_keys=unit_keys,
        nonnegative_keys=('r_beam',),
    )
    lb, ub = _apply_sorted_dphi_bounds(lb, ub, tree, nfp, stellsym)
    names = CoilSupport._make_names(None, tree)

    expected = {
        'dphis_start_cc(0,0)': (-0.5, 0.5),
        'dphis_start_cc(0,1)': (0.0, 1.0),
        'dphis_end_cc(0,0)': (-0.5, 0.5),
        'dphis_end_cc(0,1)': (0.0, 1.0),
        'dphis_end_cc(0,2)': (0.0, 1.0),
        'dphis_start_cf(0,0)': (-0.5, 0.5),
        'dphis_start_cr(0,0)': (-0.5, 0.5),
        'dphis_start_cr(0,1)': (0.0, 1.0),
        'dphis_end_cr(0,0)': (-0.5 * s, 0.5 * s),
        'dphis_end_cr(0,1)': (-0.5 * s, 0.5 * s),
        'dphis_end_cr(1,0)': (0.0, s),
        'dphis_end_cr(1,1)': (0.0, s),
    }
    for name, lo, hi in zip(names, lb, ub):
        key = name.split('(', 1)[0]
        if name in expected:
            exp_lo, exp_hi = expected[name]
            assert lo == exp_lo and hi == exp_hi, (name, lo, hi, exp_lo, exp_hi)
        elif key == 'r_beam':
            assert lo == 0.0 and np.isposinf(hi), name
        else:
            raise AssertionError(f'unexpected name {name}')


def test_apply_sorted_dphi_bounds_noop_on_phis():
    tree = {'phis_start_cc': [jnp.array([0.1, 0.2])]}
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    lb2, ub2 = _apply_sorted_dphi_bounds(lb, ub, tree, nfp=5, stellsym=True)
    np.testing.assert_array_equal(lb2, lb)
    np.testing.assert_array_equal(ub2, ub)
