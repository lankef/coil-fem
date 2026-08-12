"""Checks for incremental dphis Sorted support DOFs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from coil_fem.simsopt.coil_support import (
    CoilSupport,
    _SortedDphisMixin,
    _cumsum_last_vjp,
    _decode_dphis,
    _encode_dphis,
    _vjp_dphis,
)
from coil_fem.simsopt.coil_support_beams import _uniform_list


def _make_names(tree):
    return CoilSupport._make_names(None, tree)


def test_encode_decode_roundtrip_dense():
    phis = {'phis': jnp.array([[0.1, 0.3, 0.6], [0.0, 0.2, 0.9]])}
    raw = _encode_dphis(phis)
    assert 'dphis' in raw and 'phis' not in raw
    back = _decode_dphis(raw)
    np.testing.assert_allclose(back['phis'], phis['phis'], atol=1e-12)


def test_encode_decode_roundtrip_ragged():
    phis = {
        'phis_start_cc': [jnp.array([0.1, 0.4]), jnp.array([0.2])],
        'phis_end_cc': [jnp.array([0.5, 0.8]), jnp.array([0.3])],
        'phis_start_cf': [jnp.array([0.25]), jnp.array([0.1, 0.7])],
        'x_foundation': [jnp.zeros((1, 3)), jnp.zeros((2, 3))],
        'r_beam': [jnp.array([0.01, 0.02]), jnp.array([0.03])],
    }
    raw = _encode_dphis(phis)
    assert set(raw) == {
        'dphis_start_cc', 'dphis_end_cc', 'dphis_start_cf',
        'x_foundation', 'r_beam',
    }
    back = _decode_dphis(raw)
    for k in ('phis_start_cc', 'phis_end_cc', 'phis_start_cf'):
        for a, b in zip(back[k], phis[k]):
            np.testing.assert_allclose(a, b, atol=1e-12)
    for a, b in zip(back['x_foundation'], phis['x_foundation']):
        np.testing.assert_allclose(a, b)
    for a, b in zip(back['r_beam'], phis['r_beam']):
        np.testing.assert_allclose(a, b)


def test_encoded_names_use_dphis():
    phis = {'phis': jnp.zeros((2, 3)), 'phis_start_cc': [jnp.zeros(2)]}
    names = _make_names(_encode_dphis(phis))
    assert all(n.startswith('dphis') for n in names)
    assert 'dphis(0,0)' in names
    assert 'dphis_start_cc(0,0)' in names


def test_vjp_matches_reverse_cumsum():
    g_phi = {
        'phis': jnp.array([[1.0, 2.0, 3.0], [0.5, 0.0, 1.5]]),
        'x_foundation': [jnp.ones((1, 3))],
    }
    g_d = _vjp_dphis(g_phi)
    assert 'dphis' in g_d and 'phis' not in g_d
    np.testing.assert_allclose(
        g_d['dphis'], _cumsum_last_vjp(g_phi['phis']),
    )
    np.testing.assert_allclose(g_d['x_foundation'][0], g_phi['x_foundation'][0])


def test_sorted_mixin_flatten_grad_taylor():
    """flatten_grad must be the VJP of support_dofs decode (dJ contract)."""

    class _SortedStub(_SortedDphisMixin):
        def __init__(self, dphis):
            flat, unravel = ravel_pytree({'dphis': dphis})
            self._unravel = unravel
            self._local_full_x = np.asarray(flat, dtype=float)

        @property
        def local_full_x(self):
            return self._local_full_x

    dphis = jnp.array([[0.1, 0.2, 0.15], [0.05, 0.25, 0.1]])
    stub = _SortedStub(dphis)
    flat = stub.local_full_x

    def J_from_phis(phis):
        return jnp.sum(phis ** 2)

    phis = stub.support_dofs['phis']
    g_phi = {'phis': jax.grad(J_from_phis)(phis)}
    g_flat = stub.flatten_grad(g_phi)

    def J_from_d(d_flat):
        raw = stub._unravel(d_flat)
        return jnp.sum(_decode_dphis(raw)['phis'] ** 2)

    g_flat_ref = np.asarray(jax.grad(J_from_d)(jnp.asarray(flat)), dtype=float)
    np.testing.assert_allclose(g_flat, g_flat_ref, atol=1e-12)

    eps = 1e-6
    j0 = float(J_from_d(jnp.asarray(flat)))
    for i in range(len(flat)):
        e = np.zeros_like(flat)
        e[i] = eps
        j1 = float(J_from_d(jnp.asarray(flat) + e))
        assert abs((j1 - j0) / eps - g_flat[i]) < 1e-4


def test_make_bounds_dphis_unit_interval():
    tree = _encode_dphis({
        'phis': jnp.zeros((2, 2)),
        'phis_start_cc': [jnp.zeros(1)],
        'r_beam': [jnp.array([0.01])],
        'x_foundation': [jnp.zeros((1, 3))],
    })
    lb, ub = CoilSupport._make_bounds(
        None, tree,
        unit_interval_keys=tuple(k for k in tree if k.startswith('dphis')),
        nonnegative_keys=('r_beam',),
    )
    names = _make_names(tree)
    for name, lo, hi in zip(names, lb, ub):
        key = name.split('(', 1)[0]
        if key.startswith('dphis'):
            assert lo == 0.0 and hi == 1.0, name
        elif key == 'r_beam':
            assert lo == 0.0 and np.isposinf(hi), name
        elif key == 'x_foundation':
            assert np.isneginf(lo) and np.isposinf(hi), name


def test_uniform_list_wrap_ends_ascending():
    """Stellsym wrap-end defaults must be ascending for Sorted dphis >= 0."""
    # Mimic n_beam_cc after stellsym halving: (4, 4, 4, 4, 2, 2).
    counts = (4, 4, 4, 4, 2, 2)
    starts = _uniform_list(counts, cc_stellsym=True, cc_end=False)
    ends = _uniform_list(counts, cc_stellsym=True, cc_end=True)

    np.testing.assert_allclose(starts[-2], [0.125, 0.375])
    np.testing.assert_allclose(starts[-1], [0.125, 0.375])
    np.testing.assert_allclose(ends[-2], [0.625, 0.875])
    np.testing.assert_allclose(ends[-1], [0.625, 0.875])

    for arr in starts + ends:
        dphi = np.diff(np.asarray(arr), prepend=0.0)
        assert np.all(dphi >= -1e-15), (arr, dphi)


def test_sorted_stellsym_defaults_inside_box_bounds():
    """CoilSupportBeamsSorted defaults must satisfy dphis box bounds."""
    pytest.importorskip("simsopt")
    from simsopt.field import Coil, Current
    from simsopt.geo import create_equally_spaced_curves
    from coil_fem.simsopt import CoilSupportBeamsSorted

    n_base, nfp = 2, 2
    curves = create_equally_spaced_curves(
        n_base, nfp, stellsym=True, R0=1.0, R1=0.5, order=2, numquadpoints=16,
    )
    cs = CoilSupportBeamsSorted(
        base_coils=[Coil(c, Current(1e5)) for c in curves],
        nfp=nfp,
        stellsym=True,
        beam_options={
            'n_beam_cc': 4,
            'n_beam_cf': 0,
            'E': 200e9,
            'nu': 0.3,
            'cross_section_type': 'solid_circle',
            'attachment_type': 'direct',
        },
        r_beam=0.05,
    )
    x = np.asarray(cs.local_x)
    lb, ub = cs.local_bounds
    assert np.all(x >= np.asarray(lb) - 1e-14)
    assert np.all(x <= np.asarray(ub) + 1e-14)
    for pe in cs.support_dofs['phis_end_cc'][-2:]:
        np.testing.assert_allclose(np.asarray(pe), [0.625, 0.875])
