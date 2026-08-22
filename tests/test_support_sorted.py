"""Checks for incremental dphis Sorted support DOFs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from coil_fem.simsopt.coil_support import CoilSupport
from coil_fem.simsopt.sorted_dphis import (
    _SortedDphisMixin,
    _cumsum_last_vjp,
    _decode_dphis,
    _decode_end_cc,
    _decode_end_cr,
    _encode_dphis,
    _encode_end_cc,
    _encode_end_cr,
    _fold_into_interval,
    _sector_width,
    _vjp_dphis,
    _vjp_end_cc,
    _vjp_end_cr,
)
from coil_fem.simsopt.coil_support_beams import _uniform_list


def _make_names(tree):
    return CoilSupport._make_names(None, tree)


def _beam_options(n_beam_cc=4, n_beam_cf=0):
    return {
        'n_beam_cc': n_beam_cc,
        'n_beam_cf': n_beam_cf,
        'E': 200e9,
        'nu': 0.3,
        'cross_section_type': 'solid_circle',
        'attachment_type': 'direct',
    }


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


def test_uniform_list_wrap_ends_pair_complement():
    """Stellsym wrap ends are 1 - starts (descending); encoded dphis >= 0."""
    # Mimic n_beam_cc after stellsym halving: (4, 4, 4, 4, 2, 2).
    counts = (4, 4, 4, 4, 2, 2)
    starts = _uniform_list(counts, cc_stellsym=True, cc_end=False)
    ends = _uniform_list(counts, cc_stellsym=True, cc_end=True)

    np.testing.assert_allclose(starts[-2], [0.125, 0.375])
    np.testing.assert_allclose(starts[-1], [0.125, 0.375])
    np.testing.assert_allclose(ends[-2], [0.875, 0.625])
    np.testing.assert_allclose(ends[-1], [0.875, 0.625])

    for g in (-2, -1):
        np.testing.assert_allclose(
            np.asarray(ends[g]), 1.0 - np.asarray(starts[g]), atol=1e-12,
        )

    # Interior groups stay ascending; wrap starts ascending; wrap ends
    # descending in absolute phi — but Sorted encode must stay >= 0.
    for arr in starts[:-2] + ends[:-2] + starts[-2:]:
        dphi = np.diff(np.asarray(arr), prepend=0.0)
        assert np.all(dphi >= -1e-15), (arr, dphi)

    encoded_ends = _encode_end_cc(ends, stellsym=True)
    for d in encoded_ends:
        assert np.all(np.asarray(d) >= -1e-15), d


def test_encode_decode_end_cc_roundtrip():
    """Wrap-aware end codec roundtrips for stellsym True/False."""
    # n_groups=3, stellsym: wrap = {1, 2}; n_base=1 stellsym: wrap = {0, 1}.
    cases = [
        (
            True,
            [
                jnp.array([0.1, 0.15, 0.2]),
                jnp.array([0.05, 0.1]),
                jnp.zeros(0),
            ],
        ),
        (
            False,
            [
                jnp.array([0.1, 0.2]),
                jnp.array([0.3]),
            ],
        ),
        (
            True,
            [
                jnp.array([0.08, 0.12]),
                jnp.array([0.2]),
            ],
        ),
    ]
    for stellsym, d_list in cases:
        phi = _decode_end_cc(d_list, stellsym)
        back = _encode_end_cc(phi, stellsym)
        for a, b in zip(back, d_list):
            np.testing.assert_allclose(a, b, atol=1e-12)


def test_vjp_end_cc_matches_decode():
    """Analytic VJP of wrap-end decode matches reverse-mode AD."""
    d_list = [
        jnp.array([0.1, 0.2]),
        jnp.array([0.05, 0.1, 0.15]),
        jnp.array([0.2, 0.1]),
    ]
    stellsym = True

    def J(d_flat):
        # Reconstruct ragged list from flat concat of known lengths.
        sizes = [2, 3, 2]
        parts, i = [], 0
        for n in sizes:
            parts.append(d_flat[i:i + n])
            i += n
        phi = _decode_end_cc(parts, stellsym)
        return sum(jnp.sum(p ** 2) for p in phi)

    d_flat = jnp.concatenate(d_list)
    g_ad = jax.grad(J)(d_flat)

    phi = _decode_end_cc(d_list, stellsym)
    g_phi = [2.0 * p for p in phi]
    g_d = _vjp_end_cc(g_phi, stellsym)
    g_manual = jnp.concatenate(g_d)
    np.testing.assert_allclose(g_manual, g_ad, atol=1e-12)


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
        beam_options=_beam_options(n_beam_cc=4),
        r_beam=0.05,
    )
    x = np.asarray(cs.local_x)
    lb, ub = cs.local_bounds
    assert np.all(x >= np.asarray(lb) - 1e-14)
    assert np.all(x <= np.asarray(ub) + 1e-14)
    # After halving, wrap groups have c=2 → ends = 1 - [0.125, 0.375].
    for pe in cs.support_dofs['phis_end_cc'][-2:]:
        np.testing.assert_allclose(np.asarray(pe), [0.875, 0.625])


def test_sorted_stellsym_wrap_pairing():
    """Default Sorted wrap groups pair phi_end[j] = 1 - phi_start[j]."""
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
        beam_options=_beam_options(n_beam_cc=6),
        r_beam=0.05,
    )
    sd = cs.support_dofs
    n_groups = len(sd['phis_start_cc'])
    assert n_groups == n_base + 1

    for g in range(n_groups - 2):
        ps = np.asarray(sd['phis_start_cc'][g])
        pe = np.asarray(sd['phis_end_cc'][g])
        assert np.all(np.diff(ps, prepend=0.0) >= -1e-15)
        assert np.all(np.diff(pe, prepend=0.0) >= -1e-15)

    for g in (n_groups - 2, n_groups - 1):
        ps = np.asarray(sd['phis_start_cc'][g])
        pe = np.asarray(sd['phis_end_cc'][g])
        np.testing.assert_allclose(pe, 1.0 - ps, atol=1e-12)
        assert pe[0] > pe[-1]  # descending absolute ends


def test_sorted_wrap_end_flatten_grad_fd():
    """flatten_grad VJP matches FD on a wrap dphis_end_cc DOF."""
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
        beam_options=_beam_options(n_beam_cc=4),
        r_beam=0.05,
    )

    # Full DOF vector includes fixed entries; use local_full_x.
    flat0 = np.asarray(cs.local_full_x, dtype=float).copy()
    names = list(cs.local_full_dof_names)
    # Pick first wrap-group end increment: group n_base-1 = 1, beam 0.
    target = 'dphis_end_cc(1,0)'
    matches = [i for i, n in enumerate(names) if n.endswith(':' + target) or n == target]
    # simsopt may prefix with Optimizable name; match by suffix.
    if not matches:
        matches = [i for i, n in enumerate(names) if target in n]
    assert matches, names
    idx = matches[0]

    def J_from_full(x_full):
        cs.local_full_x = np.asarray(x_full, dtype=float)
        pe = cs.support_dofs['phis_end_cc']
        return float(sum(jnp.sum(p ** 2) for p in pe))

    j0 = J_from_full(flat0)
    eps = 1e-6
    e = np.zeros_like(flat0)
    e[idx] = eps
    j1 = J_from_full(flat0 + e)
    fd = (j1 - j0) / eps

    # Restore and evaluate analytic flatten_grad.
    cs.local_full_x = flat0
    pe = cs.support_dofs['phis_end_cc']
    g_phi = {
        'phis_end_cc': [2.0 * p for p in pe],
        # Other keys present in support_dofs with zero grad so ravel aligns
        # only if flatten_grad expects the full phi tree — it takes whatever
        # keys _vjp_dphis / ravel see.  Pass the full support_dofs structure
        # with zeros elsewhere.
    }
    sd = cs.support_dofs
    g_full = {k: jax.tree_util.tree_map(jnp.zeros_like, v) for k, v in sd.items()}
    g_full['phis_end_cc'] = [2.0 * p for p in pe]
    g_flat = cs.flatten_grad(g_full)
    assert abs(fd - g_flat[idx]) < 1e-4, (fd, g_flat[idx], names[idx])


def _csr_beam_options(n_beam_cc=2, n_beam_cf=0, n_beam_cr=2):
    return {
        'n_beam_cc': n_beam_cc,
        'n_beam_cf': n_beam_cf,
        'n_beam_cr': n_beam_cr,
        'E': 200e9,
        'nu': 0.3,
        'cross_section_type': 'solid_circle',
        'attachment_type': 'direct',
    }


def _csr_options(nfp=2, order=1, n_phi=4):
    return {
        'order': order,
        'w1': 0.08,
        'w2': 0.08,
        'n_phi': n_phi,
        'n_grid_1': 1,
        'n_grid_2': 1,
        'E': 200e9,
        'nu': 0.3,
    }


def test_csr_sorted_defaults_inside_box_bounds():
    """CoilSupportBeamsCSRSorted defaults must satisfy dphis box bounds."""
    pytest.importorskip("simsopt")
    from simsopt.field import Coil, Current
    from simsopt.geo import create_equally_spaced_curves
    from coil_fem.simsopt import CoilSupportBeamsCSRSorted

    n_base, nfp = 2, 2
    curves = create_equally_spaced_curves(
        n_base, nfp, stellsym=True, R0=1.0, R1=0.5, order=2, numquadpoints=16,
    )
    cs = CoilSupportBeamsCSRSorted(
        base_coils=[Coil(c, Current(1e5)) for c in curves],
        nfp=nfp,
        stellsym=True,
        beam_options=_csr_beam_options(n_beam_cc=4, n_beam_cr=2),
        csr_options=_csr_options(nfp=nfp),
        problem_options={'solver': 'umfpack'},
        r_beam=0.05,
    )
    x = np.asarray(cs.local_x)
    lb, ub = cs.local_bounds
    assert np.all(x >= np.asarray(lb) - 1e-14)
    assert np.all(x <= np.asarray(ub) + 1e-14)

    sd = cs.support_dofs
    assert 'phis_start_cr' in sd and 'phis_end_cr' in sd
    ps = np.asarray(sd['phis_start_cr'])
    pe = np.asarray(sd['phis_end_cr'])
    # Start: first beam of each coil may be negative after fold; later beams >= 0.
    if ps.size:
        assert np.all((-0.5 - 1e-14 <= ps[:, 0]) & (ps[:, 0] <= 0.5 + 1e-14))
    if ps.shape[1] > 1:
        assert np.all(np.diff(ps, axis=1) >= -1e-15)
    # End: coil 0 (all beams) in the first-increment box; later coils ascend.
    s = 1.0 / nfp / 2.0
    if pe.size:
        assert np.all((-0.5 * s - 1e-14 <= pe[0]) & (pe[0] <= 0.5 * s + 1e-14))
    if pe.shape[0] > 1:
        assert np.all(np.diff(pe, axis=0) >= -1e-15)


def test_csr_default_phis_end_cr_at_coil_center():
    """Default phis_end_cr seeds at each coil's cylindrical angle."""
    pytest.importorskip("simsopt")
    from simsopt.field import Coil, Current
    from simsopt.geo import create_equally_spaced_curves
    from coil_fem.geo import CurveXYZFourierJAX
    from coil_fem.simsopt import CoilSupportBeamsCSRSorted
    from coil_fem.simsopt.sorted_dphis import _encode_dphis

    n_base, nfp, n_cr = 2, 2, 2
    curves = create_equally_spaced_curves(
        n_base, nfp, stellsym=False, R0=1.0, R1=0.5, order=2, numquadpoints=16,
    )
    base_coils = [Coil(c, Current(1e5)) for c in curves]
    expected = []
    for coil in base_coils:
        c = CurveXYZFourierJAX.from_simsopt(coil.curve).curve_center()
        phi_i = float((np.arctan2(c[1], c[0]) / (2.0 * np.pi)) % 1.0)
        expected.append(np.full((n_cr,), phi_i))
    expected = np.stack(expected, axis=0)

    # Sorted constructs via base CoilSupportBeamsCSR defaults, then encodes.
    cs = CoilSupportBeamsCSRSorted(
        base_coils=base_coils,
        nfp=nfp,
        stellsym=False,
        beam_options=_csr_beam_options(n_beam_cc=0, n_beam_cr=n_cr),
        csr_options=_csr_options(nfp=nfp),
        problem_options={'solver': 'umfpack'},
        r_beam=0.05,
    )
    s = _sector_width(nfp, False)
    sd = cs.support_dofs
    pe = np.asarray(sd['phis_end_cr'])
    folded0 = float(_fold_into_interval(expected[0, 0], -0.5 * s, 0.5 * s))
    expected_phi = folded0 + (expected - expected[0])
    np.testing.assert_allclose(pe, expected_phi, atol=1e-12)

    encoded = _encode_dphis({'phis_end_cr': sd['phis_end_cr']})
    d = np.asarray(encoded['dphis_end_cr'])
    expected_d = np.diff(
        expected_phi, axis=0, prepend=np.zeros_like(expected_phi[:1]),
    )
    np.testing.assert_allclose(d, expected_d, atol=1e-12)


def test_csr_default_phis_start_cr_min_R_window_and_v_end():
    """phis_start_cr in width-1/4 min-R window; v_end_cr = linspace(-1, 1)."""
    pytest.importorskip("simsopt")
    from simsopt.field import Coil, Current
    from simsopt.geo import create_equally_spaced_curves
    from coil_fem.geo import CurveXYZFourierJAX
    from coil_fem.simsopt import CoilSupportBeamsCSRSorted
    from coil_fem.simsopt.sorted_dphis import _encode_dphis

    n_base, nfp, n_cr = 1, 2, 3
    curves = create_equally_spaced_curves(
        n_base, nfp, stellsym=False, R0=1.0, R1=0.5, order=2, numquadpoints=32,
    )
    base_coils = [Coil(c, Current(1e5)) for c in curves]
    curve = CurveXYZFourierJAX.from_simsopt(base_coils[0].curve)
    gamma = np.asarray(curve.gamma())
    R = np.sqrt(gamma[:, 0] ** 2 + gamma[:, 1] ** 2)
    phi0 = float(np.asarray(curve.quadpoints)[np.argmin(R)])

    cs = CoilSupportBeamsCSRSorted(
        base_coils=base_coils,
        nfp=nfp,
        stellsym=False,
        beam_options=_csr_beam_options(n_beam_cc=0, n_beam_cr=n_cr),
        csr_options=_csr_options(nfp=nfp),
        problem_options={'solver': 'umfpack'},
        r_beam=0.05,
    )
    sd = cs.support_dofs
    np.testing.assert_allclose(
        np.asarray(sd['v_end_cr'][0]), [-1.0, 0.0, 1.0], atol=1e-12,
    )

    ps = np.asarray(sd['phis_start_cr'][0])
    assert ps.shape == (n_cr,)
    # Circular distance to phi0 ≤ half-width (+ tiny tol).
    half = 0.125
    dcirc = np.minimum(np.abs(ps - phi0) % 1.0, 1.0 - (np.abs(ps - phi0) % 1.0))
    assert np.all(dcirc <= half + 1e-9), (ps, phi0, dcirc)

    # Remaining increments stay non-negative; first may be folded into [-0.5, 0.5].
    assert np.all(np.diff(ps) >= -1e-15)
    encoded = _encode_dphis({'phis_start_cr': sd['phis_start_cr']})
    dphis = np.asarray(encoded['dphis_start_cr'][0])
    assert -0.5 - 1e-14 <= dphis[0] <= 0.5 + 1e-14
    assert np.all(dphis[1:] >= -1e-15)
    np.testing.assert_allclose(np.cumsum(dphis), ps, atol=1e-12)


def test_csr_sorted_dphis_cr_flatten_grad_fd():
    """flatten_grad VJP matches FD on dphis_start_cr / dphis_end_cr DOFs."""
    pytest.importorskip("simsopt")
    from simsopt.field import Coil, Current
    from simsopt.geo import create_equally_spaced_curves
    from coil_fem.simsopt import CoilSupportBeamsCSRSorted

    n_base, nfp = 1, 2
    curves = create_equally_spaced_curves(
        n_base, nfp, stellsym=False, R0=1.0, R1=0.5, order=2, numquadpoints=16,
    )
    cs = CoilSupportBeamsCSRSorted(
        base_coils=[Coil(c, Current(1e5)) for c in curves],
        nfp=nfp,
        stellsym=False,
        beam_options=_csr_beam_options(n_beam_cc=0, n_beam_cr=3),
        csr_options=_csr_options(nfp=nfp),
        problem_options={'solver': 'umfpack'},
        r_beam=0.05,
    )

    flat0 = np.asarray(cs.local_full_x, dtype=float).copy()
    names = list(cs.local_full_dof_names)

    def _find(target):
        matches = [
            i for i, n in enumerate(names)
            if n.endswith(':' + target) or n == target or target in n
        ]
        assert matches, (target, names)
        return matches[0]

    idx_start = _find('dphis_start_cr(0,0)')
    idx_end = _find('dphis_end_cr(0,0)')

    def J_from_full(x_full):
        cs.local_full_x = np.asarray(x_full, dtype=float)
        sd = cs.support_dofs
        return float(
            jnp.sum(sd['phis_start_cr'] ** 2)
            + jnp.sum(sd['phis_end_cr'] ** 2)
        )

    j0 = J_from_full(flat0)
    eps = 1e-6

    sd = cs.support_dofs
    g_full = {k: jax.tree_util.tree_map(jnp.zeros_like, v) for k, v in sd.items()}
    g_full['phis_start_cr'] = 2.0 * sd['phis_start_cr']
    g_full['phis_end_cr'] = 2.0 * sd['phis_end_cr']
    g_flat = cs.flatten_grad(g_full)

    for idx in (idx_start, idx_end):
        e = np.zeros_like(flat0)
        e[idx] = eps
        j1 = J_from_full(flat0 + e)
        fd = (j1 - j0) / eps
        cs.local_full_x = flat0
        assert abs(fd - g_flat[idx]) < 1e-4, (fd, g_flat[idx], names[idx])


def test_encode_decode_end_cr_roundtrip():
    """Cross-coil CR-end codec inverts on a (n_coil, n_beam) array."""
    phi = jnp.array([[0.01, 0.02], [0.04, 0.06], [0.09, 0.11]])
    d = _encode_end_cr(phi)
    np.testing.assert_allclose(d[0], phi[0])
    np.testing.assert_allclose(d[1], phi[1] - phi[0])
    np.testing.assert_allclose(d[2], phi[2] - phi[1])
    np.testing.assert_allclose(_decode_end_cr(d), phi, atol=1e-12)


def test_vjp_end_cr_matches_decode():
    """Analytic VJP of coil-axis decode matches reverse-mode AD."""
    d = jnp.array([[0.1, 0.2], [0.05, 0.1], [0.02, 0.03]])

    def J(d_arr):
        return jnp.sum(_decode_end_cr(d_arr) ** 2)

    g_ad = jax.grad(J)(d)
    phi = _decode_end_cr(d)
    g_manual = _vjp_end_cr(2.0 * phi)
    np.testing.assert_allclose(g_manual, g_ad, atol=1e-12)
