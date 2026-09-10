"""Tests for stellarator-symmetric inter-coil (CS) beams."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from coil_fem.coupling import SupportBeams
from coil_fem.coupling.beam_network_csr import SupportBeamsCSR
from coil_fem.geo import CurveXYZFourierJAX
from coil_fem.simsopt.coil_support import CoilSupport
from coil_fem.simsopt.sorted_dphis import _apply_sorted_dphi_bounds

from tests.test_beam_networks import (
    _constant_section_fn,
    _make_curves,
    _make_support_dofs,
    _uniform_clamp_fn,
)


def _beam_options_cs(n_base=2, n_beam_cc=2, n_beam_cf=0, **extra):
    n_groups = n_base + 1  # stellsym
    return {
        'n_beam_cc': n_beam_cc,
        'n_beam_cf': n_beam_cf,
        'E': 200e9,
        'nu': 0.3,
        'k_attachment': 1e8,
        'i_beam_cs': [(0, 1), (1, 0)],
        's_beam_cs': [True, False],
        **extra,
    }


def test_cs_rejects_stellsym_false():
    opts = _beam_options_cs()
    with pytest.raises(
        ValueError,
        match="Stellarator-symmetric inter-coil beams are only supported",
    ):
        SupportBeams(
            nfp=2,
            stellsym=False,
            beam_options={
                **opts,
                'n_beam_cc': 2,  # n_base groups when stellsym=False
            },
            n_base=2,
            cross_section_fn=_constant_section_fn(),
            attachment_fn=_uniform_clamp_fn,
        )


def test_cs_rejects_on_csr():
    opts = {
        **_beam_options_cs(),
        'n_beam_cr': 1,
    }
    with pytest.raises(ValueError, match="not supported on SupportBeamsCSR"):
        SupportBeamsCSR(
            nfp=2,
            stellsym=True,
            beam_options=opts,
            n_base=2,
            cross_section_fn=_constant_section_fn(),
            attachment_fn=_uniform_clamp_fn,
            csr_options={
                'order': 1, 'w1': 0.05, 'w2': 0.05, 'n_phi': 8,
                'E': 200e9, 'nu': 0.3,
            },
            problem_options={},
        )


def test_cs_default_none_unchanged_total():
    sb0 = SupportBeams(
        nfp=2, stellsym=True,
        beam_options={
            'n_beam_cc': 2, 'n_beam_cf': 1, 'E': 200e9, 'nu': 0.3,
            'k_attachment': 1e8,
        },
        n_base=2,
        cross_section_fn=_constant_section_fn(),
        attachment_fn=_uniform_clamp_fn,
    )
    assert sb0.n_beam_cs == 0
    assert sb0.cs_beam_offset == sb0.n_beams_total
    # After wrap halving: n_cc = (1, 1, 1) for broadcast 2 on 3 groups.
    assert sb0.n_beams_total == sum(sb0.n_beams_per_coil) + sb0.n_beam_cc[2]


def test_cs_counts_and_offset():
    n_base = 2
    opts = _beam_options_cs(n_base=n_base, n_beam_cc=2, n_beam_cf=0)
    sb = SupportBeams(
        nfp=2, stellsym=True, beam_options=opts, n_base=n_base,
        cross_section_fn=_constant_section_fn(),
        attachment_fn=_uniform_clamp_fn,
    )
    assert sb.n_beam_cs == 2
    assert sb.i_beam_cs == ((0, 1), (1, 0))
    assert sb.s_beam_cs == (True, False)
    n_wrap = sb.n_beam_cc[n_base]
    assert sb.cs_beam_offset == sum(sb.n_beams_per_coil) + n_wrap
    assert sb.n_beams_total == sb.cs_beam_offset + 2


def test_cs_geometry_q_transform():
    n_base = 2
    opts = _beam_options_cs(n_base=n_base, n_beam_cc=0, n_beam_cf=0)
    # With n_beam_cc=0 broadcast, wrap groups also 0 after ceil(0/2).
    sb = SupportBeams(
        nfp=2, stellsym=True, beam_options=opts, n_base=n_base,
        cross_section_fn=_constant_section_fn(),
        attachment_fn=_uniform_clamp_fn,
    )
    curves = _make_curves(n_base)
    sdofs = _make_support_dofs(n_base, n_beam_cc=0, n_beam_cf=0, stellsym=True)
    phi_s = jnp.array([0.2, 0.3])
    phi_e = 1.0 - phi_s
    sdofs['phis_start_cs'] = phi_s
    sdofs['phis_end_cs'] = phi_e
    sdofs['thetas_orientation_cs'] = jnp.zeros(2)

    geom = sb.beam_geometry(curves, sdofs)
    assert geom['x_start'].shape == (2, 3)
    assert geom['x_end'].shape == (2, 3)

    Q_flip = np.asarray(sb._tfm_Q['flip'])
    Q_half = np.asarray(sb._tfm_Q['flip_half'])

    x_s0 = np.asarray(curves[0].gamma_eval(jnp.array([0.2]))[0])
    x_e0 = Q_flip @ np.asarray(curves[1].gamma_eval(jnp.array([0.8]))[0])
    np.testing.assert_allclose(geom['x_start'][0], x_s0, atol=1e-12)
    np.testing.assert_allclose(geom['x_end'][0], x_e0, atol=1e-12)

    x_s1 = np.asarray(curves[1].gamma_eval(jnp.array([0.3]))[0])
    x_e1 = Q_half @ np.asarray(curves[0].gamma_eval(jnp.array([0.7]))[0])
    np.testing.assert_allclose(geom['x_start'][1], x_s1, atol=1e-12)
    np.testing.assert_allclose(geom['x_end'][1], x_e1, atol=1e-12)


def test_cs_beam_labels():
    n_base = 2
    opts = _beam_options_cs(n_base=n_base, n_beam_cc=0, n_beam_cf=0)
    sb = SupportBeams(
        nfp=2, stellsym=True, beam_options=opts, n_base=n_base,
        cross_section_fn=_constant_section_fn(),
        attachment_fn=_uniform_clamp_fn,
    )
    coil_idx, beam_type = sb.beam_labels()
    np.testing.assert_array_equal(coil_idx, [0, 1])
    np.testing.assert_array_equal(beam_type, [0, 0])


def test_sorted_cs_bounds_seed_pm_half():
    tree = {
        'dphis_start_cc': [jnp.array([0.1])],
        'phis_start_cs': jnp.array([0.25, 0.40]),
        'phis_end_cs': jnp.array([0.75, 0.60]),
        'r_beam': [jnp.array([0.01])],
    }
    unit_keys = (
        'dphis_start_cc', 'phis_start_cs', 'phis_end_cs',
    )
    lb, ub = CoilSupport._make_bounds(
        None, tree, unit_interval_keys=unit_keys, nonnegative_keys=('r_beam',),
    )
    lb, ub = _apply_sorted_dphi_bounds(lb, ub, tree, nfp=2, stellsym=True)
    names = CoilSupport._make_names(None, tree)
    expected = {
        'dphis_start_cc(0,0)': (-0.5, 0.5),
        'phis_start_cs(0)': (0.25 - 0.5, 0.25 + 0.5),
        'phis_start_cs(1)': (0.40 - 0.5, 0.40 + 0.5),
        'phis_end_cs(0)': (0.75 - 0.5, 0.75 + 0.5),
        'phis_end_cs(1)': (0.60 - 0.5, 0.60 + 0.5),
    }
    for name, lo, hi in zip(names, lb, ub):
        key = name.split('(', 1)[0]
        if name in expected:
            exp_lo, exp_hi = expected[name]
            assert lo == exp_lo and hi == exp_hi, (name, lo, hi)
        elif key == 'r_beam':
            assert lo == 0.0 and np.isposinf(hi)
        else:
            raise AssertionError(name)


def test_unsorted_cs_bounds_unit_interval():
    """Non-Sorted phis_*_cs stay in [0, 1] (no seed±0.5 overlay)."""
    tree = {
        'phis_start_cs': jnp.array([0.25]),
        'phis_end_cs': jnp.array([0.75]),
    }
    lb, ub = CoilSupport._make_bounds(
        None, tree,
        unit_interval_keys=('phis_start_cs', 'phis_end_cs'),
    )
    lb2, ub2 = _apply_sorted_dphi_bounds(lb, ub, tree, nfp=2, stellsym=True)
    np.testing.assert_array_equal(lb2, [0.0, 0.0])
    np.testing.assert_array_equal(ub2, [1.0, 1.0])


def test_coil_support_beams_cs_defaults():
    pytest.importorskip("simsopt")
    from simsopt.field import Coil, Current
    from simsopt.geo import create_equally_spaced_curves
    from coil_fem.simsopt import CoilSupportBeams, CoilSupportBeamsSorted

    n_base, nfp = 2, 2
    curves = create_equally_spaced_curves(
        n_base, nfp, stellsym=True, R0=1.0, R1=0.5, order=2, numquadpoints=32,
    )
    base_coils = [Coil(c, Current(1e5)) for c in curves]
    beam_options = {
        'n_beam_cc': 0,
        'n_beam_cf': 0,
        'E': 200e9,
        'nu': 0.3,
        'cross_section_type': 'solid_circle',
        'attachment_type': 'direct',
        'i_beam_cs': [(0, 0), (1, 1)],
        's_beam_cs': [True, False],
    }

    cs = CoilSupportBeams(
        base_coils=base_coils, nfp=nfp, stellsym=True,
        beam_options=beam_options, r_beam=0.05,
    )
    sd = cs.support_dofs
    assert 'phis_start_cs' in sd and 'phis_end_cs' in sd
    ps = np.asarray(sd['phis_start_cs'])
    pe = np.asarray(sd['phis_end_cs'])
    assert ps.shape == (2,)
    np.testing.assert_allclose(pe, 1.0 - ps, atol=1e-12)

    # Inboard point: argmin R on each start coil.
    for j, (i0, _) in enumerate(cs.support.i_beam_cs):
        curve = CurveXYZFourierJAX.from_simsopt(base_coils[i0].curve)
        gamma = curve.gamma()
        R = np.sqrt(gamma[:, 0] ** 2 + gamma[:, 1] ** 2)
        phi0 = float(curve.quadpoints[np.argmin(R)] % 1.0)
        np.testing.assert_allclose(ps[j], phi0, atol=1e-10)

    # Sorted: same absolute keys, bounds seed ± 0.5.
    cs_s = CoilSupportBeamsSorted(
        base_coils=base_coils, nfp=nfp, stellsym=True,
        beam_options=beam_options, r_beam=0.05,
    )
    sd_s = cs_s.support_dofs
    assert 'phis_start_cs' in sd_s
    raw = cs_s._unravel(jnp.asarray(cs_s.local_full_x))
    assert 'dphis_start_cs' not in raw
    assert 'phis_start_cs' in raw
    x_free = np.asarray(cs_s.local_x)
    lb_free, ub_free = cs_s.local_bounds
    assert np.all(x_free >= np.asarray(lb_free) - 1e-14)
    assert np.all(x_free <= np.asarray(ub_free) + 1e-14)
    names = list(cs_s.local_full_dof_names)
    full_lo = np.asarray(cs_s.local_full_lower_bounds)
    full_hi = np.asarray(cs_s.local_full_upper_bounds)
    for key, seed_arr in (
        ('phis_start_cs', sd_s['phis_start_cs']),
        ('phis_end_cs', sd_s['phis_end_cs']),
    ):
        seed = np.asarray(seed_arr)
        for j in range(seed.shape[0]):
            target = f'{key}({j})'
            matches = [i for i, n in enumerate(names) if target in n]
            assert matches, (target, names)
            idx = matches[0]
            assert abs(full_lo[idx] - (seed[j] - 0.5)) < 1e-12, (
                target, full_lo[idx], seed[j],
            )
            assert abs(full_hi[idx] - (seed[j] + 0.5)) < 1e-12, (
                target, full_hi[idx], seed[j],
            )
