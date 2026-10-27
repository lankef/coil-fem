"""Checks for the BeamCurveDistance support-beam clearance penalty."""

import jax.numpy as jnp
import numpy as np
import pytest

from coil_fem.simsopt.objectives import _curve_segment_hinge, _segment_point_dists

simsopt = pytest.importorskip("simsopt")

from simsopt.field import Coil, Current                       # noqa: E402
from simsopt.geo import create_equally_spaced_curves          # noqa: E402

from coil_fem.simsopt import (                                # noqa: E402
    BeamCurveDistance,
    CoilSupportBeams,
    CoilSupportBeamsSorted,
)

NFP = 2
N_BASE = 2

BEAM_OPTIONS = {
    'n_beam_cc': 1,
    'n_beam_cf': 1,
    'E': 200e9,
    'nu': 0.3,
    'cross_section_type': 'solid_circle',
    'attachment_type': 'direct',
}

# Trim attachment neighbourhood so free-span endpoints are not on the coils
# (dead_length=0 puts chord ends on γ(φ), making shortest distance 0 and dJ NaN).
DEAD_LENGTH = 0.05


def _make_coil_support(cls=CoilSupportBeams, **beam_overrides):
    """A 2-coil, nfp=2, stellsym beam support with one CC and one CF beam each."""
    curves = create_equally_spaced_curves(
        N_BASE, NFP, stellsym=True, R0=1.0, R1=0.5, order=2, numquadpoints=32,
    )
    base_coils = [Coil(c, Current(1e5)) for c in curves]
    opts = {**BEAM_OPTIONS, **beam_overrides}
    return cls(
        base_coils=base_coils,
        nfp=NFP,
        stellsym=True,
        beam_options=opts,
        r_beam=0.05,
    )


def _dof_index(opt, pattern, prefix=False):
    """Index into ``opt.x`` of the first DOF whose name matches ``pattern``."""
    for i, name in enumerate(opt.dof_names):
        if name.startswith(pattern) if prefix else name.endswith(pattern):
            return i
    raise AssertionError(f"no DOF matching {pattern!r} in {opt.dof_names}")


# ============================================================================
# Pure hinge helper
# ============================================================================

def test_curve_segment_hinge_zero_when_clear():
    """Points far from a short segment give a zero hinge."""
    x_a = jnp.array([[0.0, 0.0, 0.0]])
    x_b = jnp.array([[1.0, 0.0, 0.0]])
    gamma = jnp.array([[0.5, 10.0, 0.0], [0.0, 10.0, 0.0]])
    gammadash = jnp.ones_like(gamma)
    h = _curve_segment_hinge(x_a, x_b, gamma, gammadash, minimum_distance=1.0)
    assert h.shape == (1,)
    assert float(h[0]) == 0.0


def test_curve_segment_hinge_matches_mean_formula():
    x_a = jnp.array([[0.0, 0.0, 0.0]])
    x_b = jnp.array([[2.0, 0.0, 0.0]])
    gamma = jnp.array([[1.0, 0.5, 0.0], [1.0, 1.0, 0.0]])
    gammadash = jnp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    dmin = 2.0
    h = float(_curve_segment_hinge(x_a, x_b, gamma, gammadash, dmin)[0])
    dists = np.asarray(_segment_point_dists(x_a, x_b, gamma)[0])
    alen = np.linalg.norm(np.asarray(gammadash), axis=1)
    expected = np.mean(alen * np.maximum(dmin - dists, 0.0) ** 2)
    np.testing.assert_allclose(h, expected, atol=1e-12)


# ============================================================================
# J
# ============================================================================

def test_J_zero_when_clear_and_positive_when_violated():
    cs = _make_coil_support()
    d_clear = 0.5 * BeamCurveDistance(cs, DEAD_LENGTH, 0.0).shortest_distance()
    assert d_clear > 0.0
    assert np.isfinite(d_clear)

    assert BeamCurveDistance(cs, DEAD_LENGTH, d_clear).J() == 0.0
    assert BeamCurveDistance(cs, DEAD_LENGTH, 4.0 * d_clear).J() > 0.0


def test_oversized_dead_length_is_capped():
    """dead_length > L/2 is clipped so the free span stays non-empty and dJ finite."""
    cs = _make_coil_support()
    Jb = BeamCurveDistance(cs, dead_length=1e3, minimum_distance=10.0)
    assert np.isfinite(Jb.J())
    assert np.all(np.isfinite(np.asarray(Jb.dJ())))

    d = BeamCurveDistance(cs, dead_length=1e3, minimum_distance=0.0).shortest_distance()
    assert np.isfinite(d)
    assert d > 0.0

    cdofs, sdofs = Jb._read_dofs()
    curves_jax = Jb._curves_jax(cdofs)
    geom = cs.support.beam_geometry(curves_jax, sdofs)
    x_a, x_b, active = Jb._effective_segments(geom)
    span = np.linalg.norm(np.asarray(x_b) - np.asarray(x_a), axis=1)
    assert np.all(np.asarray(active))
    assert np.all(span > 0.0)


def test_cf_omits_end_curve_integral():
    """CF-only supports only hinge against the start coil (no second curve)."""
    cs = _make_coil_support(n_beam_cc=0, n_beam_cf=1)
    dmin = 1.0
    Jb = BeamCurveDistance(cs, dead_length=DEAD_LENGTH, minimum_distance=dmin)
    cdofs, sdofs = Jb._read_dofs()
    curves_jax = Jb._curves_jax(cdofs)
    geom = cs.support.beam_geometry(curves_jax, sdofs)
    x_a, x_b, active = Jb._effective_segments(geom)

    # Host reconstruction: only start-curve hinges for each CF beam.
    expected = 0.0
    b = 0
    for i in range(cs.support.n_base):
        n_cf = cs.support.n_beam_cf[i]
        if n_cf == 0:
            continue
        sl = slice(b, b + n_cf)
        hs = np.asarray(_curve_segment_hinge(
            x_a[sl], x_b[sl],
            curves_jax[i].gamma(), curves_jax[i].gammadash(),
            dmin,
        ))
        expected += float(np.sum(np.where(np.asarray(active[sl]), hs, 0.0)))
        b += n_cf

    np.testing.assert_allclose(Jb.J(), expected, rtol=1e-10)


def test_shortest_distance_matches_brute_force():
    cs = _make_coil_support()
    Jb = BeamCurveDistance(cs, dead_length=DEAD_LENGTH, minimum_distance=0.1)
    cdofs, sdofs = Jb._read_dofs()
    curves_jax = Jb._curves_jax(cdofs)
    geom = cs.support.beam_geometry(curves_jax, sdofs)
    x_a, x_b, active = Jb._effective_segments(geom)
    x_a = np.asarray(x_a)
    x_b = np.asarray(x_b)
    active = np.asarray(active)

    support = cs.support
    best = np.inf
    b = 0
    n_base = support.n_base

    def sweep(gamma, sl):
        nonlocal best
        pts = np.asarray(gamma)
        for k in range(sl.start, sl.stop):
            if not active[k]:
                continue
            a, c = x_a[k], x_b[k]
            ab = c - a
            t = np.clip((pts - a) @ ab / (ab @ ab + 1e-300), 0.0, 1.0)
            best = min(best, np.min(np.linalg.norm(a + t[:, None] * ab - pts, axis=1)))

    for i in range(n_base):
        n_g = support.n_beam_cc[i]
        if n_g > 0:
            start_idx, end_idx, end_tfm = support.cc_groups[i]
            sl = slice(b, b + n_g)
            sweep(curves_jax[start_idx].gamma(), sl)
            gamma_e = support._apply_end_transform(
                curves_jax[end_idx].gamma(), end_tfm,
            )
            sweep(gamma_e, sl)
            b += n_g
        n_cf = support.n_beam_cf[i]
        if n_cf > 0:
            sl = slice(b, b + n_cf)
            sweep(curves_jax[i].gamma(), sl)
            b += n_cf
    if support.stellsym:
        n_wrap = support.n_beam_cc[n_base]
        if n_wrap > 0:
            start_idx, end_idx, end_tfm = support.cc_groups[n_base]
            sl = slice(b, b + n_wrap)
            sweep(curves_jax[start_idx].gamma(), sl)
            gamma_e = support._apply_end_transform(
                curves_jax[end_idx].gamma(), end_tfm,
            )
            sweep(gamma_e, sl)

    np.testing.assert_allclose(Jb.shortest_distance(), best, rtol=1e-10)


def test_J_matches_hinge_formula_on_host():
    cs = _make_coil_support()
    d_min = 3.0 * BeamCurveDistance(cs, DEAD_LENGTH, 0.0).shortest_distance()
    Jb = BeamCurveDistance(cs, dead_length=DEAD_LENGTH, minimum_distance=d_min)

    cdofs, sdofs = Jb._read_dofs()
    curves_jax = Jb._curves_jax(cdofs)
    geom = cs.support.beam_geometry(curves_jax, sdofs)
    x_a, x_b, active = Jb._effective_segments(geom)

    support = cs.support
    expected = 0.0
    b = 0
    n_base = support.n_base

    def add_cc(g, b0):
        nonlocal expected
        n_g = support.n_beam_cc[g]
        if n_g == 0:
            return b0
        start_idx, end_idx, end_tfm = support.cc_groups[g]
        sl = slice(b0, b0 + n_g)
        gamma_e = support._apply_end_transform(
            curves_jax[end_idx].gamma(), end_tfm,
        )
        hs = np.asarray(_curve_segment_hinge(
            x_a[sl], x_b[sl],
            curves_jax[start_idx].gamma(), curves_jax[start_idx].gammadash(),
            d_min,
        ))
        he = np.asarray(_curve_segment_hinge(
            x_a[sl], x_b[sl],
            gamma_e, curves_jax[end_idx].gammadash(),
            d_min,
        ))
        mask = np.asarray(active[sl])
        expected += float(np.sum(np.where(mask, hs + he, 0.0)))
        return b0 + n_g

    for i in range(n_base):
        b = add_cc(i, b)
        n_cf = support.n_beam_cf[i]
        if n_cf > 0:
            sl = slice(b, b + n_cf)
            hs = np.asarray(_curve_segment_hinge(
                x_a[sl], x_b[sl],
                curves_jax[i].gamma(), curves_jax[i].gammadash(),
                d_min,
            ))
            expected += float(np.sum(np.where(np.asarray(active[sl]), hs, 0.0)))
            b += n_cf
    if support.stellsym:
        b = add_cc(n_base, b)

    np.testing.assert_allclose(Jb.J(), expected, rtol=1e-10)


def test_J_cache_invalidated_on_dof_change():
    cs = _make_coil_support()
    Jb = BeamCurveDistance(cs, dead_length=DEAD_LENGTH, minimum_distance=1.0)
    J0 = Jb.J()

    i = _dof_index(Jb, ':x_foundation(0,0,0)')
    x = np.array(Jb.x)
    x[i] += 0.05
    Jb.x = x
    assert Jb.J() != J0


def test_J_independent_of_currents():
    cs = _make_coil_support()
    Jb = BeamCurveDistance(cs, dead_length=DEAD_LENGTH, minimum_distance=1.0)
    J0 = Jb.J()

    i = _dof_index(Jb, 'Current', prefix=True)
    x = np.array(Jb.x)
    x[i] *= 2.0
    Jb.x = x
    assert Jb.J() == J0
    assert np.asarray(Jb.dJ())[i] == 0.0


def test_rejects_nonpositive_dead_length_and_negative_dmin():
    cs = _make_coil_support()
    with pytest.raises(ValueError, match="dead_length"):
        BeamCurveDistance(cs, dead_length=-0.1, minimum_distance=0.0)
    with pytest.raises(ValueError, match="dead_length"):
        BeamCurveDistance(cs, dead_length=0.0, minimum_distance=0.0)
    with pytest.raises(ValueError, match="minimum_distance"):
        BeamCurveDistance(cs, dead_length=DEAD_LENGTH, minimum_distance=-0.1)


# ============================================================================
# dJ
# ============================================================================

@pytest.mark.parametrize("cls", [CoilSupportBeams, CoilSupportBeamsSorted])
def test_dJ_taylor_test(cls):
    """Central differences of J must match dJ to second order."""
    cs = _make_coil_support(cls)
    d0 = BeamCurveDistance(cs, DEAD_LENGTH, 0.0).shortest_distance()
    Jb = BeamCurveDistance(cs, dead_length=DEAD_LENGTH, minimum_distance=4.0 * d0)

    x0 = np.array(Jb.x)
    assert x0.size > 0
    rng = np.random.default_rng(0)
    dx = rng.standard_normal(x0.size)
    dJdx = np.asarray(Jb.dJ()) @ dx
    assert np.isfinite(dJdx)

    errs = []
    for eps in [1e-4, 1e-5, 1e-6]:
        Jb.x = x0 + eps * dx
        Jp = Jb.J()
        Jb.x = x0 - eps * dx
        Jm = Jb.J()
        errs.append(abs((Jp - Jm) / (2 * eps) - dJdx))
    Jb.x = x0

    assert errs[0] > 0.0
    for e_coarse, e_fine in zip(errs[:-1], errs[1:]):
        assert e_fine < e_coarse * 0.05 or e_fine < 1e-9 * abs(dJdx)


# ============================================================================
# effective_endpoint
# ============================================================================

def _synthetic_geom(xi_start, xi_end, L=1.0):
    """One beam along +x of length ``L`` with the given surface-exit stations."""
    return {
        'x_start': jnp.array([[0.0, 0.0, 0.0]]),
        'x_end': jnp.array([[L, 0.0, 0.0]]),
        'L': jnp.array([L]),
        'xi_start': jnp.array([xi_start]),
        'xi_end': jnp.array([xi_end]),
    }


def _span_stations(Jb, geom):
    x_a, x_b, _ = Jb._effective_segments(geom)
    L = float(geom['L'][0])
    return float(x_a[0, 0]) / L, float(x_b[0, 0]) / L


def test_effective_endpoint_measures_from_surface_exits():
    cs = _make_coil_support()
    geom = _synthetic_geom(0.1, 0.8)
    Jc = BeamCurveDistance(cs, dead_length=0.05, minimum_distance=0.1)
    Je = BeamCurveDistance(
        cs, dead_length=0.05, minimum_distance=0.1, effective_endpoint=True,
    )
    np.testing.assert_allclose(_span_stations(Jc, geom), (0.1, 0.8), atol=1e-12)
    np.testing.assert_allclose(_span_stations(Je, geom), (0.15, 0.75), atol=1e-12)


def test_effective_endpoint_caps_oversized_dead_length():
    """2*dead_length >= L_free collapses to a min_span sliver at the free-span midpoint."""
    cs = _make_coil_support()
    Je = BeamCurveDistance(
        cs, dead_length=10.0, minimum_distance=0.1, effective_endpoint=True,
    )
    a, b = _span_stations(Je, _synthetic_geom(0.1, 0.8))
    np.testing.assert_allclose(0.5 * (a + b), 0.45, atol=1e-12)
    np.testing.assert_allclose(b - a, 2e-3, atol=1e-9)


@pytest.mark.parametrize("xi_start, xi_end", [(0.6, 0.4), (0.5, 0.5)])
def test_effective_endpoint_overlapping_exits_keep_sliver(xi_start, xi_end):
    cs = _make_coil_support()
    Je = BeamCurveDistance(
        cs, dead_length=0.05, minimum_distance=0.1, effective_endpoint=True,
    )
    a, b = _span_stations(Je, _synthetic_geom(xi_start, xi_end))
    np.testing.assert_allclose(0.5 * (a + b), 0.5, atol=1e-12)
    np.testing.assert_allclose(b - a, 1e-3, atol=1e-9)


def test_effective_endpoint_matches_default_when_meshes_unbound():
    """Unbound meshes give xi = (0, 1), so both modes agree and dJ is finite."""
    cs = _make_coil_support()
    d0 = BeamCurveDistance(cs, DEAD_LENGTH, 0.0).shortest_distance()
    Jc = BeamCurveDistance(cs, DEAD_LENGTH, 4.0 * d0)
    Je = BeamCurveDistance(cs, DEAD_LENGTH, 4.0 * d0, effective_endpoint=True)
    np.testing.assert_allclose(Je.J(), Jc.J(), rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(Je.dJ()), np.asarray(Jc.dJ()), rtol=1e-10, atol=1e-14,
    )
    assert np.all(np.isfinite(np.asarray(Je.dJ())))


def test_dJ_reaches_both_curve_and_support_dofs():
    cs = _make_coil_support()
    d0 = BeamCurveDistance(cs, DEAD_LENGTH, 0.0).shortest_distance()
    Jb = BeamCurveDistance(cs, dead_length=DEAD_LENGTH, minimum_distance=4.0 * d0)

    partials = Jb.dJ(partials=True)
    for curve in cs.base_curves:
        assert np.any(np.asarray(partials(curve)) != 0.0)
    assert np.any(np.asarray(partials(cs)) != 0.0)
