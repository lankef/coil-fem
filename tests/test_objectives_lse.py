"""Checks for the LSE soft-minimum clearance objectives."""

import numpy as np
import pytest
from scipy.spatial.distance import cdist

import jax.numpy as jnp

from coil_fem.simsopt.objectives import _chord_quadrature, _pooled_endpoints
from coil_fem.simsopt.objectives_geo import max_lse

simsopt = pytest.importorskip("simsopt")

from simsopt.field import Coil, Current                       # noqa: E402
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves  # noqa: E402

from coil_fem.simsopt import (                                # noqa: E402
    BeamBeamDistanceMinLSE,
    BeamSurfaceDistanceMinLSE,
    CoilSupportBeams,
    CurveCurveDistanceMinLSE,
    CurveCurvatureMaxSqLSE,
    CurveSurfaceDistanceMinLSE,
)

EPS = 1e-2
NFP = 2
N_BASE = 2


def _curves(numquadpoints=16):
    return create_equally_spaced_curves(
        N_BASE, NFP, stellsym=True, R0=1.0, R1=0.5, order=2,
        numquadpoints=numquadpoints,
    )


def _surface():
    surf = SurfaceRZFourier(
        nfp=NFP, stellsym=True, mpol=1, ntor=0,
        quadpoints_phi=np.linspace(0, 1, 8, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, 8, endpoint=False),
    )
    surf.set_rc(0, 0, 1.0)
    surf.set_rc(1, 0, 0.15)
    surf.set_zs(1, 0, 0.15)
    return surf


def _support():
    """Two CC beams per group and no CF beams, so both beam objectives accept it."""
    base_coils = [Coil(c, Current(1e5)) for c in _curves(numquadpoints=32)]
    return CoilSupportBeams(
        base_coils=base_coils,
        nfp=NFP,
        stellsym=True,
        beam_options={
            'n_beam_cc': 2,
            'n_beam_cf': 0,
            'E': 200e9,
            'nu': 0.3,
            'cross_section_type': 'solid_circle',
            'attachment_type': 'direct',
        },
        r_beam=0.05,
    )


def _assert_softmin(J, true_min, epsilon):
    assert np.isfinite(J)
    assert true_min - epsilon - 1e-8 <= J <= true_min + 1e-8


def _assert_fd(obj):
    """Central differences of J must match dJ to second order."""
    x0 = np.array(obj.x)
    assert x0.size > 0
    rng = np.random.default_rng(0)
    dx = rng.standard_normal(x0.size)
    dJdx = float(np.asarray(obj.dJ()) @ dx)
    assert np.isfinite(dJdx)

    errs = []
    for step in (1e-4, 1e-5, 1e-6):
        obj.x = x0 + step * dx
        Jp = float(obj.J())
        obj.x = x0 - step * dx
        Jm = float(obj.J())
        errs.append(abs((Jp - Jm) / (2 * step) - dJdx))
    obj.x = x0

    assert errs[0] > 0.0
    for coarse, fine in zip(errs[:-1], errs[1:]):
        assert fine < coarse * 0.05 or fine < 1e-9 * abs(dJdx)


def test_max_lse_bounds_and_singleton():
    x = jnp.asarray(np.random.default_rng(0).normal(size=50))
    approx = float(max_lse(x, EPS))
    maximum = float(np.max(x))
    assert maximum - 1e-12 <= approx <= maximum + EPS + 1e-9
    np.testing.assert_allclose(float(max_lse(x[:1], EPS)), float(x[0]), atol=1e-12)


def test_rejects_nonpositive_epsilon():
    curves = _curves()
    surf = _surface()
    cs = _support()
    with pytest.raises(ValueError, match="epsilon"):
        CurveCurveDistanceMinLSE(curves, 0.0)
    with pytest.raises(ValueError, match="epsilon"):
        CurveCurvatureMaxSqLSE(curves, 0.0)
    with pytest.raises(ValueError, match="epsilon"):
        CurveSurfaceDistanceMinLSE(curves, surf, -0.1)
    with pytest.raises(ValueError, match="epsilon"):
        BeamSurfaceDistanceMinLSE(cs, surf, 0.0)
    with pytest.raises(ValueError, match="epsilon"):
        BeamBeamDistanceMinLSE(cs, -1.0)


def test_curve_curve_bound_and_dJ():
    curves = _curves()
    obj = CurveCurveDistanceMinLSE(curves, EPS, downsample=2)
    sampled = [c.gamma()[::2] for c in curves]
    true_min = min(cdist(sampled[i], sampled[j]).min() for i, j in obj.pairs)
    _assert_softmin(float(obj.J()), true_min, EPS)
    _assert_fd(obj)


def test_curve_surface_bound_and_dJ():
    curves = _curves()
    surf = _surface()
    obj = CurveSurfaceDistanceMinLSE(curves, surf, EPS)
    surf_pts = surf.gamma().reshape((-1, 3))
    true_min = min(cdist(c.gamma(), surf_pts).min() for c in curves)
    _assert_softmin(float(obj.J()), true_min, EPS)
    _assert_fd(obj)


def test_beam_surface_bound_and_dJ():
    cs = _support()
    surf = _surface()
    obj = BeamSurfaceDistanceMinLSE(cs, surf, EPS)
    _assert_softmin(obj.J(), obj.shortest_distance(), EPS)
    _assert_fd(obj)


def test_beam_beam_bound_and_dJ():
    obj = BeamBeamDistanceMinLSE(_support(), EPS, n_quad=8)
    cdofs, sdofs = obj._read_dofs()
    geom = obj._support.beam_geometry(obj._curves_jax(cdofs), sdofs)
    x_s, x_e = _pooled_endpoints(geom, obj._support)
    pts, _ = _chord_quadrature(x_s, x_e, obj.n_quad)
    pts = np.asarray(pts)
    ia = np.asarray(obj._ia)
    ib = np.asarray(obj._ib)
    assert ia.size > 0
    true_min = min(cdist(pts[i], pts[j]).min() for i, j in zip(ia, ib))
    _assert_softmin(obj.J(), true_min, EPS)
    _assert_fd(obj)


def test_curvature_max_sq_bound_and_dJ():
    curves = _curves()
    obj = CurveCurvatureMaxSqLSE(curves, EPS)
    true_max = max(float(np.max(np.asarray(c.kappa()) ** 2)) for c in curves)
    J = float(obj.J())
    assert true_max - 1e-8 <= J <= true_max + EPS + 1e-8

    # Curvature squared is more nonlinear than the distance objectives, so the
    # shared 1e-4 step sits outside the linear range.
    x0 = np.array(obj.x)
    dx = np.random.default_rng(0).standard_normal(x0.size)
    dJdx = float(np.asarray(obj.dJ()) @ dx)
    errs = []
    for step in (1e-5, 1e-6, 1e-7):
        obj.x = x0 + step * dx
        Jp = float(obj.J())
        obj.x = x0 - step * dx
        Jm = float(obj.J())
        errs.append(abs((Jp - Jm) / (2 * step) - dJdx))
    obj.x = x0
    assert errs[0] > 0.0
    for coarse, fine in zip(errs[:-1], errs[1:]):
        assert fine < coarse * 0.05 or fine < 1e-9 * abs(dJdx)
