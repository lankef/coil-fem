"""Checks for the BeamSurfaceDistance support-beam clearance penalty."""

import jax.numpy as jnp
import numpy as np
import pytest

from coil_fem.simsopt.objectives import _segment_point_dists

simsopt = pytest.importorskip("simsopt")

from simsopt.field import Coil, Current                       # noqa: E402
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves  # noqa: E402

from coil_fem.simsopt import (                                # noqa: E402
    BeamSurfaceDistance,
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


def _make_coil_support(cls=CoilSupportBeams):
    """A 2-coil, nfp=2, stellsym beam support with one CC and one CF beam each."""
    curves = create_equally_spaced_curves(
        N_BASE, NFP, stellsym=True, R0=1.0, R1=0.5, order=2, numquadpoints=32,
    )
    base_coils = [Coil(c, Current(1e5)) for c in curves]
    return cls(
        base_coils=base_coils,
        nfp=NFP,
        stellsym=True,
        beam_options=BEAM_OPTIONS,
        r_beam=0.05,
    )


def _dof_index(opt, pattern, prefix=False):
    """Index into ``opt.x`` of the first DOF whose name matches ``pattern``.

    Simsopt auto-numbers Optimizable names per process, so tests match on the
    stable part of the name rather than the full string.
    """
    for i, name in enumerate(opt.dof_names):
        if name.startswith(pattern) if prefix else name.endswith(pattern):
            return i
    raise AssertionError(f"no DOF matching {pattern!r} in {opt.dof_names}")


def _make_surface(nphi=8, ntheta=8):
    """A small torus well inside the coil set."""
    surf = SurfaceRZFourier(
        nfp=NFP, stellsym=True, mpol=1, ntor=0,
        quadpoints_phi=np.linspace(0, 1, nphi, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, ntheta, endpoint=False),
    )
    surf.set_rc(0, 0, 1.0)
    surf.set_rc(1, 0, 0.15)
    surf.set_zs(1, 0, 0.15)
    return surf


# ============================================================================
# Pure segment-distance formula
# ============================================================================

def test_segment_distance_uses_perpendicular_foot():
    """A point beside the chord midpoint gets the perpendicular distance."""
    x_start = jnp.array([[0.0, 0.0, 0.0]])
    x_end = jnp.array([[10.0, 0.0, 0.0]])
    pts = jnp.array([[5.0, 2.0, 0.0]])
    d = _segment_point_dists(x_start, x_end, pts)
    assert d.shape == (1, 1)
    np.testing.assert_allclose(float(d[0, 0]), 2.0, atol=1e-12)


def test_segment_distance_clamps_beyond_endpoints():
    """Past the ends, the distance falls back to the nearest endpoint."""
    x_start = jnp.array([[0.0, 0.0, 0.0]])
    x_end = jnp.array([[10.0, 0.0, 0.0]])
    pts = jnp.array([[-3.0, 4.0, 0.0], [14.0, 0.0, 3.0]])
    d = _segment_point_dists(x_start, x_end, pts)
    np.testing.assert_allclose(np.asarray(d[0]), [5.0, 5.0], atol=1e-12)


# ============================================================================
# J
# ============================================================================

def test_J_zero_when_clear_and_positive_when_violated():
    cs = _make_coil_support()
    surf = _make_surface()

    d_min_clear = 0.5 * BeamSurfaceDistance(cs, surf, 0.0).shortest_distance()
    assert d_min_clear > 0.0

    assert BeamSurfaceDistance(cs, surf, d_min_clear).J() == 0.0
    assert BeamSurfaceDistance(cs, surf, 4.0 * d_min_clear).J() > 0.0


def test_shortest_distance_matches_brute_force_segment_distance():
    """shortest_distance() agrees with a host-side point-to-segment sweep."""
    cs = _make_coil_support()
    surf = _make_surface()
    Jb = BeamSurfaceDistance(cs, surf, 0.1)

    cdofs, sdofs = Jb._read_dofs()
    x_start, x_end, _ = Jb._beam_chords(cdofs, sdofs)
    x_start = np.asarray(x_start)
    x_end = np.asarray(x_end)
    pts = np.asarray(surf.gamma().reshape((-1, 3)))

    best = np.inf
    for a, b in zip(x_start, x_end):
        ab = b - a
        t = np.clip((pts - a) @ ab / (ab @ ab), 0.0, 1.0)
        best = min(best, np.min(np.linalg.norm(a + t[:, None] * ab - pts, axis=1)))

    np.testing.assert_allclose(Jb.shortest_distance(), best, rtol=1e-10)


def test_J_matches_hinge_formula_on_host():
    """J reproduces mean(L * |n| * max(0, d_min - d)^2) computed with numpy."""
    cs = _make_coil_support()
    surf = _make_surface()
    d_min = 3.0 * BeamSurfaceDistance(cs, surf, 0.0).shortest_distance()
    Jb = BeamSurfaceDistance(cs, surf, d_min)

    cdofs, sdofs = Jb._read_dofs()
    x_start, x_end, L = Jb._beam_chords(cdofs, sdofs)
    gammas = jnp.asarray(surf.gamma().reshape((-1, 3)))
    dists = np.asarray(_segment_point_dists(x_start, x_end, gammas))
    ns = np.linalg.norm(surf.normal().reshape((-1, 3)), axis=1)
    expected = np.mean(
        np.asarray(L)[:, None] * ns[None, :]
        * np.maximum(d_min - dists, 0.0) ** 2
    )
    np.testing.assert_allclose(Jb.J(), expected, rtol=1e-10)


def test_J_cache_invalidated_on_dof_change():
    cs = _make_coil_support()
    Jb = BeamSurfaceDistance(cs, _make_surface(), 1.0)
    J0 = Jb.J()

    i = _dof_index(Jb, ':x_foundation(0,0,0)')
    x = np.array(Jb.x)
    x[i] += 0.05
    Jb.x = x
    assert Jb.J() != J0


def test_J_independent_of_currents():
    """The penalty is purely geometric, so current DOFs must not move it."""
    cs = _make_coil_support()
    Jb = BeamSurfaceDistance(cs, _make_surface(), 1.0)
    J0 = Jb.J()

    i = _dof_index(Jb, 'Current', prefix=True)
    x = np.array(Jb.x)
    x[i] *= 2.0
    Jb.x = x
    assert Jb.J() == J0
    assert np.asarray(Jb.dJ())[i] == 0.0


# ============================================================================
# dJ
# ============================================================================

@pytest.mark.parametrize("cls", [CoilSupportBeams, CoilSupportBeamsSorted])
def test_dJ_taylor_test(cls):
    """Central differences of J must match dJ to second order."""
    cs = _make_coil_support(cls)
    surf = _make_surface()
    # Large enough that the hinge is active for every beam/surface pair.
    Jb = BeamSurfaceDistance(cs, surf, 4.0 * BeamSurfaceDistance(
        cs, surf, 0.0).shortest_distance())

    x0 = np.array(Jb.x)
    assert x0.size > 0
    rng = np.random.default_rng(0)
    dx = rng.standard_normal(x0.size)
    dJdx = np.asarray(Jb.dJ()) @ dx

    errs = []
    for eps in [1e-4, 1e-5, 1e-6]:
        Jb.x = x0 + eps * dx
        Jp = Jb.J()
        Jb.x = x0 - eps * dx
        Jm = Jb.J()
        errs.append(abs((Jp - Jm) / (2 * eps) - dJdx))
    Jb.x = x0

    assert errs[0] > 0.0
    # Second-order convergence: each 10x smaller eps cuts the error ~100x.
    for e_coarse, e_fine in zip(errs[:-1], errs[1:]):
        assert e_fine < e_coarse * 0.05 or e_fine < 1e-9 * abs(dJdx)


def test_dJ_reaches_both_curve_and_support_dofs():
    cs = _make_coil_support()
    surf = _make_surface()
    Jb = BeamSurfaceDistance(cs, surf, 4.0 * BeamSurfaceDistance(
        cs, surf, 0.0).shortest_distance())

    partials = Jb.dJ(partials=True)
    for curve in cs.base_curves:
        assert np.any(np.asarray(partials(curve)) != 0.0)
    assert np.any(np.asarray(partials(cs)) != 0.0)
