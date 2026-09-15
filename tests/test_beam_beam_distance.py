"""Checks for the BeamBeamDistance support-beam clearance penalty."""

import jax.numpy as jnp
import numpy as np
import pytest

from coil_fem.simsopt.objectives import (
    _beam_beam_pair_indices,
    _chord_quadrature,
    _pooled_endpoints,
    _segment_point_dists,
)

simsopt = pytest.importorskip("simsopt")

from simsopt.field import Coil, Current                       # noqa: E402
from simsopt.geo import create_equally_spaced_curves          # noqa: E402
from simsopt.geo.curveobjectives import cc_distance_pure      # noqa: E402

from coil_fem.simsopt import (                                # noqa: E402
    BeamBeamDistance,
    CoilSupportBeams,
    CoilSupportBeamsCSR,
    CoilSupportBeamsSorted,
)

NFP = 2
N_BASE = 2
N_QUAD = 16

BEAM_OPTIONS = {
    'n_beam_cc': 2,
    'n_beam_cf': 0,
    'E': 200e9,
    'nu': 0.3,
    'cross_section_type': 'solid_circle',
    'attachment_type': 'direct',
}


def _make_coil_support(cls=CoilSupportBeams, **beam_overrides):
    """A 2-coil, nfp=2, stellsym beam support with two CC beams per group."""
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


def _make_csr_support():
    """Minimal CSR support used only to check the TypeError path."""
    curves = create_equally_spaced_curves(
        1, NFP, stellsym=False, R0=1.2, R1=0.4, order=1, numquadpoints=16,
    )
    base_coils = [Coil(c, Current(1e5)) for c in curves]
    return CoilSupportBeamsCSR(
        base_coils=base_coils,
        nfp=NFP,
        stellsym=False,
        beam_options={
            'n_beam_cc': 0,
            'n_beam_cf': 0,
            'n_beam_cr': 1,
            'E': 200e9,
            'nu': 0.3,
            'cross_section_type': 'solid_circle',
            'attachment_type': 'direct',
        },
        csr_options={
            'order': 1, 'w1': 0.08, 'w2': 0.06, 'n_phi': 4,
            'n_grid_1': 1, 'n_grid_2': 1, 'E': 200e9, 'nu': 0.3,
        },
        problem_options={'solver': 'umfpack'},
        r_beam=0.05,
    )


def _dof_index(opt, pattern, prefix=False):
    """Index into ``opt.x`` of the first DOF whose name matches ``pattern``."""
    for i, name in enumerate(opt.dof_names):
        if name.startswith(pattern) if prefix else name.endswith(pattern):
            return i
    raise AssertionError(f"no DOF matching {pattern!r} in {opt.dof_names}")


def _numpy_cc_distance(pts_a, tan_a, pts_b, tan_b, dmin):
    """Host reconstruction of ``cc_distance_pure`` for one pair."""
    d = np.linalg.norm(pts_a[:, None, :] - pts_b[None, :, :], axis=-1)
    alen = (
        np.linalg.norm(tan_a, axis=1)[:, None]
        * np.linalg.norm(tan_b, axis=1)[None, :]
    )
    return float(np.sum(alen * np.maximum(dmin - d, 0.0) ** 2) / (len(pts_a) * len(pts_b)))


def _host_J(Jb):
    """Sum the numpy double-integral over the static pair list."""
    cdofs, sdofs = Jb._read_dofs()
    curves_jax = Jb._curves_jax(cdofs)
    geom = Jb._support.beam_geometry(curves_jax, sdofs)
    x_s, x_e = _pooled_endpoints(geom, Jb._support)
    pts, tan = _chord_quadrature(x_s, x_e, Jb.n_quad)
    pts = np.asarray(pts)
    tan = np.asarray(tan)
    ia = np.asarray(Jb._ia)
    ib = np.asarray(Jb._ib)
    return sum(
        _numpy_cc_distance(pts[i], tan[i], pts[j], tan[j], Jb.minimum_distance)
        for i, j in zip(ia, ib)
    )


# ============================================================================
# Pure helpers
# ============================================================================

def test_chord_quadrature_midpoints_and_tangents():
    x_s = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    x_e = jnp.array([[2.0, 0.0, 0.0], [1.0, 4.0, 0.0]])
    pts, tan = _chord_quadrature(x_s, x_e, n_quad=4)
    assert pts.shape == (2, 4, 3)
    assert tan.shape == (2, 4, 3)
    xi = (np.arange(4) + 0.5) / 4
    expected0 = xi[:, None] * np.array([2.0, 0.0, 0.0])
    np.testing.assert_allclose(np.asarray(pts[0]), expected0)
    np.testing.assert_allclose(np.asarray(tan[0, 0]), [2.0, 0.0, 0.0])
    np.testing.assert_allclose(np.asarray(tan[1, 0]), [0.0, 4.0, 0.0])



def test_cc_distance_pure_matches_numpy():
    pts, tan = _chord_quadrature(
        jnp.array([[0.0, 0.0, 0.0]]),
        jnp.array([[1.0, 0.0, 0.0]]),
        n_quad=5,
    )
    pts_b, tan_b = _chord_quadrature(
        jnp.array([[0.0, 0.5, 0.0]]),
        jnp.array([[1.0, 0.5, 0.0]]),
        n_quad=5,
    )
    dmin = 1.0
    val = float(cc_distance_pure(pts[0], tan[0], pts_b[0], tan_b[0], dmin))
    expected = _numpy_cc_distance(
        np.asarray(pts[0]), np.asarray(tan[0]),
        np.asarray(pts_b[0]), np.asarray(tan_b[0]),
        dmin,
    )
    np.testing.assert_allclose(val, expected, atol=1e-12)


# ============================================================================
# J
# ============================================================================

def test_J_matches_numpy_double_sum_and_zero_when_clear():
    cs = _make_coil_support()
    d0 = BeamBeamDistance(cs, 0.0, n_quad=N_QUAD).shortest_distance()
    assert np.isfinite(d0)
    assert d0 > 0.0

    J_clear = BeamBeamDistance(cs, 0.5 * d0, n_quad=N_QUAD)
    assert J_clear.J() == 0.0

    Jb = BeamBeamDistance(cs, 4.0 * d0, n_quad=N_QUAD)
    assert Jb.J() > 0.0
    np.testing.assert_allclose(Jb.J(), _host_J(Jb), rtol=1e-10)


def test_J_is_intra_group_only():
    """Cross-group CC pairs must not enter J (wrap beams stay unpaired)."""
    cs = _make_coil_support()
    support = cs.support
    # n_beam_cc=2 with stellsym halves the wrap groups to 1: only group 0 pairs.
    assert support.n_beam_cc[0] == 2
    assert all(n < 2 for n in support.n_beam_cc[1:])
    ia, ib = _beam_beam_pair_indices(support)
    assert ia.shape == (1,)
    assert int(ia[0]) == support.beam_offsets[0]
    assert int(ib[0]) == support.beam_offsets[0] + 1


def test_cs_pairs_include_symmetry_images():
    cs = _make_coil_support(
        n_beam_cc=0, n_beam_cf=0,
        i_beam_cs=[(0, 1)], s_beam_cs=[True],
    )
    n_cs = cs.support.n_beam_cs
    assert n_cs == 1
    n_images = n_cs * NFP * 2
    ia, ib = _beam_beam_pair_indices(cs.support)
    assert ia.size == n_images - 1
    Jb = BeamBeamDistance(cs, minimum_distance=10.0, n_quad=N_QUAD)
    assert Jb.J() > 0.0
    np.testing.assert_allclose(Jb.J(), _host_J(Jb), rtol=1e-10)


def test_shortest_distance_matches_point_to_segment():
    cs = _make_coil_support()
    Jb = BeamBeamDistance(cs, minimum_distance=0.1, n_quad=N_QUAD)
    cdofs, sdofs = Jb._read_dofs()
    curves_jax = Jb._curves_jax(cdofs)
    geom = cs.support.beam_geometry(curves_jax, sdofs)
    x_s, x_e = _pooled_endpoints(geom, cs.support)
    pts, _ = _chord_quadrature(x_s, x_e, N_QUAD)
    best = np.inf
    for i, j in zip(np.asarray(Jb._ia), np.asarray(Jb._ib)):
        d_ab = float(jnp.min(_segment_point_dists(x_s[i][None], x_e[i][None], pts[j])))
        d_ba = float(jnp.min(_segment_point_dists(x_s[j][None], x_e[j][None], pts[i])))
        best = min(best, d_ab, d_ba)
    np.testing.assert_allclose(Jb.shortest_distance(), best, rtol=1e-10)


def test_J_cache_invalidated_on_dof_change():
    cs = _make_coil_support()
    Jb = BeamBeamDistance(cs, minimum_distance=1.0, n_quad=N_QUAD)
    J0 = Jb.J()
    i = _dof_index(Jb, ':phis_start_cc(0,0)')
    x = np.array(Jb.x)
    x[i] += 0.05
    Jb.x = x
    assert Jb.J() != J0


def test_J_independent_of_currents():
    cs = _make_coil_support()
    Jb = BeamBeamDistance(cs, minimum_distance=1.0, n_quad=N_QUAD)
    J0 = Jb.J()
    i = _dof_index(Jb, 'Current', prefix=True)
    x = np.array(Jb.x)
    x[i] *= 2.0
    Jb.x = x
    assert Jb.J() == J0
    assert np.asarray(Jb.dJ())[i] == 0.0


def test_rejects_cf_csr_and_negative_dmin():
    with pytest.raises(NotImplementedError, match="CF beams"):
        BeamBeamDistance(_make_coil_support(n_beam_cf=1), minimum_distance=0.1)
    with pytest.raises(TypeError, match="SupportBeamsCSR"):
        BeamBeamDistance(_make_csr_support(), minimum_distance=0.1)
    cs = _make_coil_support()
    with pytest.raises(ValueError, match="minimum_distance"):
        BeamBeamDistance(cs, minimum_distance=-0.1)
    with pytest.raises(ValueError, match="n_quad"):
        BeamBeamDistance(cs, minimum_distance=0.1, n_quad=0)


# ============================================================================
# dJ
# ============================================================================

@pytest.mark.parametrize("cls", [CoilSupportBeams, CoilSupportBeamsSorted])
def test_dJ_taylor_test(cls):
    """Central differences of J must match dJ to second order."""
    cs = _make_coil_support(cls)
    d0 = BeamBeamDistance(cs, 0.0, n_quad=N_QUAD).shortest_distance()
    Jb = BeamBeamDistance(cs, minimum_distance=4.0 * d0, n_quad=N_QUAD)

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


def test_dJ_reaches_both_curve_and_support_dofs():
    cs = _make_coil_support()
    d0 = BeamBeamDistance(cs, 0.0, n_quad=N_QUAD).shortest_distance()
    Jb = BeamBeamDistance(cs, minimum_distance=4.0 * d0, n_quad=N_QUAD)
    partials = Jb.dJ(partials=True)
    for curve in cs.base_curves:
        assert np.any(np.asarray(partials(curve)) != 0.0)
    assert np.any(np.asarray(partials(cs)) != 0.0)
