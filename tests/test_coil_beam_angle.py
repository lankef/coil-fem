"""Checks for the CoilBeamAngle beam–coil attachment-angle penalty."""

import math

import numpy as np
import pytest

simsopt = pytest.importorskip("simsopt")

from simsopt.field import Coil, Current                       # noqa: E402
from simsopt.geo import create_equally_spaced_curves  # noqa: E402

from coil_fem.simsopt import (                                # noqa: E402
    CoilBeamAngle,
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
    """Index into ``opt.x`` of the first DOF whose name matches ``pattern``."""
    for i, name in enumerate(opt.dof_names):
        if name.startswith(pattern) if prefix else name.endswith(pattern):
            return i
    raise AssertionError(f"no DOF matching {pattern!r} in {opt.dof_names}")


# ============================================================================
# Constructor validation
# ============================================================================

def test_rejects_minimum_angle_out_of_range():
    cs = _make_coil_support()
    with pytest.raises(ValueError, match="minimum_angle"):
        CoilBeamAngle(cs, minimum_angle=-0.1)
    with pytest.raises(ValueError, match="minimum_angle"):
        CoilBeamAngle(cs, minimum_angle=0.5 * math.pi)
    with pytest.raises(ValueError, match="minimum_angle"):
        CoilBeamAngle(cs, minimum_angle=math.pi)


def test_rejects_invalid_mode():
    cs = _make_coil_support()
    with pytest.raises(ValueError, match="mode"):
        CoilBeamAngle(cs, minimum_angle=0.2, mode='coil')


# ============================================================================
# J / smallest_angle
# ============================================================================

def test_J_zero_when_clear_and_positive_when_violated():
    cs = _make_coil_support()
    alpha = CoilBeamAngle(cs, minimum_angle=0.0, mode='all').smallest_angle()
    assert 0.0 < alpha < 0.5 * math.pi

    assert CoilBeamAngle(cs, minimum_angle=0.5 * alpha, mode='all').J() == 0.0
    theta_violate = 0.5 * (alpha + 0.5 * math.pi)
    assert CoilBeamAngle(cs, minimum_angle=theta_violate, mode='all').J() > 0.0


def test_smallest_angle_matches_host_arccos():
    """smallest_angle agrees with a host-side sweep of active attachments."""
    cs = _make_coil_support()
    Jang = CoilBeamAngle(cs, minimum_angle=0.1, mode='all')

    cdofs, sdofs = Jang._read_dofs()
    t_beam, t_start, t_end = Jang._geom(cdofs, sdofs)
    t_beam = np.asarray(t_beam)
    t_start = np.asarray(t_start)
    t_end = np.asarray(t_end)
    is_cc = np.asarray(Jang._is_cc)
    is_cf = np.asarray(Jang._is_cf)

    angles = []
    for i in range(t_beam.shape[0]):
        if is_cc[i]:
            angles.append(math.acos(min(1.0, abs(t_beam[i] @ t_start[i]))))
            angles.append(math.acos(min(1.0, abs(t_beam[i] @ t_end[i]))))
        if is_cf[i]:
            angles.append(math.acos(min(1.0, abs(t_beam[i] @ t_start[i]))))

    np.testing.assert_allclose(Jang.smallest_angle(), min(angles), rtol=1e-10)


def test_J_matches_hinge_formula_on_host():
    cs = _make_coil_support()
    alpha = CoilBeamAngle(cs, minimum_angle=0.0).smallest_angle()
    theta = 0.5 * (alpha + 0.5 * math.pi)
    Jang = CoilBeamAngle(cs, minimum_angle=theta, mode='all')
    cos_min = math.cos(theta)

    cdofs, sdofs = Jang._read_dofs()
    t_beam, t_start, t_end = Jang._geom(cdofs, sdofs)
    t_beam = np.asarray(t_beam)
    t_start = np.asarray(t_start)
    t_end = np.asarray(t_end)
    is_cc = np.asarray(Jang._is_cc)
    is_cf = np.asarray(Jang._is_cf)

    expected = 0.0
    for i in range(t_beam.shape[0]):
        if is_cc[i]:
            expected += max(abs(t_beam[i] @ t_start[i]) - cos_min, 0.0) ** 2
            expected += max(abs(t_beam[i] @ t_end[i]) - cos_min, 0.0) ** 2
        if is_cf[i]:
            expected += max(abs(t_beam[i] @ t_start[i]) - cos_min, 0.0) ** 2

    np.testing.assert_allclose(Jang.J(), expected, rtol=1e-10)


def test_mode_switches_cc_cf_all():
    cs = _make_coil_support()
    # Default CF chords are orthogonal to t_coil (planar offset to z=0
    # foundations).  Shift foundations in z so CF angles leave π/2.
    x = np.array(cs.x)
    for i, name in enumerate(cs.dof_names):
        if name.endswith(':x_foundation(0,0,2)') or name.endswith(
            ':x_foundation(1,0,2)'
        ):
            x[i] += 0.5
    cs.x = x

    alpha_cc = CoilBeamAngle(cs, 0.0, mode='cc').smallest_angle()
    alpha_cf = CoilBeamAngle(cs, 0.0, mode='cf').smallest_angle()
    assert alpha_cf < 0.5 * math.pi - 1e-6
    theta = 0.5 * (max(alpha_cc, alpha_cf) + 0.5 * math.pi)

    J_cc = CoilBeamAngle(cs, theta, mode='cc').J()
    J_cf = CoilBeamAngle(cs, theta, mode='cf').J()
    J_all = CoilBeamAngle(cs, theta, mode='all').J()

    assert J_cc > 0.0
    assert J_cf > 0.0
    np.testing.assert_allclose(J_all, J_cc + J_cf, rtol=1e-12)
    assert (
        CoilBeamAngle(cs, theta, mode='cf').smallest_angle()
        >= CoilBeamAngle(cs, theta, mode='all').smallest_angle() - 1e-15
    )


def test_smallest_angle_empty_cf_mode_returns_half_pi():
    """mode='cf' with no CF beams reports π/2 (no contributing endpoints)."""
    curves = create_equally_spaced_curves(
        N_BASE, NFP, stellsym=True, R0=1.0, R1=0.5, order=2, numquadpoints=32,
    )
    base_coils = [Coil(c, Current(1e5)) for c in curves]
    cs = CoilSupportBeams(
        base_coils=base_coils,
        nfp=NFP,
        stellsym=True,
        beam_options={**BEAM_OPTIONS, 'n_beam_cf': 0},
        r_beam=0.05,
        fixed_clamp_options={
            'enabled': True,
            'k_clamp': 1e9,
            'n_clamp': 1,
            'r_clamp': 0.1,
        },
    )
    Jang = CoilBeamAngle(cs, minimum_angle=0.2, mode='cf')
    np.testing.assert_allclose(Jang.smallest_angle(), 0.5 * math.pi, atol=1e-15)
    assert Jang.J() == 0.0


def test_J_cache_invalidated_on_dof_change():
    cs = _make_coil_support()
    alpha = CoilBeamAngle(cs, 0.0).smallest_angle()
    Jang = CoilBeamAngle(cs, 0.5 * (alpha + 0.5 * math.pi), mode='all')
    J0 = Jang.J()

    i = _dof_index(Jang, ':phis_start_cc(0,0)')
    x = np.array(Jang.x)
    x[i] += 0.05
    Jang.x = x
    assert Jang.J() != J0


def test_J_independent_of_currents():
    cs = _make_coil_support()
    alpha = CoilBeamAngle(cs, 0.0).smallest_angle()
    Jang = CoilBeamAngle(cs, 0.5 * (alpha + 0.5 * math.pi), mode='all')
    J0 = Jang.J()

    i = _dof_index(Jang, 'Current', prefix=True)
    x = np.array(Jang.x)
    x[i] *= 2.0
    Jang.x = x
    assert Jang.J() == J0
    assert np.asarray(Jang.dJ())[i] == 0.0


# ============================================================================
# dJ
# ============================================================================

@pytest.mark.parametrize("cls", [CoilSupportBeams, CoilSupportBeamsSorted])
def test_dJ_taylor_test(cls):
    """Central differences of J must match dJ to second order."""
    cs = _make_coil_support(cls)
    alpha = CoilBeamAngle(cs, 0.0).smallest_angle()
    Jang = CoilBeamAngle(cs, 0.5 * (alpha + 0.5 * math.pi), mode='all')

    x0 = np.array(Jang.x)
    assert x0.size > 0
    assert len(np.asarray(Jang.dJ())) == Jang.dof_size

    rng = np.random.default_rng(0)
    dx = rng.standard_normal(x0.size)
    dJdx = np.asarray(Jang.dJ()) @ dx

    errs = []
    for eps in [1e-4, 1e-5, 1e-6]:
        Jang.x = x0 + eps * dx
        Jp = Jang.J()
        Jang.x = x0 - eps * dx
        Jm = Jang.J()
        errs.append(abs((Jp - Jm) / (2 * eps) - dJdx))
    Jang.x = x0

    assert errs[0] > 0.0
    for e_coarse, e_fine in zip(errs[:-1], errs[1:]):
        assert e_fine < e_coarse * 0.05 or e_fine < 1e-9 * abs(dJdx)


def test_dJ_reaches_both_curve_and_support_dofs():
    cs = _make_coil_support()
    alpha = CoilBeamAngle(cs, 0.0).smallest_angle()
    Jang = CoilBeamAngle(cs, 0.5 * (alpha + 0.5 * math.pi), mode='all')

    partials = Jang.dJ(partials=True)
    for curve in cs.base_curves:
        assert np.any(np.asarray(partials(curve)) != 0.0)
    assert np.any(np.asarray(partials(cs)) != 0.0)
