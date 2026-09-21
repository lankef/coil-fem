"""Checks for the CSRSurfaceDistance CSR–surface hinge."""

from __future__ import annotations

import numpy as np
import pytest

simsopt = pytest.importorskip("simsopt")

from simsopt.field import Coil, Current                       # noqa: E402
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves  # noqa: E402

from coil_fem.simsopt import (                                # noqa: E402
    CSRSurfaceDistance,
    CoilSupportBeamsCSR,
)

NFP = 2
W1 = 0.08
W2 = 0.06
DMIN = 0.3


def _csr_dofs(R, order=1, stellsym=False):
    n = (2 * order + 1) if stellsym else (4 * order + 2)
    dofs = np.zeros(n)
    dofs[0] = R
    return dofs


def _make_coil_support(*, R=3.0, stellsym=False):
    """CSR support with a circular ring of radius ``R``."""
    curves = create_equally_spaced_curves(
        1, NFP, stellsym=stellsym,
        R0=1.2, R1=0.4, order=1, numquadpoints=16,
    )
    base_coils = [Coil(c, Current(1e5)) for c in curves]
    return CoilSupportBeamsCSR(
        base_coils=base_coils,
        nfp=NFP,
        stellsym=stellsym,
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
            'order': 1,
            'w1': W1,
            'w2': W2,
            'n_phi': 4,
            'n_grid_1': 1,
            'n_grid_2': 1,
            'E': 200e9,
            'nu': 0.3,
        },
        problem_options={'solver': 'umfpack'},
        csr_curve_dofs=_csr_dofs(R, stellsym=stellsym),
        fixed_dof_names=[
            'thetas_orientation_cc',
            'thetas_orientation_cf',
            'thetas_orientation_cr',
            'phis_start_cc', 'phis_end_cc',
            'phis_start_cf', 'x_foundation',
            'phis_start_cr', 'phis_end_cr',
            'r_beam',
        ],
        r_beam=0.05,
    )


def _make_surface(nphi=8, ntheta=8):
    """A small torus well inside a large CSR ring."""
    surf = SurfaceRZFourier(
        nfp=NFP, stellsym=True, mpol=1, ntor=0,
        quadpoints_phi=np.linspace(0, 1, nphi, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, ntheta, endpoint=False),
    )
    surf.set_rc(0, 0, 1.0)
    surf.set_rc(1, 0, 0.15)
    surf.set_zs(1, 0, 0.15)
    return surf


def _rc0_index(opt):
    """Index of ``csr_curve_dofs(0)`` (``rc_0``) in ``opt.x``."""
    for i, name in enumerate(opt.dof_names):
        if name.endswith('csr_curve_dofs(0)'):
            return i
    raise AssertionError(f"no csr_curve_dofs(0) in {opt.dof_names}")


def test_far_csr_is_feasible():
    """Circular CSR well outside the torus → J == 0."""
    cs = _make_coil_support(R=3.0)
    Jcs = CSRSurfaceDistance(cs, _make_surface(), DMIN)
    assert Jcs.J() == 0.0
    assert Jcs.shortest_distance() > DMIN


def test_near_csr_is_penalised():
    """CSR through the torus → J > 0."""
    cs = _make_coil_support(R=1.0)
    Jcs = CSRSurfaceDistance(cs, _make_surface(), DMIN)
    assert Jcs.J() > 0.0
    assert Jcs.shortest_distance() < DMIN


def test_taylor_dJ_d_rc0():
    """Centered FD on ``csr_curve_dofs(0)`` matches ``dJ`` while the hinge is active."""
    cs = _make_coil_support(R=1.0)
    Jcs = CSRSurfaceDistance(cs, _make_surface(), DMIN)
    assert Jcs.J() > 0.0
    g = Jcs.dJ()
    i = _rc0_index(Jcs)
    eps = 1e-6
    x0 = Jcs.x.copy()
    x_p = x0.copy()
    x_m = x0.copy()
    x_p[i] += eps
    x_m[i] -= eps
    Jcs.x = x_p
    Jp = Jcs.J()
    Jcs.x = x_m
    Jm = Jcs.J()
    Jcs.x = x0
    fd = (Jp - Jm) / (2.0 * eps)
    np.testing.assert_allclose(g[i], fd, rtol=1e-4, atol=1e-8)


@pytest.mark.parametrize('stellsym', [False, True])
def test_quadpoints_are_one_sector(stellsym):
    """CSR samples stay in the first field period / stellsym half."""
    cs = _make_coil_support(R=3.0, stellsym=stellsym)
    Jcs = CSRSurfaceDistance(cs, _make_surface(), DMIN)
    phi_max = 1.0 / NFP / (2.0 if stellsym else 1.0)
    qp = np.asarray(Jcs._qp_sector)
    assert qp.size > 0
    assert np.max(qp) < phi_max


def test_rejects_non_csr_support():
    from coil_fem.simsopt import CoilSupportBeams

    curves = create_equally_spaced_curves(
        1, NFP, stellsym=False, R0=1.2, R1=0.4, order=1, numquadpoints=16,
    )
    base_coils = [Coil(c, Current(1e5)) for c in curves]
    cs = CoilSupportBeams(
        base_coils=base_coils,
        nfp=NFP,
        stellsym=False,
        beam_options={
            'n_beam_cc': 0,
            'n_beam_cf': 1,
            'E': 200e9,
            'nu': 0.3,
            'cross_section_type': 'solid_circle',
            'attachment_type': 'direct',
        },
        r_beam=0.05,
    )
    with pytest.raises(TypeError, match='SupportBeamsCSR'):
        CSRSurfaceDistance(cs, _make_surface(), DMIN)
