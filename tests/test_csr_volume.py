"""Checks for the CSRVolume rectangular-section volume estimate."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

simsopt = pytest.importorskip("simsopt")

from simsopt.field import Coil, Current                       # noqa: E402
from simsopt.geo import create_equally_spaced_curves          # noqa: E402

from coil_fem.simsopt import CSRVolume, CoilSupportBeamsCSR   # noqa: E402

NFP = 2
W1 = 0.08
W2 = 0.06
R0 = 1.0


def _make_coil_support(*, R: float = R0, fix_csr: bool = False):
    """Minimal CSR support with a circular ring of radius ``R``."""
    curves = create_equally_spaced_curves(
        1, NFP, stellsym=False, R0=1.2, R1=0.4, order=1, numquadpoints=16,
    )
    base_coils = [Coil(c, Current(1e5)) for c in curves]
    order = 1
    # Non-stellsym CurveRZFourier DOFs: [rc_0, rc_1, rs_1, zc_0, zc_1, zs_1]
    csr_dofs = np.zeros(4 * order + 2)
    csr_dofs[0] = R
    fixed = [
        'thetas_orientation_cc',
        'thetas_orientation_cf',
        'thetas_orientation_cr',
        'phis_start_cc', 'phis_end_cc',
        'phis_start_cf', 'x_foundation',
        'phis_start_cr', 'phis_end_cr', 'v_end_cr',
        'r_beam',
    ]
    if fix_csr:
        fixed = fixed + ['csr_curve_dofs']
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
            'order': order,
            'w1': W1,
            'w2': W2,
            'n_phi': 4,
            'n_grid_1': 1,
            'n_grid_2': 1,
            'E': 200e9,
            'nu': 0.3,
        },
        problem_options={'solver': 'umfpack'},
        csr_curve_dofs=csr_dofs,
        fixed_dof_names=fixed,
        r_beam=0.05,
    )


def _rc0_index(opt):
    """Index of ``csr_curve_dofs(0)`` (``rc_0``) in ``opt.x``."""
    for i, name in enumerate(opt.dof_names):
        if name.endswith('csr_curve_dofs(0)'):
            return i
    raise AssertionError(
        f"no csr_curve_dofs(0) in {opt.dof_names}"
    )


def test_J_matches_circle_volume():
    """Circular ring of radius R → V = w1 * w2 * 2πR."""
    cs = _make_coil_support(R=R0)
    Jvol = CSRVolume(cs)
    expected = W1 * W2 * 2.0 * math.pi * R0
    np.testing.assert_allclose(Jvol.J(), expected, rtol=1e-10)
    np.testing.assert_allclose(Jvol.length(), 2.0 * math.pi * R0, rtol=1e-10)


def test_dJ_dR_matches_analytic():
    """∂V/∂R = w1 * w2 * 2π for a circular ring."""
    cs = _make_coil_support(R=R0)
    Jvol = CSRVolume(cs)
    g = Jvol.dJ()
    i = _rc0_index(Jvol)
    expected = W1 * W2 * 2.0 * math.pi
    np.testing.assert_allclose(g[i], expected, rtol=1e-8)

    # Centered FD on the free DOF vector.
    eps = 1e-6
    x0 = Jvol.x.copy()
    x_p = x0.copy()
    x_m = x0.copy()
    x_p[i] += eps
    x_m[i] -= eps
    Jvol.x = x_p
    Jp = Jvol.J()
    Jvol.x = x_m
    Jm = Jvol.J()
    Jvol.x = x0
    fd = (Jp - Jm) / (2.0 * eps)
    np.testing.assert_allclose(g[i], fd, rtol=1e-5)


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
        CSRVolume(cs)
