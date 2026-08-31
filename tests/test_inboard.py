"""Checks for ClampInboard and CRBeamInboard radial-outboard hinges."""

from __future__ import annotations

import numpy as np
import pytest

simsopt = pytest.importorskip("simsopt")

from simsopt.field import Coil, Current                       # noqa: E402
from simsopt.geo import create_equally_spaced_curves          # noqa: E402

from coil_fem.simsopt import (                                # noqa: E402
    ClampInboard,
    CoilSupportBeams,
    CoilSupportBeamsCSR,
    CoilSupportFixed,
    CoilSupportTopBottom,
    CRBeamInboard,
)

NFP = 2
R0 = 1.2
R1 = 0.4


def _base_coils():
    curves = create_equally_spaced_curves(
        1, NFP, stellsym=False, R0=R0, R1=R1, order=1, numquadpoints=16,
    )
    return [Coil(c, Current(1e5)) for c in curves]


def _make_fixed(*, phis):
    return CoilSupportFixed(
        base_coils=_base_coils(),
        nfp=NFP,
        stellsym=False,
        fixed_clamp_options={
            'k_clamp': 1e9,
            'r_clamp': 0.05,
            'n_clamp': 2,
        },
        phis=phis,
    )


def _make_csr(*, phis_start_cr):
    order = 1
    csr_dofs = np.zeros(4 * order + 2)
    csr_dofs[0] = 1.0
    return CoilSupportBeamsCSR(
        base_coils=_base_coils(),
        nfp=NFP,
        stellsym=False,
        beam_options={
            'n_beam_cc': 0,
            'n_beam_cf': 0,
            'n_beam_cr': 2,
            'E': 200e9,
            'nu': 0.3,
            'cross_section_type': 'solid_circle',
            'attachment_type': 'direct',
        },
        csr_options={
            'order': order,
            'w1': 0.08,
            'w2': 0.06,
            'n_phi': 4,
            'n_grid_1': 1,
            'n_grid_2': 1,
            'E': 200e9,
            'nu': 0.3,
        },
        problem_options={'solver': 'umfpack'},
        phis_start_cr=phis_start_cr,
        csr_curve_dofs=csr_dofs,
        fixed_dof_names=[
            'thetas_orientation_cc',
            'thetas_orientation_cf',
            'thetas_orientation_cr',
            'phis_start_cc', 'phis_end_cc',
            'phis_start_cf', 'x_foundation',
            'phis_end_cr',
            'csr_curve_dofs',
            'r_beam',
        ],
        r_beam=0.05,
    )


def test_clamp_inboard_J_matches_analytic():
    """phi={0, 0.5} on a circular coil → J = R1^2 (only outboard contributes)."""
    cs = _make_fixed(phis=np.array([[0.0, 0.5]]))
    J = ClampInboard(cs)
    np.testing.assert_allclose(J.J(), R1 ** 2, rtol=1e-10)
    np.testing.assert_allclose(J.max_overhang(), R1, rtol=1e-10)


def test_cr_beam_inboard_J_matches_analytic():
    """Same hinge for CR starts at phi={0, 0.5}."""
    cs = _make_csr(phis_start_cr=np.array([[0.0, 0.5]]))
    J = CRBeamInboard(cs)
    np.testing.assert_allclose(J.J(), R1 ** 2, rtol=1e-10)
    np.testing.assert_allclose(J.max_overhang(), R1, rtol=1e-10)


def test_clamp_inboard_dJ_matches_fd():
    """Centred FD on one free clamp angle versus dJ."""
    # Start slightly off the extrema so the hinge is active and smooth.
    cs = _make_fixed(phis=np.array([[0.1, 0.5]]))
    J = ClampInboard(cs)
    g = J.dJ()
    names = J.dof_names
    i = next(k for k, n in enumerate(names) if 'phis(0,0)' in n)

    eps = 1e-6
    x0 = J.x.copy()
    x_p = x0.copy()
    x_m = x0.copy()
    x_p[i] += eps
    x_m[i] -= eps
    J.x = x_p
    Jp = J.J()
    J.x = x_m
    Jm = J.J()
    J.x = x0
    fd = (Jp - Jm) / (2.0 * eps)
    np.testing.assert_allclose(g[i], fd, rtol=1e-5)


def test_clamp_inboard_rejects_top_bottom():
    cs = CoilSupportTopBottom(
        base_coils=_base_coils(),
        nfp=NFP,
        stellsym=False,
        fixed_clamp_options={'k_clamp': 1e9, 'r_clamp': 0.05},
    )
    with pytest.raises(TypeError, match='phis'):
        ClampInboard(cs)


def test_cr_beam_inboard_rejects_non_csr():
    cs = CoilSupportBeams(
        base_coils=_base_coils(),
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
    with pytest.raises(TypeError, match='phis_start_cr'):
        CRBeamInboard(cs)
