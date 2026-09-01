"""Checks for :meth:`CoilSupportBeamsCSRSorted.from_clamps`."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("simsopt")

from simsopt.field import Coil, Current
from simsopt.geo import create_equally_spaced_curves

from coil_fem.geo import CurveRZFourierJAX, CurveXYZFourierJAX
from coil_fem.simsopt import (
    CoilSupportBeams,
    CoilSupportBeamsCSRSorted,
    CoilSupportBeamsSorted,
)

N_BASE = 3
NFP = 2
CLAMP_PHI = 0.5
DISTANCE = 0.2


def _base_coils():
    curves = create_equally_spaced_curves(
        N_BASE, NFP, stellsym=True, R0=1.0, R1=0.5, order=2, numquadpoints=32,
    )
    return [Coil(c, Current(1e5)) for c in curves]


def _csr_options():
    return {
        'order': 1, 'w1': 0.08, 'w2': 0.08, 'n_phi': 4,
        'n_grid_1': 1, 'n_grid_2': 1, 'E': 200e9, 'nu': 0.3,
    }


def _source(cls=CoilSupportBeams, n_clamp=1, enabled=True, base_coils=None):
    base_coils = _base_coils() if base_coils is None else base_coils
    phis = np.full((N_BASE, n_clamp), CLAMP_PHI)
    angles = (
        {'dphis': np.diff(phis, axis=-1, prepend=0.0)}
        if cls is CoilSupportBeamsSorted else {'phis': phis}
    )
    return cls(
        base_coils=base_coils,
        nfp=NFP,
        stellsym=True,
        beam_options={
            'n_beam_cc': 2,
            'n_beam_cf': 1,
            'E': 200e9,
            'nu': 0.3,
            'k_attachment': 1e10,
            'cross_section_type': 'solid_circle',
            'attachment_type': 'direct',
        },
        fixed_clamp_options={
            'enabled': enabled, 'k_clamp': 1e12,
            'r_clamp': 0.1, 'n_clamp': n_clamp,
        },
        r_beam=0.05,
        **angles,
    )


def _convert(source):
    return CoilSupportBeamsCSRSorted.from_clamps(
        source,
        coil_csr_distance=DISTANCE,
        csr_options=_csr_options(),
        problem_options={'solver': 'umfpack'},
        r_beam=0.05,
    )


def _circular_diff(a, b):
    """Signed difference of two angle fractions, reduced to ``[-0.5, 0.5)``."""
    return np.mod(np.asarray(a) - np.asarray(b) + 0.5, 1.0) - 0.5


def _move_beams_off_defaults(source):
    """Shift the CC/CF DOFs so a missing copy is distinguishable from a default."""
    x = np.asarray(source.local_full_x, dtype=float)
    for i, name in enumerate(source.local_full_dof_names):
        key = name.split(':')[-1].split('(')[0]
        if key.endswith(('_cc', '_cf')) and 'phis' in key:
            x[i] += 0.01
        elif key == 'x_foundation':
            x[i] += 0.1
    source.local_full_x = x
    return source


def test_from_clamps_cr_beams_span_clamp_to_ring():
    """CR beams start at the clamps and end ``coil_csr_distance`` outward."""
    base_coils = _base_coils()
    source = _source(base_coils=base_coils)
    out = _convert(source)

    sd = out.support_dofs
    np.testing.assert_allclose(
        _circular_diff(sd['phis_start_cr'], np.full((N_BASE, 1), CLAMP_PHI)),
        0.0, atol=1e-6,
    )

    curves = [CurveXYZFourierJAX.from_simsopt(c.curve) for c in base_coils]
    x_clamp = np.stack([np.asarray(c.gamma_eval(CLAMP_PHI)) for c in curves])
    centers = np.stack([np.asarray(c.curve_center()) for c in curves])

    ring = CurveRZFourierJAX(
        np.zeros(1), sd['csr_curve_dofs'], _csr_options()['order'], NFP, True,
    )
    x_ring = np.asarray(ring.gamma_eval(np.asarray(sd['phis_end_cr'])[:, 0]))

    # Ring end sits at coil_csr_distance from the clamp, directly away from
    # the coil centre.  Tolerance covers the phi-grid resolution of the search.
    step = np.linalg.norm(x_ring, axis=1).max() * 2.0 * np.pi / 4096
    np.testing.assert_allclose(
        np.linalg.norm(x_ring - x_clamp, axis=1), DISTANCE, atol=10 * step,
    )
    outward = (x_clamp - centers) / np.linalg.norm(
        x_clamp - centers, axis=1, keepdims=True,
    )
    moved = (x_ring - x_clamp) / np.linalg.norm(
        x_ring - x_clamp, axis=1, keepdims=True,
    )
    np.testing.assert_allclose(np.sum(outward * moved, axis=1), 1.0, atol=1e-3)
    assert np.all(
        np.linalg.norm(x_ring - centers, axis=1)
        > np.linalg.norm(x_clamp - centers, axis=1)
    )


@pytest.mark.parametrize("cls", [CoilSupportBeams, CoilSupportBeamsSorted])
def test_from_clamps_copies_cc_and_cf_beams(cls):
    """CC/CF counts and attachment DOFs survive the conversion."""
    source = _move_beams_off_defaults(_source(cls=cls))
    src = source.support_dofs
    out = _convert(source)
    sd = out.support_dofs

    assert out.support.n_beam_cc == source.support.n_beam_cc
    assert out.support.n_beam_cf == source.support.n_beam_cf
    assert out.support.n_beam_cr == 1

    for key in ('phis_start_cc', 'phis_end_cc', 'phis_start_cf'):
        assert len(sd[key]) == len(src[key])
        for got, want in zip(sd[key], src[key]):
            np.testing.assert_allclose(
                _circular_diff(got, want), 0.0, atol=1e-6, err_msg=key,
            )
    for got, want in zip(sd['x_foundation'], src['x_foundation']):
        np.testing.assert_allclose(got, want)


def test_from_clamps_rejects_bad_sources():
    """Wrong type, no clamps, or n_clamp != 1 all raise."""
    with pytest.raises(TypeError):
        _convert(object())
    with pytest.raises(ValueError, match="no fixed-sphere clamps"):
        _convert(_source(enabled=False))
    with pytest.raises(ValueError, match="n_clamp == 1"):
        _convert(_source(n_clamp=2))
