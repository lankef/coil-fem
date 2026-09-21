"""Empty-array reshape guards for GSON round-trips of support DOF seeds."""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

simsopt = pytest.importorskip("simsopt")

from simsopt import load, save                                              # noqa: E402
from simsopt.field import Coil, Current                                     # noqa: E402
from simsopt.geo import create_equally_spaced_curves                        # noqa: E402

from coil_fem.simsopt.coil_support_beams import _check_ragged_shape         # noqa: E402
from coil_fem.simsopt.coil_support_beams_csr import (                       # noqa: E402
    CoilSupportBeamsCSR,
    _check_rect_shape,
)


def test_check_ragged_shape_reshapes_empty_trailing():
    """GSON drops trailing axes of empty arrays: (0, 3) → (0,)."""
    got = _check_ragged_shape(
        [np.array([]), np.array([])],
        counts=(0, 0),
        name='x_foundation',
        trailing=(3,),
    )
    assert len(got) == 2
    for arr in got:
        assert arr.shape == (0, 3)


def test_check_ragged_shape_still_rejects_bad_nonzero():
    with pytest.raises(ValueError, match='x_foundation\\[0\\]'):
        _check_ragged_shape(
            [np.zeros(3)],
            counts=(1,),
            name='x_foundation',
            trailing=(3,),
        )


def test_check_rect_shape_reshapes_empty_cr():
    """GSON drops zero-size axes: (n, 0) → (0,)."""
    got = _check_rect_shape(np.array([]), (3, 0), 'phis_start_cr')
    assert got.shape == (3, 0)


def test_csr_gson_roundtrip_zero_cf():
    """CoilSupportBeamsCSR with n_beam_cf=0 survives simsopt save/load."""
    curves = create_equally_spaced_curves(
        2, 2, stellsym=False, R0=1.2, R1=0.4, order=1, numquadpoints=16,
    )
    base_coils = [Coil(c, Current(1e5)) for c in curves]
    # Explicit empty foundations match what GSON reloads (shape (0,) data).
    x_foundation = [np.array([]), np.array([])]
    cs = CoilSupportBeamsCSR(
        base_coils=base_coils,
        nfp=2,
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
            'order': 1,
            'w1': 0.08,
            'w2': 0.06,
            'n_phi': 4,
            'n_grid_1': 1,
            'n_grid_2': 1,
            'E': 200e9,
            'nu': 0.3,
        },
        problem_options={'solver': 'umfpack'},
        x_foundation=x_foundation,
        fixed_dof_names=[
            'thetas_orientation_cc',
            'thetas_orientation_cf',
            'thetas_orientation_cr',
            'phis_start_cc', 'phis_end_cc',
            'phis_start_cf', 'x_foundation',
            'phis_start_cr', 'phis_end_cr',
            'csr_curve_dofs',
            'r_beam',
        ],
        r_beam=0.05,
    )
    for xf in cs.support_dofs['x_foundation']:
        assert tuple(xf.shape) == (0, 3)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'cs.json'
        save([cs], str(path))
        cs2 = load(str(path))[0]
    for xf in cs2.support_dofs['x_foundation']:
        assert tuple(xf.shape) == (0, 3)
