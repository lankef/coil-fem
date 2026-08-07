"""Checks for Biot–Savart scan rematerialization knobs."""

from __future__ import annotations

import jax.numpy as jnp

from coil_fem.coil_fem import _broadcast_problem_options
from coil_fem.magnetic import biot_savart


def test_remat_bs_default_true():
    assert _broadcast_problem_options(None)['remat_bs'] is True
    assert _broadcast_problem_options({'solver': 'umfpack'})['remat_bs'] is True
    assert _broadcast_problem_options({'remat_bs': False})['remat_bs'] is False


def test_biot_savart_remat_matches_noremat():
    # Tiny synthetic filaments: 2 sources × 4 quads, 3 targets.
    phi = jnp.linspace(0.0, 2.0 * jnp.pi, 4, endpoint=False)
    circ = jnp.stack([jnp.cos(phi), jnp.sin(phi), jnp.zeros_like(phi)], axis=-1)
    tang = jnp.stack([-jnp.sin(phi), jnp.cos(phi), jnp.zeros_like(phi)], axis=-1)
    source_gammas = jnp.stack([circ, circ + jnp.array([2.0, 0.0, 0.0])])
    source_gammadashs = jnp.stack([tang, tang])
    source_currents = jnp.array([1e6, 1e6])
    targets = jnp.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.5],
        [2.0, 0.0, 1.0],
    ])

    B_remat = biot_savart(
        targets, source_gammas, source_gammadashs, source_currents, remat=True,
    )
    B_store = biot_savart(
        targets, source_gammas, source_gammadashs, source_currents, remat=False,
    )
    assert jnp.allclose(B_remat, B_store, rtol=0.0, atol=0.0)
