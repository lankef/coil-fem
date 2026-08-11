"""Tests for src/coil_fem/presets/cross_section_fns.py."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from coil_fem.presets import cross_section_fns as cs


def test_hollow_rectangle_A_I_match_grid_quadrature():
    """A, Iy, Iz of hollow_rectangle match a fine 2-D grid over the wall."""
    w1, w2, t = 0.04, 0.03, 0.004
    A, Iy, Iz, _ = cs.hollow_rectangle({
        'w1_beam': [jnp.array([w1])],
        'w2_beam': [jnp.array([w2])],
        't_beam': [jnp.array([t])],
    })

    # Cell-centred Riemann sum over the outer rectangle, masking the void.
    n = 800
    dz, dy = w1 / n, w2 / n
    z = -w1 / 2 + dz / 2 + np.arange(n) * dz
    y = -w2 / 2 + dy / 2 + np.arange(n) * dy
    Z, Y = np.meshgrid(z, y, indexing='ij')
    wall = (np.abs(Z) >= w1 / 2 - t) | (np.abs(Y) >= w2 / 2 - t)
    A_q = np.sum(wall) * dz * dy
    Iy_q = np.sum(wall * Z**2) * dz * dy   # ∫ z² dA about y
    Iz_q = np.sum(wall * Y**2) * dz * dy   # ∫ y² dA about z

    assert np.isclose(float(A[0][0]), A_q, rtol=2e-3)
    assert np.isclose(float(Iy[0][0]), Iy_q, rtol=2e-3)
    assert np.isclose(float(Iz[0][0]), Iz_q, rtol=2e-3)


def test_hollow_rectangle_J_matches_bredt():
    """J equals 4 A_m² / ∮(ds/t); square tube reduces to t (a − t)³."""
    w1, w2, t = 0.05, 0.03, 0.003
    _, _, _, J = cs.hollow_rectangle({
        'w1_beam': [jnp.array([w1])],
        'w2_beam': [jnp.array([w2])],
        't_beam': [jnp.array([t])],
    })
    A_m = (w1 - t) * (w2 - t)
    oint = 2.0 * ((w1 - t) + (w2 - t)) / t
    J_bredt = 4.0 * A_m**2 / oint
    assert np.isclose(float(J[0][0]), J_bredt, rtol=1e-12)

    a, t_sq = 0.04, 0.002
    _, _, _, J_sq = cs.hollow_rectangle({
        'w1_beam': [jnp.array([a])],
        'w2_beam': [jnp.array([a])],
        't_beam': [jnp.array([t_sq])],
    })
    assert np.isclose(float(J_sq[0][0]), t_sq * (a - t_sq) ** 3, rtol=1e-12)


def test_hollow_rectangle_ragged_structure():
    """Outputs are per-group lists with shapes matching the DOF inputs."""
    dofs = {
        'w1_beam': [jnp.array([0.04, 0.05]), jnp.array([0.03])],
        'w2_beam': [jnp.array([0.03, 0.04]), jnp.array([0.02])],
        't_beam': [jnp.array([0.004, 0.005]), jnp.array([0.003])],
    }
    A, Iy, Iz, J = cs.hollow_rectangle(dofs)
    for out in (A, Iy, Iz, J):
        assert len(out) == 2
        assert out[0].shape == (2,)
        assert out[1].shape == (1,)
        assert jnp.all(out[0] > 0) and jnp.all(out[1] > 0)


def test_solid_rectangle_attachment_weights():
    """Weight ~1 at centre on the correct side; ~0 outside / wrong side."""
    w1, w2 = 0.04, 0.03
    dofs = {'w1_beam': w1, 'w2_beam': w2}
    opts = {'eps_sigmoid': 0.05}
    # (x, y, z): centre on +x, outside in z, outside in y, centre on -x
    pts = jnp.array([
        [0.01, 0.0, 0.0],
        [0.01, 0.0, w1 / 2 + 0.01],
        [0.01, w2 / 2 + 0.01, 0.0],
        [-0.01, 0.0, 0.0],
    ])
    w = cs.solid_rectangle_attachment(pts, dofs, True, opts)
    assert float(w[0]) > 0.95
    assert float(w[1]) < 0.05
    assert float(w[2]) < 0.05
    assert float(w[3]) == 0.0

    # hollow_rectangle_attachment is the same function
    assert cs.hollow_rectangle_attachment is cs.solid_rectangle_attachment


if __name__ == "__main__":
    test_hollow_rectangle_A_I_match_grid_quadrature()
    test_hollow_rectangle_J_matches_bredt()
    test_hollow_rectangle_ragged_structure()
    test_solid_rectangle_attachment_weights()
    print("All cross_section_fns checks passed.")
