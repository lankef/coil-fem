"""Tests for src/coil_fem/utils.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from coil_fem.utils import cubic_hermite_interp


def test_cubic_hermite_interp_matches_endpoint_values():
    """y(0) == y0 and y(1) == y1 regardless of slopes or span."""
    L, y0, dy0, y1, dy1 = 2.5, 3.0, -1.5, -4.0, 0.7
    assert jnp.allclose(cubic_hermite_interp(0.0, L, y0, dy0, y1, dy1), y0)
    assert jnp.allclose(cubic_hermite_interp(1.0, L, y0, dy0, y1, dy1), y1)


def test_cubic_hermite_interp_matches_endpoint_slopes():
    """dy/dxi at xi=0,1 recovers L*dy0 and L*dy1 (chain rule dy/dx = (dy/dxi)/L)."""
    L, y0, dy0, y1, dy1 = 2.5, 3.0, -1.5, -4.0, 0.7
    dydxi = jax.grad(cubic_hermite_interp, argnums=0)
    assert jnp.allclose(dydxi(0.0, L, y0, dy0, y1, dy1), L * dy0)
    assert jnp.allclose(dydxi(1.0, L, y0, dy0, y1, dy1), L * dy1)


def test_cubic_hermite_interp_reduces_to_linear_for_consistent_slopes():
    """A linear function is reproduced exactly when slopes match the secant."""
    L, y0, y1 = 4.0, 1.0, 9.0
    secant = (y1 - y0) / L
    xi = jnp.linspace(0.0, 1.0, 11)
    y = cubic_hermite_interp(xi, L, y0, secant, y1, secant)
    expected = y0 + xi * (y1 - y0)
    assert jnp.allclose(y, expected)


if __name__ == "__main__":
    test_cubic_hermite_interp_matches_endpoint_values()
    test_cubic_hermite_interp_matches_endpoint_slopes()
    test_cubic_hermite_interp_reduces_to_linear_for_consistent_slopes()
    print("OK")
