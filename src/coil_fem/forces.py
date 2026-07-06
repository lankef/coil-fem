"""
Lorentz body force from per-quadrature current density and magnetic field.

The exact law is f = J x B.  For a coil with uniform current I distributed
uniformly over cross-section area A with tangent t_hat, J = (I / A) t_hat —
but the caller is responsible for building J, so this module is agnostic to
the current model.  This keeps the door open for non-uniform current density
(cable models, skin effect, etc.) without changes here.

For magnetic field computation see :mod:`coil_fem.magnetic`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def lorentz_body_force(J: jax.Array, B: jax.Array) -> jax.Array:
    """Lorentz body force density **f = J × B**.

    Parameters
    ----------
    J : jax.Array, shape ``(..., 3)``
        Current density [A/m²].
    B : jax.Array, shape ``(..., 3)``
        Magnetic field [T].

    Returns
    -------
    jax.Array, shape ``(..., 3)`` [N/m³]
    """
    return jnp.cross(J, B)
