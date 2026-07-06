"""
Thermal eigenstrain hooks for linear elasticity (staged loading).

Small-strain additive decomposition ``ε = ε_mech + ε_th`` with
``ε_th = α ΔT I`` (isotropic) shifts the stress without changing the FE tangent
for linear Hooke:

    σ = λ tr(ε - ε_th) I + 2μ (ε - ε_th).

A full JAX-FEM ``Problem`` subclass can override ``get_tensor_map`` to subtract
``ε_th`` from ``ε`` before applying Hooke's law; this module keeps the formulas
explicit for reuse.
"""

from __future__ import annotations

import jax.numpy as jnp


def itc_strain(itc: jnp.ndarray) -> jnp.ndarray:
    """Isotropic thermal eigenstrain from a single integral thermal contraction.

    ``itc`` (integral thermal contraction) is the (positive) dimensionless linear
    thermal contraction ``ΔL/L`` accumulated on cooldown to the service
    temperature.  The eigenstrain is ``ε_th = −itc · I`` (negative = contraction),
    Parametrised directly by the measured integral contraction (no constant-CTE
    assumption) — equivalent to ``α ΔT = −itc`` without requiring a constant
    coefficient of thermal expansion.

    Parameters
    ----------
    itc : float
        Integral thermal contraction ``ΔL/L`` (e.g. ``0.0029`` for 0.29 %).

    Returns
    -------
    jnp.ndarray, shape (3, 3)
    """
    s = jnp.asarray(itc, dtype=float).reshape(())
    return -s * jnp.eye(3, dtype=jnp.float64)


def cauchy_stress_with_thermal_strain(
    u_grad: jnp.ndarray,
    lam: float,
    mu: float,
    epsilon_th: jnp.ndarray,
) -> jnp.ndarray:
    """Hooke law using mechanical strain ``ε - ε_th``."""
    eps = 0.5 * (u_grad + jnp.swapaxes(u_grad, -1, -2))
    eps_m = eps - epsilon_th
    tr = jnp.trace(eps_m, axis1=-2, axis2=-1)
    eye = jnp.eye(3, dtype=u_grad.dtype)
    return lam * tr[..., None, None] * eye + 2.0 * mu * eps_m
