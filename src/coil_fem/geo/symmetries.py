"""Pure JAX stellarator-symmetry expansion for coil geometry and currents.

Applies ``nfp``-fold rotational symmetry about the z-axis and optional
stellarator symmetry (y, z flip + current sign reversal) to arrays of
pre-evaluated curve geometry and current values.  All operations are pure
``jnp`` — fully differentiable through any upstream DOF computation.

Expansion order mirrors simsopt ``apply_symmetries_to_curves``:
the first ``n_base`` entries are always the original base coils
(``k=0``, ``flip=False``).
"""

from __future__ import annotations

import jax.numpy as jnp
import jax


# ============================================================================
# Low-level geometry transforms (pure jnp, differentiable)
# ============================================================================

def rotate_points_z(pts: jax.Array, phi: float) -> jax.Array:
    """Rotate ``(N, 3)`` point array by angle *phi* about the z-axis.

    Uses the same convention as simsopt ``RotatedCurve``:
        x_new =  x * cos(phi) - y * sin(phi)
        y_new =  x * sin(phi) + y * cos(phi)
        z_new =  z
    """
    c, s = jnp.cos(phi), jnp.sin(phi)
    # Build rotation as einsum to stay differentiable w.r.t. pts
    x = pts[..., 0] * c - pts[..., 1] * s
    y = pts[..., 0] * s + pts[..., 1] * c
    z = pts[..., 2]
    return jnp.stack([x, y, z], axis=-1)


# Keep private aliases for backward compatibility with any internal callers.
_rotate_points_z = rotate_points_z


def flip_points(pts: jax.Array) -> jax.Array:
    """Apply stellarator reflection: negate y and z components.

    Matches simsopt's flip matrix ``diag(1, -1, -1)`` applied after rotation.
    """
    signs = jnp.array([1.0, -1.0, -1.0])
    return pts * signs


_flip_points = flip_points


def rodrigues(axis: jax.Array, angle: jax.Array) -> jax.Array:
    """Rotation matrix that rotates by ``angle`` (radians) about unit ``axis``.

    Uses the Rodrigues formula::

        R = cos(θ) I + (1 − cos(θ)) (n ⊗ n) + sin(θ) [n]×

    Parameters
    ----------
    axis : jax.Array, shape (3,)
        Unit rotation axis (caller is responsible for normalising).
    angle : jax.Array, scalar
        Rotation angle in radians.

    Returns
    -------
    jax.Array, shape (3, 3)
    """
    c, s = jnp.cos(angle), jnp.sin(angle)
    x, y, z = axis[0], axis[1], axis[2]
    outer = jnp.outer(axis, axis)
    skew = jnp.array([[ 0., -z,  y],
                      [ z,  0., -x],
                      [-y,  x,  0.]])
    return c * jnp.eye(3) + (1.0 - c) * outer + s * skew


# ============================================================================
# Symmetry expansion
# ============================================================================

def apply_symmetries_to_gammas(
    base_gammas: jax.Array,
    nfp: int,
    stellsym: bool,
) -> jax.Array:
    """Expand ``(n_base, n_quad, 3)`` positions to all symmetry images.

    Parameters
    ----------
    base_gammas : jax.Array, shape ``(n_base, n_quad, 3)``
        Positions of the base coils at their quadrature points.
    nfp : int
        Number of field periods (rotational symmetry order).
    stellsym : bool
        Whether to include stellarator symmetry (flip through the phi=0
        half-plane, with simultaneous current sign reversal).

    Returns
    -------
    jax.Array, shape ``(n_total, n_quad, 3)``
        Expanded positions in the same expansion order as simsopt.
        ``n_total = n_base * nfp * (1 + int(stellsym))``.
    """
    n_base = base_gammas.shape[0]
    flip_list = [False, True] if stellsym else [False]
    images = []
    for k in range(nfp):
        phi = 2.0 * jnp.pi * k / nfp
        for flip in flip_list:
            for i in range(n_base):
                g = base_gammas[i]
                if k == 0 and not flip:
                    # Identity: keep original (avoids unnecessary computation)
                    images.append(g)
                else:
                    g = _rotate_points_z(g, phi)
                    if flip:
                        g = _flip_points(g)
                    images.append(g)
    return jnp.stack(images, axis=0)  # (n_total, n_quad, 3)


def apply_symmetries_to_gammadashs(
    base_gammadashs: jax.Array,
    nfp: int,
    stellsym: bool,
) -> jax.Array:
    """Expand ``(n_base, n_quad, 3)`` tangent vectors to all symmetry images.

    Parameters
    ----------
    base_gammadashs : jax.Array, shape ``(n_base, n_quad, 3)``
        First derivatives (tangent vectors) of the base coils.
    nfp : int
    stellsym : bool

    Returns
    -------
    jax.Array, shape ``(n_total, n_quad, 3)``

    Notes
    -----
    Tangent vectors transform identically to position vectors under rotation
    and the stellarator flip (both are linear maps).
    """
    # Tangent vectors obey the same linear transformation as positions
    return apply_symmetries_to_gammas(base_gammadashs, nfp, stellsym)


def apply_symmetries_to_currents(
    base_currents_jax: jax.Array,
    nfp: int,
    stellsym: bool,
) -> jax.Array:
    """Expand ``(n_base,)`` current array to all symmetry images.

    Parameters
    ----------
    base_currents_jax : jax.Array, shape ``(n_base,)``
        Currents of the base coils.
    nfp : int
    stellsym : bool
        Stellarator symmetry reverses current sign on all flipped images,
        matching simsopt ``ScaledCurrent(..., -1.0)``.

    Returns
    -------
    jax.Array, shape ``(n_total,)``
    """
    n_base = base_currents_jax.shape[0]
    flip_list = [False, True] if stellsym else [False]
    images = []
    for k in range(nfp):
        for flip in flip_list:
            for i in range(n_base):
                I = base_currents_jax[i]
                images.append(-I if flip else I)
    return jnp.stack(images, axis=0)  # (n_total,)


def n_coils_total(n_base: int, nfp: int, stellsym: bool) -> int:
    """Return the total number of coils after symmetry expansion."""
    return n_base * nfp * (1 + int(stellsym))
