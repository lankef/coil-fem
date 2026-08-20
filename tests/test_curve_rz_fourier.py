"""Tests for CurveRZFourierJAX vs simsopt CurveRZFourier and framed-curve smoke."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from simsopt.geo import CurveRZFourier

from coil_fem.geo import (
    CurveXYZFourierJAX,
    CurveRZFourierJAX,
    FramedCurveCentroidJAX,
    make_framed_curve,
)


def _make_simsopt_rz(order, nfp, stellsym, nquad=64, seed=0):
    """Build a random simsopt CurveRZFourier with non-trivial DOFs."""
    rng = np.random.default_rng(seed)
    if isinstance(nquad, int):
        # Full [0, 1) grid so nfp>1 curves are closed in Cartesian space for RMF.
        quadpoints = np.linspace(0.0, 1.0, nquad, endpoint=False)
    else:
        quadpoints = np.asarray(nquad)
    curve = CurveRZFourier(quadpoints, order, nfp, stellsym)
    dofs = curve.get_dofs()
    dofs = rng.normal(size=dofs.shape)
    # Ensure a positive major radius (rc0 > 0) so the curve is well-formed.
    dofs[0] = abs(dofs[0]) + 1.0
    curve.set_dofs(dofs)
    return curve


@pytest.mark.parametrize("stellsym", [True, False])
@pytest.mark.parametrize("nfp", [1, 2])
def test_dofs_round_trip(stellsym, nfp):
    c_sim = _make_simsopt_rz(order=3, nfp=nfp, stellsym=stellsym)
    c_jax = CurveRZFourierJAX.from_simsopt(c_sim)
    np.testing.assert_array_equal(np.asarray(c_jax.get_dofs()), c_sim.get_dofs())

    c_back = c_jax.to_simsopt()
    np.testing.assert_allclose(c_back.get_dofs(), c_sim.get_dofs(), atol=0, rtol=0)
    np.testing.assert_allclose(c_back.gamma(), c_sim.gamma(), atol=1e-14, rtol=0)


@pytest.mark.parametrize("stellsym", [True, False])
@pytest.mark.parametrize("nfp", [1, 2])
def test_gamma_derivatives_match_simsopt(stellsym, nfp):
    c_sim = _make_simsopt_rz(order=4, nfp=nfp, stellsym=stellsym, seed=1)
    c_jax = CurveRZFourierJAX.from_simsopt(c_sim)

    np.testing.assert_allclose(
        np.asarray(c_jax.gamma()), c_sim.gamma(), atol=1e-12, rtol=0,
    )
    np.testing.assert_allclose(
        np.asarray(c_jax.gammadash()), c_sim.gammadash(), atol=1e-10, rtol=0,
    )
    np.testing.assert_allclose(
        np.asarray(c_jax.gammadashdash()), c_sim.gammadashdash(), atol=1e-9, rtol=0,
    )
    np.testing.assert_allclose(
        np.asarray(c_jax.gammadashdashdash()),
        c_sim.gammadashdashdash(),
        atol=1e-8,
        rtol=0,
    )


@pytest.mark.parametrize("stellsym", [True, False])
def test_kappa_torsion_match_simsopt(stellsym):
    c_sim = _make_simsopt_rz(order=3, nfp=1, stellsym=stellsym, seed=2)
    c_jax = CurveRZFourierJAX.from_simsopt(c_sim)

    np.testing.assert_allclose(
        np.asarray(c_jax.kappa()), c_sim.kappa(), atol=1e-10, rtol=0,
    )
    np.testing.assert_allclose(
        np.asarray(c_jax.torsion()), c_sim.torsion(), atol=1e-9, rtol=0,
    )


def test_pytree_and_with_dofs():
    c_sim = _make_simsopt_rz(order=2, nfp=1, stellsym=True, seed=3)
    c_jax = CurveRZFourierJAX.from_simsopt(c_sim)

    leaves, treedef = jax.tree_util.tree_flatten(c_jax)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_allclose(
        np.asarray(rebuilt.gamma()), np.asarray(c_jax.gamma()), atol=0, rtol=0,
    )
    assert rebuilt.order == c_jax.order
    assert rebuilt.nfp == c_jax.nfp
    assert rebuilt.stellsym == c_jax.stellsym

    new_dofs = c_jax.dofs * 1.1
    c2 = c_jax.with_dofs(new_dofs)
    np.testing.assert_allclose(np.asarray(c2.dofs), np.asarray(new_dofs))
    assert c2.order == c_jax.order
    assert c2 is not c_jax


def test_curve_center_rz_is_origin():
    c_sim = _make_simsopt_rz(order=2, nfp=2, stellsym=True)
    c_jax = CurveRZFourierJAX.from_simsopt(c_sim)
    np.testing.assert_array_equal(np.asarray(c_jax.curve_center()), np.zeros(3))


def test_curve_center_xyz_matches_packed_dofs():
    order = 2
    quadpoints = jnp.linspace(0.0, 1.0, 16, endpoint=False)
    # xc0=1, yc0=2, zc0=3; remaining coeffs zero
    k = 2 * order + 1
    dofs = jnp.zeros(3 * k)
    dofs = dofs.at[0].set(1.0)
    dofs = dofs.at[k].set(2.0)
    dofs = dofs.at[2 * k].set(3.0)
    c = CurveXYZFourierJAX(quadpoints, dofs, order)
    np.testing.assert_allclose(
        np.asarray(c.curve_center()), np.array([1.0, 2.0, 3.0]), atol=0, rtol=0,
    )


def test_framed_centroid_and_rmf_on_circular_rz():
    """Circular RZ axis (nfp=1, order=0) works with both framed-curve types."""
    R = 1.5
    nquad = 64
    quadpoints = np.linspace(0.0, 1.0, nquad, endpoint=False)
    c_sim = CurveRZFourier(quadpoints, 0, 1, True)
    c_sim.set_dofs([R])
    c_jax = CurveRZFourierJAX.from_simsopt(c_sim)

    for frame_type in ("centroid", "rmf"):
        fc = make_framed_curve(c_jax, frame_type)
        t, p, q = fc.rotated_frame()
        assert jnp.all(jnp.isfinite(t))
        assert jnp.all(jnp.isfinite(p))
        assert jnp.all(jnp.isfinite(q))
        # Orthonormality
        np.testing.assert_allclose(
            np.asarray(jnp.sum(t * t, axis=1)), 1.0, atol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(jnp.sum(p * p, axis=1)), 1.0, atol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(jnp.sum(q * q, axis=1)), 1.0, atol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(jnp.sum(t * p, axis=1)), 0.0, atol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(jnp.sum(t * q, axis=1)), 0.0, atol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(jnp.sum(p * q, axis=1)), 0.0, atol=1e-10,
        )


def test_framed_with_dofs_preserves_rz_type():
    R = 1.0
    nquad = 32
    quadpoints = np.linspace(0.0, 1.0, nquad, endpoint=False)
    c = CurveRZFourierJAX(quadpoints, jnp.array([R]), 0, 1, True)
    fc = FramedCurveCentroidJAX(c)
    fc2 = fc.with_dofs(jnp.array([1.2]))
    assert isinstance(fc2.curve, CurveRZFourierJAX)
    assert fc2.curve.nfp == 1
    assert fc2.curve.stellsym is True
    np.testing.assert_allclose(np.asarray(fc2.curve.dofs), [1.2])
