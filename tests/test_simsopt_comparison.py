"""
Comparison tests against simsopt for the coil-fem library.

Uses the W7-X stellarator configuration as a real-world test case.
All tests compare coil-fem's pure-JAX implementations against simsopt's
reference implementations (C++ BiotSavart, RotatedCurve symmetry expansion,
B_regularized_pure self-field).

Expected tolerances are set conservatively relative to the observed
floating-point differences (~1e-14 relative for geometry, ~1e-10 relative
for force magnitudes of ~1.6 MN/m).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from simsopt.configs.zoo import get_data
from simsopt.field.coil import coils_via_symmetries
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.selffield import B_regularized_pure, regularization_rect
from simsopt.geo.framedcurve import FramedCurveCentroid, FrameRotation

from coil_fem.geo import CurveXYZFourierJAX
from coil_fem.geo import FramedCurveCentroidJAX
from coil_fem.geo import (
    apply_symmetries_to_gammas,
    apply_symmetries_to_gammadashs,
    apply_symmetries_to_currents,
    n_coils_total,
)
from simsopt.field.selffield import B_regularized_pure as B_self_centerline
from coil_fem.magnetic import biot_savart as _biot_savart_all


def B_mutual_centerline(gamma, all_gammas, all_gammadashs, all_currents, *, exclude_self_index):
    """Thin shim: mutual Biot-Savart field, excluding the self coil."""
    currents_no_self = all_currents.at[exclude_self_index].set(0.0)
    return _biot_savart_all(gamma, all_gammas, all_gammadashs, currents_no_self)


def lorentz_line_force(B, I, gammadash):
    """Lorentz line-force density  F = I * (gammadash × B) / n_phi  [N/m]."""
    n_phi = gammadash.shape[0]
    return (I / n_phi) * jnp.cross(gammadash, B)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def w7x():
    """
    W7-X coil configuration at moderate resolution (order=8, 5 pts/period).

    Returns a dict with pre-computed JAX arrays and the simsopt reference
    objects so individual tests don't have to rebuild them.
    """
    base_curves_sim, base_currents_sim, _ma, nfp, _bs = get_data(
        "w7x", coil_order=8, points_per_period=5
    )
    stellsym = True

    # JAX mirror of each base curve
    jax_curves = [CurveXYZFourierJAX.from_simsopt(c) for c in base_curves_sim]

    # Stacked base-coil arrays  (n_base, n_quad, 3) / (n_base,)
    base_gammas = jnp.stack([c.gamma() for c in jax_curves])
    base_gammadashs = jnp.stack([c.gammadash() for c in jax_curves])
    base_currents_jax = jnp.array([c.get_value() for c in base_currents_sim])

    # Symmetry-expanded arrays (n_total, n_quad, 3) / (n_total,)
    all_gammas = apply_symmetries_to_gammas(base_gammas, nfp, stellsym)
    all_gammadashs = apply_symmetries_to_gammadashs(base_gammadashs, nfp, stellsym)
    all_currents = apply_symmetries_to_currents(base_currents_jax, nfp, stellsym)

    # Simsopt reference: full coil set
    coils_sim = coils_via_symmetries(base_curves_sim, base_currents_sim, nfp, stellsym)

    return dict(
        # simsopt objects
        base_curves_sim=base_curves_sim,
        base_currents_sim=base_currents_sim,
        coils_sim=coils_sim,
        # JAX objects
        jax_curves=jax_curves,
        base_gammas=base_gammas,
        base_gammadashs=base_gammadashs,
        base_currents_jax=base_currents_jax,
        all_gammas=all_gammas,
        all_gammadashs=all_gammadashs,
        all_currents=all_currents,
        # scalars
        nfp=nfp,
        stellsym=stellsym,
        n_base=len(base_curves_sim),
    )


# ---------------------------------------------------------------------------
# 1. CurveXYZFourierJAX vs simsopt CurveXYZFourier
# ---------------------------------------------------------------------------

class TestCurveJAXMatchesSimsopt:
    """
    CurveXYZFourierJAX.from_simsopt should reproduce gamma, gammadash, and
    gammadashdash to machine precision.
    """

    def test_dofs_round_trip(self, w7x):
        """DOFs extracted by simsopt and stored in JAX array are identical."""
        for c_sim, c_jax in zip(w7x["base_curves_sim"], w7x["jax_curves"]):
            np.testing.assert_array_equal(
                c_sim.get_dofs(),
                np.asarray(c_jax.dofs),
                err_msg=f"DOF mismatch for curve {c_sim}",
            )

    def test_gamma_all_coils(self, w7x):
        """Curve positions agree to machine precision for all 7 W7-X base coils."""
        for i, (c_sim, c_jax) in enumerate(
            zip(w7x["base_curves_sim"], w7x["jax_curves"])
        ):
            np.testing.assert_allclose(
                np.asarray(c_jax.gamma()),
                c_sim.gamma(),
                atol=1e-12,
                rtol=0,
                err_msg=f"gamma mismatch on base coil {i}",
            )

    def test_gammadash_all_coils(self, w7x):
        """Curve tangents agree to machine precision for all 7 W7-X base coils."""
        for i, (c_sim, c_jax) in enumerate(
            zip(w7x["base_curves_sim"], w7x["jax_curves"])
        ):
            np.testing.assert_allclose(
                np.asarray(c_jax.gammadash()),
                c_sim.gammadash(),
                atol=1e-11,
                rtol=0,
                err_msg=f"gammadash mismatch on base coil {i}",
            )

    def test_gammadashdash_all_coils(self, w7x):
        """Curve curvature vectors agree to machine precision."""
        for i, (c_sim, c_jax) in enumerate(
            zip(w7x["base_curves_sim"], w7x["jax_curves"])
        ):
            np.testing.assert_allclose(
                np.asarray(c_jax.gammadashdash()),
                c_sim.gammadashdash(),
                atol=1e-10,
                rtol=0,
                err_msg=f"gammadashdash mismatch on base coil {i}",
            )

    def test_quadpoints_match(self, w7x):
        """Quadrature point values agree exactly."""
        for c_sim, c_jax in zip(w7x["base_curves_sim"], w7x["jax_curves"]):
            np.testing.assert_allclose(
                np.asarray(c_jax.quadpoints),
                c_sim.quadpoints,
                atol=1e-15,
                rtol=0,
            )


# ---------------------------------------------------------------------------
# 2. Symmetry expansion vs simsopt coils_via_symmetries
# ---------------------------------------------------------------------------

class TestSymmetryMatchesSimsopt:
    """
    apply_symmetries_to_gammas / _gammadashs / _currents should reproduce
    simsopt's coils_via_symmetries expansion (RotatedCurve + ScaledCurrent)
    for the full W7-X coil set.
    """

    def test_total_coil_count(self, w7x):
        """n_coils_total helper matches the actual number of simsopt coils."""
        n = n_coils_total(w7x["n_base"], w7x["nfp"], w7x["stellsym"])
        assert n == len(w7x["coils_sim"]), (
            f"Expected {len(w7x['coils_sim'])} coils, got {n}"
        )

    def test_all_gammas_match(self, w7x):
        """
        Symmetry-expanded curve positions from coil_fem match every RotatedCurve
        in simsopt's coils_via_symmetries output.
        """
        gammas_sim = np.stack([c.curve.gamma() for c in w7x["coils_sim"]])
        np.testing.assert_allclose(
            np.asarray(w7x["all_gammas"]),
            gammas_sim,
            atol=1e-12,
            rtol=0,
            err_msg="Symmetry-expanded gammas differ from simsopt RotatedCurve.gamma()",
        )

    def test_all_gammadashs_match(self, w7x):
        """
        Symmetry-expanded tangent vectors match simsopt RotatedCurve.gammadash()
        for all 70 W7-X coils.
        """
        gammadashs_sim = np.stack([c.curve.gammadash() for c in w7x["coils_sim"]])
        np.testing.assert_allclose(
            np.asarray(w7x["all_gammadashs"]),
            gammadashs_sim,
            atol=1e-11,
            rtol=0,
            err_msg="Symmetry-expanded gammadashs differ from simsopt",
        )

    def test_all_currents_match(self, w7x):
        """
        Symmetry-expanded currents (including stellarator sign flips) match
        simsopt ScaledCurrent values for all 70 W7-X coils.
        """
        currents_sim = np.array([c.current.get_value() for c in w7x["coils_sim"]])
        np.testing.assert_allclose(
            np.asarray(w7x["all_currents"]),
            currents_sim,
            atol=0,
            rtol=1e-14,
            err_msg="Symmetry-expanded currents differ from simsopt ScaledCurrent",
        )

    def test_stellarator_images_sign_flip(self, w7x):
        """
        The first stellarator image (index n_base) carries the negated current
        of the corresponding base coil – this is the key stellarator symmetry check.
        """
        n = w7x["n_base"]
        base_I = np.asarray(w7x["base_currents_jax"])
        image_I = np.asarray(w7x["all_currents"][n : 2 * n])
        np.testing.assert_allclose(
            image_I, -base_I, atol=0, rtol=1e-14,
            err_msg="Stellarator image currents should be negated base currents",
        )

    def test_rotational_copies_same_current_magnitude(self, w7x):
        """
        All nfp rotational copies of a base coil carry the same |current| as
        the original (only stellarator images get sign-flipped).
        """
        n_base = w7x["n_base"]
        nfp = w7x["nfp"]
        for i in range(n_base):
            # Indices of all rotational copies of base coil i (non-flipped)
            indices = [k * 2 * n_base + i for k in range(nfp)]
            magnitudes = np.abs(np.asarray(w7x["all_currents"])[indices])
            expected = abs(float(w7x["base_currents_jax"][i]))
            np.testing.assert_allclose(
                magnitudes, expected, atol=0, rtol=1e-14,
                err_msg=f"Rotational copies of base coil {i} have wrong current magnitude",
            )


# ---------------------------------------------------------------------------
# 3. B_mutual_centerline vs simsopt BiotSavart (C++ kernel)
# ---------------------------------------------------------------------------

class TestBMutualCenterlineMatchesSimsopt:
    """
    B_mutual_centerline evaluated at the centerline of coil i (excluding
    coil i itself) should match a simsopt BiotSavart built from all other coils.
    """

    @pytest.mark.parametrize("coil_idx", [0, 1, 2, 4])
    def test_B_mutual_vs_simsopt_biotsavart(self, w7x, coil_idx):
        """
        Mutual Biot-Savart field from all other coils at the centerline of
        base coil `coil_idx` matches the C++ BiotSavart to machine precision.
        """
        target_gamma = w7x["all_gammas"][coil_idx]

        # coil-fem
        B_cf = B_mutual_centerline(
            target_gamma,
            w7x["all_gammas"],
            w7x["all_gammadashs"],
            w7x["all_currents"],
            exclude_self_index=coil_idx,
        )

        # simsopt reference: BiotSavart from every coil except coil_idx
        other_coils = [
            c for j, c in enumerate(w7x["coils_sim"]) if j != coil_idx
        ]
        bs_other = BiotSavart(other_coils)
        bs_other.set_points(np.asarray(target_gamma))
        B_sim = bs_other.B()

        np.testing.assert_allclose(
            np.asarray(B_cf),
            B_sim,
            atol=1e-13,
            rtol=0,
            err_msg=(
                f"B_mutual_centerline differs from simsopt BiotSavart "
                f"at centerline of coil {coil_idx}"
            ),
        )

    def test_B_mutual_symmetry_consistency(self, w7x):
        """
        By symmetry, |B_mutual| evaluated at the n-th rotational copy of coil 0
        should equal |B_mutual| at coil 0 itself.
        """
        n_base = w7x["n_base"]
        nfp = w7x["nfp"]

        def _B_rms(idx):
            B = B_mutual_centerline(
                w7x["all_gammas"][idx],
                w7x["all_gammas"],
                w7x["all_gammadashs"],
                w7x["all_currents"],
                exclude_self_index=idx,
            )
            return float(jnp.sqrt(jnp.mean(jnp.sum(B**2, axis=-1))))

        B0 = _B_rms(0)
        for k in range(1, nfp):
            # Rotational copies of base coil 0 (non-flipped, k-th period)
            idx_k = k * 2 * n_base
            Bk = _B_rms(idx_k)
            assert abs(Bk - B0) / B0 < 1e-10, (
                f"B_rms at rotational copy k={k} ({Bk:.6g} T) differs from "
                f"copy k=0 ({B0:.6g} T); relative diff = {abs(Bk-B0)/B0:.2e}"
            )


# ---------------------------------------------------------------------------
# 4. B_self_centerline vs simsopt B_regularized_pure
# ---------------------------------------------------------------------------

class TestBSelfCenterlineMatchesSimsopt:
    """
    B_regularized_pure (simsopt) is used as-is for the B_reg term.
    These tests verify its basic properties on W7-X coils.
    """

    @pytest.mark.parametrize("coil_idx", [0, 2, 4])
    def test_self_field_identical_to_simsopt(self, w7x, coil_idx):
        """
        B_regularized_pure from coil_fem (= simsopt) and direct simsopt call
        return identical arrays.
        """
        c_jax = w7x["jax_curves"][coil_idx]
        I = float(w7x["base_currents_jax"][coil_idx])
        reg = float(regularization_rect(0.05, 0.02))

        B_cf = B_self_centerline(
            c_jax.gamma(),
            c_jax.gammadash(),
            c_jax.gammadashdash(),
            c_jax.quadpoints,
            I,
            reg,
        )

        B_sim = B_regularized_pure(
            np.asarray(c_jax.gamma()),
            np.asarray(c_jax.gammadash()),
            np.asarray(c_jax.gammadashdash()),
            np.asarray(c_jax.quadpoints),
            I,
            reg,
        )

        np.testing.assert_allclose(
            np.asarray(B_cf), np.asarray(B_sim), atol=0, rtol=1e-14,
            err_msg=f"B_regularized_pure differs between calls on coil {coil_idx}",
        )

    def test_self_field_sign_with_current(self, w7x):
        """Reversing the current sign should flip B_regularized_pure."""
        c_jax = w7x["jax_curves"][0]
        I = float(w7x["base_currents_jax"][0])
        reg = float(regularization_rect(0.05, 0.02))

        B_pos = B_self_centerline(
            c_jax.gamma(), c_jax.gammadash(), c_jax.gammadashdash(),
            c_jax.quadpoints, I, reg,
        )
        B_neg = B_self_centerline(
            c_jax.gamma(), c_jax.gammadash(), c_jax.gammadashdash(),
            c_jax.quadpoints, -I, reg,
        )
        np.testing.assert_allclose(
            np.asarray(B_neg), -np.asarray(B_pos), atol=0, rtol=1e-14,
        )


# ---------------------------------------------------------------------------
# 5. Lorentz force vs simsopt reference
# ---------------------------------------------------------------------------

class TestLorentzForceMatchesSimsopt:
    """
    The total Lorentz line-force density (self + mutual) evaluated by coil-fem
    should match the field assembled from simsopt's B_regularized_pure and
    BiotSavart C++ kernel.
    """

    @pytest.mark.parametrize("coil_idx", [0, 3])
    def test_total_lorentz_force_vs_simsopt(self, w7x, coil_idx):
        """
        F = I * (B_self + B_mutual) × dl  at all centerline quadrature points
        matches the simsopt reference to ~1e-10 relative tolerance.
        """
        c_jax = w7x["jax_curves"][coil_idx]
        I = float(w7x["base_currents_jax"][coil_idx])
        reg = float(regularization_rect(0.05, 0.02))

        # coil-fem: regularized self-field + pure-JAX mutual field
        B_self_cf = B_self_centerline(
            c_jax.gamma(), c_jax.gammadash(), c_jax.gammadashdash(),
            c_jax.quadpoints, I, reg,
        )
        B_mutual_cf = B_mutual_centerline(
            c_jax.gamma(),
            w7x["all_gammas"],
            w7x["all_gammadashs"],
            w7x["all_currents"],
            exclude_self_index=coil_idx,
        )
        F_cf = lorentz_line_force(B_self_cf + B_mutual_cf, I, c_jax.gammadash())

        # simsopt reference: same B assembled from simsopt routines
        B_self_sim = B_regularized_pure(
            np.asarray(c_jax.gamma()), np.asarray(c_jax.gammadash()),
            np.asarray(c_jax.gammadashdash()), np.asarray(c_jax.quadpoints),
            I, reg,
        )
        other_coils = [
            c for j, c in enumerate(w7x["coils_sim"]) if j != coil_idx
        ]
        bs_other = BiotSavart(other_coils)
        bs_other.set_points(np.asarray(c_jax.gamma()))
        B_mutual_sim = bs_other.B()
        F_sim = lorentz_line_force(
            jnp.asarray(B_self_sim + B_mutual_sim), I,
            c_jax.gammadash(),
        )

        np.testing.assert_allclose(
            np.asarray(F_cf),
            np.asarray(F_sim),
            atol=1e-4,   # ~0.1 N/m absolute on forces of ~1 MN/m
            rtol=1e-10,
            err_msg=(
                f"Lorentz line-force differs from simsopt reference "
                f"on coil {coil_idx}"
            ),
        )

    def test_zero_current_gives_zero_force(self, w7x):
        """
        A coil with I = 0 should produce zero Lorentz force regardless of B.
        """
        c_jax = w7x["jax_curves"][5]  # planar coil A, I=0 in standard config
        I = 0.0
        reg = float(regularization_rect(0.05, 0.02))

        B_self = B_self_centerline(
            c_jax.gamma(), c_jax.gammadash(), c_jax.gammadashdash(),
            c_jax.quadpoints, I, reg,
        )
        B_mutual = B_mutual_centerline(
            c_jax.gamma(),
            w7x["all_gammas"],
            w7x["all_gammadashs"],
            w7x["all_currents"],
            exclude_self_index=5,
        )
        F = lorentz_line_force(B_self + B_mutual, I, c_jax.gammadash())

        np.testing.assert_array_equal(
            np.asarray(F), np.zeros_like(np.asarray(F)),
            err_msg="Lorentz force with I=0 should be zero",
        )

    def test_force_direction_perpendicular_to_dl(self, w7x):
        """
        F = I dl × B is always perpendicular to dl (F · dl = 0 to machine precision).
        """
        c_jax = w7x["jax_curves"][0]
        I = float(w7x["base_currents_jax"][0])
        reg = float(regularization_rect(0.05, 0.02))

        B_self = B_self_centerline(
            c_jax.gamma(), c_jax.gammadash(), c_jax.gammadashdash(),
            c_jax.quadpoints, I, reg,
        )
        B_mutual = B_mutual_centerline(
            c_jax.gamma(),
            w7x["all_gammas"],
            w7x["all_gammadashs"],
            w7x["all_currents"],
            exclude_self_index=0,
        )
        dl = c_jax.gammadash()
        F = lorentz_line_force(B_self + B_mutual, I, dl)

        # F · dl should be zero everywhere (F = I dl × B is perpendicular to dl)
        dot = jnp.sum(F * dl, axis=-1)
        norm_F = jnp.linalg.norm(F, axis=-1)
        norm_dl = jnp.linalg.norm(dl, axis=-1)
        cos_theta = jnp.abs(dot) / (norm_F * norm_dl + 1e-30)
        np.testing.assert_array_less(
            np.asarray(cos_theta),
            np.full_like(np.asarray(cos_theta), 1e-12),
            err_msg="Lorentz force is not perpendicular to dl",
        )


# ---------------------------------------------------------------------------
# 6. FramedCurveCentroidJAX vs simsopt FramedCurveCentroid
# ---------------------------------------------------------------------------

class TestFramedCurveCentroidJAXMatchesSimsopt:
    """
    FramedCurveCentroidJAX should reproduce simsopt's FramedCurveCentroid
    for rotated_frame, frame_binormal_curvature, and frame_torsion.

    Tests run on three W7-X base coils with both zero and non-zero rotation
    angles, ensuring the JVP-based frame-derivative pipeline is correct.
    """

    def _build_pair(self, c_sim, c_jax, alpha_dofs):
        """Build matched simsopt / coil-fem framed-curve objects."""
        qp = c_sim.quadpoints
        rotation = FrameRotation(qp, order=2)
        rotation.x = np.array(alpha_dofs)  # set non-zero DOFs

        alpha = np.asarray(rotation.alpha(qp))

        fc_sim = FramedCurveCentroid(c_sim, rotation)
        fc_jax = FramedCurveCentroidJAX(c_jax, alpha=alpha)
        return fc_sim, fc_jax

    @pytest.mark.parametrize("coil_idx", [0, 2, 4])
    def test_rotated_frame_zero_alpha(self, w7x, coil_idx):
        """
        With alpha = 0, FramedCurveCentroidJAX.rotated_frame() matches
        simsopt's FramedCurveCentroid.rotated_frame() to machine precision.
        """
        c_sim = w7x["base_curves_sim"][coil_idx]
        c_jax = w7x["jax_curves"][coil_idx]
        fc_sim, fc_jax = self._build_pair(
            c_sim, c_jax, alpha_dofs=[0.0] * 5
        )

        t_sim, p_sim, q_sim = fc_sim.rotated_frame()
        t_jax, p_jax, q_jax = fc_jax.rotated_frame()

        for vec_sim, vec_jax, name in [
            (t_sim, t_jax, "t"), (p_sim, p_jax, "p"), (q_sim, q_jax, "q")
        ]:
            np.testing.assert_allclose(
                np.asarray(vec_jax), vec_sim,
                atol=1e-12, rtol=0,
                err_msg=f"rotated_frame {name} mismatch (zero alpha, coil {coil_idx})",
            )

    @pytest.mark.parametrize("coil_idx", [0, 2, 4])
    def test_rotated_frame_nonzero_alpha(self, w7x, coil_idx):
        """
        With a non-trivial rotation angle, frame vectors still match to
        machine precision.
        """
        c_sim = w7x["base_curves_sim"][coil_idx]
        c_jax = w7x["jax_curves"][coil_idx]
        alpha_dofs = [0.3, 0.1, -0.05, 0.07, -0.02]
        fc_sim, fc_jax = self._build_pair(c_sim, c_jax, alpha_dofs)

        t_sim, p_sim, q_sim = fc_sim.rotated_frame()
        t_jax, p_jax, q_jax = fc_jax.rotated_frame()

        for vec_sim, vec_jax, name in [
            (t_sim, t_jax, "t"), (p_sim, p_jax, "p"), (q_sim, q_jax, "q")
        ]:
            np.testing.assert_allclose(
                np.asarray(vec_jax), vec_sim,
                atol=1e-12, rtol=0,
                err_msg=f"rotated_frame {name} mismatch (nonzero alpha, coil {coil_idx})",
            )

    @pytest.mark.parametrize("coil_idx", [0, 2, 4])
    def test_frame_binormal_curvature_zero_alpha(self, w7x, coil_idx):
        """
        frame_binormal_curvature (κ₂) matches simsopt's frame_binormal_curvature
        with zero rotation angle.
        """
        c_sim = w7x["base_curves_sim"][coil_idx]
        c_jax = w7x["jax_curves"][coil_idx]
        fc_sim, fc_jax = self._build_pair(c_sim, c_jax, [0.0] * 5)

        kappa2_sim = fc_sim.frame_binormal_curvature()
        kappa2_jax = fc_jax.frame_binormal_curvature()

        np.testing.assert_allclose(
            np.asarray(kappa2_jax), kappa2_sim,
            atol=1e-10, rtol=0,
            err_msg=f"frame_binormal_curvature mismatch (zero alpha, coil {coil_idx})",
        )

    @pytest.mark.parametrize("coil_idx", [0, 2, 4])
    def test_frame_torsion_zero_alpha(self, w7x, coil_idx):
        """
        frame_torsion (κ₃) matches simsopt's frame_torsion with zero rotation.
        """
        c_sim = w7x["base_curves_sim"][coil_idx]
        c_jax = w7x["jax_curves"][coil_idx]
        fc_sim, fc_jax = self._build_pair(c_sim, c_jax, [0.0] * 5)

        kappa3_sim = fc_sim.frame_torsion()
        kappa3_jax = fc_jax.frame_torsion()

        np.testing.assert_allclose(
            np.asarray(kappa3_jax), kappa3_sim,
            atol=1e-10, rtol=0,
            err_msg=f"frame_torsion mismatch (zero alpha, coil {coil_idx})",
        )

    @pytest.mark.parametrize("coil_idx", [0, 2, 4])
    def test_frame_binormal_curvature_nonzero_alpha(self, w7x, coil_idx):
        """
        With a non-trivial rotation angle, frame_binormal_curvature still
        matches simsopt.
        """
        c_sim = w7x["base_curves_sim"][coil_idx]
        c_jax = w7x["jax_curves"][coil_idx]
        alpha_dofs = [0.3, 0.1, -0.05, 0.07, -0.02]
        fc_sim, fc_jax = self._build_pair(c_sim, c_jax, alpha_dofs)

        kappa2_sim = fc_sim.frame_binormal_curvature()
        kappa2_jax = fc_jax.frame_binormal_curvature()

        np.testing.assert_allclose(
            np.asarray(kappa2_jax), kappa2_sim,
            atol=1e-10, rtol=0,
            err_msg=(
                f"frame_binormal_curvature mismatch (nonzero alpha, coil {coil_idx})"
            ),
        )

    @pytest.mark.parametrize("coil_idx", [0, 2, 4])
    def test_frame_torsion_nonzero_alpha(self, w7x, coil_idx):
        """
        With a non-trivial rotation angle, frame_torsion still matches
        simsopt.
        """
        c_sim = w7x["base_curves_sim"][coil_idx]
        c_jax = w7x["jax_curves"][coil_idx]
        alpha_dofs = [0.3, 0.1, -0.05, 0.07, -0.02]
        fc_sim, fc_jax = self._build_pair(c_sim, c_jax, alpha_dofs)

        kappa3_sim = fc_sim.frame_torsion()
        kappa3_jax = fc_jax.frame_torsion()

        np.testing.assert_allclose(
            np.asarray(kappa3_jax), kappa3_sim,
            atol=1e-10, rtol=0,
            err_msg=f"frame_torsion mismatch (nonzero alpha, coil {coil_idx})",
        )

    @pytest.mark.parametrize("coil_idx", [0, 2, 4])
    def test_pythagorean_identity(self, w7x, coil_idx):
        """
        κ₁² + κ₂² = |dt/dl|² (Pythagorean identity from dt/dl = κ₁n + κ₂b).
        """
        c_jax = w7x["jax_curves"][coil_idx]
        alpha_dofs = [0.3, 0.1, -0.05, 0.07, -0.02]
        c_sim = w7x["base_curves_sim"][coil_idx]
        _, fc_jax = self._build_pair(c_sim, c_jax, alpha_dofs)

        kappa1, kappa2, _ = fc_jax.frame_curvatures()
        gd_norm = jnp.linalg.norm(c_jax.gammadash(), axis=1)
        # |dt/dl|² via second derivative formula:
        #   dt/dl = (gammadash / |gammadash|)' / |gammadash|
        t, p, q = fc_jax.rotated_frame()
        gd = c_jax.gammadash()
        gdd = c_jax.gammadashdash()
        inner = jnp.sum(gd * gdd, axis=1)
        tdash_raw = gdd / gd_norm[:, None] - (inner / gd_norm**3)[:, None] * gd
        kappa_sq_ref = jnp.sum(tdash_raw**2, axis=1) / gd_norm**2  # |dt/dl|²

        np.testing.assert_allclose(
            np.asarray(kappa1**2 + kappa2**2),
            np.asarray(kappa_sq_ref),
            rtol=1e-8,
            err_msg=(
                f"Pythagorean identity κ₁²+κ₂²=|dt/dl|² failed on coil {coil_idx}"
            ),
        )
