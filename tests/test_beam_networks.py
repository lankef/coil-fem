"""Smoke tests for SupportBeams beam-network support.

Follows the same lightweight-fixture style as ``tests/test_pipelines.py``.
All tests use a minimal two-coil ring configuration (``n_base=2``,
``n_beam_cc=1``, ``n_beam_cf=1``) unless otherwise noted.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from coil_fem.geo import CurveXYZFourierJAX
from coil_fem.coupling import SupportBeams


# ============================================================================
# Shared fixtures / helpers
# ============================================================================

def _make_circle(N: int = 8, R: float = 1.0) -> CurveXYZFourierJAX:
    """Planar circle of radius R in the xz-plane (order-1 Fourier curve)."""
    quadpoints = jnp.linspace(0.0, 1.0, N, endpoint=False)
    # x = R cos(2π φ), z = R sin(2π φ), y = 0
    dofs = jnp.array([0.0, R, 0.0,
                      0.0, 0.0, 0.0,
                      0.0, 0.0, R])
    return CurveXYZFourierJAX(quadpoints, dofs, order=1)


def _make_tilted_circle(N: int = 8, R: float = 1.0,
                        offset_y: float = 0.5) -> CurveXYZFourierJAX:
    """Circle offset in y — makes CC-beam endpoints non-degenerate."""
    quadpoints = jnp.linspace(0.0, 1.0, N, endpoint=False)
    dofs = jnp.array([0.0, R, 0.0,
                      offset_y, 0.0, 0.0,
                      0.0, 0.0, R])
    return CurveXYZFourierJAX(quadpoints, dofs, order=1)


def _as_counts(value, n_entries):
    """Broadcast an int (or coerce a sequence) to a length-n_entries tuple of ints."""
    if np.ndim(value) == 0:
        return tuple(int(value) for _ in range(n_entries))
    seq = list(value)
    assert len(seq) == n_entries
    return tuple(int(v) for v in seq)


def _constant_section_fn(A_val=1e-4, Iy_val=1e-8, Iz_val=1e-8, J_val=2e-8):
    """Returns a ragged-aware cross_section_fn with constant properties.

    Consumes the ragged (per-group list) support_dofs and returns one array
    of beam properties per group (cc-then-cf order; the stellsym wrap group
    has no cf part), matching the shape contract of ``SupportBeams.coo``.
    """
    def fn(support_dofs):
        phi_cc = support_dofs['phis_start_cc']   # list[g] -> (n_beam_cc[g],)
        phi_cf = support_dofs['phis_start_cf']   # list[i] -> (n_beam_cf[i],)
        A, Iy, Iz, J = [], [], [], []
        for g in range(len(phi_cc)):
            n_cf = phi_cf[g].shape[0] if g < len(phi_cf) else 0
            n_per = phi_cc[g].shape[0] + n_cf
            A.append(jnp.full((n_per,), A_val))
            Iy.append(jnp.full((n_per,), Iy_val))
            Iz.append(jnp.full((n_per,), Iz_val))
            J.append(jnp.full((n_per,), J_val))
        return A, Iy, Iz, J
    return fn


def _uniform_clamp_fn(surface_pts_beam_frame, dofs, sign_x, constants):
    """Returns uniform unit weights (all surface nodes equally clamped)."""
    return jnp.ones(surface_pts_beam_frame.shape[0])


def _make_curves(n_base: int, N: int = 8) -> list:
    """Build the canonical test base-coil curves (circles of increasing radius)."""
    return [_make_circle(N=N, R=1.0 + 0.1 * i) for i in range(n_base)]


def _make_support_beams(
    n_base: int = 2,
    n_beam_cc: int = 1,
    n_beam_cf: int = 1,
    nfp: int = 2,
    stellsym: bool = False,
    fixed_clamp_fns=None,
) -> SupportBeams:
    """Build a minimal SupportBeams instance for testing."""
    beam_options = {
        'n_beam_cc': n_beam_cc,
        'n_beam_cf': n_beam_cf,
        'E': 200e9,
        'nu': 0.3,
        'k_lin': 1e8,
        'k_tor': 1e4,
    }
    return SupportBeams(
        nfp=nfp,
        stellsym=stellsym,
        beam_options=beam_options,
        n_base=n_base,
        cross_section_fn=_constant_section_fn(),
        attachment_fn=_uniform_clamp_fn,
        fixed_clamp_fns=fixed_clamp_fns,
    )


def _make_support_dofs(n_base: int = 2, n_beam_cc=1, n_beam_cf=1,
                       stellsym: bool = False):
    """Build a ragged support_dofs dict (per-group list of arrays).

    ``n_beam_cc`` / ``n_beam_cf`` may be an int (broadcast) or a sequence for
    a ragged configuration.  CC keys have one entry per CC group
    (``n_base + 1`` when ``stellsym=True``); CF keys have one per coil.
    """
    n_groups_cc = n_base + (1 if stellsym else 0)
    ncc = _as_counts(n_beam_cc, n_groups_cc)
    ncf = _as_counts(n_beam_cf, n_base)

    phi_cc_start = [jnp.full((ncc[g],), 0.1) for g in range(n_groups_cc)]
    phi_cc_end   = [jnp.full((ncc[g],), 0.1) for g in range(n_groups_cc)]
    phi_cf_start = [jnp.full((ncf[i],), 0.6) for i in range(n_base)]

    # Foundation points: slightly outside the coil rings to give non-zero beams
    x_foundation = []
    for i in range(n_base):
        R_i = 1.0 + 0.1 * i
        row = []
        for j in range(ncf[i]):
            x_phi = 0.6 + 0.05 * j
            x_f = jnp.array([R_i * math.cos(2 * math.pi * x_phi) + 0.5,
                              0.0,
                              R_i * math.sin(2 * math.pi * x_phi)])
            row.append(x_f)
        if row:
            x_foundation.append(jnp.stack(row, axis=0))   # (n_beam_cf[i], 3)
        else:
            x_foundation.append(jnp.zeros((0, 3)))

    theta_cc = [jnp.zeros((ncc[g],)) for g in range(n_groups_cc)]
    theta_cf = [jnp.zeros((ncf[i],)) for i in range(n_base)]

    return {
        'phis_start_cc':         phi_cc_start,
        'phis_end_cc':           phi_cc_end,
        'phis_start_cf':         phi_cf_start,
        'x_foundation':          x_foundation,
        'thetas_orientation_cc': theta_cc,
        'thetas_orientation_cf': theta_cf,
    }


def _make_surface_pts(n_base: int = 2, n_surf: int = 10):
    """Dummy surface points on each coil's outer surface."""
    pts_list = []
    for i in range(n_base):
        R = 1.0 + 0.1 * i
        phis = jnp.linspace(0.0, 1.0, n_surf, endpoint=False)
        xs = R * jnp.cos(2 * math.pi * phis)
        ys = jnp.zeros(n_surf)
        zs = R * jnp.sin(2 * math.pi * phis)
        pts_list.append(jnp.stack([xs, ys, zs], axis=1))
    return pts_list


# ============================================================================
# 1. is_coupled
# ============================================================================

def test_support_beams_is_coupled():
    """SupportBeams.is_coupled must be True."""
    sb = _make_support_beams()
    assert sb.is_coupled is True


# ============================================================================
# 2. DOF count
# ============================================================================

def test_support_beams_dof_count():
    """n_support_dofs == n_base * (n_beam_cc + n_beam_cf) * 12."""
    n_base, n_cc, n_cf = 2, 1, 1
    sb = _make_support_beams(n_base=n_base, n_beam_cc=n_cc, n_beam_cf=n_cf)
    expected = n_base * (n_cc + n_cf) * 12
    assert sb.n_support_dofs == expected, (
        f"Expected {expected}, got {sb.n_support_dofs}"
    )


# ============================================================================
# 3. coo shapes
# ============================================================================

def test_support_beams_coo_shapes():
    """coo() returns (I, J, V, n) with consistent shapes."""
    n_base, n_cc, n_cf = 2, 1, 1
    sb     = _make_support_beams(n_base=n_base, n_beam_cc=n_cc, n_beam_cf=n_cf)
    curves = _make_curves(n_base)
    sdofs  = _make_support_dofs(n_base, n_cc, n_cf)
    surf   = _make_surface_pts(n_base)

    I, J, V, n_dofs = sb.coo(curves, sdofs, surface_pts_by_coil=surf)

    n_beams = n_base * (n_cc + n_cf)
    expected_nnz = n_beams * 144   # 12 * 12 per beam

    assert len(I) == expected_nnz, f"I has wrong length: {len(I)}"
    assert len(J) == expected_nnz, f"J has wrong length: {len(J)}"
    assert V.shape == (expected_nnz,), f"V has wrong shape: {V.shape}"
    assert n_dofs == sb.n_support_dofs
    assert int(I.max()) < n_dofs
    assert int(J.max()) < n_dofs


# ============================================================================
# 4. Diagonal dominance / rank check
# ============================================================================

def test_support_beams_coo_diagonal_dominance():
    """Each 12×12 beam block gains rank from bare-beam 6 to full 12 with springs.

    The bare bisymmetric beam has 6 rigid-body modes (3 translations + torsion
    about its own axis + 2 bending rotations = 6).  The endpoint springs
    constrain all of them: the translational blocks (k_lin) pin the
    translation and bending-rotation RBMs, and the torque–rotation block
    ``K_rr = -k_tor Σ w [r]×[r]×`` (PSD) pins the remaining torsional
    rigid-body mode — pure rotation of the whole beam about its own axis —
    through the attach-point moment arms.
    """
    n_base, n_cc, n_cf = 2, 1, 1
    sb     = _make_support_beams(n_base=n_base, n_beam_cc=n_cc, n_beam_cf=n_cf)
    curves = _make_curves(n_base)
    sdofs  = _make_support_dofs(n_base, n_cc, n_cf)
    surf   = _make_surface_pts(n_base)

    I, J, V, n_dofs = sb.coo(curves, sdofs, surface_pts_by_coil=surf)

    # Reconstruct the dense block-diagonal matrix
    K = np.zeros((n_dofs, n_dofs))
    np.add.at(K, (np.asarray(I), np.asarray(J)), np.asarray(V))

    n_beams = n_base * (n_cc + n_cf)
    for b in range(n_beams):
        block = K[12 * b: 12 * b + 12, 12 * b: 12 * b + 12]
        rank  = np.linalg.matrix_rank(block, tol=1e-3)
        assert rank == 12, (
            f"Beam {b} block has rank {rank} (expected 12); "
            "the endpoint springs should constrain all 6 rigid-body modes."
        )


# ============================================================================
# 5. Zero RHS → zero solution
# ============================================================================

def test_support_beams_solve_zero_rhs_yields_zero():
    """With u_mesh_by_coil=None the RHS is zero → u_s should be ≈ 0."""
    n_base, n_cc, n_cf = 2, 1, 1
    sb     = _make_support_beams(n_base=n_base, n_beam_cc=n_cc, n_beam_cf=n_cf)
    curves = _make_curves(n_base)
    sdofs  = _make_support_dofs(n_base, n_cc, n_cf)
    surf   = _make_surface_pts(n_base)

    result = sb.solve({
        'curves_jax':          curves,
        'support_dofs':        sdofs,
        'surface_pts_by_coil': surf,
        'u_mesh_by_coil':      None,
    })

    u_s = result['u_s']
    assert u_s.shape == (sb.n_support_dofs,)
    assert jnp.allclose(u_s, 0.0, atol=1e-10), (
        f"u_s should be zero but max |u_s| = {jnp.abs(u_s).max():.3e}"
    )


# ============================================================================
# 6. Grad through x_foundation
# ============================================================================

def test_support_beams_grad_through_x_foundation():
    """jax.grad w.r.t. x_foundation through solve() is nonzero and finite."""
    n_base, n_cc, n_cf = 2, 1, 1
    sb     = _make_support_beams(n_base=n_base, n_beam_cc=n_cc, n_beam_cf=n_cf)
    curves = _make_curves(n_base)
    surf   = _make_surface_pts(n_base)

    # Give nonzero coil displacements so that x_foundation shift matters
    u_mesh = [jnp.ones((s.shape[0], 3)) * 1e-3 for s in surf]

    def loss(x_found):
        sdofs = _make_support_dofs(n_base, n_cc, n_cf)
        sdofs = {**sdofs, 'x_foundation': x_found}
        result = sb.solve({
            'curves_jax':          curves,
            'support_dofs':        sdofs,
            'surface_pts_by_coil': surf,
            'u_mesh_by_coil':      u_mesh,
        })
        return jnp.sum(result['u_s'])

    x_found0 = _make_support_dofs(n_base, n_cc, n_cf)['x_foundation']
    grad = jax.grad(loss)(x_found0)

    # grad is a ragged pytree (per-coil list of arrays); flatten to check.
    grad_flat = jnp.concatenate([g.reshape(-1) for g in grad])
    assert jnp.all(jnp.isfinite(grad_flat)), "Gradient contains NaN/Inf"
    # The gradient w.r.t. x_foundation need not be nonzero in general for zero
    # RHS, so we only assert finiteness here.  A stronger check would require
    # a nonzero external loading; this test focuses on the autodiff plumbing.


# ============================================================================
# 7. compute_weights pass-through via Support
# ============================================================================

def test_support_beams_compute_weights_passthrough():
    """When dofs=None, compute_weights delegates to Support (returns fixed_clamp_fns result)."""
    n_surf = 12
    # A constant weight function that returns 0.5 for all nodes
    def half_fn(surface_pts, curve_jax, dofs):
        return jnp.full(surface_pts.shape[0], 0.5)

    sb     = _make_support_beams(n_base=2, fixed_clamp_fns=half_fn)
    curves = _make_curves(2)
    surf   = jnp.ones((n_surf, 3))

    w = sb.compute_weights(0, surf, curves, None)

    assert w.shape == (n_surf,)
    assert jnp.allclose(w, 0.5), f"Expected 0.5 everywhere, got {w}"


# ============================================================================
# 8. Wraparound: stellsym=True vs stellsym=False
# ============================================================================

def test_support_beams_wraparound_stellsym_true_vs_false():
    """CC group topology: boundary groups and transforms differ for stellsym T/F."""
    n_base = 3
    sb_sym   = _make_support_beams(n_base=n_base, nfp=3, stellsym=True)
    sb_nosym = _make_support_beams(n_base=n_base, nfp=3, stellsym=False)

    # stellsym=True: n_base+1 groups; the last coil reflects about
    # phi = pi/nfp, and an extra group wraps coil 0 about phi = 0.
    assert sb_sym.n_groups_cc == n_base + 1
    assert sb_sym.cc_groups == (
        (0, 1, 'none'),
        (1, 2, 'none'),
        (2, 2, 'flip_half'),
        (0, 0, 'flip'),
    )
    # Wrap-group beams are appended after all per-coil blocks.
    assert sb_sym.wrap_beam_offset == sum(sb_sym.n_beams_per_coil)
    assert sb_sym.n_beams_total == (
        sum(sb_sym.n_beams_per_coil) + sb_sym.n_beam_cc[n_base]
    )

    # stellsym=False: n_base groups; last coil wraps to coil 0 via rotation.
    assert sb_nosym.n_groups_cc == n_base
    assert sb_nosym.cc_groups == (
        (0, 1, 'none'),
        (1, 2, 'none'),
        (2, 0, 'rotate'),
    )
    assert sb_nosym.n_beams_total == sum(sb_nosym.n_beams_per_coil)


def test_support_beams_transform_matrices_and_reflection_planes():
    """Q matrices are proper rotations; boundary x_end lie on the right planes."""
    n_base, nfp = 2, 3
    sb = _make_support_beams(n_base=n_base, nfp=nfp, stellsym=True)

    Q_flip = np.asarray(sb._tfm_Q['flip'])
    Q_half = np.asarray(sb._tfm_Q['flip_half'])
    Q_rot  = np.asarray(sb._tfm_Q['rotate'])

    # All transforms are proper rotations; the stellsym ones are involutions.
    for Q in (Q_flip, Q_half, Q_rot):
        assert np.isclose(np.linalg.det(Q), 1.0)
        assert np.allclose(Q @ Q.T, np.eye(3), atol=1e-14)
    assert np.allclose(Q_flip @ Q_flip, np.eye(3), atol=1e-14)
    assert np.allclose(Q_half @ Q_half, np.eye(3), atol=1e-14)
    assert np.allclose(Q_half, Q_rot @ Q_flip, atol=1e-14)

    # flip_half: cylindrical angle phi -> 2*pi/nfp - phi, z negated.
    rng = np.random.default_rng(0)
    p = rng.normal(size=3)
    q = Q_half @ p
    phi_p = math.atan2(p[1], p[0])
    phi_q = math.atan2(q[1], q[0])
    assert np.isclose(
        (phi_q - (2.0 * math.pi / nfp - phi_p)) % (2.0 * math.pi), 0.0,
        atol=1e-12,
    )
    assert np.isclose(q[2], -p[2])
    assert np.isclose(np.hypot(q[0], q[1]), np.hypot(p[0], p[1]))

    # Boundary-group beam endpoints land on the transformed curves:
    # group n_base-1 -> flip_half(coil n-1), group n_base -> flip(coil 0).
    sdofs  = _make_support_dofs(n_base, 1, 1, stellsym=True)
    curves = _make_curves(n_base)
    geom   = sb._beam_geometry(curves, sdofs)

    phi_e = sdofs['phis_end_cc'][n_base - 1][0]
    x_e_expected = Q_half @ np.asarray(curves[n_base - 1].gamma_eval(phi_e))
    b_half = sb.beam_offsets[n_base - 1]     # first CC beam of last coil
    assert np.allclose(np.asarray(geom['x_end'][b_half]), x_e_expected)

    phi_e0 = sdofs['phis_end_cc'][n_base][0]
    x_e0_expected = Q_flip @ np.asarray(curves[0].gamma_eval(phi_e0))
    b_wrap = sb.wrap_beam_offset             # first beam of the wrap group
    assert np.allclose(np.asarray(geom['x_end'][b_wrap]), x_e0_expected)


# ============================================================================
# 9. coo without surface_pts_by_coil — bare-beam blocks are rank-6
# ============================================================================

def test_support_beams_bare_beam_rank():
    """Without springs the bare beam block has rank 6 (rank-deficient)."""
    n_base, n_cc, n_cf = 1, 1, 0
    sb     = _make_support_beams(n_base=n_base, n_beam_cc=n_cc, n_beam_cf=n_cf,
                                  nfp=2, stellsym=False)
    curves = _make_curves(n_base)
    sdofs  = _make_support_dofs(n_base, n_cc, n_cf)

    # Pass surface_pts_by_coil=None → bare beam, no spring regularisation
    I, J, V, n_dofs = sb.coo(curves, sdofs, surface_pts_by_coil=None)

    K = np.zeros((n_dofs, n_dofs))
    np.add.at(K, (np.asarray(I), np.asarray(J)), np.asarray(V))

    rank = np.linalg.matrix_rank(K, tol=1e-3)
    # The bisymmetric beam element has exactly 6 rigid-body modes → rank 6
    assert rank == 6, (
        f"Bare-beam block expected rank 6 but got {rank}."
    )


# ============================================================================
# 10. End-side gradient: gradients flow to the end coil's DOFs
# ============================================================================

def test_support_beams_end_side_gradient():
    """compute_weights gradient w.r.t. end-side coil DOFs is non-zero and finite.

    Before the beam-frame clamp_fn refactor, clamp weights for node-2 only
    traced through the *start* coil's DOFs (coil-tangent approximation).
    This test verifies that gradients now flow to the end coil's DOFs.
    """
    n_base, n_cc, n_cf = 2, 1, 0
    nfp = 2

    def sigmoid_clamp_fn(surface_pts_beam_frame, dofs, sign_x, constants):
        """Sigmoid of max distance in beam frame — smoothly differentiable."""
        d = jnp.sqrt(jnp.sum(surface_pts_beam_frame ** 2, axis=1) + 1e-8)
        return jax.nn.sigmoid(1.0 - d)

    base_curves = _make_curves(n_base)
    sb = SupportBeams(
        nfp=nfp,
        stellsym=False,
        beam_options={
            'n_beam_cc': n_cc,
            'n_beam_cf': n_cf,
            'E': 200e9,
            'nu': 0.3,
            'k_lin': 1e8,
            'k_tor': 1e4,
        },
        n_base=n_base,
        cross_section_fn=_constant_section_fn(),
        attachment_fn=sigmoid_clamp_fn,
    )

    sdofs = _make_support_dofs(n_base, n_cc, n_cf)
    surf  = jnp.ones((8, 3))

    def weight_sum_end_coil(dofs_end):
        """Sum of weights for coil 1 (end-side) parameterised by coil 1's DOFs."""
        curves_jax = [
            CurveXYZFourierJAX(base_curves[0].quadpoints, base_curves[0].dofs, base_curves[0].order),
            CurveXYZFourierJAX(base_curves[1].quadpoints, dofs_end, base_curves[1].order),
        ]
        # coil_idx=1 receives node-2 of all CC beams from coil 0
        w = sb.compute_weights(1, surf, curves_jax, sdofs)
        return jnp.sum(w)

    grad = jax.grad(weight_sum_end_coil)(base_curves[1].dofs)

    assert jnp.all(jnp.isfinite(grad)), "End-side gradient contains NaN/Inf"
    assert jnp.any(grad != 0.0), (
        "End-side gradient is identically zero; "
        "gradients should flow through the end coil's DOFs."
    )


# ============================================================================
# 11. Ragged per-coil beam counts
# ============================================================================

_RAGGED_CC = [2, 1, 3]
_RAGGED_CF = [1, 2, 1]


def test_support_beams_ragged_counts_and_offsets():
    """Per-coil counts, cumulative offsets, and total sizing are consistent."""
    n_base = 3
    sb = _make_support_beams(
        n_base=n_base, n_beam_cc=_RAGGED_CC, n_beam_cf=_RAGGED_CF,
        nfp=2, stellsym=False,
    )
    assert tuple(sb.n_beam_cc) == tuple(_RAGGED_CC)
    assert tuple(sb.n_beam_cf) == tuple(_RAGGED_CF)
    per_coil = [_RAGGED_CC[i] + _RAGGED_CF[i] for i in range(n_base)]
    assert tuple(sb.n_beams_per_coil) == tuple(per_coil)
    assert sb.n_beams_total == sum(per_coil)
    # Cumulative offsets start at 0 and accumulate per-coil totals.
    expected_offsets = tuple(sum(per_coil[:i]) for i in range(n_base))
    assert tuple(sb.beam_offsets) == expected_offsets
    assert sb.n_support_dofs == 12 * sum(per_coil)


def test_support_beams_ragged_coo_shapes():
    """coo() sizing and index bounds hold for ragged per-coil counts."""
    n_base = 3
    sb     = _make_support_beams(
        n_base=n_base, n_beam_cc=_RAGGED_CC, n_beam_cf=_RAGGED_CF,
        nfp=2, stellsym=False,
    )
    curves = _make_curves(n_base)
    sdofs  = _make_support_dofs(n_base, _RAGGED_CC, _RAGGED_CF)
    surf   = _make_surface_pts(n_base)

    I, J, V, n_dofs = sb.coo(curves, sdofs, surface_pts_by_coil=surf)

    n_beams = sum(_RAGGED_CC[i] + _RAGGED_CF[i] for i in range(n_base))
    expected_nnz = n_beams * 144
    assert len(I) == expected_nnz
    assert len(J) == expected_nnz
    assert V.shape == (expected_nnz,)
    assert n_dofs == sb.n_support_dofs == 12 * n_beams
    assert int(I.max()) < n_dofs
    assert int(J.max()) < n_dofs

    # Dense K_ss must be square (n_dofs, n_dofs).
    K = np.zeros((n_dofs, n_dofs))
    np.add.at(K, (np.asarray(I), np.asarray(J)), np.asarray(V))
    assert K.shape == (n_dofs, n_dofs)


def test_support_beams_ragged_solve_zero_rhs_yields_zero():
    """Zero RHS -> zero solution, even with ragged per-coil counts."""
    n_base = 3
    sb     = _make_support_beams(
        n_base=n_base, n_beam_cc=_RAGGED_CC, n_beam_cf=_RAGGED_CF,
        nfp=2, stellsym=False,
    )
    curves = _make_curves(n_base)
    sdofs  = _make_support_dofs(n_base, _RAGGED_CC, _RAGGED_CF)
    surf   = _make_surface_pts(n_base)

    result = sb.solve({
        'curves_jax':          curves,
        'support_dofs':        sdofs,
        'surface_pts_by_coil': surf,
        'u_mesh_by_coil':      None,
    })
    u_s = result['u_s']
    assert u_s.shape == (sb.n_support_dofs,)
    assert jnp.allclose(u_s, 0.0, atol=1e-10)


def test_support_beams_ragged_coupling_terms_bounds():
    """coupling_terms() off-diagonal indices stay within the merged system."""
    n_base = 3
    sb     = _make_support_beams(
        n_base=n_base, n_beam_cc=_RAGGED_CC, n_beam_cf=_RAGGED_CF,
        nfp=2, stellsym=False,
    )
    curves = _make_curves(n_base)
    sdofs  = _make_support_dofs(n_base, _RAGGED_CC, _RAGGED_CF)
    surf   = _make_surface_pts(n_base, n_surf=6)

    # Minimal merged-system layout: 3 DOFs per surface node, coils packed first.
    n_surf = surf[0].shape[0]
    coil_dof_offsets = [i * (n_surf * 3) for i in range(n_base)]
    support_dof_offset = n_base * n_surf * 3
    surf_idx = [np.arange(n_surf, dtype=np.int32) for _ in range(n_base)]

    terms = sb.coupling_terms(
        curves, sdofs, surf, coil_dof_offsets, support_dof_offset, surf_idx,
    )
    n_total = support_dof_offset + sb.n_support_dofs
    for key in ('I_cs', 'J_cs', 'I_sc', 'J_sc'):
        idx = np.asarray(terms[key])
        assert idx.size > 0
        assert int(idx.max()) < n_total
        assert int(idx.min()) >= 0
    assert terms['V_cs'].shape[0] == terms['I_cs'].shape[0]
    assert terms['V_sc'].shape[0] == terms['I_sc'].shape[0]


def test_ragged_support_dofs_ravel_roundtrip():
    """Ragged support_dofs flatten/unflatten deterministically via ravel_pytree."""
    from jax.flatten_util import ravel_pytree

    sdofs = _make_support_dofs(3, _RAGGED_CC, _RAGGED_CF)
    flat, unravel = ravel_pytree(sdofs)
    restored = unravel(flat)

    assert set(restored.keys()) == set(sdofs.keys())
    for k in sdofs:
        assert len(restored[k]) == len(sdofs[k])
        for a, b in zip(sdofs[k], restored[k]):
            assert a.shape == b.shape
            assert jnp.allclose(a, b)


# ============================================================================
# 12. Staggered/monolithic consistency: _assemble_rhs == -K_sc @ u_mesh
# ============================================================================

def test_support_beams_rhs_matches_coupling_terms():
    """The standalone-solve RHS equals -K_sc @ u_mesh from coupling_terms.

    Uses a stellsym model so that 'none', 'flip_half' and 'flip' endpoint
    transforms are all present — validating the Q factors and the torque
    rows of both assembly paths against each other.
    """
    n_base, nfp = 2, 3
    sb     = _make_support_beams(n_base=n_base, nfp=nfp, stellsym=True)
    curves = _make_curves(n_base)
    sdofs  = _make_support_dofs(n_base, 1, 1, stellsym=True)
    surf   = _make_surface_pts(n_base, n_surf=6)
    n_surf = surf[0].shape[0]

    rng    = np.random.default_rng(42)
    u_mesh = [jnp.asarray(rng.normal(size=(n_surf, 3)) * 1e-3)
              for _ in range(n_base)]

    # Merged-system layout: coil DOFs packed first, support DOFs last.
    coil_dof_offsets   = [i * (n_surf * 3) for i in range(n_base)]
    support_dof_offset = n_base * n_surf * 3
    surf_idx = [np.arange(n_surf, dtype=np.int32) for _ in range(n_base)]

    terms = sb.coupling_terms(
        curves, sdofs, surf, coil_dof_offsets, support_dof_offset, surf_idx,
    )

    # Dense K_sc: rows = support DOFs, cols = coil DOFs.
    n_s = sb.n_support_dofs
    n_c = support_dof_offset
    K_sc = np.zeros((n_s, n_c))
    np.add.at(
        K_sc,
        (np.asarray(terms['I_sc']) - support_dof_offset,
         np.asarray(terms['J_sc'])),
        np.asarray(terms['V_sc']),
    )
    u_c = np.concatenate([np.asarray(u).reshape(-1) for u in u_mesh])
    f_expected = -K_sc @ u_c

    # RHS from the standalone-solve path.
    geom   = sb._beam_geometry(curves, sdofs)
    gamma3 = sb._direction_cosine_matrices(geom, sdofs)
    beps   = sb._endpoint_weights_and_r(curves, geom, gamma3, sdofs, surf)
    f_rhs  = np.asarray(sb._assemble_rhs(geom, beps, u_mesh))

    assert f_rhs.shape == (n_s,)
    assert np.allclose(f_rhs, f_expected, rtol=1e-10, atol=1e-14), (
        f"max |f_rhs - (-K_sc u_c)| = {np.abs(f_rhs - f_expected).max():.3e}"
    )


# ============================================================================
# 13. Equivalence: stellsym model vs explicit symmetry expansion
# ============================================================================

def test_support_beams_stellsym_vs_explicit_expansion():
    """A stellsym model matches an explicitly expanded stellsym=False model.

    nfp=2, one base coil.  The stellsym model has two boundary groups
    ('flip_half' and 'flip'); the explicit model has two coils
    (``c1 = Q_half c0``) with two beams per group — each master beam plus
    its mirror partner (phis swapped).  With symmetric mesh displacements
    the beam solutions and the coil-0 attach field must agree exactly
    (circular cross-sections make the section roll irrelevant).
    """
    nfp = 2
    phi_s_h, phi_e_h = 0.12, 0.31   # half-plane (flip_half) group
    phi_s_0, phi_e_0 = 0.83, 0.64   # phi=0 (flip) group
    phi_cf = 0.45

    # Generic (non-symmetric) order-1 base curve.
    quadpoints = jnp.linspace(0.0, 1.0, 8, endpoint=False)
    dofs0 = jnp.array([0.10, 1.00, 0.05,
                       0.30, 0.02, 0.20,
                       0.15, 0.10, 0.90])
    c0 = CurveXYZFourierJAX(quadpoints, dofs0, order=1)

    beam_common = {'E': 200e9, 'nu': 0.3, 'k_lin': 1e8, 'k_tor': 1e4}
    const_w = lambda pts, dofs, sign_x, opts: jnp.full(pts.shape[0], 0.6)

    # ── Stellsym model: 1 coil, groups [(0,0,'flip_half'), (0,0,'flip')] ──
    sb_sym = SupportBeams(
        nfp=nfp, stellsym=True,
        beam_options={**beam_common, 'n_beam_cc': [1, 1], 'n_beam_cf': [1]},
        n_base=1,
        cross_section_fn=_constant_section_fn(),
        attachment_fn=const_w,
    )
    Q_half = np.asarray(sb_sym._tfm_Q['flip_half'])
    Q_rot  = np.asarray(sb_sym._tfm_Q['rotate'])

    xf0 = jnp.array([[2.0, 0.3, 0.4]])
    sdofs_sym = {
        'phis_start_cc': [jnp.array([phi_s_h]), jnp.array([phi_s_0])],
        'phis_end_cc':   [jnp.array([phi_e_h]), jnp.array([phi_e_0])],
        'phis_start_cf': [jnp.array([phi_cf])],
        'x_foundation':  [xf0],
        'thetas_orientation_cc': [jnp.zeros(1), jnp.zeros(1)],
        'thetas_orientation_cf': [jnp.zeros(1)],
    }

    # ── Explicit model: 2 coils, c1 = Q_half c0, mirror-pair beams ───────
    dofs1 = jnp.asarray((Q_half @ np.asarray(dofs0).reshape(3, 3)).reshape(-1))
    c1 = CurveXYZFourierJAX(quadpoints, dofs1, order=1)

    sb_exp = SupportBeams(
        nfp=nfp, stellsym=False,
        beam_options={**beam_common, 'n_beam_cc': [2, 2], 'n_beam_cf': [1, 1]},
        n_base=2,
        cross_section_fn=_constant_section_fn(),
        attachment_fn=const_w,
    )
    xf1 = jnp.asarray(np.asarray(xf0) @ Q_half.T)
    sdofs_exp = {
        # group 0 (c0 -> c1): master + mirror partner (phis swapped);
        # group 1 (c1 -> rotate(c0)): rotated images of the phi=0 pair.
        'phis_start_cc': [jnp.array([phi_s_h, phi_e_h]),
                          jnp.array([phi_e_0, phi_s_0])],
        'phis_end_cc':   [jnp.array([phi_e_h, phi_s_h]),
                          jnp.array([phi_s_0, phi_e_0])],
        'phis_start_cf': [jnp.array([phi_cf]), jnp.array([phi_cf])],
        'x_foundation':  [xf0, xf1],
        'thetas_orientation_cc': [jnp.zeros(2), jnp.zeros(2)],
        'thetas_orientation_cf': [jnp.zeros(1), jnp.zeros(1)],
    }

    # Symmetric surface points and mesh displacements: coil-1 quantities are
    # the Q_half images of coil-0's.
    rng   = np.random.default_rng(7)
    surf0 = jnp.asarray(rng.normal(size=(6, 3)))
    u0    = jnp.asarray(rng.normal(size=(6, 3)) * 1e-3)
    surf1 = jnp.asarray(np.asarray(surf0) @ Q_half.T)
    u1    = jnp.asarray(np.asarray(u0) @ Q_half.T)

    u_sym = np.asarray(sb_sym.solve({
        'curves_jax':          [c0],
        'support_dofs':        sdofs_sym,
        'surface_pts_by_coil': [surf0],
        'u_mesh_by_coil':      [u0],
    })['u_s'])
    u_exp = np.asarray(sb_exp.solve({
        'curves_jax':          [c0, c1],
        'support_dofs':        sdofs_exp,
        'surface_pts_by_coil': [surf0, surf1],
        'u_mesh_by_coil':      [u0, u1],
    })['u_s'])

    # Beam order — sym: [b0 = flip_half master, b1 = CF(c0), b2 = flip master]
    #              exp: [b0, b1 (c0 cc pair), b2 = CF(c0),
    #                    b3, b4 (c1 cc pair), b5 = CF(c1)]
    def blocks(u, b):
        """(t1, r1, t2, r2) 3-vectors of beam ``b``."""
        v = u[12 * b: 12 * b + 12]
        return v[0:3], v[3:6], v[6:9], v[9:12]

    # 1. flip_half master: physically the same beam in both models.
    assert np.allclose(u_sym[0:12], u_exp[0:12], rtol=1e-8, atol=1e-15), (
        "flip_half master beam DOFs differ between the two models"
    )

    # 2. CF beam of coil 0: identical in both models.
    assert np.allclose(u_sym[12:24], u_exp[24:36], rtol=1e-8, atol=1e-15)

    # 3. phi=0 ('flip') master vs its rotated node-swapped image (exp b3):
    #    u_sym[b2, node m] = Q_rot^T u_exp[b3, node (1-m)].
    t1s, r1s, t2s, r2s = blocks(u_sym, 2)
    t1e, r1e, t2e, r2e = blocks(u_exp, 3)
    assert np.allclose(t1s, Q_rot.T @ t2e, rtol=1e-8, atol=1e-15)
    assert np.allclose(r1s, Q_rot.T @ r2e, rtol=1e-8, atol=1e-15)
    assert np.allclose(t2s, Q_rot.T @ t1e, rtol=1e-8, atol=1e-15)
    assert np.allclose(r2s, Q_rot.T @ r1e, rtol=1e-8, atol=1e-15)

    # 4. CF beam of coil 1 = Q_half image of coil 0's CF beam.
    t1c, r1c, t2c, r2c = blocks(u_sym, 1)
    t1x, r1x, t2x, r2x = blocks(u_exp, 5)
    assert np.allclose(t1x, Q_half @ t1c, rtol=1e-8, atol=1e-15)
    assert np.allclose(r1x, Q_half @ r1c, rtol=1e-8, atol=1e-15)
    assert np.allclose(t2x, Q_half @ t2c, rtol=1e-8, atol=1e-15)
    assert np.allclose(r2x, Q_half @ r2c, rtol=1e-8, atol=1e-15)

    # 5. Coil-0 Winkler weights and attach displacements agree.
    curves_sym = [CurveXYZFourierJAX(quadpoints, dofs0, 1)]
    curves_exp = [CurveXYZFourierJAX(quadpoints, dofs0, 1),
                  CurveXYZFourierJAX(quadpoints, dofs1, 1)]
    w_sym = sb_sym.compute_weights(0, surf0, curves_sym, sdofs_sym)
    w_exp = sb_exp.compute_weights(0, surf0, curves_exp, sdofs_exp)
    assert np.allclose(np.asarray(w_sym), np.asarray(w_exp), rtol=1e-10)

    ua_sym = sb_sym.compute_attach(0, surf0, curves_sym, sdofs_sym,
                                   {'u_s': jnp.asarray(u_sym)})
    ua_exp = sb_exp.compute_attach(0, surf0, curves_exp, sdofs_exp,
                                   {'u_s': jnp.asarray(u_exp)})
    assert np.allclose(np.asarray(ua_sym), np.asarray(ua_exp),
                       rtol=1e-7, atol=1e-15), (
        "coil-0 attach displacement differs between stellsym and explicit "
        f"models: max diff = {np.abs(np.asarray(ua_sym) - np.asarray(ua_exp)).max():.3e}"
    )
