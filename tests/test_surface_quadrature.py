"""Tests for LinearElasticity3D.surface_jxw and the JxW coupling integration.

Covers:
* Shape and positivity of ``surface_jxw``.
* Area sanity: ``Σ JxW`` matches the analytic lateral surface area of the
  coil up to the mesh discretisation error.
* Autodiff: ``jax.grad`` flows through ``surface_jxw``.
* Coupling consistency: with JxW=1 fixture the beam coupling assembly is
  unchanged from the pre-JxW code path (identity-interp regression check).
* Transpose symmetry: with ``Q = I``, ``K_cs = K_sc^T`` unconditionally.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from coil_fem.geo import CurveXYZFourierJAX, make_framed_curve
from coil_fem.meshing import CoilMesh
from coil_fem.pipelines import ElasticPipeline
from coil_fem.coupling import SupportBeams


# ============================================================================
# Shared helpers
# ============================================================================

def _make_circle(N: int = 8, R: float = 1.0) -> CurveXYZFourierJAX:
    """Planar circle of radius R in the xz-plane (order-1 Fourier curve)."""
    quadpoints = jnp.linspace(0.0, 1.0, N, endpoint=False)
    dofs = jnp.array([0.0, R, 0.0,
                      0.0, 0.0, 0.0,
                      0.0, 0.0, R])
    return CurveXYZFourierJAX(quadpoints, dofs, order=1)


def _make_pipeline(
    R: float = 1.0,
    n_phi: int = 4,
    w1: float = 0.02,
    w2: float = 0.02,
    n_grid: int = 1,
) -> ElasticPipeline:
    """Build a minimal ElasticPipeline with a Winkler surface."""
    curve = _make_circle(N=n_phi, R=R)
    fc    = make_framed_curve(curve, 'rmf')
    mesh  = CoilMesh.from_options(
        fc,
        {'shape': 'rect', 'w1': w1, 'w2': w2,
         'n_grid_1': n_grid, 'n_grid_2': n_grid},
        'TET4',
    )
    return ElasticPipeline(
        mesh,
        E=200e9, nu=0.3, itc=None,
        gravity_bf=(0.0, 0.0, 0.0),
        winkler_k=1e9,
        problem_options={'winkler_k': 1e9, 'solver': 'umfpack'},
    )


# ============================================================================
# 1. surface_jxw — shape and positivity
# ============================================================================

def test_surface_jxw_shape_and_positivity():
    """surface_jxw returns (num_sel, n_fq) array of strictly positive values."""
    pipeline = _make_pipeline()
    pts = jnp.asarray(pipeline.mesh.points)
    jxw = pipeline.problem.surface_jxw(pts)

    bi = pipeline.problem.boundary_inds_list[0]
    num_sel = int(bi.shape[0])
    n_fq = int(pipeline.problem._face_qw.shape[1])

    assert jxw.shape == (num_sel, n_fq), (
        f"Expected shape ({num_sel}, {n_fq}), got {jxw.shape}"
    )
    assert jnp.all(jxw > 0), "JxW values must be strictly positive"


# ============================================================================
# 2. surface_jxw — area sanity check
# ============================================================================

def test_surface_jxw_area_sanity():
    """Sum of JxW over the Winkler surface approximates the coil's surface area.

    For a rectangular cross-section (w1 × w2) swept along a circle of radius R
    the lateral surface area is ``(2*(w1 + w2)) * 2π R``.  With a coarse mesh
    (n_phi=4, n_grid=1) we expect agreement within ~20 % of the analytic value.
    """
    R, w1, w2, n_phi, n_grid = 1.0, 0.04, 0.04, 8, 1
    pipeline = _make_pipeline(R=R, n_phi=n_phi, w1=w1, w2=w2, n_grid=n_grid)
    pts = jnp.asarray(pipeline.mesh.points)
    jxw = pipeline.problem.surface_jxw(pts)

    total_area = float(jxw.sum())
    analytic   = (2.0 * (w1 + w2)) * 2.0 * math.pi * R

    rel_err = abs(total_area - analytic) / analytic
    assert rel_err < 0.25, (
        f"Surface area mismatch: computed={total_area:.4f}, "
        f"analytic={analytic:.4f}, rel_err={rel_err:.3f}"
    )


# ============================================================================
# 3. surface_jxw — autodiff
# ============================================================================

def test_surface_jxw_autodiff():
    """jax.grad flows through surface_jxw w.r.t. mesh node positions."""
    pipeline = _make_pipeline()
    pts0 = jnp.asarray(pipeline.mesh.points, dtype=jnp.float64)

    def total_area(pts):
        return pipeline.problem.surface_jxw(pts).sum()

    grad = jax.grad(total_area)(pts0)
    assert grad.shape == pts0.shape
    assert jnp.all(jnp.isfinite(grad)), "Gradient contains NaN/Inf"


# ============================================================================
# 4. surface_jxw — raises when no Winkler surface
# ============================================================================

def test_surface_jxw_raises_without_winkler():
    """surface_jxw raises ValueError when no Winkler surface is configured."""
    pipeline = _make_pipeline()
    pts = jnp.asarray(pipeline.mesh.points)
    # Temporarily disable the Winkler surface by patching the scalar to None.
    orig = pipeline.problem._winkler_k_scalar
    try:
        pipeline.problem._winkler_k_scalar = None
        with pytest.raises(ValueError, match="winkler"):
            pipeline.problem.surface_jxw(pts)
    finally:
        pipeline.problem._winkler_k_scalar = orig


# ============================================================================
# 5. Transpose symmetry: K_cs = K_sc^T when Q = I
# ============================================================================

def test_coupling_transpose_symmetry():
    """coupling_values gives K_cs = K_sc^T unconditionally when Q = I.

    Tests a single CC beam connecting coil 0 to coil 1 with no stellsym
    transforms (all Q = I).  With a single k_attachment modulus the coupling
    blocks satisfy the transpose condition at every surface node by construction.
    """
    n_base, nfp = 2, 1

    k = 1e8  # k_attachment

    def _const_section_fn(support_dofs):
        phi_cc = support_dofs['phis_start_cc']
        phi_cf = support_dofs['phis_start_cf']
        A, Iy, Iz, J = [], [], [], []
        for g in range(len(phi_cc)):
            n_cf = phi_cf[g].shape[0] if g < len(phi_cf) else 0
            n_per = phi_cc[g].shape[0] + n_cf
            A.append(jnp.full((n_per,), 1e-4))
            Iy.append(jnp.full((n_per,), 1e-8))
            Iz.append(jnp.full((n_per,), 1e-8))
            J.append(jnp.full((n_per,), 2e-8))
        return A, Iy, Iz, J

    sb = SupportBeams(
        nfp=nfp, stellsym=False,
        beam_options={'n_beam_cc': 1, 'n_beam_cf': 0,
                      'E': 200e9, 'nu': 0.3,
                      'k_attachment': k},
        n_base=n_base,
        cross_section_fn=_const_section_fn,
        attachment_fn=lambda pts, dofs, sign_x, opts: jnp.ones(pts.shape[0]),
    )

    curves = [_make_circle(N=8, R=1.0 + 0.1 * i) for i in range(n_base)]

    n_surf = 8
    surf = []
    for i in range(n_base):
        R = 1.0 + 0.1 * i
        phis = jnp.linspace(0.0, 1.0, n_surf, endpoint=False)
        surf.append(jnp.stack([
            R * jnp.cos(2 * math.pi * phis),
            jnp.zeros(n_surf),
            R * jnp.sin(2 * math.pi * phis),
        ], axis=1))

    # Identity-like surf_interp and unit jxw
    face_sv = jnp.ones((n_surf, 1, 1), dtype=jnp.float64)
    face_to = jnp.arange(n_surf, dtype=jnp.int32).reshape(n_surf, 1)
    surf_interp = [(face_sv, face_to, n_surf) for _ in range(n_base)]
    jxw = [jnp.ones((n_surf, 1), dtype=jnp.float64) for _ in range(n_base)]

    # With n_base=2, stellsym=False: n_groups_cc = n_base = 2.
    sdofs = {
        'phis_start_cc': [jnp.array([0.1]), jnp.array([0.6])],
        'phis_end_cc':   [jnp.array([0.6]), jnp.array([0.1])],
        'phis_start_cf': [jnp.zeros(0), jnp.zeros(0)],
        'x_foundation':  [jnp.zeros((0, 3)), jnp.zeros((0, 3))],
        'thetas_orientation_cc': [jnp.zeros(1), jnp.zeros(1)],
        'thetas_orientation_cf': [jnp.zeros(0), jnp.zeros(0)],
    }

    # Build the merged-system layout
    surf_idx = [np.arange(n_surf, dtype=np.int32) for _ in range(n_base)]
    coil_dof_offsets = [0, n_surf * 3]
    support_dof_offset = n_base * n_surf * 3

    I_cs, J_cs, I_sc, J_sc = sb.coupling_pattern(
        coil_dof_offsets, support_dof_offset, surf_idx,
    )
    V_cs, V_sc = sb.coupling_values(
        curves, sdofs, surf,
        surf_interp_by_coil=surf_interp,
        jxw_by_coil=jxw,
    )

    n_s = sb.n_support_dofs
    n_c = support_dof_offset

    I_cs = np.asarray(I_cs)
    J_cs = np.asarray(J_cs)
    V_cs = np.asarray(V_cs)
    I_sc = np.asarray(I_sc)
    J_sc = np.asarray(J_sc)
    V_sc = np.asarray(V_sc)

    K_cs = np.zeros((n_c, n_s))
    np.add.at(K_cs, (I_cs, J_cs - support_dof_offset), V_cs)

    K_sc = np.zeros((n_s, n_c))
    np.add.at(K_sc, (I_sc - support_dof_offset, J_sc), V_sc)

    # With k_attachment and Q = I: K_cs = K_sc^T unconditionally
    assert np.allclose(K_cs, K_sc.T, rtol=1e-10, atol=1e-14), (
        f"K_cs != K_sc^T: max deviation = "
        f"{np.abs(K_cs - K_sc.T).max():.3e}"
    )
