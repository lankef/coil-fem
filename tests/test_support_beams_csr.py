"""Tests for SupportBeamsCSR (central support ring + CR beams).

Covers the seam-reduction projection, the nfp-periodic rigid nullspace,
zero body-force RHS, Taylor tests on ``v_end_cr`` / ``csr_curve_dofs``, and
a stellsym seam-symmetry canary (GPU-gated when a full solve is required).
"""

from __future__ import annotations

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from coil_fem.geo import CurveXYZFourierJAX, make_centroid_frame
from coil_fem.meshing import FramedCurveMeshRectangle
from coil_fem.pipelines import ElasticPipeline
from coil_fem.coupling import SupportBeamsCSR

_HAS_SPINEAX = importlib.util.find_spec("spineax") is not None
_HAS_GPU = any(d.platform == "gpu" for d in jax.devices())
_GPU_REASON = "requires spineax + a CUDA device"


# ============================================================================
# Fixtures / helpers
# ============================================================================

def _make_circle(N: int = 8, R: float = 1.2) -> CurveXYZFourierJAX:
    """Planar circle of radius R in the xz-plane."""
    qp = jnp.linspace(0.0, 1.0, N, endpoint=False)
    dofs = jnp.array([0.0, R, 0.0,
                      0.0, 0.0, 0.0,
                      0.0, 0.0, R])
    return CurveXYZFourierJAX(qp, dofs, order=1)


def _section_fn_with_cr(A_val=1e-4, Iy_val=1e-8, Iz_val=1e-8, J_val=2e-8):
    """Ragged cross_section_fn including CR beams in each base-coil group."""

    def fn(support_dofs):
        phi_cc = support_dofs['phis_start_cc']
        phi_cf = support_dofs['phis_start_cf']
        phi_cr = support_dofs.get('phis_start_cr', [jnp.zeros(0)] * len(phi_cc))
        A, Iy, Iz, J = [], [], [], []
        for g in range(len(phi_cc)):
            n_cf = phi_cf[g].shape[0] if g < len(phi_cf) else 0
            n_cr = phi_cr[g].shape[0] if g < len(phi_cr) else 0
            n_per = phi_cc[g].shape[0] + n_cf + n_cr
            A.append(jnp.full((n_per,), A_val))
            Iy.append(jnp.full((n_per,), Iy_val))
            Iz.append(jnp.full((n_per,), Iz_val))
            J.append(jnp.full((n_per,), J_val))
        return A, Iy, Iz, J

    return fn


def _uniform_clamp(surface_pts_beam_frame, dofs, sign_x, constants):
    """Attachment weight for beam endpoints (SupportBeams.attachment_fn)."""
    return jnp.ones(surface_pts_beam_frame.shape[0])


def _ground_clamp_fn(surface_pts, curve, dofs):
    """Uniform grounded Winkler weight for coil surfaces (fixed_clamp_fns)."""
    return jnp.ones(surface_pts.shape[0])


def _csr_options(nfp: int = 2, n_phi: int = 4, order: int = 1):
    return {
        'order': order,
        'w1': 0.08,
        'w2': 0.08,
        'n_phi': n_phi,
        'n_grid_1': 1,
        'n_grid_2': 1,
        'E': 200e9,
        'nu': 0.3,
    }


def _make_csr(
    *,
    nfp: int = 2,
    stellsym: bool = False,
    n_beam_cr: int = 0,
    n_beam_cc: int = 0,
    n_beam_cf: int = 0,
    n_base: int = 1,
    k_attachment: float = 1e8,
    k_clamp: float = 0.0,
    n_phi: int = 4,
) -> SupportBeamsCSR:
    beam_options = {
        'n_beam_cc': n_beam_cc,
        'n_beam_cf': n_beam_cf,
        'n_beam_cr': n_beam_cr,
        'E': 200e9,
        'nu': 0.3,
        'k_attachment': k_attachment,
    }
    fco = {'k_clamp': k_clamp} if k_clamp else None
    return SupportBeamsCSR(
        nfp=nfp,
        stellsym=stellsym,
        beam_options=beam_options,
        n_base=n_base,
        cross_section_fn=_section_fn_with_cr(),
        attachment_fn=_uniform_clamp,
        csr_options=_csr_options(nfp=nfp, n_phi=n_phi),
        problem_options={'solver': 'umfpack'},
        fixed_clamp_options=fco,
    )


def _csr_curve_dofs(sb: SupportBeamsCSR, R: float = 1.0) -> jax.Array:
    dofs = jnp.zeros(sb._csr_curve_template.dofs.shape)
    return dofs.at[0].set(R)


def _empty_cc_cf(n_base: int):
    z0 = [jnp.zeros(0) for _ in range(n_base)]
    return {
        'phis_start_cc': list(z0),
        'phis_end_cc': list(z0),
        'phis_start_cf': list(z0),
        'x_foundation': [jnp.zeros((0, 3)) for _ in range(n_base)],
        'thetas_orientation_cc': list(z0),
        'thetas_orientation_cf': list(z0),
    }


def _support_dofs_cr(
    sb: SupportBeamsCSR,
    *,
    phi_start: float = 0.25,
    phi_end: float = 0.1,
    v_end: float = 0.0,
    R_csr: float = 1.0,
):
    n_base = sb.n_base
    sd = _empty_cc_cf(n_base)
    n_cr = sb._n_beam_cr
    sd['phis_start_cr'] = [
        jnp.full((n_cr[i],), phi_start) for i in range(n_base)
    ]
    sd['phis_end_cr'] = [
        jnp.full((n_cr[i],), phi_end) for i in range(n_base)
    ]
    sd['v_end_cr'] = [
        jnp.full((n_cr[i],), v_end) for i in range(n_base)
    ]
    sd['thetas_orientation_cr'] = [
        jnp.zeros((n_cr[i],)) for i in range(n_base)
    ]
    sd['csr_curve_dofs'] = _csr_curve_dofs(sb, R=R_csr)
    return sd


def _bind_one_coil(sb: SupportBeamsCSR, R: float = 1.2):
    curve = _make_circle(N=8, R=R)
    fc = make_centroid_frame(curve)
    mesh = FramedCurveMeshRectangle(
        fc, 0.15, 0.15, n_grid_1=1, n_grid_2=1,
    )
    sb.bind_coil_meshes([mesh])
    pipe = ElasticPipeline(
        mesh, 200e9, 0.3, None, (0.0, 0.0, 0.0),
        {'solver': 'umfpack'},
    )
    pts = mesh.mesh_points_from_dofs(curve.dofs)
    surf = pipe.surface_quad_points(pts)
    jxw = pipe.problem.surface_jxw(pts)
    return [curve], [surf], [jxw], pts, mesh, pipe


def _dense_from_coo(I, J, V, n):
    K = np.zeros((n, n), dtype=np.float64)
    np.add.at(K, (np.asarray(I), np.asarray(J)), np.asarray(V, dtype=np.float64))
    return K


def _projection_T(sb: SupportBeamsCSR) -> np.ndarray:
    """Full→reduced displacement map: u_full = T @ u_red."""
    n_nodes = sb.csr_mesh.points.shape[0]
    n_full = 3 * n_nodes
    n_red = sb._n_csr_dofs
    T = np.zeros((n_full, n_red), dtype=np.float64)
    for i in range(n_nodes):
        r = int(sb._csr_red_node[i])
        Q = sb._csr_node_Q[i]
        T[3 * i:3 * i + 3, 3 * r:3 * r + 3] = Q
    return T


def _reduced_ring_K(sb: SupportBeamsCSR, csr_pts, support_k=None) -> np.ndarray:
    """Dense reduced CSR stiffness (no ``w_sym``), zero attachments by default."""
    pipe = sb._csr_pipeline
    n_cells = pipe.problem.fes[0].num_cells
    n_quads = pipe.problem.fes[0].num_quads
    if support_k is None:
        support_k = jnp.zeros(pipe.n_surface_quads)
    _, _, V_full, _, _ = pipe.assemble_coo({
        'points': csr_pts,
        'body_force': jnp.zeros((n_cells, n_quads, 3)),
        'support_k': support_k,
    })
    V_full = np.asarray(V_full, dtype=np.float64)
    I_full = np.asarray(pipe.problem.I)
    J_full = np.asarray(pipe.problem.J)
    n_full = 3 * sb.csr_mesh.points.shape[0]
    K = _dense_from_coo(I_full, J_full, V_full, n_full)
    T = _projection_T(sb)
    return T.T @ K @ T, V_full


# ============================================================================
# 1. Reduction self-consistency
# ============================================================================

def test_csr_seam_pairing_bijection():
    sb = _make_csr(nfp=2, n_beam_cr=0, n_phi=4)
    far = np.where(sb.csr_mesh.phi_idx_per_node == sb.csr_mesh.phi_idx_per_node.max())[0]
    free = np.where(~sb._csr_is_slave)[0]
    assert sb._n_csr_dofs == 3 * free.shape[0]
    # Far→near via red_node is injective into the free-node reduced set.
    far_red = {int(sb._csr_red_node[i]) for i in far}
    free_red = {int(sb._csr_red_node[i]) for i in free}
    assert len(far_red) == far.shape[0]
    assert far_red.issubset(free_red)


def test_csr_reduction_matches_dense_projection():
    sb = _make_csr(nfp=2, n_beam_cr=0, n_phi=4)
    csr_pts = sb.csr_mesh.mesh_points_from_dofs(_csr_curve_dofs(sb))
    K_ref, V_full = _reduced_ring_K(sb, csr_pts)

    V_plain = V_full[sb._csr_plain_idx]
    V_exp = (V_full[sb._csr_exp_idx][:, None, None] * sb._csr_W).reshape(-1)
    V_red = np.concatenate([V_plain, V_exp])
    K_impl = _dense_from_coo(sb._csr_I, sb._csr_J, V_red, sb._n_csr_dofs)

    err = np.max(np.abs(K_ref - K_impl))
    scale = max(np.max(np.abs(K_ref)), 1.0)
    assert err / scale < 1e-10, f"rel err={err / scale}, abs={err}"


# ============================================================================
# 2. Rigid-motion nullspace (nfp-periodic)
# ============================================================================

def test_csr_reduced_nullspace_z_rigid_only():
    """With no attachments the reduced ring admits only z-translation / z-rotation."""
    sb = _make_csr(nfp=2, n_beam_cr=0, n_phi=6, k_attachment=0.0)
    csr_pts = np.asarray(
        sb.csr_mesh.mesh_points_from_dofs(_csr_curve_dofs(sb)),
        dtype=np.float64,
    )
    K_red, _ = _reduced_ring_K(sb, jnp.asarray(csr_pts))

    free = np.where(~sb._csr_is_slave)[0]
    pts_free = csr_pts[free]

    u_tz = np.zeros(sb._n_csr_dofs)
    u_tz[2::3] = 1.0

    u_rz = np.zeros(sb._n_csr_dofs)
    u_rz[0::3] = -pts_free[:, 1]
    u_rz[1::3] = pts_free[:, 0]

    u_tx = np.zeros(sb._n_csr_dofs)
    u_tx[0::3] = 1.0
    u_ty = np.zeros(sb._n_csr_dofs)
    u_ty[1::3] = 1.0

    def _rayleigh(u):
        return float(u @ (K_red @ u) / max(u @ u, 1e-30))

    r_tz, r_rz = _rayleigh(u_tz), _rayleigh(u_rz)
    r_tx, r_ty = _rayleigh(u_tx), _rayleigh(u_ty)
    # Null modes should be many orders below the stiff x/y translations.
    assert r_tz < 1e-6 * max(r_tx, r_ty, 1.0)
    assert r_rz < 1e-6 * max(r_tx, r_ty, 1.0)
    assert r_tx > 1.0 and r_ty > 1.0



# ============================================================================
# 3. Zero support RHS
# ============================================================================

def test_csr_assemble_coo_zero_body_force_rhs():
    sb = _make_csr(nfp=2, n_beam_cr=0, n_phi=4)
    csr_pts = sb.csr_mesh.mesh_points_from_dofs(_csr_curve_dofs(sb))
    pipe = sb._csr_pipeline
    n_cells = pipe.problem.fes[0].num_cells
    n_quads = pipe.problem.fes[0].num_quads
    _, _, _, _, load = pipe.assemble_coo({
        'points': csr_pts,
        'body_force': jnp.zeros((n_cells, n_quads, 3)),
        'support_k': jnp.zeros(pipe.n_surface_quads),
    })
    assert float(jnp.max(jnp.abs(load))) == 0.0


# ============================================================================
# 4. Taylor tests (CPU: energy through support_values; GPU: objective)
# ============================================================================

def test_taylor_v_end_cr_and_csr_curve_dofs_via_support_energy():
    """Analytic ∂J/∂v_end_cr and ∂J/∂R_csr match centered FD on Σ V²."""
    sb = _make_csr(nfp=2, stellsym=False, n_beam_cr=1, n_beam_cc=0, n_beam_cf=0)
    curves, surf, jxw, _, _, _ = _bind_one_coil(sb, R=1.2)
    sd0 = _support_dofs_cr(sb, phi_start=0.3, phi_end=0.12, v_end=0.1, R_csr=1.0)

    def J_of(v_end, R_csr):
        sd = {
            **sd0,
            'v_end_cr': [jnp.array([v_end])],
            'csr_curve_dofs': _csr_curve_dofs(sb, R=R_csr),
        }
        geom = sb.beam_geometry(curves, sd)
        V = sb.support_values(
            curves, sd, surf, geom=geom, jxw_by_coil=jxw,
        )
        return jnp.sum(V * V)

    v0 = float(sd0['v_end_cr'][0][0])
    R0 = float(sd0['csr_curve_dofs'][0])

    g_v, g_R = jax.grad(J_of, argnums=(0, 1))(v0, R0)
    eps = 1e-5
    fd_v = (float(J_of(v0 + eps, R0)) - float(J_of(v0 - eps, R0))) / (2 * eps)
    fd_R = (float(J_of(v0, R0 + eps)) - float(J_of(v0, R0 - eps))) / (2 * eps)

    for name, analytic, fd in (('v_end', float(g_v), fd_v), ('R', float(g_R), fd_R)):
        scale = max(abs(fd), abs(analytic), 1.0)
        assert abs(analytic - fd) / scale < 5e-2, (
            f"{name}: analytic={analytic!r}, fd={fd!r}"
        )


@pytest.mark.skipif(not (_HAS_SPINEAX and _HAS_GPU), reason=_GPU_REASON)
def test_taylor_objective_v_end_cr_and_csr_curve_dofs():
    """Full CoilFEM objective gradients vs FD (monolithic / cuDSS).

    Skips on GPU OOM; the CPU energy Taylor above covers the same DOFs.
    """
    from coil_fem import CoilFEM

    jax.clear_caches()
    n_base = 1
    try:
        curves = [_make_circle(N=4, R=1.2)]
        support = SupportBeamsCSR(
            nfp=2, stellsym=False,
            beam_options={
                'n_beam_cc': 0, 'n_beam_cf': 0, 'n_beam_cr': 1,
                'E': 200e9, 'nu': 0.3, 'k_attachment': 1e10,
            },
            n_base=n_base,
            cross_section_fn=_section_fn_with_cr(),
            attachment_fn=_uniform_clamp,
            csr_options=_csr_options(nfp=2, n_phi=3),
            problem_options={'solver': 'cudss'},
            fixed_clamp_fns=[_ground_clamp_fn],
            fixed_clamp_options={'k_clamp': 1e9},
        )
        fem = CoilFEM(
            base_curves_jax=curves,
            base_currents_jax=jnp.ones(n_base),
            nfp=2,
            stellsym=False,
            mesh_options={
                'shape': 'rect', 'w1': 0.05, 'w2': 0.05,
                'n_grid_1': 1, 'n_grid_2': 1,
            },
            support=support,
            material_options={'E': 200e9, 'nu': 0.3, 'density': 8900.0},
            problem_options={'solver': 'cudss'},
            coupling='monolithic',
        )
        sd0 = _support_dofs_cr(
            support, phi_start=0.3, phi_end=0.12, v_end=0.05, R_csr=0.9,
        )
        cdofs = [c.dofs for c in curves]
        idofs = jnp.ones(n_base)

        def J_of(v_end, R_csr):
            sd = {
                **sd0,
                'v_end_cr': [jnp.array([v_end])],
                'csr_curve_dofs': _csr_curve_dofs(support, R=R_csr),
            }
            out = fem.objective(cdofs, idofs, sd, metrics=('l2_von_mises',))
            return out['l2_von_mises']

        v0 = float(sd0['v_end_cr'][0][0])
        R0 = float(sd0['csr_curve_dofs'][0])
        g_v, g_R = jax.grad(J_of, argnums=(0, 1))(v0, R0)
        eps = 1e-5
        fd_v = (float(J_of(v0 + eps, R0)) - float(J_of(v0 - eps, R0))) / (2 * eps)
        fd_R = (float(J_of(v0, R0 + eps)) - float(J_of(v0, R0 - eps))) / (2 * eps)
    except Exception as exc:  # noqa: BLE001
        if 'OUT_OF_MEMORY' in str(exc) or 'RESOURCE_EXHAUSTED' in str(exc):
            pytest.skip(f"GPU OOM during CSR objective Taylor: {exc}")
        raise

    for name, analytic, fd in (('v_end', float(g_v), fd_v), ('R', float(g_R), fd_R)):
        assert np.isfinite(analytic) and np.isfinite(fd)
        scale = max(abs(fd), abs(analytic), 1.0)
        assert abs(analytic - fd) / scale < 5e-2, (
            f"{name}: analytic={analytic!r}, fd={fd!r}"
        )


# ============================================================================
# 5. Symmetry / seam canaries
# ============================================================================

def test_csr_displacement_enforces_periodic_tie():
    """``continuum_members[0].solution`` always expands slaves as ``Q @ u_near``."""
    sb = _make_csr(nfp=2, n_beam_cr=0, n_phi=4)
    rng = np.random.default_rng(0)
    u_s = jnp.zeros(sb.n_support_dofs)
    u_red = jnp.asarray(rng.normal(size=(sb._n_csr_dofs,)))
    u_s = u_s.at[sb._csr_dof_offset:].set(u_red)
    member = sb.continuum_members[0]
    u = np.asarray(member.solution(u_s, {})[0])
    phi_idx = sb.csr_mesh.phi_idx_per_node
    near = np.where(phi_idx == 0)[0]
    far = np.where(phi_idx == phi_idx.max())[0]
    Q = sb._csr_Q_period
    for i_far in far:
        r = int(sb._csr_red_node[i_far])
        masters = [i for i in near if int(sb._csr_red_node[i]) == r]
        assert len(masters) == 1
        np.testing.assert_allclose(u[i_far], Q @ u[masters[0]], atol=1e-12)


def test_w_sym_and_flip_image_specs():
    """``w_sym`` and the stellsym flip_half ring attachment are present."""
    sb0 = _make_csr(nfp=2, stellsym=False, n_beam_cr=1)
    sb1 = _make_csr(nfp=2, stellsym=True, n_beam_cr=1)
    assert sb0._w_sym == pytest.approx(1.0)
    assert sb1._w_sym == pytest.approx(0.5)

    curves0, _, _, _, _, _ = _bind_one_coil(sb0)
    sd0 = _support_dofs_cr(sb0)
    geom0 = sb0.beam_geometry(curves0, sd0)
    specs0 = sb0._csr_endpoint_specs(geom0)
    assert len(specs0) == 1 and specs0[0].tfm == 'none'

    # stellsym wrap group needs empty CC slots in support_dofs for geometry.
    n_wrap = sb1.n_beam_cc[sb1.n_base]
    curves1 = [_make_circle(N=8, R=1.2)]
    fc = make_centroid_frame(curves1[0])
    mesh = FramedCurveMeshRectangle(fc, 0.15, 0.15, n_grid_1=1, n_grid_2=1)
    sb1.bind_coil_meshes([mesh])
    sd1 = _support_dofs_cr(sb1)
    sd1['phis_start_cc'] = [jnp.zeros(0), jnp.zeros(n_wrap)]
    sd1['phis_end_cc'] = [jnp.zeros(0), jnp.zeros(n_wrap)]
    sd1['thetas_orientation_cc'] = [jnp.zeros(0), jnp.zeros(n_wrap)]
    geom1 = sb1.beam_geometry(curves1, sd1)
    specs1 = sb1._csr_endpoint_specs(geom1)
    assert len(specs1) == 2
    assert {s.tfm for s in specs1} == {'none', 'flip_half'}


def test_w_sym_scales_reduced_ring_values():
    """Ring COO values from ``support_values`` carry the ``w_sym`` factor."""
    sb = _make_csr(
        nfp=2, stellsym=True, n_beam_cr=1, n_phi=4, k_attachment=0.0,
    )
    assert sb._w_sym == pytest.approx(0.5)
    csr_pts = sb.csr_mesh.mesh_points_from_dofs(_csr_curve_dofs(sb))
    pipe = sb._csr_pipeline
    n_cells = pipe.problem.fes[0].num_cells
    n_quads = pipe.problem.fes[0].num_quads
    _, _, V_full, _, _ = pipe.assemble_coo({
        'points': csr_pts,
        'body_force': jnp.zeros((n_cells, n_quads, 3)),
        'support_k': jnp.zeros(pipe.n_surface_quads),
    })
    V_full = np.asarray(V_full)
    V_plain = V_full[sb._csr_plain_idx]
    V_exp = (V_full[sb._csr_exp_idx][:, None, None] * sb._csr_W).reshape(-1)
    V_unscaled = np.concatenate([V_plain, V_exp])

    curves, surf, jxw, _, _, _ = _bind_one_coil(sb)
    sd = _support_dofs_cr(sb)
    n_wrap = sb.n_beam_cc[sb.n_base]
    sd['phis_start_cc'] = [jnp.zeros(0), jnp.zeros(n_wrap)]
    sd['phis_end_cc'] = [jnp.zeros(0), jnp.zeros(n_wrap)]
    sd['thetas_orientation_cc'] = [jnp.zeros(0), jnp.zeros(n_wrap)]
    geom = sb.beam_geometry(curves, sd)
    V = np.asarray(sb.support_values(
        curves, sd, surf, geom=geom, jxw_by_coil=jxw,
    ))
    n_beam = sb._support_I.shape[0]
    n_csr = sb._csr_I.shape[0]
    V_csr = V[n_beam:n_beam + n_csr]
    # assemble_coo may run in float32 on GPU; compare at assembly precision.
    scale = max(float(np.max(np.abs(V_unscaled))), 1.0)
    np.testing.assert_allclose(
        V_csr, V_unscaled * sb._w_sym, rtol=1e-5, atol=1e-5 * scale,
    )


@pytest.mark.skipif(not (_HAS_SPINEAX and _HAS_GPU), reason=_GPU_REASON)
def test_monolithic_csr_forward_finite_and_seam():
    """Tiny monolithic CSR solve stays finite; displacement respects the seam.

    Also peeks at cuDSS inertia when the handle exposes it.  Skips on GPU OOM
    (CSR + coil pipelines are memory-heavy on shared devices).
    """
    from coil_fem import CoilFEM

    jax.clear_caches()
    try:
        curves = [_make_circle(N=4, R=1.2)]
        support = SupportBeamsCSR(
            nfp=2, stellsym=True,
            beam_options={
                'n_beam_cc': 0, 'n_beam_cf': 0, 'n_beam_cr': 1,
                'E': 200e9, 'nu': 0.3, 'k_attachment': 1e10,
            },
            n_base=1,
            cross_section_fn=_section_fn_with_cr(),
            attachment_fn=_uniform_clamp,
            csr_options=_csr_options(nfp=2, n_phi=3),
            problem_options={'solver': 'cudss'},
            fixed_clamp_fns=[_ground_clamp_fn],
            fixed_clamp_options={'k_clamp': 1e9},
        )
        fem = CoilFEM(
            base_curves_jax=curves,
            base_currents_jax=jnp.ones(1),
            nfp=2, stellsym=True,
            mesh_options={
                'shape': 'rect', 'w1': 0.05, 'w2': 0.05,
                'n_grid_1': 1, 'n_grid_2': 1,
            },
            support=support,
            material_options={'E': 200e9, 'nu': 0.3, 'density': 8900.0},
            problem_options={'solver': 'cudss'},
            coupling='monolithic',
        )
        n_wrap = support.n_beam_cc[support.n_base]
        sd = _support_dofs_cr(support, phi_start=0.25, phi_end=0.1, R_csr=0.9)
        sd['phis_start_cc'] = [jnp.zeros(0), jnp.zeros(n_wrap)]
        sd['phis_end_cc'] = [jnp.zeros(0), jnp.zeros(n_wrap)]
        sd['thetas_orientation_cc'] = [jnp.zeros(0), jnp.zeros(n_wrap)]
        out = fem.run(
            base_curves_dofs=[c.dofs for c in curves],
            base_currents_dofs=jnp.ones(1),
            base_support_dofs=sd,
        )
    except Exception as exc:  # noqa: BLE001 — OOM is environmental
        if 'OUT_OF_MEMORY' in str(exc) or 'RESOURCE_EXHAUSTED' in str(exc):
            pytest.skip(f"GPU OOM during CSR monolithic solve: {exc}")
        raise

    assert jnp.all(jnp.isfinite(out['u_s']))
    member = support.continuum_members[0]
    u = np.asarray(member.solution(out['u_s'], sd)[0])
    phi_idx = support.csr_mesh.phi_idx_per_node
    near = np.where(phi_idx == 0)[0]
    far = np.where(phi_idx == phi_idx.max())[0]
    Q = support._csr_Q_period
    err = 0.0
    for i_far in far:
        r = int(support._csr_red_node[i_far])
        masters = [i for i in near if int(support._csr_red_node[i]) == r]
        assert len(masters) == 1
        err = max(err, float(np.linalg.norm(u[i_far] - Q @ u[masters[0]])))
    u_scale = max(float(np.max(np.abs(u))), 1e-12)
    assert err / u_scale < 1e-6, f"seam err={err}, scale={u_scale}"

    assert len(out['support_continuum']) == 1
    assert out['support_continuum'][0]['name'] == 'csr'
    assert out['support_continuum'][0]['von_mises'].shape[0] > 0

    static = fem.monolithic_static
    inertia = getattr(static, 'inertia', None)
    if inertia is None and getattr(static, 'solver', None) is not None:
        inertia = getattr(static.solver, 'inertia', None)
    if inertia is not None:
        n_neg, n_zero, _n_pos = inertia
        assert n_neg == 0 and n_zero == 0, f"inertia={inertia}"


# ============================================================================
# Layout / mesh smoke
# ============================================================================

def test_n_support_dofs_and_beam_labels():
    sb = _make_csr(nfp=2, n_beam_cr=2, n_beam_cc=1, n_beam_cf=1, n_base=1)
    assert sb.n_support_dofs == 12 * sb.n_beams_total + sb._n_csr_dofs
    coil_idx, beam_type = sb.beam_labels()
    assert coil_idx.shape == (sb.n_beams_total,)
    assert set(np.unique(beam_type)).issubset({0, 1, 2})
    assert np.sum(beam_type == 2) == 2


def test_open_phi_span_mesh_has_free_ends():
    """Sector sweep exposes two distinct end faces (K+1 phi slices)."""
    sb = _make_csr(nfp=3, n_phi=5)
    assert sb.csr_mesh.phi_span == pytest.approx(1.0 / 3.0)
    assert int(sb.csr_mesh.phi_idx_per_node.max()) == 5
    near = np.sum(sb.csr_mesh.phi_idx_per_node == 0)
    far = np.sum(sb.csr_mesh.phi_idx_per_node == 5)
    assert near == far > 0


# ============================================================================
# Continuum members
# ============================================================================

def test_continuum_members_shape():
    """SupportBeamsCSR publishes one CSR continuum member."""
    sb = _make_csr(nfp=2, stellsym=True, n_beam_cr=1)
    members = sb.continuum_members
    assert len(members) == 1
    m = members[0]
    assert m.name == 'csr'
    assert m.pipeline is sb._csr_pipeline
    assert m.sym_weight == pytest.approx(2 * (1 + 1))


def test_continuum_members_empty_for_base_support():
    """Grounded Support and beam-only SupportBeams expose no continuum."""
    from coil_fem.coupling import Support, SupportBeams

    assert Support(k_clamp=1.0).continuum_members == ()

    beams = SupportBeams(
        nfp=2,
        stellsym=False,
        beam_options={
            'n_beam_cc': 0, 'n_beam_cf': 1,
            'E': 200e9, 'nu': 0.3, 'k_attachment': 1e8,
        },
        n_base=1,
        cross_section_fn=lambda sd: (
            [jnp.full((1,), 1e-4)],
            [jnp.full((1,), 1e-8)],
            [jnp.full((1,), 1e-8)],
            [jnp.full((1,), 2e-8)],
        ),
        attachment_fn=_uniform_clamp,
        fixed_clamp_options={'k_clamp': 1e8},
    )
    assert beams.continuum_members == ()


def test_csr_l2_von_mises_taylor_csr_dofs():
    """∂(sym_weight · l2_von_mises)/∂R_csr is finite via continuum_members."""
    from coil_fem.metrics import l2_von_mises
    from coil_fem.problems import recompute_fe_geometry

    sb = _make_csr(nfp=2, stellsym=False, n_beam_cr=1, n_phi=4)
    member = sb.continuum_members[0]
    sd0 = _support_dofs_cr(sb, phi_start=0.3, phi_end=0.12, v_end=0.0, R_csr=1.0)

    rng = np.random.default_rng(1)
    u_s = jnp.zeros(sb.n_support_dofs)
    u_s = u_s.at[sb._csr_dof_offset:].set(
        jnp.asarray(1e-4 * rng.normal(size=(sb._n_csr_dofs,)))
    )

    def J_of(R):
        sd = {**sd0, 'csr_curve_dofs': _csr_curve_dofs(sb, R=R)}
        pts = member.mesh_points(sd)
        sol = member.solution(u_s, sd)
        prob = member.pipeline.problem
        sg, jxw, _, _ = recompute_fe_geometry(
            pts, prob._cells_jnp, prob._sg_ref, prob._sv, prob._qw,
        )
        val = l2_von_mises(
            prob, sol, member.pipeline.lam, member.pipeline.mu,
            shape_grads=sg, JxW=jxw,
        )
        return member.sym_weight * val

    R0 = 1.0
    g = float(jax.grad(J_of)(R0))
    eps = 1e-5
    fd = (float(J_of(R0 + eps)) - float(J_of(R0 - eps))) / (2 * eps)
    assert np.isfinite(g) and np.isfinite(fd)
    assert abs(g) > 0.0 or abs(fd) > 0.0
    scale = max(abs(g), abs(fd), 1.0)
    assert abs(g - fd) / scale < 5e-2, f"analytic={g!r}, fd={fd!r}"
