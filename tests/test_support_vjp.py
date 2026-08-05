"""Forward/adjoint consistency checks for the SupportBeams Winkler coupling.

Covers the invariants that must hold when the grounded Winkler stiffness
``k = k_clamp*w_g + k_attachment*w_a`` also drives ``K_cs``/``K_sc``/``K_ss``:
the assembled ``K`` matches the residual Jacobian, the per-block residual VJPs
match finite differences at a frozen solution, and ``k(phi)`` differentiates
correctly outside the ``custom_vjp``.  Also covers the simsopt-free Taylor JIT
mask fix (concrete integer indices).

GPU-gated.  Prints numeric tables.
"""

from __future__ import annotations

import importlib.util
import math

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from coil_fem.coupling import SupportBeams
from coil_fem.geo import CurveXYZFourierJAX, make_framed_curve
from coil_fem.meshing import CoilMesh

_HAS_SPINEAX = importlib.util.find_spec("spineax") is not None
_HAS_GPU = any(d.platform == "gpu" for d in jax.devices())
_GPU_REASON = "requires spineax + a CUDA device"


def _make_circle(N: int = 4, R: float = 1.0) -> CurveXYZFourierJAX:
    quadpoints = jnp.linspace(0.0, 1.0, N, endpoint=False)
    dofs = jnp.array([0.0, R, 0.0,
                      0.0, 0.0, 0.0,
                      0.0, 0.0, R])
    return CurveXYZFourierJAX(quadpoints, dofs, order=1)


def _constant_section_fn(A_val=1e-4, Iy_val=1e-8, Iz_val=1e-8, J_val=2e-8):
    def fn(support_dofs):
        phi_cc = support_dofs['phis_start_cc']
        phi_cf = support_dofs['phis_start_cf']
        A, Iy, Iz, Jt = [], [], [], []
        for g in range(len(phi_cc)):
            n_cf = phi_cf[g].shape[0] if g < len(phi_cf) else 0
            n_per = phi_cc[g].shape[0] + n_cf
            A.append(jnp.full((n_per,), A_val))
            Iy.append(jnp.full((n_per,), Iy_val))
            Iz.append(jnp.full((n_per,), Iz_val))
            Jt.append(jnp.full((n_per,), J_val))
        return A, Iy, Iz, Jt
    return fn


def _tiny_support_beams_fem(*, k_attachment: float = 1e8):
    from coil_fem import CoilFEM
    from coil_fem.presets.cross_section_fns import wrap_attachment

    n_base = 2
    radii = [1.0, 1.1]
    curves = [_make_circle(N=4, R=R) for R in radii]
    # wrap_attachment: geometry-dependent w_a (beam-frame d²), unlike a
    # constant ones() clamp that makes ∂(k_att w_a)/∂φ ≡ 0.
    beam_options = {
        'n_beam_cc': 1, 'n_beam_cf': 1,
        'E': 200e9, 'nu': 0.3,
        'k_attachment': k_attachment,
        # Tiny mesh: beam-frame |r| is O(0.3–1.7); r_attachment must cover that
        # or w_a is identically zero and the grounded-w_a path is invisible.
        'r_attachment': 1.5,
        'eps_sigmoid': 0.05,
    }
    support = SupportBeams(
        nfp=1, stellsym=False,
        beam_options=beam_options,
        n_base=n_base,
        cross_section_fn=_constant_section_fn(),
        attachment_fn=wrap_attachment,
    )
    fem = CoilFEM(
        base_curves_jax=curves,
        base_currents_jax=jnp.ones(n_base),
        nfp=1,
        stellsym=False,
        mesh_options={
            'shape': 'rect', 'w1': 0.01, 'w2': 0.01,
            'n_grid_1': 1, 'n_grid_2': 1,
        },
        support=support,
        material_options={'E': 200e9, 'nu': 0.3, 'density': 8900.0},
        problem_options={'solver': 'cudss'},
        coupling='monolithic',
    )
    phi = [jnp.full((1,), 0.1) for _ in range(n_base)]
    x_found = []
    for R in radii:
        xf = jnp.array([
            R * math.cos(2 * math.pi * 0.6) + 0.5, 0.0,
            R * math.sin(2 * math.pi * 0.6),
        ])
        x_found.append(xf[None, :])
    support_dofs = {
        'phis_start_cc':         list(phi),
        'phis_end_cc':           list(phi),
        'phis_start_cf':         [jnp.full((1,), 0.6) for _ in range(n_base)],
        'x_foundation':          x_found,
        'thetas_orientation_cc': [jnp.zeros((1,)) for _ in range(n_base)],
        'thetas_orientation_cf': [jnp.zeros((1,)) for _ in range(n_base)],
    }
    return fem, curves, support_dofs


def _sol_flat_from_run(fem, out):
    """Concatenate coil displacements + support DOFs into merged sol_flat."""
    chunks = [jnp.ravel(u) for u in out['displacements']]
    u_s = out['u_s']
    if u_s is not None:
        chunks.append(jnp.ravel(u_s))
    return jnp.concatenate(chunks)


def _dir_fd(scalar_fn, x0, eps=1e-6):
    return (float(scalar_fn(x0 + eps)) - float(scalar_fn(x0 - eps))) / (2.0 * eps)


def test_simsopt_free_mask_uses_concrete_indices_under_jit():
    """Boolean free-masks under JIT raise NonConcreteBooleanIndexError.

    ``taylor.py --simsopt-free`` must scatter with ``np.flatnonzero(free)``.
    """
    free_mask = np.array([True, False, True, False, True], dtype=bool)
    free_idx = np.flatnonzero(free_mask)
    x_full0 = jnp.arange(5.0, dtype=jnp.float64)

    def J_free(x_free):
        x_full = x_full0.at[free_idx].set(x_free)
        return jnp.sum(x_full ** 2)

    g = jax.jit(jax.grad(J_free))(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64))
    assert g.shape == (3,)
    assert jnp.all(jnp.isfinite(g))


def _print_row(name, analytic, fd):
    scale = max(abs(fd), abs(analytic), 1.0)
    ratio = analytic / fd if abs(fd) > 0 else float('nan')
    print(
        f"  {name:28s}  analytic={analytic: .6e}  FD={fd: .6e}  "
        f"ratio={ratio: .6f}  |err|/scale={abs(analytic - fd) / scale:.3e}",
        flush=True,
    )
    return analytic, fd, ratio


# ---------------------------------------------------------------------------
# Step 4 — k(φ) outside custom_vjp + assemble_coo vs residual Jacobian
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (_HAS_SPINEAX and _HAS_GPU), reason=_GPU_REASON)
def test_k_phi_grad_outside_custom_vjp():
    """jax.grad through stiffness(weights(beam_geometry(φ))) must match FD."""
    fem, curves, support_dofs0 = _tiny_support_beams_fem()
    support = fem.support
    cdofs = [c.dofs for c in curves]
    curves_live = fem.curves_from_dofs(cdofs)
    pts0 = [fem.meshes[i].mesh_points_from_dofs(cdofs[i]) for i in range(2)]

    def sum_k(phi0):
        sdofs = {
            **support_dofs0,
            'phis_start_cc': [
                jnp.full((1,), phi0),
                support_dofs0['phis_start_cc'][1],
            ],
        }
        geom = support.beam_geometry(curves_live, sdofs)
        total = 0.0
        for i in range(2):
            w_g, w_a = fem._support_weights(
                i, pts0[i], curves_live, sdofs, geom=geom,
            )
            k_i = support.stiffness(w_g, w_a)
            total = total + jnp.sum(k_i) + jnp.sum(w_a)
        return total

    phi0 = float(support_dofs0['phis_start_cc'][0][0])
    d_an = float(jax.grad(sum_k)(phi0))
    d_fd = _dir_fd(sum_k, phi0, eps=1e-6)
    print("\n=== k(φ) / w_a(φ) outside custom_vjp ===", flush=True)
    _print_row("sum(k)+sum(w_a)", d_an, d_fd)
    sens = max(abs(d_fd), abs(d_an))
    assert sens > 1e-12, "wrap_attachment must make k/w_a φ-sensitive"
    scale = max(sens, 1.0)
    assert abs(d_an - d_fd) / scale < 1e-5, (d_an, d_fd)


@pytest.mark.skipif(not (_HAS_SPINEAX and _HAS_GPU), reason=_GPU_REASON)
def test_assemble_coo_matches_residual_jacobian_with_wa():
    """K from assemble_coo must match ∂R/∂u of compute_residual_vars (nonzero w_a)."""
    fem, curves, support_dofs0 = _tiny_support_beams_fem(k_attachment=1e10)
    support = fem.support
    cdofs = [c.dofs for c in curves]
    curves_live = fem.curves_from_dofs(cdofs)
    i = 0
    pipeline = fem.pipelines[i]
    pts = fem.meshes[i].mesh_points_from_dofs(cdofs[i])
    geom = support.beam_geometry(curves_live, support_dofs0)
    w_g, w_a = fem._support_weights(
        i, pts, curves_live, support_dofs0, geom=geom,
    )
    k = support.stiffness(w_g, w_a)
    assert float(jnp.max(w_a)) > 0.0, "fixture must have nonzero w_a"
    assert float(jnp.max(k)) > float(support.k_clamp) * 0.5

    bf = jnp.zeros((fem.meshes[i].n_cells, fem.meshes[i].n_quads, 3))
    sg, jxw, vgj, pqp = fem._jit_fe_geom_fns[i](pts)
    params = {
        'points': pts,
        'body_force': bf,
        'support_k': k,
        '_fe_geom': (sg, jxw, vgj, pqp),
    }
    I, J, V, _, _ = pipeline.assemble_coo(params)
    n_nodes = pipeline.problem.fes[0].num_total_nodes
    n = pipeline.problem.num_total_dofs_all_vars
    K = jnp.zeros((n, n)).at[I, J].add(V)

    pipeline.problem.set_params(params)

    def residual(u_flat):
        u = u_flat.reshape(n_nodes, 3)
        res = pipeline.problem.compute_residual_vars(
            [u],
            pipeline.problem.internal_vars,
            pipeline.problem.internal_vars_surfaces,
        )
        return jax.flatten_util.ravel_pytree(res)[0]

    R0 = residual(jnp.zeros(n))
    u = jax.random.normal(jax.random.PRNGKey(1), (n,)) * 1e-7
    lhs = K @ u
    rhs = residual(u) - R0
    err = float(jnp.max(jnp.abs(lhs - rhs)))
    print("\n=== assemble_coo vs residual Jacobian (w_a>0) ===", flush=True)
    print(f"  max|K u - (R(u)-R(0))| = {err:.3e}", flush=True)
    print(f"  max(w_a)={float(jnp.max(w_a)):.3e}  max(k)={float(jnp.max(k)):.3e}", flush=True)
    assert err < 1e-6 * max(float(jnp.max(jnp.abs(lhs))), 1.0)


# ---------------------------------------------------------------------------
# Step 3 — residual-level FD vs VJP at frozen u*
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (_HAS_SPINEAX and _HAS_GPU), reason=_GPU_REASON)
def test_residual_fd_vs_vjp_coil_vs_coupling():
    """At frozen u*, FD vs VJP of R_coil (with live k(φ)) and R_coupling."""
    fem, curves, support_dofs0 = _tiny_support_beams_fem(k_attachment=1e10)
    support = fem.support
    static = fem.monolithic_static
    cdofs = [c.dofs for c in curves]
    idofs = jnp.ones(len(curves))

    out = fem.run(
        base_curves_dofs=cdofs,
        base_currents_dofs=idofs,
        base_support_dofs=support_dofs0,
    )
    sol_flat = _sol_flat_from_run(fem, out)
    assert jnp.all(jnp.isfinite(sol_flat))

    pipelines = fem.pipelines
    n_base = len(pipelines)
    curves_live0 = fem.curves_from_dofs(cdofs)
    pts0 = [fem.meshes[i].mesh_points_from_dofs(cdofs[i]) for i in range(n_base)]
    bf0 = []
    fe_geom0 = []
    for i in range(n_base):
        sg, jxw, vgj, pqp = fem._jit_fe_geom_fns[i](pts0[i])
        # Zero body force — residual still sees Winkler via support_k.
        bf0.append(jnp.zeros((fem.meshes[i].n_cells, fem.meshes[i].n_quads, 3)))
        fe_geom0.append((sg, jxw, vgj, pqp))

    coil_dof_offsets = static.coil_dof_offsets
    n_dofs_per_coil = static.n_dofs_per_coil
    support_dof_offset = static.support_dof_offset
    n_s = static.n_s
    has_cs = static.has_cs
    has_sc = static.has_sc
    I_cs = jnp.asarray(static.I_cs_pat) if has_cs else None
    J_cs = jnp.asarray(static.J_cs_pat) if has_cs else None
    I_sc = jnp.asarray(static.I_sc_pat) if has_sc else None
    J_sc = jnp.asarray(static.J_sc_pat) if has_sc else None
    surf_interp = [
        (
            pipelines[i].problem._sel_face_sv,
            pipelines[i].problem._surf_face_to_surf_node,
            int(pipelines[i].problem._surf_unique_global_nodes.shape[0]),
        )
        for i in range(n_base)
    ]

    def _sdofs_phi(phi0):
        return {
            **support_dofs0,
            'phis_start_cc': [
                jnp.full((1,), phi0),
                support_dofs0['phis_start_cc'][1],
            ],
        }

    def _k_list(sdofs, geom):
        ks = []
        for i in range(n_base):
            w_g, w_a = fem._support_weights(
                i, pts0[i], curves_live0, sdofs, geom=geom,
            )
            ks.append(support.stiffness(w_g, w_a))
        return ks

    def R_coil_live_k(phi0):
        """Coil residual block with k recomputed from φ (outer wiring)."""
        sdofs = _sdofs_phi(phi0)
        geom = support.beam_geometry(curves_live0, sdofs)
        k_list = _k_list(sdofs, geom)
        residuals = []
        for i, pipeline in enumerate(pipelines):
            p_par = {
                'points': pts0[i],
                'body_force': bf0[i],
                'support_k': k_list[i],
                '_fe_geom': fe_geom0[i],
            }
            u_c_i = sol_flat[
                coil_dof_offsets[i]:coil_dof_offsets[i] + n_dofs_per_coil[i]
            ].reshape(pipeline.problem.fes[0].num_total_nodes, 3)
            pipeline.problem.set_params(p_par)
            res_i = pipeline.problem.compute_residual_vars(
                [u_c_i],
                pipeline.problem.internal_vars,
                pipeline.problem.internal_vars_surfaces,
            )
            residuals.append(jax.flatten_util.ravel_pytree(res_i)[0])
        return jnp.concatenate(residuals)

    def R_coil_frozen_k(phi0, k_frozen):
        """Coil residual with k held fixed (constraint-internal view)."""
        residuals = []
        for i, pipeline in enumerate(pipelines):
            p_par = {
                'points': pts0[i],
                'body_force': bf0[i],
                'support_k': k_frozen[i],
                '_fe_geom': fe_geom0[i],
            }
            u_c_i = sol_flat[
                coil_dof_offsets[i]:coil_dof_offsets[i] + n_dofs_per_coil[i]
            ].reshape(pipeline.problem.fes[0].num_total_nodes, 3)
            pipeline.problem.set_params(p_par)
            res_i = pipeline.problem.compute_residual_vars(
                [u_c_i],
                pipeline.problem.internal_vars,
                pipeline.problem.internal_vars_surfaces,
            )
            residuals.append(jax.flatten_util.ravel_pytree(res_i)[0])
        return jnp.concatenate(residuals)

    def R_coupling(phi0):
        sdofs = _sdofs_phi(phi0)
        geom = support.beam_geometry(curves_live0, sdofs)
        geom_kw = {'geom': geom}
        s_quad = [pipelines[i].surface_quad_points(pts0[i]) for i in range(n_base)]
        jxw = [pipelines[i].problem.surface_jxw(pts0[i]) for i in range(n_base)]
        Iss, Jss = support.support_pattern()
        Vss = support.support_values(
            curves_live0, sdofs, s_quad, **geom_kw, jxw_by_coil=jxw,
        )
        u_s = sol_flat[support_dof_offset:]
        r_s = jnp.zeros(n_s, dtype=Vss.dtype).at[Iss].add(Vss * u_s[Jss])
        # Pad to full residual length for a common cotangent; coil block zero.
        n_c = int(support_dof_offset)
        r_full = jnp.concatenate([jnp.zeros(n_c, dtype=r_s.dtype), r_s])
        V_cs, V_sc = support.coupling_values(
            curves_live0, sdofs, s_quad,
            surf_interp_by_coil=surf_interp,
            jxw_by_coil=jxw,
            **geom_kw,
        )
        if has_cs:
            r_full = r_full.at[I_cs].add(V_cs * sol_flat[J_cs])
        if has_sc:
            r_full = r_full.at[I_sc].add(V_sc * sol_flat[J_sc])
        return r_full

    phi0 = float(support_dofs0['phis_start_cc'][0][0])
    geom0 = support.beam_geometry(curves_live0, support_dofs0)
    k0 = _k_list(support_dofs0, geom0)

    # Fixed random cotangents (stand-in for -λ restricted to each block).
    key = jax.random.PRNGKey(7)
    v_coil = jax.random.normal(key, R_coil_live_k(phi0).shape)
    v_coup = jax.random.normal(jax.random.fold_in(key, 1), R_coupling(phi0).shape)

    def s_coil_live(phi):
        return jnp.dot(v_coil, R_coil_live_k(phi))

    def s_coil_frozen(phi):
        return jnp.dot(v_coil, R_coil_frozen_k(phi, k0))

    def s_coup(phi):
        return jnp.dot(v_coup, R_coupling(phi))

    print("\n=== Residual FD vs VJP at frozen u* (tiny SupportBeams) ===", flush=True)
    rows = {}
    for name, sfn in (
        ("R_coil live k(φ)", s_coil_live),
        ("R_coil frozen k", s_coil_frozen),
        ("R_coupling", s_coup),
    ):
        d_an = float(jax.grad(sfn)(phi0))
        d_fd = _dir_fd(sfn, phi0, eps=1e-6)
        rows[name] = _print_row(name, d_an, d_fd)

    # Frozen-k coil residual must be φ-insensitive (k is the only φ path).
    assert abs(rows["R_coil frozen k"][0]) < 1e-6, rows["R_coil frozen k"]
    assert abs(rows["R_coil frozen k"][1]) < 1e-6, rows["R_coil frozen k"]

    # Live-k coil residual and coupling should both match their own VJPs.
    for name in ("R_coil live k(φ)", "R_coupling"):
        an, fd, ratio = rows[name]
        sens = max(abs(fd), abs(an))
        if sens < 1e-12:
            print(f"  NOTE: {name} sensitivity ~0 on tiny mesh; skip ratio assert.", flush=True)
            continue
        scale = max(sens, 1.0)
        assert abs(an - fd) / scale < 5e-3, (name, an, fd, ratio)
    # Grounded-w_a path must show up in live-k coil residual for this fixture.
    assert max(abs(rows["R_coil live k(φ)"][0]), abs(rows["R_coil live k(φ)"][1])) > 1e-12, (
        "expected nonzero ∂R_coil/∂φ through k(φ); check wrap_attachment / r_attachment"
    )
