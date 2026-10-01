"""Casing layer around a rectangular winding pack."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from coil_fem.coil_fem import CoilFEM
from coil_fem.coupling import Support
from coil_fem.geo import CurveXYZFourierJAX, make_framed_curve
from coil_fem.meshing import FramedCurveMeshRectangle
from coil_fem.pipelines import ElasticPipeline


def _circle(N=16, R=1.0):
    qp = jnp.linspace(0.0, 1.0, N, endpoint=False)
    dofs = jnp.array([0.0, R, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, R])
    return CurveXYZFourierJAX(qp, dofs, order=1)


def _clamp(surf_pts, curve_jax, dofs):
    return jnp.ones(surf_pts.shape[0])


def test_casing_mesh_layout():
    w1, w2, t = 0.04, 0.02, 0.01
    curve = _circle(N=16)
    fc = make_framed_curve(curve, 'centroid')
    mesh = FramedCurveMeshRectangle(
        fc, w1, w2, n_grid_1=2, n_grid_2=2, casing_thickness=t,
    )
    ds = fc.curve.incremental_arclength()
    target = float(jnp.mean(ds) / ds.shape[0])
    n_casing = max(1, int(round(t / target)))
    assert mesh.n_casing == n_casing
    N_tot = mesh.n_grid_1 + 2 * n_casing
    O_tot = mesh.n_grid_2 + 2 * n_casing
    assert mesh.n_cross == N_tot * O_tot

    # Outer nodes sit at the cased half-widths.
    sl = mesh.phi_idx_per_node == 0
    u = mesh.u_per_node[sl]
    v = mesh.v_per_node[sl]
    assert np.isclose(np.max(np.abs(u)) * w1 / 2, mesh.w1_outer / 2)
    assert np.isclose(np.max(np.abs(v)) * w2 / 2, mesh.w2_outer / 2)

    # Casing steps are the same physical size in both directions (squares).
    du = np.diff(np.unique(np.round(u, 12)))
    dv = np.diff(np.unique(np.round(v, 12)))
    assert np.allclose(du[:n_casing] * w1 / 2, dv[:n_casing] * w2 / 2)

    pipe = ElasticPipeline(
        mesh,
        [{'E': 1.0, 'nu': 0.3}, {'E': 2.0, 'nu': 0.3}],
        (0., 0., 0.), {'solver': 'umfpack'},
    )
    uv = np.asarray(mesh.uv_quad)
    wp = mesh.material_id == 0
    cs = mesh.material_id == 1
    assert np.all(np.abs(uv[wp]) <= 1.0 + 1e-10)
    assert np.all(np.maximum(np.abs(uv[cs, :, 0]), np.abs(uv[cs, :, 1])) > 1.0)
    assert pipe.problem.material_id_q.shape == (mesh.n_cells, mesh.n_quads)


def test_no_casing_unchanged():
    curve = _circle(N=8)
    fc = make_framed_curve(curve, 'centroid')
    mesh = FramedCurveMeshRectangle(fc, 0.05, 0.03, n_grid_1=3, n_grid_2=2)
    N, O = mesh.n_grid_1, mesh.n_grid_2
    assert mesh.n_casing == 0
    assert np.all(mesh.material_id == 0)
    assert np.isclose(mesh.w1_outer, mesh.w1)
    # Old uniform formula, recovered from the node index in the cross-section.
    j = (np.arange(mesh.points.shape[0]) % mesh.n_cross) // O
    k = np.arange(mesh.points.shape[0]) % O
    # TET4: every node is a corner.  Midside nodes only exist for TET10.
    assert np.allclose(mesh.u_per_node, 2.0 * j / (N - 1) - 1.0)
    assert np.allclose(mesh.v_per_node, 2.0 * k / (O - 1) - 1.0)


def test_casing_solve():
    curve = _circle(N=8)
    wp = {'E': 200e9, 'nu': 0.3, 'density': 8000.0}
    mesh_opt = {'shape': 'rect', 'w1': 0.02, 'w2': 0.02, 'n_grid_1': 1, 'n_grid_2': 1}
    common = dict(
        base_curves_jax=[curve],
        base_currents_jax=jnp.array([1e6]),
        nfp=1, stellsym=False,
        mesh_options=mesh_opt,
        support=Support(k_clamp=1e9, fixed_clamp_fns=_clamp),
        winding_pack_options=wp,
    )
    bare = CoilFEM(**common)
    cased = CoilFEM(**common, casing_options={**wp, 'E': 2e12, 'thickness': 0.01})
    dofs = [curve.dofs]
    u_bare = bare.run(base_curves_dofs=dofs)['displacements'][0]
    out = cased.run(base_curves_dofs=dofs)
    u_cased = out['displacements'][0]
    assert float(jnp.max(jnp.abs(u_cased))) < float(jnp.max(jnp.abs(u_bare)))
    vm = out['von_mises'][0]
    assert bool(jnp.all(jnp.isfinite(vm)))
    i_mat = cased.meshes[0].material_id
    f = out['f_vol'][0]
    assert float(jnp.max(jnp.abs(f[i_mat == 1]))) == 0.0


def _transition_materials():
    return [
        {'E': 200e9, 'nu': 0.3, 'density': 8000.0, 'itc': 0.003, 'current_weight': 1.0},
        {'E': 2e12, 'nu': 0.3, 'density': 7800.0, 'itc': 0.001, 'current_weight': 0.0},
    ]


def _cased_rect():
    fc = make_framed_curve(_circle(N=8), 'centroid')
    return FramedCurveMeshRectangle(
        fc, 0.04, 0.04, n_grid_1=4, n_grid_2=4, casing_thickness=0.01,
    )


def _pipeline(mesh, **kw):
    return ElasticPipeline(
        mesh, _transition_materials(), (0.0, 0.0, 0.0), {'solver': 'umfpack'}, **kw,
    )


def _edge_distance(mesh):
    """Per-quad distance to the winding-pack edge, hard max and log-sum-exp."""
    uv = np.asarray(mesh.uv_quad)
    t = mesh.casing_thickness
    beta = 20.0
    dd = (np.abs(uv) - 1.0) * np.array([mesh.w1, mesh.w2]) / 2.0
    d_hard = dd.max(axis=-1)
    d = (t / beta) * np.logaddexp(beta * dd[..., 0] / t, beta * dd[..., 1] / t)
    return uv, d_hard, d


def test_eps_sigmoid_none_is_hard_lookup():
    mesh = _cased_rect()
    hard = _pipeline(mesh)
    same = _pipeline(mesh, eps_sigmoid=None)
    prob = same.problem
    mid = np.asarray(prob.material_id_q)
    for name, table in (
        ('lam_q', 'lam_list'),
        ('mu_q', 'mu_list'),
        ('rho_q', 'rho_list'),
        ('eps_th_q', 'eps_th_list'),
    ):
        got = np.asarray(getattr(prob, name))
        expect = np.asarray(getattr(prob, table))[mid]
        assert np.array_equal(got, expect)
        assert np.array_equal(got, np.asarray(getattr(hard.problem, name)))


def test_small_eps_matches_hard_away_from_interface():
    mesh = _cased_rect()
    hard = _pipeline(mesh)
    eps, beta = 1e-3, 200.0
    sharp = _pipeline(mesh, eps_sigmoid=eps, beta=beta)
    uv = np.asarray(mesh.uv_quad)
    t = mesh.casing_thickness
    dd = (np.abs(uv) - 1.0) * np.array([mesh.w1, mesh.w2]) / 2.0
    d_hard = dd.max(axis=-1)
    band = t * (np.log(2.0) / beta + 10.0 * eps)
    away = np.abs(d_hard) > band
    assert away.any()
    assert np.allclose(
        np.asarray(sharp.problem.lam_q)[away],
        np.asarray(hard.problem.lam_q)[away],
        rtol=1e-6,
        atol=0.0,
    )


def test_blended_lam_bounded_monotone_and_centred():
    mesh = _cased_rect()
    eps, beta = 1.0, 20.0
    pipe = _pipeline(mesh, eps_sigmoid=eps, beta=beta)
    lam = np.asarray(pipe.problem.lam_q)
    tab = np.asarray(pipe.problem.lam_list)
    lo, hi = tab.min(), tab.max()
    assert np.all((lam >= lo) & (lam <= hi))

    uv, _, d = _edge_distance(mesh)
    lam_sorted = lam.ravel()[np.argsort(d.ravel(), kind='mergesort')]
    assert np.all(np.diff(lam_sorted) >= -1e-8 * hi)

    # On the flat faces (innermost quad-point row, away from the corners)
    # the smooth max tracks max(|u|, |v|), so lam_q is monotone in that radius.
    r = np.maximum(np.abs(uv[..., 0]), np.abs(uv[..., 1]))
    for axis in (0, 1):
        other = 1 - axis
        band = np.min(np.abs(uv[..., other])) + 1e-8
        face = (np.abs(uv[..., other]) <= band) & (
            np.abs(uv[..., axis]) >= np.abs(uv[..., other])
        )
        assert face.any()
        ordered = lam[face][np.argsort(r[face], kind='mergesort')]
        assert np.all(np.diff(ordered) >= -1e-8 * hi)

    # Centred on the interface: s = 0.5 at d = 0, away from corners.
    s = (lam - tab[0]) / (tab[1] - tab[0])
    band = np.min(np.abs(uv[..., 1])) + 1e-8
    face = np.abs(uv[..., 1]) <= band
    i = np.argmin(np.abs(d[face]))
    assert abs(float(d[face][i])) < 0.3 * mesh.casing_thickness
    assert abs(float(s[face][i]) - 0.5) < 0.15

    eps_th = np.asarray(pipe.problem.eps_th_q)
    assert np.array_equal(np.asarray(pipe.problem.epsilon_th), eps_th)


def test_current_weight_unchanged_by_smoothing():
    mesh = _cased_rect()
    hard = _pipeline(mesh)
    soft = _pipeline(mesh, eps_sigmoid=1.0, beta=20.0)
    assert np.array_equal(
        np.asarray(hard.problem.current_weight_q),
        np.asarray(soft.problem.current_weight_q),
    )
    mid = np.asarray(soft.problem.material_id_q)
    expect = np.asarray(soft.problem.current_weight_list)[mid]
    assert np.array_equal(np.asarray(soft.problem.current_weight_q), expect)


def test_casing_transition_validation():
    curve = _circle(N=8)
    wp = {'E': 200e9, 'nu': 0.3, 'density': 8000.0}
    common = dict(
        base_curves_jax=[curve],
        base_currents_jax=jnp.array([1e6]),
        nfp=1, stellsym=False,
        mesh_options={'shape': 'rect', 'w1': 0.02, 'w2': 0.02, 'n_grid_1': 1, 'n_grid_2': 1},
        support=Support(k_clamp=1e9, fixed_clamp_fns=_clamp),
        winding_pack_options=wp,
    )
    with pytest.raises(ValueError, match="eps_sigmoid"):
        CoilFEM(**common, casing_options={**wp, 'thickness': 0.01, 'eps_sigmoid': 0})
    with pytest.raises(ValueError, match="beta"):
        CoilFEM(**common, casing_options={**wp, 'thickness': 0.01, 'beta': 0})
