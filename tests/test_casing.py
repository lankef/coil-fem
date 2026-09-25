"""Casing layer around a rectangular winding pack."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

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
