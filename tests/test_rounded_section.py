"""Rounded-corner winding pack: gmsh cross-section swept in uniform phi layers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from coil_fem.coil_fem import CoilFEM
from coil_fem.coupling import Support
from coil_fem.geo import CurveXYZFourierJAX, make_framed_curve
from coil_fem.meshing import (
    FramedCurveMeshRectangle, FramedCurveMeshSection, corner_polygon,
    rounded_rect_section,
)

W1, W2, T = 0.04, 0.02, 0.005


def _circle(N=16, R=1.0):
    qp = jnp.linspace(0.0, 1.0, N, endpoint=False)
    dofs = jnp.array([0.0, R, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, R])
    return CurveXYZFourierJAX(qp, dofs, order=1)


def _clamp(surf_pts, curve_jax, dofs):
    return jnp.ones(surf_pts.shape[0])


def _wp_area(r, n):
    m = n + 1
    return W1 * W2 - 4.0 * r**2 * (1.0 - m * np.tan(np.pi / (4 * m)))


def test_corner_polygon_angles():
    assert np.allclose(np.abs(corner_polygon(W1, W2, 0.005, 0)), [W1 / 2, W2 / 2])
    for n in (1, 2, 3):
        xy = corner_polygon(W1, W2, 0.005, n)
        assert xy.shape == (4 * (n + 1), 2)
        d_in = xy - np.roll(xy, 1, axis=0)
        d_out = np.roll(xy, -1, axis=0) - xy
        cross = d_in[:, 0] * d_out[:, 1] - d_in[:, 1] * d_out[:, 0]
        turn = np.arctan2(cross, np.sum(d_in * d_out, axis=1))
        assert np.allclose(turn, np.pi / (2 * (n + 1)))      # equal CCW turns
        x, y = xy.T
        area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        assert np.isclose(area, _wp_area(0.005, n))


def test_section_areas_and_sizes():
    r, n = 0.005, 2
    s = 2 * r * np.tan(np.pi / (4 * (n + 1)))
    sec = rounded_rect_section(
        W1, W2, r, n, casing_thickness=T, h_min=s / 4, h_max=0.01,
    )
    xy = sec.uv * np.array([W1 / 2, W2 / 2])
    e1 = xy[sec.tris[:, 1]] - xy[sec.tris[:, 0]]
    e2 = xy[sec.tris[:, 2]] - xy[sec.tris[:, 0]]
    area = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    assert np.all(area > 0.0)
    assert np.isclose(sec.conductor_area, _wp_area(r, n), rtol=1e-10)
    assert np.isclose(area.sum(), (W1 + 2 * T) * (W2 + 2 * T), rtol=1e-10)
    wp = np.repeat(sec.material_id == 0, 3)
    assert np.all(np.abs(sec.uv[sec.tris.ravel()[wp]]) <= 1.0 + 1e-12)
    # Graded: corner edges are much shorter than core edges.
    edge = np.linalg.norm(np.concatenate([e1, e2]), axis=1)
    assert edge.min() < 0.6 * s and edge.max() > 2.5 * s


@pytest.mark.parametrize("mesh_type", ["TET4", "TET10"])
def test_sweep_conforming(mesh_type):
    fc = make_framed_curve(_circle(N=16), 'centroid')
    r, n = 0.005, 1
    sec = rounded_rect_section(
        W1, W2, r, n, casing_thickness=T, h_min=0.002, h_max=0.008,
    )
    mesh = FramedCurveMeshSection(fc, sec, mesh_type=mesh_type)
    pts = np.asarray(mesh.points)
    c4 = np.asarray(mesh.cells)[:, :4]

    # Every face is interior (2 tets) or on the outer casing surface (1 tet).
    faces = np.sort(c4[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]].reshape(-1, 3), 1)
    uniq, count = np.unique(faces, axis=0, return_counts=True)
    assert set(count) <= {1, 2}
    outer = np.maximum(
        np.abs(mesh.u_per_node) * W1 / (W1 + 2 * T),
        np.abs(mesh.v_per_node) * W2 / (W2 + 2 * T),
    )
    assert np.allclose(outer[uniq[count == 1]], 1.0)

    x = pts[c4]
    vol = np.einsum('ci,ci->c', np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), x[:, 3] - x[:, 0]) / 6
    assert np.all(vol > 0.0)
    # Straight-edged corner tets on a 16-slice circle: exact chordal factor.
    N = mesh.n_phi
    wp_vol = vol[mesh.material_id == 0].sum()
    assert np.isclose(wp_vol, sec.conductor_area * N * np.sin(2 * np.pi / N), rtol=1e-10)
    assert mesh.cross_section_area == sec.conductor_area

    if mesh_type == 'TET10':
        mid = np.asarray(mesh.cells)[:, 4:]
        e = np.array([[0, 1], [1, 2], [0, 2], [0, 3], [1, 3], [2, 3]])
        assert np.unique(mid).shape[0] == np.unique(np.sort(c4[:, e].reshape(-1, 2), 1), axis=0).shape[0]
    assert np.allclose(mesh.mesh_points_from_dofs(fc.curve.dofs), pts)


def test_coilfem_rounding_wiring():
    curve = _circle(N=8)
    wp = {'E': 200e9, 'nu': 0.3, 'density': 8000.0}
    mesh_opt = {'shape': 'rect', 'w1': 0.02, 'w2': 0.02, 'rounding_subdivision': 1}
    common = dict(
        base_curves_jax=[curve],
        base_currents_jax=jnp.array([1e6]),
        nfp=1, stellsym=False,
        support=Support(k_clamp=1e9, fixed_clamp_fns=_clamp),
        casing_options={**wp, 'E': 2e12, 'thickness': 0.005},
    )
    plain = CoilFEM(**common, mesh_options=mesh_opt, winding_pack_options=wp)
    assert isinstance(plain.meshes[0], FramedCurveMeshRectangle)

    rounded_wp = {**wp, 'r_rounding': 0.004, 'n_rounding': 1}
    fem = CoilFEM(**common, mesh_options=mesh_opt, winding_pack_options=rounded_wp)
    mesh = fem.meshes[0]
    assert isinstance(mesh, FramedCurveMeshSection) and mesh.shape == 'rect'
    assert 'r_rounding' not in fem.materials[0]
    out = fem.run(base_curves_dofs=[curve.dofs])
    assert bool(jnp.all(jnp.isfinite(out['von_mises'][0])))
    assert float(jnp.max(jnp.abs(out['f_vol'][0][mesh.material_id == 1]))) == 0.0

    bad = [
        ({**wp, 'r_rounding': 0.004}, mesh_opt),
        ({**wp, 'r_rounding': 0.004, 'n_rounding': 0}, mesh_opt),
        ({**wp, 'r_rounding': 0.01, 'n_rounding': 1}, mesh_opt),
        (rounded_wp, {**mesh_opt, 'n_grid_1': 2}),
    ]
    for wpo, mo in bad:
        with pytest.raises(ValueError):
            CoilFEM(**common, mesh_options=mo, winding_pack_options=wpo)
