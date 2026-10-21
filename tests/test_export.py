"""Welded export: options, beam loops, casing distance, coil contour, and one joint end to end."""

from __future__ import annotations

from types import SimpleNamespace
import warnings

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

import coil_fem.io._export as ex
from coil_fem.coupling.beam_network import SupportBeams
from coil_fem.io._export import (
    _beam_loops,
    _beam_solid,
    _casing_sdf,
    _coil_contour,
    _coil_loft,
    _export_simsopt,
    _Hull,
    _hull_solid,
    _options,
    _planes_in_frame,
    _section,
)
from coil_fem.meshing import FramedCurveMeshRectangle


def _circle_mesh(**attrs):
    """A unit-circle coil with a radial frame and a 0.4 x 0.4 casing around a 0.3 x 0.3 winding pack."""

    class _Curve:
        @staticmethod
        def gamma_eval(phi):
            phi = np.asarray(phi, dtype=float)
            ang = 2.0 * np.pi * phi
            return np.stack([np.cos(ang), np.sin(ang), np.zeros_like(phi)], axis=-1)

    class _Frame:
        curve = _Curve()

        @staticmethod
        def rotated_frame_eval(phi):
            phi = np.asarray(phi, dtype=float)
            ang = 2.0 * np.pi * phi
            n = len(phi)
            tangent = np.stack([-np.sin(ang), np.cos(ang), np.zeros(n)], axis=-1)
            radial = np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], axis=-1)
            return tangent, radial, np.tile([0.0, 0.0, 1.0], (n, 1))

    mesh = SimpleNamespace(
        framed_curve=_Frame(), w1=0.3, w2=0.3, w1_outer=0.4, w2_outer=0.4,
        casing_thickness=0.05, n_casing=1,
    )
    mesh.__dict__.update(attrs)
    return mesh


def _gmsh_model():
    try:
        owned = not gmsh.isInitialized()
    except AttributeError:
        owned = True
    if owned:
        gmsh.initialize()
    else:
        gmsh.clear()
    gmsh.model.add("export_test")
    return owned


def _gmsh_done(owned):
    if owned:
        gmsh.finalize()
    else:
        gmsh.clear()


def test_no_file_exported_warns():
    """Both output flags off returns before any geometry work."""
    with pytest.warns(UserWarning, match="No file exported"):
        assert _export_simsopt(None, save_step=False, save_vtk=False) is None
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(ValueError, match="CoilFEMObjective"):
            _export_simsopt(None, prefix="/tmp/x", save_step=True)


def test_unsupported_inputs_raise():
    """A missing casing is a ValueError."""

    def objective(**attrs):
        mesh = object.__new__(FramedCurveMeshRectangle)
        for key, val in {"n_casing": 1, "rounding": None, **attrs}.items():
            object.__setattr__(mesh, key, val)
        fem = SimpleNamespace(support=object.__new__(SupportBeams), meshes=[mesh])
        return SimpleNamespace(coil_support=SimpleNamespace(), fem=fem)

    with pytest.raises(ValueError, match="casing"):
        _export_simsopt(objective(n_casing=0), "/tmp/x", save_step=True)


def test_options_defaults_and_unknown_keys():
    """``delta`` defaults to the smallest ``min(w1, w2) / 4`` or ``r / 2``; skirt keys are gone."""
    rect = {"kind": "rect", "w1": 0.05, "w2": 0.1}
    circle = {"kind": "circle", "r": 0.04, "ri": 0.0, "n_sides": 12}
    opt = _options(None, [rect])
    assert opt["delta"] == pytest.approx(0.0125)
    assert opt["h_contour"] == opt["h_hull"] == opt["delta"]
    assert opt["edge_snap_fraction"] == pytest.approx(0.5)
    assert _options(None, [rect, circle])["delta"] == pytest.approx(0.0125)
    assert _options({"delta": 0.01}, [])["delta"] == 0.01
    for key in ("skirt_depth", "shrink_beam"):
        with pytest.raises(ValueError, match="unrecognized"):
            _options({key: 0.01}, [rect])
    with pytest.raises(ValueError, match="edge_snap_fraction"):
        _options({"edge_snap_fraction": 0.0}, [rect])


@pytest.mark.parametrize("dofs", [
    {"w1_beam": 0.1, "w2_beam": 0.2},
    {"r_beam": 0.08},
])
def test_beam_loops_halfspace(dofs):
    """Rings sit on the side planes at the grown-coil offsets; ring 2 is inset by ``shrink``."""
    sec = _section(dofs, 7)
    eye = np.eye(3)
    planes = _planes_in_frame(sec, eye, eye)
    joint = {"origin": np.zeros(3), "frame": eye, "planes": planes, "beam": 0, "side": "start"}
    delta, shrink = 0.02, 0.01

    def sdf(P):
        # Coil fills x < 0.2.
        return np.asarray(P)[:, 0] - 0.2

    ring1, ring2 = _beam_loops(sdf, joint, 1.0, delta, shrink, h=0.01)
    normals, offsets = planes
    for ring, inset, x in ((ring1, 0.0, 0.2 + delta), (ring2, shrink, 0.2 + delta + shrink)):
        assert np.allclose(ring[:, 0], x, atol=1e-8)
        signed = ring[:, 1:] @ normals.T - (offsets - inset)
        assert np.all(signed <= 1e-12)
        assert np.all(np.abs(signed).min(axis=1) < 1e-12)
        # Every corner is a sample: two planes at once.
        assert np.count_nonzero(np.sum(np.abs(signed) < 1e-12, axis=1) >= 2) == len(normals)

    # Edge clearance: every non-corner point is at least 0.005 from the corners.
    clear1, _ = _beam_loops(sdf, joint, 1.0, delta, shrink, h=0.01, edge_clear=0.005)
    signed = clear1[:, 1:] @ normals.T - offsets
    corners = clear1[np.sum(np.abs(signed) < 1e-12, axis=1) >= 2][:, 1:]
    others = clear1[np.sum(np.abs(signed) < 1e-12, axis=1) < 2][:, 1:]
    assert len(corners) == len(normals) and len(clear1) <= len(ring1)
    for pt in others:
        assert np.linalg.norm(corners - pt, axis=1).min() >= 0.005 - 1e-12

    with pytest.raises(RuntimeError, match="outside the coil"):
        _beam_loops(lambda P: np.asarray(P)[:, 0] + 1.0, joint, 1.0, delta, shrink, h=0.01)
    with pytest.raises(RuntimeError, match="never leave"):
        _beam_loops(lambda P: np.full(len(P), -1.0), joint, 1.0, delta, shrink, h=0.01)
    with pytest.raises(RuntimeError, match="half-width"):
        _beam_loops(sdf, joint, 1.0, delta, 10.0, h=0.01)


def test_casing_sdf_matches_the_sharp_casing():
    """Signed distance to the swept outer rectangle: negative inside, exact offsets outside."""
    sdf = _casing_sdf(_circle_mesh(), np.eye(3), 0.01)
    pts = np.array([
        [1.0, 0.0, 0.0],     # centreline
        [1.25, 0.0, 0.0],    # 0.05 outboard of the outer face
        [1.0, 0.0, 0.3],     # 0.1 above the top face
        [1.0, -1.2, 0.0],    # centreline of another station
    ])
    d = sdf(pts)
    assert d[0] < 0.0 and d[3] > 0.0
    assert d[1] == pytest.approx(0.05, abs=1e-3)
    assert d[2] == pytest.approx(0.1, abs=1e-3)


def test_coil_contour_hits_the_level_set_on_the_given_section():
    """The contour sits on the requested ``(w1, w2)`` rectangle at the requested distance."""
    mesh = _circle_mesh()
    hull = _Hull(np.array([
        [1.3, -0.05, -0.05], [1.3, 0.05, -0.05], [1.3, -0.05, 0.05],
        [1.3, 0.05, 0.05], [1.35, 0.0, 0.0],
    ]))
    for w, radius in ((0.4, 0.12), (0.3, 0.2)):
        coil = _coil_contour(mesh, np.eye(3), hull, radius, w, w, 0.005, "test", 1e-6)
        pts = coil["pts"]
        assert coil["n_loops"] == 1 and coil["resid_max"] < 1e-3
        assert np.allclose(hull.distance(pts), radius, atol=1e-3)
        rho = np.hypot(pts[:, 0], pts[:, 1])
        face = np.maximum(np.abs(rho - 1.0), np.abs(pts[:, 2]))
        assert np.allclose(face, 0.5 * w, atol=1e-8)

        # Edge clearance: every node is on a corner or at least 0.02 from an edge.
        clear = _coil_contour(mesh, np.eye(3), hull, radius, w, w, 0.005, "test", 1e-6, 0.02)
        u, v = np.hypot(clear["pts"][:, 0], clear["pts"][:, 1]) - 1.0, clear["pts"][:, 2]
        to_edge = np.hypot(0.5 * w - np.abs(u), 0.5 * w - np.abs(v))
        assert np.all((to_edge < 1e-9) | (to_edge >= 0.02 - 1e-9))
        assert len(clear["pts"]) <= len(pts)


def test_one_joint_weld_trims_the_beam_and_shares_faces():
    """Loops and the weld hull, then the merge sequence: fragment beam and weld, fuse into coil_big, fragment the winding pack."""
    delta = 0.025
    shrink = 0.025
    mesh = _circle_mesh()
    Q = np.eye(3)
    sec = _section({"w1_beam": 0.1, "w2_beam": 0.1}, 12)
    rec = {
        "section": sec, "L": 0.6, "gamma3": np.eye(3), "x_start": np.array([1.1, 0.0, 0.0]),
    }
    joint = {
        "origin": rec["x_start"], "frame": np.eye(3), "planes": _planes_in_frame(sec, Q, Q),
        "beam": 0, "side": "start", "coil": 0, "image": 0,
    }
    old = ex._N_SLICES
    ex._N_SLICES = 24
    owned = _gmsh_model()
    try:
        occ = gmsh.model.occ
        wp = [int(t) for _d, t in _coil_loft(occ, mesh, 0.3, 0.3)]
        big = [int(t) for _d, t in _coil_loft(occ, mesh, 0.4, 0.4)]
        beam = _beam_solid(occ, rec)[0][1]
        occ.synchronize()

        sdf = _casing_sdf(mesh, Q, 0.5 * delta)
        ring1, ring2 = _beam_loops(sdf, joint, rec["L"], delta, shrink, delta)
        # The outer casing face is the cylinder rho = 1.2, so ring 1 is at 1.2 + delta.
        assert np.all(np.abs(np.hypot(ring1[:, 0] + 1.1, ring1[:, 1]) - 1.225) < 2e-3)
        world1 = rec["x_start"] + ring1
        world2 = rec["x_start"] + ring2
        hull = _Hull(world1)
        # Both loops are sunk a little under their surfaces, as in production.
        coil1 = _coil_contour(
            mesh, Q, hull, np.sqrt(2.0) * delta, 0.39, 0.39, delta, "casing", 0.5 * delta,
        )
        coil2 = _coil_contour(
            mesh, Q, hull, np.sqrt(2.0) * (delta + 0.05), 0.28, 0.28, delta, "wp", 0.5 * delta,
        )
        weld, vol = _hull_solid(occ, np.vstack([world1, world2, coil1["pts"], coil2["pts"]]))
        occ.synchronize()
        assert vol > 0.0
        # The weld spans from the winding pack's face (x = 1.15) to ring 2 (x = 1.25).
        assert gmsh.model.isInside(3, weld, [1.22, 0.0, 0.0]) == 1
        assert gmsh.model.isInside(3, weld, [1.18, 0.0, 0.0]) == 1

        def faces(tags):
            return {
                abs(int(f)) for t in tags
                for _d, f in gmsh.model.getBoundary([(3, int(t))], oriented=False)
            }

        # Baseline: the coil alone, fragmented on copies.
        dup = [[t for _d, t in occ.copy([(3, t) for t in k])] for k in (wp, big)]
        _out, ovmap = occ.fragment([(3, t) for t in dup[1] + dup[0]], [])
        occ.synchronize()
        wp_faces = len(faces([t for d, t in ovmap[len(dup[1])] if d == 3] +
                             [t for d, t in ovmap[len(dup[1]) + 1] if d == 3]))
        occ.remove([(3, t) for outs in ovmap for _d, t in outs], recursive=True)
        occ.synchronize()
        wp_vol = sum(occ.getMass(3, t) for t in wp)
        big_vol = sum(occ.getMass(3, t) for t in big)

        # Pass 1: fragment prism and weld; the prism beyond the weld is a stump
        # inside the coil and is dropped, the rest is the trimmed beam.
        _out, ov1 = occ.fragment([(3, beam), (3, weld)], [])
        occ.synchronize()
        weld_tags = [int(t) for d, t in ov1[1] if d == 3]
        pieces = [int(t) for d, t in ov1[0] if d == 3 and int(t) not in weld_tags]
        keep = [t for t in pieces if sdf(np.asarray(occ.getCenterOfMass(3, t)))[0] > 0.0]
        assert len(keep) == 1 and len(pieces) > 1
        occ.remove([(3, t) for t in pieces if t != keep[0]], recursive=True)

        # Pass 2: the weld is fused into coil_big, so it never cuts the winding pack.
        out, _map = occ.fuse([(3, t) for t in big], [(3, t) for t in weld_tags])
        fused = [int(t) for d, t in out if d == 3]
        _out, ovmap = occ.fragment([(3, t) for t in fused + wp], [])
        occ.synchronize()
        owner = {}
        for label, outs in zip([1] * len(fused) + [0] * len(wp), ovmap):
            for d, t in outs:
                if d == 3:
                    owner[int(t)] = label
        assert set(owner) | set(keep) == {int(t) for _d, t in gmsh.model.getEntities(3)}
        wp_pieces = [t for t, m in owner.items() if m == 0]
        casing = [t for t, m in owner.items() if m == 1]
        assert len(faces(wp_pieces)) == wp_faces
        assert sum(occ.getMass(3, t) for t in wp_pieces) == pytest.approx(wp_vol, rel=1e-2)
        assert sum(occ.getMass(3, t) for t in wp_pieces + casing) >= (1.0 - 1e-2) * big_vol
        assert faces(keep) & faces(casing)
    finally:
        ex._N_SLICES = old
        _gmsh_done(owned)


def test_beam_task_brep_keeps_shared_faces_when_placed_by_rotation(tmp_path):
    """One master beam task, imported twice and rotated: beam and weld still share a face."""
    delta = 0.025
    # Thin casing so the sunk contour sections still reach the hull.
    mesh = _circle_mesh(casing_thickness=0.02, w1_outer=0.34, w2_outer=0.34)
    sec = _section({"w1_beam": 0.1, "w2_beam": 0.1}, 12)
    Q = np.eye(3)
    rec = {
        "index": 0, "section": sec, "L": 0.6, "gamma3": np.eye(3),
        "x_start": np.array([1.1, 0.0, 0.0]),
        "joints": [{
            "origin": np.array([1.1, 0.0, 0.0]), "frame": np.eye(3),
            "planes": _planes_in_frame(sec, Q, Q),
            "beam": 0, "side": "start", "coil": 0, "image": 0,
        }],
    }
    opt = {"h_hull": delta, "h_contour": delta}
    old = ex._N_SLICES
    ex._N_SLICES = 24
    try:
        ex._pool_init([mesh], np.eye(3)[None], opt, delta, 0.5 * delta)
        res = ex._beam_task((rec, str(tmp_path)))
    finally:
        ex._N_SLICES = old
        ex._CTX.clear()
    assert len(res["weld_pts"]) == 1 and len(res["weld_pts"][0]) > 4
    assert sorted(r[0] for r in res["roles"]).count("beam") == 1
    assert any(r[0] == "weld" for r in res["roles"])

    owned = _gmsh_model()
    try:
        occ = gmsh.model.occ
        copies = [[t for _d, t in ex._vols(occ.importShapes(res["brep"]))] for _ in range(2)]
        ex._rigid(occ, [(3, t) for t in copies[1]], np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
        occ.synchronize()
        com = [np.array([occ.getCenterOfMass(3, t) for t in c]) for c in copies]
        mass = [sum(occ.getMass(3, t) for t in c) for c in copies]
        assert mass[1] == pytest.approx(mass[0], rel=1e-9)
        # Rotating by 90 degrees about z sends (x, y) to (-y, x).
        assert np.allclose(com[1][:, 0], -com[0][:, 1], atol=1e-9)
        assert np.allclose(com[1][:, 1], com[0][:, 0], atol=1e-9)
        for c in copies:
            by_face = [
                {abs(int(f)) for _d, f in gmsh.model.getBoundary([(3, t)], oriented=False)}
                for t in c
            ]
            assert len(c) == len(res["roles"])
            # Some pair of pieces shares a face: the beam-weld interface.
            assert any(
                a & b for i, a in enumerate(by_face) for b in by_face[i + 1:]
            )
    finally:
        _gmsh_done(owned)


def test_straighten_folded_resets_shared_midpoints_and_leaves_valid_cells():
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float
    )
    tets = [(0, 1, 2, 3), (1, 2, 3, 4)]  # share face (1, 2, 3)
    pts, mid, cells = list(verts), {}, []
    for tet in tets:
        row = list(tet)
        for i, j in ex._TET10_EDGES:
            key = tuple(sorted((tet[i], tet[j])))
            if key not in mid:
                mid[key] = len(pts)
                pts.append(0.5 * (verts[key[0]] + verts[key[1]]))
            row.append(mid[key])
        cells.append(row)
    pts, cells = np.array(pts), np.array(cells)
    straight = pts.copy()
    # fold cell 0 by pushing the (0, 1) midpoint past the opposite face, and
    # bend an edge only cell 1 uses so it must stay untouched
    pts[mid[(0, 1)]] = [0.5, 0.5, 1.0]
    pts[mid[(1, 4)]] += [0.01, 0.0, 0.0]
    assert ex._min_detj_ratio(pts, cells)[0] < 0.05
    n_cells, n_nodes, ratio = ex._straighten_folded(pts, cells)
    assert n_cells >= 1 and n_nodes >= 1
    np.testing.assert_allclose(ratio, ex._min_detj_ratio(pts, cells))
    assert (ex._min_detj_ratio(pts, cells) > 0.05).all()
    np.testing.assert_allclose(pts[mid[(0, 1)]], straight[mid[(0, 1)]])
    np.testing.assert_allclose(pts[mid[(1, 4)]], straight[mid[(1, 4)]] + [0.01, 0, 0])
