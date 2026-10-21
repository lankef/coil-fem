"""Hull weld export: grown-coil cut, beam ring, convex solid, and STEP visibility."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.spatial import ConvexHull

gmsh = pytest.importorskip("gmsh")

from coil_fem.io._export import (
    _beam_ring,
    _beam_solid,
    _check_rounding,
    _coil_loft,
    _cut_by_coils,
    _cut_keep,
    _drop_free_curves,
    _export_simsopt,
    _fragment_device,
    _fuse_weld,
    _hull2_points,
    _hull_solid,
    _mesh_tet10,
    _on_side,
    _options,
    _owner_map,
    _planes_in_frame,
    _rounded_wire,
    _section,
    _section_points,
    _share_flush_contacts,
    _snap_yz,
    _trim_by_hulls,
    _witness_point,
    _write_step,
)


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
        assert _export_simsopt(None, save_step=False, save_vtu=False) is None


def test_removed_joint_keys_raise():
    """``sliver``, ``heal_tol`` and ``hull_overlap`` are no longer accepted."""
    sections = [{"kind": "rect", "w1": 0.05, "w2": 0.1}]
    for key, value in (("sliver", {"tiny": 1e-3}), ("heal_tol", 1e-6), ("hull_overlap", 0.02)):
        with pytest.raises(ValueError, match="unrecognized"):
            _options({key: value}, sections)
    opt = _options(None, sections)
    assert opt["h_hull"] == opt["delta"]


def test_default_delta_from_sections():
    """``min(w1, w2) / 4`` for rectangles, ``r / 2`` for circles, smallest over beams."""
    rect = {"kind": "rect", "w1": 0.05, "w2": 0.1}
    circle = {"kind": "circle", "r": 0.04, "ri": 0.0, "n_sides": 12}
    assert _options(None, [rect])["delta"] == pytest.approx(0.0125)
    assert _options(None, [circle])["delta"] == pytest.approx(0.02)
    assert _options(None, [circle, rect])["delta"] == pytest.approx(0.0125)
    assert _options({"delta": 0.03}, [rect])["delta"] == pytest.approx(0.03)
    assert _options(None, [rect])["h_contour"] == pytest.approx(0.0125)


def test_grown_section_area():
    """One rounded-rectangle section has the offset area of the casing rectangle."""
    a, b, delta = 0.04, 0.03, 0.01
    owned = _gmsh_model()
    try:
        occ = gmsh.model.occ
        wire, _curves = _rounded_wire(
            occ, np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
            a, b, delta,
        )
        occ.addPlaneSurface([wire])
        occ.synchronize()
        face = [t for d, t in gmsh.model.getEntities(2)][0]
        area = occ.getMass(2, face)
    finally:
        _gmsh_done(owned)
    expect = (2 * a) * (2 * b) + 2 * delta * (2 * a + 2 * b) + np.pi * delta ** 2
    assert area == pytest.approx(expect, rel=1e-8)


def _loft_cutter(occ, x_center):
    """Straight grown-coil stand-in along y. Its x faces sit at ``x_center +- 0.3``."""
    built = [
        _rounded_wire(
            occ, np.array([x_center, y, 0.0]), np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]), 0.28, 0.5, 0.02,
        )
        for y in np.linspace(-0.5, 0.5, 5)
    ]
    wires = [wire for wire, _curves in built]
    out = occ.addThruSections(wires, makeSolid=True, makeRuled=False)
    _drop_free_curves(occ)
    return [int(t) for d, t in out if int(d) == 3]


def _two_joint_box():
    section = {"kind": "rect", "w1": 0.2, "w2": 0.2, "t": None, "w_min": 0.2}
    joints = [
        {"origin": np.zeros(3), "frame": np.eye(3), "section": section,
         "beam": 0, "side": "start"},
        {"origin": np.array([1.0, 0.0, 0.0]), "frame": np.diag([-1.0, -1.0, 1.0]),
         "section": section, "beam": 0, "side": "end"},
    ]
    gamma = np.eye(3)
    for joint in joints:
        joint["planes"] = _planes_in_frame(section, joint["frame"], gamma)
    rec = {"gamma3": gamma, "cf": False, "x_start": np.zeros(3), "L": 1.0}
    return joints, rec


def _rings_by_joint(occ, box, joints, rec, tools_by_joint):
    rings = []
    for joint, tools in zip(joints, tools_by_joint):
        scratch = [t for d, t in occ.copy([(3, box)]) if d == 3][0]
        piece, _n_drop = _cut_keep(occ, scratch, tools, _witness_point(rec), 0)
        rings.append(_beam_ring(piece, joint, 0.05))
        occ.remove([(3, piece)], recursive=True)
        occ.synchronize()
    return rings


def _on_box_side(ring):
    return (
        (np.abs(np.abs(ring[:, 1]) - 0.1) <= 1e-5)
        | (np.abs(np.abs(ring[:, 2]) - 0.1) <= 1e-5)
    )


def test_beam_ring_per_joint_lies_on_loft_and_sides():
    """Each joint's ring, cut by its own loft only, lies on the loft and the beam sides."""
    joints, rec = _two_joint_box()
    owned = _gmsh_model()
    try:
        occ = gmsh.model.occ
        box = occ.addBox(0.0, -0.1, -0.1, 1.0, 0.2, 0.2)
        tools = [_loft_cutter(occ, 0.0), _loft_cutter(occ, 1.0)]
        occ.synchronize()
        rings = _rings_by_joint(occ, box, joints, rec, tools)
    finally:
        _gmsh_done(owned)
    for ring in rings:
        assert len(ring) >= 4
        assert np.allclose(ring[:, 0], 0.3, atol=1e-6)
        assert np.all(_on_box_side(ring))


def test_beam_ring_keeps_far_graze_with_its_coil():
    """A graze by the end coil that sits nearer the start origin stays in the end ring."""
    joints, rec = _two_joint_box()
    owned = _gmsh_model()
    try:
        occ = gmsh.model.occ
        box = occ.addBox(0.0, -0.1, -0.1, 1.0, 0.2, 0.2)
        graze = occ.addCylinder(0.4, -0.5, 0.15, 0.0, 1.0, 0.0, 0.08)
        tools = [_loft_cutter(occ, 0.0), _loft_cutter(occ, 1.0) + [graze]]
        occ.synchronize()
        start, end = _rings_by_joint(occ, box, joints, rec, tools)
    finally:
        _gmsh_done(owned)
    assert np.allclose(start[:, 0], 0.3, atol=1e-6)
    assert np.all(_on_box_side(end))
    x_world = 1.0 - end[:, 0]
    assert np.any(np.abs(x_world - 0.7) <= 1e-6)
    assert np.any(np.abs(x_world - 0.4) <= 0.07)


def test_hull_solid_volume_matches_convex_hull():
    """OCC mass of a convex polyhedron matches ``ConvexHull.volume``."""
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(30, 3))
    ref = float(ConvexHull(pts).volume)
    owned = _gmsh_model()
    try:
        _tag, vol = _hull_solid(gmsh.model.occ, pts)
    finally:
        _gmsh_done(owned)
    assert abs(vol - ref) <= 1e-6 * ref


def test_hull_keeps_exact_coordinates():
    """Duplicates are dropped, but kept points are not rounded."""
    from coil_fem.io._export import _Hull

    rng = np.random.default_rng(1)
    pts = rng.normal(size=(20, 3)) + 1e-11
    cloud = _Hull(np.vstack([pts, pts[:3] + 1e-13]))
    assert len(cloud.pts) == len(pts)
    assert np.array_equal(cloud.pts, pts)


def test_hull_solid_coplanar_samples():
    """Ring and patch samples on flat facets (sliver triangles) give a positive solid."""
    s = np.linspace(-1.0, 1.0, 9)
    ring = np.array(
        [(0.0, a, b) for a in s for b in s if abs(a) == 1 or abs(b) == 1]
    ) * [1.0, 0.05, 0.05]
    u, v = np.meshgrid(np.linspace(-0.1, 0.1, 7), np.linspace(-0.08, 0.08, 7))
    patch = np.c_[np.full(u.size, 0.12), u.ravel(), v.ravel()]
    base = np.vstack([ring, patch])
    rng = np.random.default_rng(0)
    for _ in range(20):
        R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        pts = base @ R.T + rng.normal(size=3)
        ref = float(ConvexHull(pts).volume)
        owned = _gmsh_model()
        try:
            _tag, vol = _hull_solid(gmsh.model.occ, pts)
        finally:
            _gmsh_done(owned)
        assert vol == pytest.approx(ref, rel=1e-6)


def test_beam_solid_stays_analytic():
    """A rotated beam keeps plane faces, so ``_beam_ring`` can tell cut faces apart."""
    from scipy.spatial.transform import Rotation

    from coil_fem.io._export import _beam_solid

    section = {"kind": "rect", "w1": 0.05, "w2": 0.1, "t": None, "w_min": 0.05}
    for det in (1.0, -1.0):
        g = Rotation.from_euler("xyz", [0.3, -0.7, 1.1]).as_matrix() @ np.diag([1.0, 1.0, det])
        rec = {"section": section, "L": 0.6, "gamma3": g, "x_start": np.array([1.0, 2.0, 0.5])}
        owned = _gmsh_model()
        try:
            occ = gmsh.model.occ
            tag = _beam_solid(occ, rec)[0][1]
            occ.synchronize()
            kinds = {
                gmsh.model.getType(2, abs(f))
                for _d, f in gmsh.model.getBoundary([(3, tag)], oriented=False)
            }
            vol = occ.getMass(3, tag)
            lo = np.array(occ.getBoundingBox(3, tag)[:3])
        finally:
            _gmsh_done(owned)
        assert kinds == {"Plane"}
        assert vol == pytest.approx(0.6 * 0.05 * 0.1, rel=1e-9)
        assert np.all(lo <= rec["x_start"] + 1e-6)


def test_write_step_exports_visible_solid_only(tmp_path):
    """A hidden box is left out of the STEP file."""
    owned = _gmsh_model()
    path = tmp_path / "one.step"
    try:
        occ = gmsh.model.occ
        keep = occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        drop = occ.addBox(3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        occ.synchronize()
        _write_step(path, [keep], ["box_keep"])
        assert drop > 0
    finally:
        _gmsh_done(owned)

    owned = _gmsh_model()
    try:
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        vols = [t for d, t in gmsh.model.getEntities(3)]
        assert len(vols) == 1
        assert gmsh.model.occ.getMass(3, vols[0]) == pytest.approx(1.0)
    finally:
        _gmsh_done(owned)


def test_n_sides_circle_rejects_too_few():
    """A prism needs at least three sides."""
    with pytest.raises(ValueError, match="n_sides_circle"):
        _export_simsopt(None, save_step=True, save_vtu=False, n_sides_circle=2)


def test_beam_solid_polygon():
    """A circular section is an n-gon prism with vertices on the circle."""
    from scipy.spatial.transform import Rotation

    g = Rotation.from_euler("xyz", [0.2, -0.4, 0.7]).as_matrix()
    origin = np.array([1.0, 2.0, 0.5])
    length = 0.6
    for n in (12, 5):
        for hollow in (False, True):
            dofs = {"r_1_beam": 0.05, "r_2_beam": 0.02} if hollow else {"r_beam": 0.05}
            sec = _section(dofs, n)
            rec = {"section": sec, "L": length, "gamma3": g, "x_start": origin}
            owned = _gmsh_model()
            try:
                occ = gmsh.model.occ
                tag = _beam_solid(occ, rec)[0][1]
                occ.synchronize()
                faces = [abs(f) for _d, f in gmsh.model.getBoundary([(3, tag)], oriented=False)]
                kinds = {gmsh.model.getType(2, f) for f in faces}
                vol = occ.getMass(3, tag)
                verts = np.array([
                    gmsh.model.getValue(0, abs(p), [])
                    for _d, p in gmsh.model.getBoundary([(3, tag)], recursive=True)
                    if _d == 0
                ])
            finally:
                _gmsh_done(owned)
            area = 0.5 * n * sec["r"] ** 2 * np.sin(2.0 * np.pi / n)
            if hollow:
                area -= 0.5 * n * sec["ri"] ** 2 * np.sin(2.0 * np.pi / n)
            expect_faces = 2 * n + 2 if hollow else n + 2
            assert kinds == {"Plane"}
            assert len(faces) == expect_faces
            assert vol == pytest.approx(area * length, rel=1e-9)
            on_plus_y = origin + sec["r"] * g[:, 1]
            assert np.min(np.linalg.norm(verts - on_plus_y, axis=1)) < 1e-8


def test_on_side_and_snap():
    """Snapping lands on a side or a corner, in the start frame and the mirrored end frame."""
    rect = {"kind": "rect", "w1": 0.05, "w2": 0.1, "t": None, "w_min": 0.05}
    polygon = _section({"r_beam": 0.05}, 5)
    gamma = np.eye(3)
    frames = [np.eye(3), np.diag([-1.0, -1.0, 1.0])]
    for section in (rect, polygon):
        for frame in frames:
            planes = _planes_in_frame(section, frame, gamma)
            normals, offsets = planes
            tol = 1e-4
            on_plane = offsets[0] * normals[0]
            off_plane = on_plane + 1e-6 * normals[0]
            assert _on_side([0.0, *off_plane], planes, tol)
            snapped = _snap_yz(off_plane.reshape(1, 2), planes, tol)[0]
            assert abs(normals[0] @ snapped - offsets[0]) < 1e-12

            pair = next(
                (i, j)
                for i in range(len(normals))
                for j in range(i + 1, len(normals))
                if abs(float(np.linalg.det(np.stack([normals[i], normals[j]])))) > 0.5
            )
            corner = np.linalg.solve(normals[list(pair)], offsets[list(pair)])
            noisy = corner + np.array([1e-6, -1e-6])
            snapped_corner = _snap_yz(noisy.reshape(1, 2), planes, tol)[0]
            assert np.allclose(snapped_corner, corner, atol=1e-12)

            inside = np.zeros(2)
            assert np.allclose(_snap_yz(inside.reshape(1, 2), planes, tol)[0], inside)
            assert not _on_side([0.0, *inside], planes, tol)


def _face_thinness(occ, tag: int) -> float:
    """Smallest ``2 * area / perimeter`` over the faces of one volume."""
    worst = np.inf
    for _d, face in gmsh.model.getBoundary([(3, int(tag))], oriented=False):
        face = abs(int(face))
        area = occ.getMass(2, face)
        edges = [
            abs(int(e))
            for dim, e in gmsh.model.getBoundary([(2, face)], oriented=False)
            if int(dim) == 1
        ]
        perimeter = sum(occ.getMass(1, e) for e in edges)
        if perimeter > 0.0:
            worst = min(worst, 2.0 * area / perimeter)
    return float(worst)


def _side_edge_angle(tag: int, planes, tol: float = 1e-7) -> float:
    """Smallest interior angle at an edge of ``tag`` that borders a beam side.

    ``planes`` are world-frame side planes ``n · (y, z) = offset`` of a beam
    along world x. Edges shorter than 1 mm are skipped.
    """
    normals, offsets = planes
    faces = [abs(int(f)) for _d, f in gmsh.model.getBoundary([(3, int(tag))], oriented=False)]
    owners: dict[int, list[int]] = {}
    for f in faces:
        for d, c in gmsh.model.getBoundary([(2, f)], oriented=False):
            if int(d) == 1:
                owners.setdefault(abs(int(c)), []).append(f)

    def _normal(f, xyz):
        uv = gmsh.model.getParametrization(2, f, list(xyz))
        n = np.asarray(gmsh.model.getNormal(f, uv), dtype=float)
        return n / np.linalg.norm(n)

    def _on_beam_side(f, n, xyz):
        if gmsh.model.getType(2, f) != "Plane" or abs(n[0]) > tol:
            return False
        return bool(np.any(
            (np.abs(normals @ n[1:3] - 1.0) <= tol)
            & (np.abs(normals @ np.asarray(xyz[1:3]) - offsets) <= 1e-6)
        ))

    worst = 180.0
    for c, fs in owners.items():
        if len(fs) != 2:
            continue
        b = gmsh.model.getParametrizationBounds(1, c)
        t0, t1 = float(b[0][0]), float(b[1][0])
        p0, p1, mid = (
            np.asarray(gmsh.model.getValue(1, c, [t]), dtype=float)
            for t in (t0, t1, 0.5 * (t0 + t1))
        )
        if np.linalg.norm(p1 - p0) < 1e-3:
            continue
        n0, n1 = _normal(fs[0], mid), _normal(fs[1], mid)
        if not (_on_beam_side(fs[0], n0, mid) or _on_beam_side(fs[1], n1, mid)):
            continue
        ang = 180.0 - float(np.degrees(np.arccos(np.clip(n0 @ n1, -1.0, 1.0))))
        worst = min(worst, ang)
    return worst


def _trim_weld(occ, length, tools, joint, coil, *, inset_ring=False):
    """Grown cut on a copy for the ring, unshifted hull2, trim, fuse.

    Returns the fused tag, the fused volume, and trimmed + hull2 volumes.
    """
    section = joint["section"]
    box = occ.addBox(0.0, -0.5 * section["w2"], -0.5 * section["w1"], length, section["w2"], section["w1"])
    occ.synchronize()
    rec = {
        "index": 0, "gamma3": np.eye(3), "cf": False, "x_start": np.zeros(3),
        "L": length, "section": section, "joints": [joint],
    }
    scratch = [t for d, t in occ.copy([(3, box)]) if d == 3][0]
    cut, _n = _cut_keep(occ, scratch, tools, _witness_point(rec), 0)
    ring = _beam_ring(cut, joint, 0.02)
    if inset_ring:
        ring[:, 1:3] *= 1.0 - 0.002 / (0.5 * section["w_min"])
    occ.remove([(3, cut)], recursive=True)
    occ.synchronize()
    hull, hull_vol = _hull_solid(occ, _hull2_points(joint, ring, coil, 0.0))
    piece, _n = _trim_by_hulls(occ, box, [hull], rec)
    piece_vol = occ.getMass(3, piece)
    fused = _fuse_weld(occ, piece, [hull], 0)
    return fused, occ.getMass(3, fused), piece_vol + hull_vol


def _start_joint(section):
    return {
        "origin": np.zeros(3), "frame": np.eye(3), "section": section,
        "beam": 0, "side": "start",
        "planes": _planes_in_frame(section, np.eye(3), np.eye(3)),
    }


def _patch(center, u, v, half_u, half_v, normal):
    pts = np.array([center + a * half_u * u + b * half_v * v for a in (-1, 1) for b in (-1, 1)])
    return {
        "pts": pts, "normals": np.tile(normal, (4, 1)),
        "inside_pts": np.zeros((0, 3)),
    }


def test_trimmed_joint_has_no_sliver():
    """The beam ends on hull2, so no curved face or thin face survives."""
    section = {"kind": "rect", "w1": 0.2, "w2": 0.2, "t": None, "w_min": 0.2}
    joint = _start_joint(section)
    coil = _patch(np.array([0.05, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                  np.array([0.0, 0.0, 1.0]), 0.1, 0.1, np.array([-1.0, 0.0, 0.0]))
    owned = _gmsh_model()
    try:
        occ = gmsh.model.occ
        tools = _loft_cutter(occ, 0.0)
        occ.synchronize()
        fused, vol, parts = _trim_weld(occ, 1.0, tools, joint, coil)
        types = {gmsh.model.getType(2, abs(int(f)))
                 for _d, f in gmsh.model.getBoundary([(3, fused)], oriented=False)}
        assert types == {"Plane"}
        assert _face_thinness(occ, fused) > 0.1 * section["w_min"]
        assert vol == pytest.approx(parts, rel=1e-9)
        assert vol == pytest.approx(0.95 * 0.2 * 0.2, rel=1e-9)
    finally:
        _gmsh_done(owned)


def test_trim_rejects_hull2_that_misses_the_sides():
    """An inset ring leaves a skin along the sides; the trim must not keep the beam end."""
    section = {"kind": "rect", "w1": 0.2, "w2": 0.2, "t": None, "w_min": 0.2}
    joint = _start_joint(section)
    coil = _patch(np.array([0.05, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                  np.array([0.0, 0.0, 1.0]), 0.04, 0.04, np.array([-1.0, 0.0, 0.0]))
    owned = _gmsh_model()
    try:
        occ = gmsh.model.occ
        tools = _loft_cutter(occ, 0.0)
        occ.synchronize()
        with pytest.raises(RuntimeError, match="does not span"):
            _trim_weld(occ, 1.0, tools, joint, coil, inset_ring=True)
    finally:
        _gmsh_done(owned)


def test_oblique_joint_trim():
    """A cut at 60 degrees to the beam: planar weld, no sharp edge at the beam sides."""
    section = {"kind": "rect", "w1": 0.2, "w2": 0.2, "t": None, "w_min": 0.2}
    joint = _start_joint(section)
    a = np.radians(60.0)
    p = np.array([np.cos(a), np.sin(a), 0.0])
    along = np.array([-np.sin(a), np.cos(a), 0.0])
    z = np.array([0.0, 0.0, 1.0])
    coil = _patch(np.array([0.28 / np.cos(a), 0.0, 0.0]), along, z, 0.25, 0.15, -p)
    owned = _gmsh_model()
    try:
        occ = gmsh.model.occ
        built = [
            _rounded_wire(occ, t * along, p, z, 0.28, 0.5, 0.02)
            for t in np.linspace(-1.0, 1.0, 5)
        ]
        out = occ.addThruSections([w for w, _c in built], makeSolid=True, makeRuled=False)
        _drop_free_curves(occ)
        tools = [int(t) for d, t in out if int(d) == 3]
        occ.synchronize()
        fused, vol, parts = _trim_weld(occ, 2.0, tools, joint, coil)
        types = {gmsh.model.getType(2, abs(int(f)))
                 for _d, f in gmsh.model.getBoundary([(3, fused)], oriented=False)}
        assert types == {"Plane"}
        assert vol == pytest.approx(parts, rel=1e-9)
        assert _side_edge_angle(fused, joint["planes"]) > 60.0
    finally:
        _gmsh_done(owned)


def _free_curves() -> list[int]:
    used = set()
    for _d, tag in gmsh.model.getEntities(3):
        faces = gmsh.model.getBoundary([(3, int(tag))], oriented=False, recursive=False)
        for dim, face in faces:
            if int(dim) != 2:
                continue
            for d, curve in gmsh.model.getBoundary(
                [(2, abs(int(face)))], oriented=False, recursive=False,
            ):
                if int(d) == 1:
                    used.add(abs(int(curve)))
    return [int(t) for _d, t in gmsh.model.getEntities(1) if abs(int(t)) not in used]


def test_loft_leaves_no_construction_curves():
    """Thru-section input curves are removed, and the solid's volume is unchanged."""
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

    class _Mesh:
        framed_curve = _Frame()

    owned = _gmsh_model()
    try:
        occ = gmsh.model.occ
        built = [
            _rounded_wire(
                occ, np.array([0.0, y, 0.0]), np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 1.0]), 0.04, 0.03, 0.01,
            )
            for y in np.linspace(0.0, 0.5, 4)
        ]
        wires = [wire for wire, _curves in built]
        out = occ.addThruSections(wires, makeSolid=True, makeRuled=False)
        occ.synchronize()
        tag = [t for d, t in out if int(d) == 3][0]
        before = occ.getMass(3, tag)
        _drop_free_curves(occ)
        occ.synchronize()
        assert occ.getMass(3, tag) == pytest.approx(before, rel=1e-12)
        assert _free_curves() == []

        import coil_fem.io._export as ex
        gmsh.clear()
        gmsh.model.add("coil_loft")
        old = ex._N_SLICES
        ex._N_SLICES = 8
        try:
            vols = _coil_loft(gmsh.model.occ, _Mesh(), 0.04, 0.03)
        finally:
            ex._N_SLICES = old
        gmsh.model.occ.synchronize()
        assert gmsh.model.occ.getMass(3, vols[0][1]) > 0.0
        assert _free_curves() == []
    finally:
        _gmsh_done(owned)


def test_write_step_does_not_warn_on_its_own():
    """Importing the helper does not emit the no-file warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert callable(_write_step)


def _circle_mesh():
    """A unit-circle coil with a radial frame, enough for a loft."""

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

    class _Mesh:
        framed_curve = _Frame()

    return _Mesh()


def test_owner_map_material_precedence():
    """Winding pack beats casing, and either beats a support, in either input order."""
    owners = [(0, 0, 0), (0, 0, 1), (-1, -1, -1)]
    ov_map = [
        [(3, 5)],
        [(3, 5), (3, 6), (3, 7)],
        [(3, 7), (3, 8)],
    ]
    owner_map = _owner_map(ov_map, owners)
    assert owner_map[5][2] == 0
    assert owner_map[6][2] == 1
    assert owner_map[7] == (0, 0, 1)
    assert owner_map[8] == (-1, -1, -1)

    owners_swapped = [(0, 0, 1), (0, 0, 0), (-1, -1, -1)]
    ov_map_swapped = [
        [(3, 5), (3, 6), (3, 7)],
        [(3, 5)],
        [(3, 7), (3, 8)],
    ]
    owner_map = _owner_map(ov_map_swapped, owners_swapped)
    assert owner_map[5][2] == 0
    assert owner_map[6][2] == 1
    assert owner_map[7] == (0, 0, 1)
    assert owner_map[8] == (-1, -1, -1)


def test_chamfered_loft_matches_corner_polygon():
    """A chamfered loft follows the corner polygon and is smaller than the sharp one."""
    from coil_fem.meshing import corner_polygon
    import coil_fem.io._export as ex

    r, n, w1, w2 = 0.004, 1, 0.04, 0.02
    owned = _gmsh_model()
    old = ex._N_SLICES
    ex._N_SLICES = 8
    try:
        mesh = _circle_mesh()
        phi = np.linspace(0.0, 1.0, ex._N_SLICES, endpoint=False)
        r0 = np.asarray(mesh.framed_curve.curve.gamma_eval(phi))
        _t, p, q = mesh.framed_curve.rotated_frame_eval(phi)
        xy = corner_polygon(w1, w2, r, n)
        expected = (
            r0[:, None, :]
            + xy[None, :, 0, None] * p[:, None, :]
            + xy[None, :, 1, None] * q[:, None, :]
        )
        got = np.stack([
            _section_points(r0[k], p[k], q[k], w1, w2, rounding=(r, n))
            for k in range(ex._N_SLICES)
        ])
        assert np.allclose(got, expected)

        occ = gmsh.model.occ
        sharp = _coil_loft(occ, mesh, w1, w2)
        occ.synchronize()
        sharp_vol = sum(occ.getMass(3, t) for _d, t in sharp)
        occ.remove(sharp, recursive=True)
        occ.synchronize()
        chamfer = _coil_loft(occ, mesh, w1, w2, rounding=(r, n))
        occ.synchronize()
        chamfer_vol = sum(occ.getMass(3, t) for _d, t in chamfer)
        assert chamfer_vol < sharp_vol
        tags = [t for _d, t in chamfer]

        def inside(pt):
            return any(
                gmsh.model.isInside(3, int(tag), [float(x) for x in pt]) > 0
                for tag in tags
            )

        # The section wires are construction curves and are removed. The solid
        # still passes through every chamfer vertex and misses the sharp corner.
        assert all(inside(pt) for pt in expected.reshape(-1, 3))
        corner = r0[0] + 0.5 * w1 * p[0] + 0.5 * w2 * q[0]
        assert not inside(corner)
    finally:
        ex._N_SLICES = old
        _gmsh_done(owned)


def test_rounding_without_casing_raises():
    """A chamfer needs a casing, because the joint outline stays a sharp rectangle."""

    class _Bare:
        rounding = (0.01, 1)
        n_casing = 0

    class _Cased:
        rounding = (0.01, 1)
        n_casing = 1

    class _Sharp:
        rounding = None
        n_casing = 0

    with pytest.raises(NotImplementedError, match="without a casing"):
        _check_rounding([_Bare()])
    _check_rounding([_Cased()])
    _check_rounding([_Sharp()])


def test_coil_cut_then_fragment_shares_one_face():
    """Cutting the overlap away leaves one face shared by the support and the coil."""
    rec = {
        "index": 0,
        "gamma3": np.eye(3),
        "cf": False,
        "x_start": np.zeros(3),
        "L": 1.0,
        "joints": [{"coil": 0, "image": 0, "side": "end"}],
    }
    owned = _gmsh_model()
    try:
        occ = gmsh.model.occ
        support = occ.addBox(0.0, -0.05, -0.05, 1.0, 0.1, 0.1)
        coil = occ.addCylinder(0.9, 0.0, -0.3, 0.0, 0.0, 0.6, 0.08)
        occ.synchronize()
        box_vol = occ.getMass(3, support)
        coil_vols = {(0, 0): [(int(coil), 1)]}
        kept = _cut_by_coils(
            occ, support, rec, coil_vols, box_vol, np.eye(3)[None, :, :], 0,
        )
        cut_vol = occ.getMass(3, kept)
        coil_vol = occ.getMass(3, coil)
        owner, claimed = _fragment_device(
            occ, [(kept, 0, 0)], coil_vols, [rec], np.eye(3)[None, :, :],
        )
        assert sum(occ.getMass(3, t) for t in owner) == pytest.approx(
            cut_vol + coil_vol, rel=1e-8,
        )
        support_tags = [t for t, lab in owner.items() if lab[0] < 0]
        coil_tags = [t for t, lab in owner.items() if lab[0] >= 0]

        def faces(tag):
            return {
                abs(int(f))
                for d, f in gmsh.model.getBoundary([(3, int(tag))], oriented=False)
                if int(d) == 2
            }

        shared = set()
        for tag in support_tags:
            shared |= faces(tag) & set().union(*(faces(c) for c in coil_tags))
        assert shared
        for face in shared:
            upward, _down = gmsh.model.getAdjacencies(2, int(face))
            vols = {abs(int(v)) for v in upward}
            assert vols & set(support_tags)
            assert vols & set(coil_tags)
        assert claimed[support_tags[0]] == [(0, 0)]
    finally:
        _gmsh_done(owned)


def test_flush_imprint_ignores_analytic_solids():
    """Planes and cylinders are not copies of a coil B-spline, so tags stay."""
    owned = _gmsh_model()
    try:
        occ = gmsh.model.occ
        support = occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        coil = occ.addBox(2.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        occ.synchronize()
        coil_vols = {(0, 0): [(int(coil), 1)]}
        placed = _share_flush_contacts(occ, [(int(support), 0, 0)], coil_vols)
        assert placed == [(int(support), 0, 0)]
        assert coil_vols[(0, 0)] == [(int(coil), 1)]
        assert {int(t) for _d, t in gmsh.model.getEntities(3)} == {
            int(support), int(coil),
        }
    finally:
        _gmsh_done(owned)


def test_one_model_mesh_shares_interface_nodes():
    """Two fragmented boxes meshed together share the nodes of their common face."""
    owned = _gmsh_model()
    try:
        occ = gmsh.model.occ
        left = occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        right = occ.addBox(1.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        occ.fragment([(3, left)], [(3, right)])
        occ.synchronize()
        vols = [int(t) for d, t in gmsh.model.getEntities(3)]
        assert len(vols) == 2
        owner = {vols[0]: (-1, -1, -1), vols[1]: (0, 0, 0)}
        _quality, _worst, _points, cells, owner_coil, _sym, _mat = _mesh_tet10(
            owner, 0.5, 0.5,
        )

        def faces(tag):
            return {
                abs(int(f))
                for d, f in gmsh.model.getBoundary([(3, int(tag))], oriented=False)
                if int(d) == 2
            }

        shared = faces(vols[0]) & faces(vols[1])
        assert len(shared) == 1
        node_tags, _coords, _ = gmsh.model.mesh.getNodes(2, next(iter(shared)))
        all_tags, _coords, _ = gmsh.model.mesh.getNodes()
        lut = {int(tag): i for i, tag in enumerate(np.asarray(all_tags))}
        interface = {lut[int(tag)] for tag in node_tags}
        assert interface
        assert interface <= set(cells[owner_coil < 0].ravel())
        assert interface <= set(cells[owner_coil >= 0].ravel())
    finally:
        _gmsh_done(owned)


