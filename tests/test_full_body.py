"""Owner labels, smoothing options, and interface rounding for the full-body mesh."""

import gmsh
import numpy as np
import pytest

from coil_fem.io.gmsh import (
    _coil_solid,
    _entity_owner_map,
    _fillet_joints,
    _fragment,
    _fragment_by_coil,
    _refine_interface_corners,
    _smoothing_radii,
)


class _Section:
    """Rectangular section stub. ``n_casing`` is 0 unless a casing is set."""

    def __init__(self, w1=0.1, w2=0.08, n_casing=1):
        self.w1 = w1
        self.w2 = w2
        self.n_casing = n_casing


class _Ring:
    """Framed circular centreline of radius 1 in the xy plane."""

    def __init__(self, w1, w2, thickness):
        self.w1 = w1
        self.w2 = w2
        self.n_casing = 1 if thickness else 0
        self.casing_thickness = thickness
        self.w1_outer = w1 + 2.0 * thickness
        self.w2_outer = w2 + 2.0 * thickness
        self.framed_curve = self._Frame()

    class _Frame:
        def __init__(self):
            self.curve = self

        def gamma_eval(self, phi):
            phi = np.asarray(phi, dtype=float)
            ang = 2.0 * np.pi * phi
            z = np.zeros_like(phi)
            return np.stack([np.cos(ang), np.sin(ang), z], axis=-1)

        def rotated_frame_eval(self, phi):
            phi = np.asarray(phi, dtype=float)
            ang = 2.0 * np.pi * phi
            z = np.zeros_like(phi)
            tangent = np.stack([-np.sin(ang), np.cos(ang), z], axis=-1)
            radial = np.stack([np.cos(ang), np.sin(ang), z], axis=-1)
            axial = np.stack([z, z, np.ones_like(phi)], axis=-1)
            return tangent, radial, axial


def _volumes(occ, dimtags) -> float:
    return sum(occ.getMass(d, t) for d, t in dimtags if d == 3)


def test_entity_owner_map_material_precedence():
    """Winding pack beats casing, and either beats a beam, in either input order."""
    # tag 5: inner (inputs 0 and 1); tag 6: shell (input 1);
    # tag 7: casing + beam; tag 8: beam only.
    owners = [(0, 0, 0), (0, 0, 1), (-1, -1, -1)]
    ov_map = [
        [(3, 5)],
        [(3, 5), (3, 6), (3, 7)],
        [(3, 7), (3, 8)],
    ]
    owner_map = _entity_owner_map(ov_map, owners)
    assert owner_map[5][2] == 0
    assert owner_map[6][2] == 1
    assert owner_map[7] == (0, 0, 1)
    assert owner_map[8] == (-1, -1, -1)

    # Casing listed before the winding pack.
    owners_swapped = [(0, 0, 1), (0, 0, 0), (-1, -1, -1)]
    ov_map_swapped = [
        [(3, 5), (3, 6), (3, 7)],
        [(3, 5)],
        [(3, 7), (3, 8)],
    ]
    owner_map = _entity_owner_map(ov_map_swapped, owners_swapped)
    assert owner_map[5][2] == 0
    assert owner_map[6][2] == 1
    assert owner_map[7] == (0, 0, 1)
    assert owner_map[8] == (-1, -1, -1)


def test_smoothing_radii_validation():
    """Each smoothing family is off unless its type key is set."""
    meshes = [_Section()]
    assert _smoothing_radii(None, meshes) == (None, None, None)
    assert _smoothing_radii(
        {"joint_type": "fillet", "joint_fillet_radius": 0.02}, meshes,
    ) == (0.02, None, None)
    assert _smoothing_radii(
        {"interface_type": "rounding", "interface_rounding_radius": 0.01},
        meshes,
    ) == (None, 0.01, 0.01 / 8)
    assert _smoothing_radii(
        {
            "joint_type": "fillet",
            "joint_fillet_radius": 0.02,
            "interface_type": "rounding",
            "interface_rounding_radius": 0.01,
            "interface_mesh_size": 0.002,
        },
        meshes,
    ) == (0.02, 0.01, 0.002)

    with pytest.raises(ValueError, match="unrecognized joint type"):
        _smoothing_radii(
            {"joint_type": "chamfer", "joint_fillet_radius": 0.01}, meshes,
        )
    with pytest.raises(ValueError, match="unrecognized interface type"):
        _smoothing_radii(
            {"interface_type": "fillet", "interface_rounding_radius": 0.01},
            meshes,
        )
    with pytest.raises(ValueError, match="unrecognized smoothing_options"):
        _smoothing_radii({"comment": "nope"}, meshes)
    with pytest.raises(ValueError):
        _smoothing_radii({"joint_type": "fillet"}, meshes)
    with pytest.raises(ValueError):
        _smoothing_radii(
            {"joint_type": "fillet", "joint_fillet_radius": 0.0}, meshes,
        )
    with pytest.raises(ValueError):
        _smoothing_radii(
            {"joint_type": "fillet", "joint_fillet_radius": -0.1}, meshes,
        )
    with pytest.raises(ValueError, match="joint_type"):
        _smoothing_radii({"joint_fillet_radius": 0.01}, meshes)
    with pytest.raises(ValueError, match="casing"):
        _smoothing_radii(
            {"interface_type": "rounding", "interface_rounding_radius": 0.01},
            [_Section(n_casing=0)],
        )
    with pytest.raises(ValueError, match="half"):
        _smoothing_radii(
            {"interface_type": "rounding", "interface_rounding_radius": 0.04},
            [_Section(w1=0.1, w2=0.08)],
        )
    with pytest.raises(ValueError):
        _smoothing_radii(
            {
                "interface_type": "rounding",
                "interface_rounding_radius": 0.01,
                "interface_mesh_size": -1.0,
            },
            meshes,
        )


def test_fillet_joints_box_cylinder():
    """A fillet adds support material and leaves the conductor volume unchanged."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("joint")
        occ = gmsh.model.occ
        box = (3, occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
        # Axis +z, start inside the box top, protrude past z = 1.
        cyl = (3, occ.addCylinder(0.5, 0.5, 0.9, 0.0, 0.0, 0.3, 0.1))
        _ov, ov_map = occ.fragment([box, cyl], [])
        occ.synchronize()
        owner_map = _entity_owner_map(ov_map, [(0, 0, 0), (-1, -1, -1)])
        before = sum(
            occ.getMass(3, tag) for _dim, tag in gmsh.model.getEntities(3)
        )
        owner_map = _fillet_joints(occ, owner_map, 0.02)
        coil = 0.0
        total = 0.0
        for _dim, tag in gmsh.model.getEntities(3):
            mass = occ.getMass(3, int(tag))
            total += mass
            assert int(tag) in owner_map
            if owner_map[int(tag)][0] >= 0:
                coil += mass
        assert total > before
        assert coil == pytest.approx(1.0)
    finally:
        gmsh.finalize()


class _SpyOcc:
    """Delegates to ``gmsh.model.occ`` and records each fragment call's inputs."""

    def __init__(self, occ):
        self._occ = occ
        self.calls = []

    def fragment(self, objects, tools):
        self.calls.append(list(objects))
        return self._occ.fragment(objects, tools)

    def __getattr__(self, name):
        return getattr(self._occ, name)


def _two_boxes_and_bar(occ):
    """Coil boxes A and C joined by a bar; returns groups, beams, A and C tags."""
    a = (3, occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
    c = (3, occ.addBox(4.0, 0.0, 0.0, 1.0, 1.0, 1.0))
    bar = (3, occ.addBox(0.5, 0.4, 0.4, 4.0, 0.2, 0.2))
    groups = [([a], [(0, 0, 0)]), ([c], [(1, 0, 0)])]
    return groups, [bar]


def _duplicate_nodes() -> int:
    _tags, xyz, _ = gmsh.model.mesh.getNodes()
    xyz = np.round(np.asarray(xyz).reshape(-1, 3), 9)
    return len(xyz) - len(np.unique(xyz, axis=0))


def test_fragment_by_coil_never_pairs_coils():
    """Each fragment call holds one coil image; the mesh stays conforming."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("by-coil")
        spy = _SpyOcc(gmsh.model.occ)
        groups, beams = _two_boxes_and_bar(spy)
        coil_tags = [[t for _d, t in g[0]] for g in groups]
        owner_map = _fragment_by_coil(spy, groups, beams, "test")
        assert len(spy.calls) == len(groups)
        for call in spy.calls:
            tags = {t for _d, t in call}
            assert sum(bool(tags & set(ct)) for ct in coil_tags) == 1

        labels = sorted(lab[0] for lab in owner_map.values())
        # Each coil = its box minus the bar plus the overlap; one bar middle.
        assert labels == [-1, 0, 0, 1, 1]
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.2)
        gmsh.model.mesh.generate(3)
        assert _duplicate_nodes() == 0
    finally:
        gmsh.finalize()


def test_fillet_joints_per_coil_keeps_mesh_conforming():
    """Filleting each coil adds support material and shares faces at both beam ends."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("by-coil-fillet")
        occ = gmsh.model.occ
        groups, beams = _two_boxes_and_bar(occ)
        owner_map = _fragment_by_coil(occ, groups, beams, "test")
        before = sum(
            occ.getMass(3, t) for _d, t in gmsh.model.getEntities(3)
        )
        owner_map = _fillet_joints(occ, owner_map, 0.02)
        total = 0.0
        coil = 0.0
        for _d, t in gmsh.model.getEntities(3):
            mass = occ.getMass(3, int(t))
            total += mass
            assert int(t) in owner_map
            if owner_map[int(t)][0] >= 0:
                coil += mass
        assert total > before
        assert coil == pytest.approx(2.0)
        # Both coils were filleted: coil ids 0 and 1 are still present.
        assert {lab[0] for lab in owner_map.values()} == {-1, 0, 1}
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.2)
        gmsh.model.mesh.generate(3)
        assert _duplicate_nodes() == 0
    finally:
        gmsh.finalize()


def test_coil_solid_rounding_area():
    """Rounding removes the four corner squares outside the quarter-circles."""
    w1, w2, radius = 0.2, 0.1, 0.01
    mesh = _Ring(w1, w2, thickness=0.0)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("round-area")
        occ = gmsh.model.occ
        sharp = _coil_solid(occ, mesh, w1, w2, n_slices=24)
        rounded = _coil_solid(occ, mesh, w1, w2, radius=radius, n_slices=24)
        occ.synchronize()
        ratio = _volumes(occ, rounded) / _volumes(occ, sharp)
        removed = (4.0 - np.pi) * radius ** 2
        assert ratio == pytest.approx((w1 * w2 - removed) / (w1 * w2), rel=1e-3)
    finally:
        gmsh.finalize()


def test_refine_interface_corners_finds_arcs():
    """A rounded winding pack inside a casing has 8 arc faces, 4 per half-loft."""
    w1, w2, radius, thickness = 0.2, 0.1, 0.01, 0.02
    mesh = _Ring(w1, w2, thickness)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("round-refine")
        occ = gmsh.model.occ
        n_slices = 16
        wp = _coil_solid(
            occ, mesh, w1, w2, radius=radius, n_slices=n_slices,
        )
        casing = _coil_solid(
            occ, mesh, mesh.w1_outer, mesh.w2_outer, n_slices=n_slices,
        )
        inputs, owners = [], []
        for dim, tag in wp:
            if dim == 3:
                inputs.append((3, tag))
                owners.append((0, 0, 0))
        for dim, tag in casing:
            if dim == 3:
                inputs.append((3, tag))
                owners.append((0, 0, 1))
        owner_map = _fragment(occ, inputs, owners, "test fragment failed")
        faces = _refine_interface_corners(owner_map, radius, radius / 8, 0.05)
        assert len(faces) == 8
        assert len(gmsh.model.mesh.field.list()) > 0
    finally:
        gmsh.finalize()
