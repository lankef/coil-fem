"""Mesh-first export topology, face-conforming toes, and node provenance."""

from __future__ import annotations

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")
import coil_fem.io._export_old as export_module

from coil_fem.io._export_old import (
    _boundary_loops,
    _export_simsopt,
    _gmsh_tets,
    _provenance_ids,
    _tet_boundary,
    _toe_mesh,
)


def _closed_plate(n=13, h=0.04, top_height=None, thickness=0.08):
    """Closed gridded plate with its top wound toward +z."""
    xs = (np.arange(n) - n // 2) * h
    xy = np.array([(x, y) for y in xs for x in xs], dtype=float)
    z = (
        np.zeros(len(xy), dtype=float)
        if top_height is None else np.asarray(top_height(xy), dtype=float)
    )
    top = np.column_stack([xy, z])
    bottom = np.column_stack([xy, np.full(len(xy), -float(thickness))])
    points = np.vstack([top, bottom])
    top_tri = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b, c, d = a + 1, a + 1 + n, a + n
            top_tri.extend(([a, b, c], [a, c, d]))
    top_tri = np.asarray(top_tri, dtype=np.int64)
    bottom_tri = top_tri[:, ::-1] + len(top)
    _components, loops = _boundary_loops(top_tri)
    loop = loops[0][0]
    side = []
    for a, b in zip(loop, loop[1:] + loop[:1]):
        side.extend(([b, a, a + len(top)], [b, a + len(top), b + len(top)]))
    exterior = np.vstack([top_tri, bottom_tri, np.asarray(side, dtype=np.int64)])
    return points, exterior


def _dome(n=8, radius=0.06, z_ring=0.18, z_apex=0.09):
    """Beam cap above the plate. The rim is the far edge; normals point down."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), np.full(n, z_ring)],
        axis=1,
    )
    points = np.vstack([np.array([[0.0, 0.0, z_apex]]), ring])
    # Clockwise from +z, so the normal has a negative z component.
    tris = [[0, 1 + (k + 1) % n, 1 + k] for k in range(n)]
    return points, np.asarray(tris, dtype=np.int64)


def _ordered(tris):
    return {tuple(int(v) for v in row) for row in np.asarray(tris, dtype=np.int64)}


def _face_keys(tris):
    """Unoriented triangle keys."""
    return {
        tuple(sorted(int(v) for v in row))
        for row in np.asarray(tris, dtype=np.int64)
    }


def test_cut_face_is_the_off_prism_disk():
    """The cut shares its rim with the tube, so the disk is the off-prism part."""
    boundary, _count = _tet_boundary(np.array([[0, 1, 2, 3]], dtype=np.int64))
    on_prism = np.ones(len(boundary), dtype=bool)
    on_prism[-1] = False
    comps, loops = _boundary_loops(boundary[~on_prism])
    assert len(comps) == 1 and len(loops[0]) == 1 and len(loops[0][0]) == 3
    _comps, closed = _boundary_loops(boundary)
    assert closed == [[]]


def test_n_sides_circle_rejects_a_digon():
    """A prism needs at least three sides, before any objective is touched."""
    with pytest.raises(ValueError, match="n_sides_circle"):
        _export_simsopt(None, n_sides_circle=2)


def test_unrecognized_joint_keys_raise():
    """Old hull options are not accepted."""
    with pytest.raises(ValueError, match="unrecognized"):
        _export_simsopt(None, joint_options={"hull": "bspline"})


def test_tet_boundary_and_loops():
    """One tet has four outward faces; a closed diamond has no boundary loop."""
    faces, count = _tet_boundary(np.array([[0, 1, 2, 3]], dtype=np.int64))
    assert len(faces) == 4
    assert np.all(count == 1)
    assert _ordered(faces) == {(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)}

    comps, loops = _boundary_loops(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64))
    assert len(comps) == 1
    assert len(loops) == 1 and len(loops[0]) == 1 and len(loops[0][0]) == 4

    diamond = np.array([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=np.int64)
    closed, _count = _tet_boundary(diamond)
    comps, loops = _boundary_loops(closed)
    assert len(comps) == 1
    assert loops == [[]]


def test_toe_mesh_plate_and_dome():
    """A dome cap on a plate becomes one toe that reuses both triangulations."""
    coil_points, coil_exterior = _closed_plate()
    cap_points, cap_tri = _dome()
    toe = _toe_mesh(
        cap_points, cap_tri, coil_points, coil_exterior,
        contact_radius=0.16, size_min=0.04, where="synthetic toe",
    )

    beam = toe["source_kind"] == 0
    coil = toe["source_kind"] == 1
    assert np.allclose(toe["points"][beam], cap_points[toe["source_index"][beam]])
    assert np.allclose(toe["points"][coil], coil_points[toe["source_index"][coil]])
    assert {
        tuple(sorted(int(v) for v in tri))
        for tri in toe["source_index"][toe["cap_faces"]]
    } == {
        tuple(sorted(int(v) for v in tri))
        for tri in cap_tri
    }
    assert _ordered(toe["patch_tri"]) <= _ordered(coil_exterior)
    assert set(np.unique(toe["patch_tri"]).tolist()) == set(
        toe["source_index"][coil].tolist()
    )
    assert len(toe["side_faces"])
    assert np.any(toe["source_kind"] == 2)
    assert _ordered(toe["shell"]) == (
        _ordered(toe["cap_faces"])
        | _ordered(toe["patch_faces"])
        | _ordered(toe["side_faces"])
    )

    boundary, _count = _tet_boundary(toe["tets"])
    assert _face_keys(boundary) == _face_keys(toe["shell"])
    volume = np.einsum(
        "ij,ij->i",
        toe["points"][toe["tets"][:, 1]] - toe["points"][toe["tets"][:, 0]],
        np.cross(
            toe["points"][toe["tets"][:, 2]] - toe["points"][toe["tets"][:, 0]],
            toe["points"][toe["tets"][:, 3]] - toe["points"][toe["tets"][:, 0]],
        ),
    )
    assert np.all(volume > 0.0)
    assert float(toe["quality"].min()) >= 0.05

    beam_offset, coil_offset = 100, 1000
    gid = _provenance_ids(
        toe["source_kind"], toe["source_index"], beam_offset, coil_offset,
    )
    assert np.array_equal(gid[coil], coil_offset + toe["source_index"][coil])
    assert np.array_equal(gid[beam], beam_offset + toe["source_index"][beam])
    assert np.all(gid[toe["source_kind"] == 2] == -1)


def test_toe_mesh_grows_patch_over_a_ridge(monkeypatch):
    """A raised casing ridge is absorbed into the complete-face patch."""
    def ridge(xy):
        radius = np.linalg.norm(xy, axis=1)
        return 0.09 * np.exp(-((radius - 0.065) / 0.011) ** 2)

    coil_points, coil_exterior = _closed_plate(
        n=15, h=0.02, top_height=ridge, thickness=0.05,
    )
    cap_points, cap_tri = _dome(radius=0.11, z_apex=0.03, z_ring=0.2)
    # This fixture intentionally creates a sharp feature to exercise patch
    # growth, not element-quality policy; the smooth fixture above covers it.
    monkeypatch.setattr(export_module, "_QUALITY_MIN", 0.0)
    toe = _toe_mesh(
        cap_points, cap_tri, coil_points, coil_exterior,
        contact_radius=0.04, size_min=0.04, where="ridge toe",
    )
    patch_nodes = np.unique(toe["patch_tri"])
    assert float(coil_points[patch_nodes, 2].max()) > 0.08


def test_gmsh_tets_retain_shared_node_tags():
    """Fragmented material volumes retain exact shared mesh-node identities."""
    owned = not gmsh.isInitialized()
    if owned:
        gmsh.initialize()
    else:
        gmsh.clear()
    try:
        gmsh.model.add("shared_tags")
        occ = gmsh.model.occ
        left = (3, occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
        right = (3, occ.addBox(1.0, 0.0, 0.0, 1.0, 1.0, 1.0))
        out, _owners = occ.fragment([left], [right])
        occ.synchronize()
        volumes = [int(tag) for dim, tag in out if int(dim) == 3]
        mesh = _gmsh_tets(volumes, 0.5, 0.5)
        tags = [set(mesh["volumes"][tag]["node_tags"].tolist()) for tag in volumes]
        assert tags[0] & tags[1]
        all_tags = np.unique(np.concatenate([
            mesh["volumes"][tag]["node_tags"] for tag in volumes
        ]))
        points = np.empty((len(all_tags), 3))
        tets = []
        for tag in volumes:
            part = mesh["volumes"][tag]
            local = np.searchsorted(all_tags, part["node_tags"])
            points[local] = part["points"]
            tets.append(local[part["tets"]])
        exterior, _counts = _tet_boundary(np.vstack(tets))
        assert not np.any(np.all(np.isclose(points[exterior, 0], 1.0), axis=1))
    finally:
        if owned:
            gmsh.finalize()
        else:
            gmsh.clear()
