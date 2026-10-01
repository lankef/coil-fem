"""Owner labels for the full-body OCC fragment."""

import gmsh
import pytest

from coil_fem.io.gmsh import _entity_owner_map, _fillet_joints, _fillet_radius


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


def test_fillet_radius_validation():
    """None skips filleting; only a positive 'fillet' radius is accepted."""
    assert _fillet_radius(None) is None
    assert _fillet_radius({"fillet_type": "fillet", "fillet_radius": 0.02}) == 0.02
    with pytest.raises(ValueError, match="unrecognized fillet type"):
        _fillet_radius({"fillet_type": "chamfer", "fillet_radius": 0.01})
    with pytest.raises(ValueError):
        _fillet_radius({"fillet_type": "fillet"})
    with pytest.raises(ValueError):
        _fillet_radius({"fillet_type": "fillet", "fillet_radius": 0.0})
    with pytest.raises(ValueError):
        _fillet_radius({"fillet_type": "fillet", "fillet_radius": -0.1})


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
