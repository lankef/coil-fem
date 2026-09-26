"""Owner labels for the full-body OCC fragment."""

from coil_fem.io.gmsh import _entity_owner_map


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
