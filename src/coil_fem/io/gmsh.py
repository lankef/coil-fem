"""Full-device TET10 mesh of coils and beams → ``full_body_fields.vtu``.

Coil and beam OCC solids are imprinted with a fragment so the contact
face is shared, then filled by gmsh.  Beam cross-sections come from
``presets.cross_section_fns``; coils must be rectangular
(:class:`~coil_fem.meshing.FramedCurveMeshRectangle`).
"""

from __future__ import annotations

import math
from pathlib import Path

import gmsh
import meshio
import numpy as np
import pyvista as pv

from coil_fem.meshing import FramedCurveMeshRectangle
from coil_fem.presets import cross_section_fns

__all__ = ["to_full_body"]

_N_SLICES = 96
_TET10 = 11
# Gmsh tet10 edge mids differ from VTK on the last two slots
# (meshio ``_gmsh_to_meshio_order``): VTK[i] = gmsh[perm[i]].
_GMSH_TET10_TO_VTK = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 8], dtype=np.int64)

_SMOOTHING_KEYS = {
    "joint_type",
    "joint_fillet_radius",
    "interface_type",
    "interface_rounding_radius",
    "interface_mesh_size",
}


def _symmetry_Qs(nfp: int, stellsym: bool) -> np.ndarray:
    """Orthogonal maps base → image; identity first. Shape ``(n_sym, 3, 3)``."""
    flip_Q = np.diag([1.0, -1.0, -1.0])
    flip_list = (False, True) if stellsym else (False,)
    out = []
    for k in range(nfp):
        phi = 2.0 * np.pi * k / nfp
        c, s = np.cos(phi), np.sin(phi)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        for flip in flip_list:
            out.append(flip_Q @ rot if flip else rot)
    return np.asarray(out)


def _coil_solid(occ, mesh, w1, w2, radius=None, n_slices: int = _N_SLICES):
    """Loft one box into two half solids; return ``(dim, tag)`` list.

    ``w1`` and ``w2`` are the full cross-section widths.  The winding pack
    uses ``mesh.w1`` / ``mesh.w2``; the casing uses the outer widths.
    ``radius`` rounds every cross-section corner (winding pack only).
    """
    fc = mesh.framed_curve
    phi = np.linspace(0.0, 1.0, n_slices, endpoint=False)
    r0 = np.asarray(fc.curve.gamma_eval(phi))
    _, p, q = fc.rotated_frame_eval(phi)
    p, q = np.asarray(p), np.asarray(q)

    corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    wires = []
    for k in range(n_slices):
        if radius is None:
            pts = [
                occ.addPoint(*(r0[k] + 0.5 * w1 * u * p[k]
                                     + 0.5 * w2 * v * q[k]))
                for (u, v) in corners
            ]
            edges = [occ.addLine(pts[a], pts[(a + 1) % 4]) for a in range(4)]
        else:
            r = float(radius)
            centres, on_u, on_v = [], [], []
            for u, v in corners:
                centre = (
                    r0[k] + (0.5 * w1 - r) * u * p[k]
                    + (0.5 * w2 - r) * v * q[k]
                )
                centres.append(occ.addPoint(*centre))
                on_u.append(occ.addPoint(*(centre + r * u * p[k])))
                on_v.append(occ.addPoint(*(centre + r * v * q[k])))
            # Even corners leave along q; odd corners leave along p.
            edges = []
            for i in range(4):
                nxt = (i + 1) % 4
                if i % 2 == 0:
                    start, end, nxt_start = on_u[i], on_v[i], on_v[nxt]
                else:
                    start, end, nxt_start = on_v[i], on_u[i], on_u[nxt]
                edges.append(occ.addCircleArc(start, centres[i], end))
                edges.append(occ.addLine(end, nxt_start))
        wires.append(occ.addWire(edges))
    half = n_slices // 2
    out = []
    out += occ.addThruSections(wires[: half + 1], makeSolid=True, makeRuled=False)
    out += occ.addThruSections(
        wires[half:] + [wires[0]], makeSolid=True, makeRuled=False,
    )
    occ.remove([(1, w) for w in wires], recursive=True)
    return out


def _beam_solids(occ, support, sdofs, geom, solid_fn, length_factor: float = 1.0):
    """Build and place every beam OCC body; return ``(dim, tag)`` list.

    Cross-section DOFs are flattened per-group → per-beam in the same order
    as ``geom['x_start']`` / ``geom['gamma3']``.  Each body is created in the
    beam-local frame by ``solid_fn`` with length ``length_factor * L``, then
    mapped to global coordinates with ``affineTransform([gamma3 | x_start])``.
    """
    x_start = np.asarray(geom['x_start'], dtype=np.float64)
    L = np.asarray(geom['L'], dtype=np.float64)
    gamma3 = np.asarray(geom['gamma3'], dtype=np.float64)
    flat = {
        k: np.concatenate([
            np.atleast_1d(np.asarray(a, dtype=np.float64))
            for a in sdofs[k]
        ])
        for k in support._cross_section_dof_keys
    }
    solids = []
    for b in range(x_start.shape[0]):
        dofs = {k: float(flat[k][b]) for k in support._cross_section_dof_keys}
        dimtags = solid_fn(occ, dofs, float(L[b]) * length_factor)
        affine = list(np.hstack([gamma3[b], x_start[b, :, None]]).ravel())
        occ.affineTransform(dimtags, affine)
        solids.extend(dimtags)
    return solids


def _apply_sym_copies(occ, dimtags, Q_list):
    """Copy ``dimtags`` under each symmetry map; image 0 is the original.

    ``Q`` acts on column vectors (``x_image = Q @ x``), which is the same
    map as the row form ``x_base @ Q.T``.
    """
    copies = [list(dimtags)]
    for Q in Q_list[1:]:
        copied = list(occ.copy(dimtags))
        affine = [
            float(Q[0, 0]), float(Q[0, 1]), float(Q[0, 2]), 0.0,
            float(Q[1, 0]), float(Q[1, 1]), float(Q[1, 2]), 0.0,
            float(Q[2, 0]), float(Q[2, 1]), float(Q[2, 2]), 0.0,
        ]
        occ.affineTransform(copied, affine)
        copies.append(copied)
    return copies


def _entity_owner_map(ov_map, owners):
    """Map fragment output volume tags to ``(owner_coil, owner_sym, material_id)``.

    ``owners[i]`` labels fragment input ``i``.  Each coil is several
    volumes (two half-lofts per material), so labels are per input
    volume, not per coil.  An output listed under several inputs (the
    overlap) is claimed by the conductor with the lowest ``material_id``
    (winding pack before casing).  Beam labels are ``(-1, -1, -1)``.
    """
    owner_map: dict[int, tuple[int, int, int]] = {}
    for src, outs in zip(owners, ov_map):
        if src[0] >= 0:
            continue
        for dim, tag in outs:
            if dim == 3:
                owner_map[int(tag)] = (-1, -1, -1)
    for src, outs in zip(owners, ov_map):
        if src[0] < 0:
            continue
        label = (int(src[0]), int(src[1]), int(src[2]))
        for dim, tag in outs:
            if dim != 3:
                continue
            tag = int(tag)
            prev = owner_map.get(tag)
            if prev is not None and prev[0] >= 0 and prev[2] <= label[2]:
                continue
            owner_map[tag] = label
    return owner_map


def _write_vtu(
    path: Path,
    *,
    points,
    cells,
    owner_coil,
    owner_sym,
    material_id,
    clamp_centers,
    r_clamp,
    eps_sigmoid,
    k_clamp,
    materials,
    g_vec,
):
    # VTU contents for beam_dolfinx.py:
    #
    # FieldData — REQUIRED by load_vtu_problem / solve:
    #   r_clamp, eps_sigmoid, k_clamp, g_vec
    #   E, nu, rho — arrays indexed by CellData material_id
    #   clamp_centers — omitted when there are no fixed clamps (empty
    #   arrays are rejected by PyVista FieldData)
    #
    # CellData:
    #   owner_coil, owner_sym — sanity / Paraview (Lorentz reclassifies
    #   quads from Jstress.json; Winkler uses FieldData spheres)
    #   material_id — REQUIRED to index E/nu/rho; -1 is support and does
    #   not use those arrays
    meshio.Mesh(
        points=points,
        cells=[("tetra10", cells)],
        cell_data={
            "owner_coil": [np.asarray(owner_coil, dtype=np.int32)],
            "owner_sym": [np.asarray(owner_sym, dtype=np.int32)],
            "material_id": [np.asarray(material_id, dtype=np.int32)],
        },
    ).write(path)
    grid = pv.read(str(path))
    clamp_centers = np.asarray(clamp_centers, dtype=np.float64).reshape(-1, 3)
    if clamp_centers.shape[0]:
        grid.field_data["clamp_centers"] = clamp_centers
    grid.field_data["r_clamp"] = np.array([r_clamp], dtype=np.float64)
    grid.field_data["eps_sigmoid"] = np.array([eps_sigmoid], dtype=np.float64)
    grid.field_data["k_clamp"] = np.array([k_clamp], dtype=np.float64)
    grid.field_data["E"] = np.array([m["E"] for m in materials], dtype=np.float64)
    grid.field_data["nu"] = np.array([m["nu"] for m in materials], dtype=np.float64)
    grid.field_data["rho"] = np.array([m["density"] for m in materials], dtype=np.float64)
    grid.field_data["g_vec"] = np.asarray(g_vec, dtype=np.float64)
    grid.save(str(path))


def _extract_tet10(owner_map):
    """Read the order-2 tet mesh and per-cell owners.

    Returns points, cells, owner_coil, owner_sym, material_id.
    """
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    if node_tags.size == 0:
        raise RuntimeError("to_full_body: gmsh produced no mesh nodes")
    points = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)
    lut = np.full(int(node_tags.max()) + 1, -1, dtype=np.int64)
    lut[node_tags] = np.arange(node_tags.size)

    cells, owner_coil, owner_sym, material_id = [], [], [], []
    for dim, tag in gmsh.model.getEntities(3):
        etypes, _, ntags = gmsh.model.mesh.getElements(dim, tag)
        etypes = [int(t) for t in etypes]
        if _TET10 not in etypes:
            raise RuntimeError(
                f"to_full_body: volume {tag} has no TET10 elements "
                f"(types {etypes})"
            )
        conn = np.asarray(ntags[etypes.index(_TET10)], dtype=np.int64)
        conn = conn.reshape(-1, 10)[:, _GMSH_TET10_TO_VTK]
        conn = lut[conn]
        if (conn < 0).any():
            raise RuntimeError(
                f"to_full_body: tet node missing from getNodes (volume {tag})"
            )
        oc, osym, mat = owner_map.get(int(tag), (-1, -1, -1))
        cells.append(conn)
        owner_coil.append(np.full(len(conn), oc, dtype=np.int32))
        owner_sym.append(np.full(len(conn), osym, dtype=np.int32))
        material_id.append(np.full(len(conn), mat, dtype=np.int32))
    if not cells:
        raise RuntimeError("to_full_body: fragment left no volumes to mesh")
    return (
        points,
        np.vstack(cells),
        np.concatenate(owner_coil),
        np.concatenate(owner_sym),
        np.concatenate(material_id),
    )


def _fragment_inputs(occ, meshes, Q_list, beam_dimtags, interface_r=None):
    """Per-coil-image fragment inputs, plus the symmetry-expanded beams.

    Returns ``(coil_groups, beams)``.  ``coil_groups`` has one
    ``(inputs, owners)`` entry per coil image: its winding-pack and casing
    half-lofts, owned by ``(base_coil, sym_image, material_id)``.  ``beams``
    is the flat ``(dim, tag)`` list of every beam image.  ``interface_r``
    rounds the winding-pack profile only; the casing loft stays a sharp box.
    """
    n_base = len(meshes)
    n_sym = len(Q_list)
    coil_vols = [[[] for _ in range(n_base)] for _ in range(n_sym)]
    for i, mesh in enumerate(meshes):
        widths = [(mesh.w1, mesh.w2)]
        if mesh.n_casing > 0:
            widths.append((mesh.w1_outer, mesh.w2_outer))
        for mat, (a, b) in enumerate(widths):
            radius = interface_r if mat == 0 else None
            copies = _apply_sym_copies(
                occ, _coil_solid(occ, mesh, a, b, radius=radius), Q_list,
            )
            for s, dts in enumerate(copies):
                coil_vols[s][i].append((dts, mat))
    if beam_dimtags:
        beam_vols = _apply_sym_copies(occ, beam_dimtags, Q_list)
    else:
        beam_vols = [[] for _ in range(n_sym)]

    coil_groups = []
    for s in range(n_sym):
        for i in range(n_base):
            inputs, owners = [], []
            for dts, mat in coil_vols[s][i]:
                for dim, tag in dts:
                    if dim == 3:
                        inputs.append((3, tag))
                        owners.append((i, s, mat))
            coil_groups.append((inputs, owners))
    beams = [
        (3, tag) for vols in beam_vols for dim, tag in vols if dim == 3
    ]
    return coil_groups, beams


def _smoothing_radii(options, meshes):
    """Return ``(joint_radius, interface_radius, interface_mesh_size)``.

    Each entry is ``None`` when that smoothing family is off.  A family is
    on only when its ``*_type`` key is set.
    """
    if options is None:
        return None, None, None

    def _positive(key: str) -> float:
        if key not in options:
            raise ValueError(
                f"to_full_body: smoothing_options must contain {key!r}"
            )
        value = float(options[key])
        if value <= 0.0:
            raise ValueError(f"{key} must be positive, got {value}")
        return value

    unknown = set(options) - _SMOOTHING_KEYS
    if unknown:
        raise ValueError(
            "to_full_body: unrecognized smoothing_options keys "
            f"{sorted(unknown)}"
        )

    joint_r = None
    if "joint_type" in options:
        joint_type = options["joint_type"]
        if joint_type != "fillet":
            raise ValueError(f"unrecognized joint type {joint_type!r}")
        joint_r = _positive("joint_fillet_radius")
    elif "joint_fillet_radius" in options:
        raise ValueError(
            "to_full_body: 'joint_fillet_radius' requires 'joint_type'"
        )

    interface_r = None
    interface_h = None
    has_interface = (
        "interface_type" in options
        or "interface_rounding_radius" in options
        or "interface_mesh_size" in options
    )
    if has_interface and "interface_type" not in options:
        raise ValueError(
            "to_full_body: interface smoothing requires 'interface_type'"
        )
    if "interface_type" in options:
        interface_type = options["interface_type"]
        if interface_type != "rounding":
            raise ValueError(f"unrecognized interface type {interface_type!r}")
        if not any(m.n_casing > 0 for m in meshes):
            raise ValueError(
                "interface rounding requires a casing (n_casing > 0)"
            )
        interface_r = _positive("interface_rounding_radius")
        for m in meshes:
            limit = 0.5 * min(float(m.w1), float(m.w2))
            if interface_r >= limit:
                raise ValueError(
                    "interface_rounding_radius must be < half the smaller "
                    f"winding-pack width ({limit}), got {interface_r}"
                )
        if "interface_mesh_size" in options:
            interface_h = _positive("interface_mesh_size")
        else:
            interface_h = interface_r / 8.0
    return joint_r, interface_r, interface_h


def _fragment(occ, inputs, owners, hint: str):
    """Fragment ``inputs`` and label each output volume from ``owners``.

    ``occ.fragment`` cuts the solids against each other so shared contacts
    become shared faces.  ``owners[i]`` labels input ``i``; the returned
    map labels each output volume.
    """
    try:
        _ov, ov_map = occ.fragment(inputs, [])
        occ.synchronize()
    except Exception as exc:
        raise RuntimeError(f"to_full_body: OCC fragment failed. {hint}") from exc
    if len(ov_map) != len(owners):
        raise RuntimeError(
            "to_full_body: fragment map length "
            f"{len(ov_map)} != {len(owners)} inputs"
        )
    return _entity_owner_map(ov_map, owners)


def _fragment_by_coil(occ, coil_groups, beam_dimtags, hint: str):
    """Fragment each coil image against the beams, one coil at a time.

    A single ``occ.fragment`` intersects every pair of its inputs.  Coils
    never overlap each other, so each call holds one coil image plus the
    current beam pieces, and coil-coil pairs are never tested.  Beam pieces
    outside the coil just fragmented feed the next call; shared faces keep
    their identity across calls, so the mesh stays conforming.
    """
    beams = list(beam_dimtags)
    owner_map: dict[int, tuple[int, int, int]] = {}
    for inputs, owners in coil_groups:
        omap = _fragment(
            occ, inputs + beams,
            owners + [(-1, -1, -1)] * len(beams), hint,
        )
        owner_map.update({t: lab for t, lab in omap.items() if lab[0] >= 0})
        beams = [(3, t) for t, lab in omap.items() if lab[0] < 0]
    owner_map.update({t: (-1, -1, -1) for _dim, t in beams})
    return owner_map


def _fillet_joints(occ, owner_map, radius: float):
    """Fillet coil–beam joints and return an updated volume owner map.

    Each coil image is filleted on its own: fused with only the beam pieces
    that touch it, filleted, then fragmented back against those pieces.
    Coils never overlap, so no coil-coil pair is ever tested.  Fillet-only
    volumes are labelled support ``(-1, -1, -1)``.  Conductor and beam
    volumes keep their labels.
    """
    occ.synchronize()
    owner_map = dict(owner_map)
    bb = gmsh.model.getBoundingBox(-1, -1)
    extent = max(bb[i + 3] - bb[i] for i in range(3))
    tol2 = (1e-4 * extent) ** 2

    def _curves_of(tag: int) -> list[int]:
        curves: list[int] = []
        got: set[int] = set()
        _up, faces = gmsh.model.getAdjacencies(3, tag)
        for face in faces:
            _fup, down = gmsh.model.getAdjacencies(2, int(face))
            for ctag in down:
                ctag = abs(int(ctag))
                if ctag not in got:
                    got.add(ctag)
                    curves.append(ctag)
        return curves

    def _midpoint(tag: int) -> list[float]:
        tmin, tmax = gmsh.model.getParametrizationBounds(1, tag)
        tmid = 0.5 * (float(tmin[0]) + float(tmax[0]))
        xyz = gmsh.model.getValue(1, tag, [tmid])
        return [float(xyz[0]), float(xyz[1]), float(xyz[2])]

    n_edges = 0
    coil_keys = sorted({(lab[0], lab[1]) for lab in owner_map.values() if lab[0] >= 0})
    for key in coil_keys:
        # ====================================================================
        # Find this coil's junction curves and the beam pieces it touches
        # ====================================================================
        coil_tags = {
            t for t, lab in owner_map.items() if (lab[0], lab[1]) == key
        }
        junctions: list[int] = []
        beam_tags: set[int] = set()
        seen: set[int] = set()
        for _dim, ftag in gmsh.model.getEntities(2):
            upward, down = gmsh.model.getAdjacencies(2, int(ftag))
            vols = [int(v) for v in upward]
            if len(vols) != 2:
                continue
            on_coil = [v for v in vols if v in coil_tags]
            on_beam = [
                v for v in vols
                if owner_map.get(v) is not None and owner_map[v][0] < 0
            ]
            if len(on_coil) != 1 or len(on_beam) != 1:
                continue
            beam_tags.add(on_beam[0])
            for curve in (abs(int(c)) for c in down):
                if curve not in seen:
                    seen.add(curve)
                    junctions.append(curve)
        if not junctions:
            continue

        # ====================================================================
        # Fuse the coil with its beam pieces and pick the junction edges
        # ====================================================================
        group = [(3, t) for t in sorted(coil_tags) + sorted(beam_tags)]
        jc_copies = list(occ.copy([(1, c) for c in junctions]))
        # The originals stay in the model: they are fragmented against the
        # filleted solid below, so faces shared with other coils survive.
        fused, _fused_map = occ.fuse(
            group[:1], group[1:], removeObject=False, removeTool=False,
        )
        occ.synchronize()
        jc_tags = [int(t) for d, t in jc_copies if d == 1]

        def _on_junction(mid: list[float]) -> bool:
            for jtag in jc_tags:
                closest, _par = gmsh.model.getClosestPoint(1, jtag, mid)
                dx = float(closest[0]) - mid[0]
                dy = float(closest[1]) - mid[1]
                dz = float(closest[2]) - mid[2]
                if dx * dx + dy * dy + dz * dz <= tol2:
                    return True
            return False

        to_fillet = []
        to_drop = []
        for dim, tag in fused:
            if dim != 3:
                continue
            tag = int(tag)
            edges = [
                c for c in _curves_of(tag) if _on_junction(_midpoint(c))
            ]
            if edges:
                to_fillet.append((tag, edges))
            else:
                to_drop.append((3, tag))
        if not to_fillet:
            raise RuntimeError(
                f"to_full_body: coil {key} junction curves did not match "
                "any edge of the fused solid"
            )
        occ.remove(jc_copies, recursive=True)
        if to_drop:
            occ.remove(to_drop, recursive=True)

        # ====================================================================
        # Fillet, then fragment against the original coil and beam pieces
        # ====================================================================
        filleted = []
        for tag, edges in to_fillet:
            n_edges += len(edges)
            try:
                out = occ.fillet([tag], edges, [radius])
            except Exception as exc:
                raise RuntimeError(
                    f"to_full_body: OCC fillet failed on coil {key}. "
                    f"Lower joint_fillet_radius (currently {radius})."
                ) from exc
            vols_out = [(d, int(t)) for d, t in out if d == 3]
            if not vols_out:
                raise RuntimeError("to_full_body: OCC fillet returned no volume")
            filleted.extend(vols_out)

        omap = _fragment(
            occ,
            group + filleted,
            [owner_map[t] for _d, t in group] + [(-1, -1, -1)] * len(filleted),
            "Fragment of the filleted solid failed; lower joint_fillet_radius.",
        )
        for _d, t in group:
            owner_map.pop(t, None)
        owner_map.update(omap)

    if n_edges == 0:
        print("to_full_body: no coil-beam joint to fillet")
    else:
        print(f"to_full_body: filleted {n_edges} joint edges, radius={radius}")
    return owner_map


def _refine_interface_corners(owner_map, radius: float, h_min: float, h_max: float):
    """Set a background mesh size on the rounded interface corners.

    Corner faces are winding-pack/casing faces whose curvature matches
    ``radius``.  The size field measures distance from the low-curvature
    seam curves of those faces (the generators along the coil), not from
    the short arcs that close each half-loft.

    Returns
    -------
    list of int
        Tags of the corner faces the field was built from.
    """
    lo, hi = 0.5 / radius, 2.0 / radius
    corner_faces = []
    for _dim, tag in gmsh.model.getEntities(2):
        tag = int(tag)
        upward, _down = gmsh.model.getAdjacencies(2, tag)
        vols = [int(v) for v in upward]
        if len(vols) != 2:
            continue
        labels = [owner_map.get(v) for v in vols]
        if any(label is None for label in labels):
            continue
        if sorted(label[2] for label in labels) != [0, 1]:
            continue
        uvmin, uvmax = gmsh.model.getParametrizationBounds(2, tag)
        mid = [
            0.5 * (float(uvmin[0]) + float(uvmax[0])),
            0.5 * (float(uvmin[1]) + float(uvmax[1])),
        ]
        curv = float(gmsh.model.getCurvature(2, tag, mid)[0])
        if lo <= curv <= hi:
            corner_faces.append(tag)
    if not corner_faces:
        raise RuntimeError(
            "to_full_body: no rounded interface corner faces found"
        )

    # Seam curves run along the coil.  The arc ends of each half-loft
    # have curvature about 1/radius and are left out.
    seams: list[int] = []
    seen: set[int] = set()
    for face in corner_faces:
        _fup, down = gmsh.model.getAdjacencies(2, face)
        for curve in (abs(int(c)) for c in down):
            if curve in seen:
                continue
            seen.add(curve)
            tmin, tmax = gmsh.model.getParametrizationBounds(1, curve)
            tmid = 0.5 * (float(tmin[0]) + float(tmax[0]))
            curv = float(gmsh.model.getCurvature(1, curve, [tmid])[0])
            if curv < lo:
                seams.append(curve)
    if not seams:
        raise RuntimeError(
            "to_full_body: rounded corner faces have no seam curves"
        )

    max_len = max(float(gmsh.model.occ.getMass(1, c)) for c in seams)
    sampling = max(2, math.ceil(max_len / (0.5 * h_min)))
    dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist, "CurvesList", seams)
    gmsh.model.mesh.field.setNumber(dist, "Sampling", sampling)

    thresh = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thresh, "InField", dist)
    gmsh.model.mesh.field.setNumber(thresh, "SizeMin", h_min)
    gmsh.model.mesh.field.setNumber(thresh, "SizeMax", h_max)
    gmsh.model.mesh.field.setNumber(thresh, "DistMin", radius)
    gmsh.model.mesh.field.setNumber(thresh, "DistMax", 3.0 * radius)
    gmsh.model.mesh.field.setAsBackgroundMesh(thresh)
    print(
        f"to_full_body: refining {len(corner_faces)} interface corner "
        f"faces, h={h_min:.4g}"
    )
    return corner_faces


def to_full_body(
    Jstress,
    mesh_scale: float = 0.5,
    path: str | Path = "full_body_fields.vtu",
    beam_length_factor: float = 0.95,
    save_step: bool = False,
    smoothing_options: dict | None = None,
) -> Path:
    """Build a full-device TET10 mesh and write ``full_body_fields.vtu``.

    OCC ``fragment`` imprints beam–coil contacts so gmsh meshes each volume
    once with shared interface nodes.  Coils are assumed not to overlap
    or touch one another, so coil–coil contacts are neither detected nor
    shared.  A cased coil is
    two nested lofts (winding pack, then casing); the overlap is labelled
    winding pack.  CellData ``owner_coil`` / ``owner_sym`` / ``material_id``
    label cells; ``-1`` is support.  FieldData ``E``, ``nu`` and ``rho``
    are arrays indexed by ``material_id``.  When fixed clamps are
    disabled, ``clamp_centers`` is omitted from FieldData; consumers must
    treat that key as optional.

    When ``save_step`` is set, the symmetry-expanded coil and beam solids
    are written to a STEP file before the fragment.  ``smoothing_options``
    rounds the winding-pack/casing corners and fillets coil–beam joints
    before meshing.  Fillet material is labelled support.

    Parameters
    ----------
    Jstress : CoilFEMObjective
        Must wrap ``CoilSupportBeams`` whose ``cross_section_type`` has a
        matching ``*_solid`` factory in
        :mod:`coil_fem.presets.cross_section_fns`, and rectangular coil
        meshes (:class:`~coil_fem.meshing.FramedCurveMeshRectangle`).
    mesh_scale : float
        Multiplier on gmsh ``MeshSizeMax``.  The size is
        ``mesh_scale * 0.5 * w1``, capped at the smallest casing
        thickness when a casing is present.
    path : path-like
        Output VTU path (default ``full_body_fields.vtu``).
    beam_length_factor : float
        Beam solids are built with length ``beam_length_factor * L``
        (default 0.95) so the far end pulls back from the mating solid.
        Must be positive.
    save_step : bool
        When True, write the symmetry-expanded coil and beam solids
        (before the fragment) next to ``path``, with a ``.step`` suffix.
        Coordinates are metres.  Overlapping solids are expected.
    smoothing_options : dict or None
        ``None`` (default) leaves corners and joints sharp.  Otherwise
        a subset of:

        * ``joint_type`` — ``'fillet'``; fillets every coil–beam joint.
        * ``joint_fillet_radius`` — positive radius, required with
          ``joint_type``.
        * ``interface_type`` — ``'rounding'``; rounds the winding-pack
          corners.  The casing outer corners stay sharp.  Requires a
          casing.
        * ``interface_rounding_radius`` — positive radius, required with
          ``interface_type``, and strictly less than half the smaller
          winding-pack width.
        * ``interface_mesh_size`` — target element size on the rounded
          corners.  Optional; default is the rounding radius over 8.

    Returns
    -------
    pathlib.Path
        Path written.

    Raises
    ------
    ValueError
        Rectangular meshes or an OCC solid factory are missing,
        ``beam_length_factor`` is not positive, or ``smoothing_options``
        has an unknown key, an unrecognized type, or a missing or
        non-positive radius.
    RuntimeError
        The OCC fragment or fillet failed.  Lower
        ``beam_length_factor`` or ``joint_fillet_radius``, or use the
        wildmeshing path in ``gmsh.py.old``.

    Notes
    -----
    An OCC fillet on a beam-into-loft intersection curve can fail.  The
    radius must be well below the beam cross-section size and the local
    coil width.  A beam end pulled back by ``beam_length_factor`` < 1
    does not touch the coil and is not filleted.  The fillet meets the
    coil and beam tangentially, so the imprint of the fillet solid can
    leave sliver faces; check the mesh when a joint looks faceted.

    Interface rounding is built into the winding-pack cross-section
    before the loft, so the fragment carries it onto the inner casing
    surface.  A gmsh size field then refines those corner faces down to
    ``interface_mesh_size``; the global ``MeshSizeMax`` cap is unchanged.
    """
    if beam_length_factor <= 0.0:
        raise ValueError(
            f"beam_length_factor must be positive, got {beam_length_factor}"
        )

    path = Path(path)
    coil_support = Jstress.coil_support
    fem = Jstress.fem
    support = fem.support
    meshes = fem.meshes
    sdofs = coil_support.support_dofs
    curves = fem.base_curves_jax

    cs_type = getattr(coil_support, "beam_options", {}).get(
        "cross_section_type", "solid_circle",
    )
    try:
        solid_fn = getattr(cross_section_fns, cs_type + "_solid")
    except AttributeError as exc:
        raise ValueError(
            f"to_full_body: no OCC solid factory for "
            f"cross_section_type={cs_type!r} "
            f"(expected {cs_type}_solid in "
            f"coil_fem.presets.cross_section_fns)."
        ) from exc
    if not all(isinstance(m, FramedCurveMeshRectangle) for m in meshes):
        raise ValueError(
            "to_full_body requires rectangular coil meshes (FramedCurveMeshRectangle)."
        )

    joint_r, interface_r, interface_h = _smoothing_radii(
        smoothing_options, meshes,
    )

    w1 = float(meshes[0].w1)
    materials = fem.materials
    g_vec = np.asarray(
        (fem.gravity_options or {}).get("g_vec", (0.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    has_clamps = (
        coil_support._r_clamp is not None and coil_support._sig_eps is not None
    )
    r_clamp = float(coil_support._r_clamp) if has_clamps else 0.0
    eps_sigmoid = float(coil_support._sig_eps) if has_clamps else 0.0
    k_clamp = float(support.k_clamp)

    geom = support.beam_geometry(curves, sdofs)
    nfp, stellsym = support.nfp, support.stellsym
    Q_list = _symmetry_Qs(nfp, stellsym)
    size_max = mesh_scale * 0.5 * w1
    t_min = min(
        (m.casing_thickness for m in meshes if m.n_casing > 0), default=None,
    )
    if t_min is not None:
        # ponytail: global cap. Rounded interface corners are refined
        # further by _refine_interface_corners; the flat casing stays
        # at this size.
        size_max = min(size_max, t_min)

    # =========================================================================
    # Phase 1–4: OCC fragment, physical groups, gmsh volume mesh
    # =========================================================================
    try:
        owned = not gmsh.isInitialized()
    except AttributeError:
        owned = True
    if owned:
        gmsh.initialize()
    else:
        gmsh.clear()
    try:
        gmsh.option.setNumber("Geometry.OCCParallel", 1)
        gmsh.model.add("device")
        occ = gmsh.model.occ
        beam_dimtags = _beam_solids(
            occ, support, sdofs, geom, solid_fn,
            length_factor=beam_length_factor,
        )
        coil_groups, beams = _fragment_inputs(
            occ, meshes, Q_list, beam_dimtags, interface_r=interface_r,
        )
        if save_step:
            occ.synchronize()
            # Default STEP units are millimetres; the model is metres.
            gmsh.option.setString("Geometry.OCCTargetUnit", "M")
            step_path = path.with_suffix(".step")
            gmsh.write(str(step_path))
            print(f"to_full_body: wrote {step_path}")
        n_solids = sum(len(g[0]) for g in coil_groups) + len(beams)
        print(
            f"to_full_body: fragment {n_solids} solids, one coil image at a "
            f"time (beam_length_factor={beam_length_factor})"
        )
        owner_map = _fragment_by_coil(
            occ, coil_groups, beams,
            f"(BOPAlgo). Lower beam_length_factor (currently "
            f"{beam_length_factor}) to shrink beam solids away from the "
            "coil surface, or use the wildmeshing path in gmsh.py.old.",
        )
        if joint_r is not None:
            owner_map = _fillet_joints(occ, owner_map, joint_r)
        groups = {
            "winding_pack": [t for t, (_c, _s, mat) in owner_map.items() if mat == 0],
            "casing": [t for t, (_c, _s, mat) in owner_map.items() if mat == 1],
            "support": [t for t, (_c, _s, mat) in owner_map.items() if mat < 0],
        }
        for name, tags in groups.items():
            if tags:
                gmsh.model.addPhysicalGroup(3, tags, name=name)

        if interface_r is not None:
            _refine_interface_corners(
                owner_map, interface_r, interface_h, size_max,
            )
        gmsh.option.setNumber("Mesh.MeshSizeMax", size_max)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        print(f"to_full_body: meshing, MeshSizeMax={size_max:.4g}")
        gmsh.model.mesh.generate(3)
        points, cells, owner_coil, owner_sym, material_id = _extract_tet10(owner_map)
    finally:
        if owned:
            gmsh.finalize()
        else:
            gmsh.clear()

    # =========================================================================
    # Phase 5: counts + VTU
    # =========================================================================
    n_nodes = int(points.shape[0])
    n_cells = int(cells.shape[0])
    coil_cell = owner_coil >= 0
    n_coil_cells = int(np.count_nonzero(coil_cell))
    n_casing_cells = int(np.count_nonzero(material_id == 1))
    n_support_cells = n_cells - n_coil_cells
    if n_coil_cells:
        n_coil_nodes = int(np.unique(cells[coil_cell]).size)
    else:
        n_coil_nodes = 0
    n_support_nodes = n_nodes - n_coil_nodes
    print(
        "to_full_body mesh counts:\n"
        f"  all bodies:       {n_nodes} nodes, {n_cells} cells\n"
        f"  conductor (coil): {n_coil_nodes} nodes, {n_coil_cells} cells\n"
        f"  casing:           {n_casing_cells} cells\n"
        f"  support:          {n_support_nodes} nodes, {n_support_cells} cells"
    )

    n_base = len(meshes)
    n_sym = Q_list.shape[0]
    if has_clamps and "phis" in sdofs:
        phis_clamp = sdofs["phis"]
        clamp_centers = []
        for i in range(n_base):
            phi_i = np.asarray(phis_clamp[i], dtype=np.float64).ravel()
            c_base = np.asarray(curves[i].gamma_eval(phi_i), dtype=np.float64)
            for s in range(n_sym):
                clamp_centers.append(c_base @ Q_list[s].T)
        clamp_centers = np.vstack(clamp_centers)
    else:
        clamp_centers = np.zeros((0, 3), dtype=np.float64)

    _write_vtu(
        path,
        points=points,
        cells=cells,
        owner_coil=owner_coil,
        owner_sym=owner_sym,
        material_id=material_id,
        clamp_centers=clamp_centers,
        r_clamp=r_clamp,
        eps_sigmoid=eps_sigmoid,
        k_clamp=k_clamp,
        materials=materials,
        g_vec=g_vec,
    )
    return path
