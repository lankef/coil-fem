"""Full-device TET10 mesh of coils and beams → ``full_body_fields.vtu``.

Coil and beam OCC solids are imprinted with a fragment so the contact
face is shared, then filled by gmsh.  Beam cross-sections come from
``presets.cross_section_fns``; coils must be rectangular
(:class:`~coil_fem.meshing.FramedCurveMeshRectangle`).
"""

from __future__ import annotations

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


def _coil_solid(occ, mesh, w1, w2, n_slices: int = _N_SLICES):
    """Loft one rectangular box into two half solids; return ``(dim, tag)`` list.

    ``w1`` and ``w2`` are the full cross-section widths.  The winding pack
    uses ``mesh.w1`` / ``mesh.w2``; the casing uses the outer widths.
    """
    fc = mesh.framed_curve
    phi = np.linspace(0.0, 1.0, n_slices, endpoint=False)
    r0 = np.asarray(fc.curve.gamma_eval(phi))
    _, p, q = fc.rotated_frame_eval(phi)
    p, q = np.asarray(p), np.asarray(q)

    corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    wires = []
    for k in range(n_slices):
        pts = [
            occ.addPoint(*(r0[k] + 0.5 * w1 * u * p[k]
                                 + 0.5 * w2 * v * q[k]))
            for (u, v) in corners
        ]
        lines = [occ.addLine(pts[a], pts[(a + 1) % 4]) for a in range(4)]
        wires.append(occ.addWire(lines))

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


def _fragment_inputs(occ, meshes, Q_list, beam_dimtags):
    """Full-device coil then beam volumes, with a parallel owner list.

    Coil owners are ``(base_coil, sym_image, material_id)``.  Beam owners
    are ``(-1, -1, -1)``.  Each coil image may contribute more than one
    volume (the two half-lofts of the winding pack, and of the casing
    when ``mesh.n_casing > 0``).
    """
    n_base = len(meshes)
    n_sym = len(Q_list)
    coil_vols = [[[] for _ in range(n_base)] for _ in range(n_sym)]
    for i, mesh in enumerate(meshes):
        widths = [(mesh.w1, mesh.w2)]
        if mesh.n_casing > 0:
            widths.append((mesh.w1_outer, mesh.w2_outer))
        for mat, (a, b) in enumerate(widths):
            copies = _apply_sym_copies(occ, _coil_solid(occ, mesh, a, b), Q_list)
            for s, dts in enumerate(copies):
                coil_vols[s][i].append((dts, mat))
    if beam_dimtags:
        beam_vols = _apply_sym_copies(occ, beam_dimtags, Q_list)
    else:
        beam_vols = [[] for _ in range(n_sym)]

    inputs, owners = [], []
    for s in range(n_sym):
        for i in range(n_base):
            for dts, mat in coil_vols[s][i]:
                for dim, tag in dts:
                    if dim == 3:
                        inputs.append((3, tag))
                        owners.append((i, s, mat))
    for s in range(n_sym):
        for dim, tag in beam_vols[s]:
            if dim == 3:
                inputs.append((3, tag))
                owners.append((-1, -1, -1))
    return inputs, owners


def _fillet_radius(fillet_options: dict | None) -> float | None:
    """Return the fillet radius, or None when filleting is disabled."""
    if fillet_options is None:
        return None
    if "fillet_type" not in fillet_options:
        raise ValueError(
            "to_full_body: fillet_options must contain 'fillet_type'"
        )
    fillet_type = fillet_options["fillet_type"]
    if fillet_type != "fillet":
        raise ValueError(f"unrecognized fillet type {fillet_type!r}")
    if "fillet_radius" not in fillet_options:
        raise ValueError(
            "to_full_body: fillet_options must contain 'fillet_radius'"
        )
    radius = float(fillet_options["fillet_radius"])
    if radius <= 0.0:
        raise ValueError(f"fillet_radius must be positive, got {radius}")
    return radius


def _boundary_curves(dim: int, tag: int) -> list[int]:
    """Unique curve tags on the boundary of one face or volume."""
    if dim == 3:
        faces = gmsh.model.getBoundary(
            [(3, tag)], combined=False, oriented=False,
        )
        dimtags = [(d, int(t)) for d, t in faces if d == 2]
    else:
        dimtags = [(dim, tag)]
    curves: list[int] = []
    seen: set[int] = set()
    for fdim, ftag in dimtags:
        for cdim, ctag in gmsh.model.getBoundary(
            [(fdim, ftag)], combined=False, oriented=False,
        ):
            if cdim != 1:
                continue
            ctag = abs(int(ctag))
            if ctag not in seen:
                seen.add(ctag)
                curves.append(ctag)
    return curves


def _junction_curves(owner_map) -> list[int]:
    """Curves of faces shared by one beam volume and one conductor."""
    curves: list[int] = []
    seen: set[int] = set()
    for dim, tag in gmsh.model.getEntities(2):
        upward, _down = gmsh.model.getAdjacencies(dim, tag)
        vols = [int(v) for v in upward]
        if len(vols) != 2:
            continue
        labels = [owner_map.get(v) for v in vols]
        if any(label is None for label in labels):
            continue
        if sum(label[0] < 0 for label in labels) != 1:
            continue
        for curve in _boundary_curves(2, int(tag)):
            if curve not in seen:
                seen.add(curve)
                curves.append(curve)
    return curves


def _curve_midpoint(tag: int) -> list[float]:
    """Parametric midpoint of a curve, as ``[x, y, z]``."""
    tmin, tmax = gmsh.model.getParametrizationBounds(1, tag)
    tmid = 0.5 * (float(tmin[0]) + float(tmax[0]))
    xyz = gmsh.model.getValue(1, tag, [tmid])
    return [float(xyz[0]), float(xyz[1]), float(xyz[2])]


def _near_junction(mid: list[float], jc_tags: list[int], tol: float) -> bool:
    """True when ``mid`` lies within ``tol`` of a junction curve."""
    tol2 = tol * tol
    for tag in jc_tags:
        closest, _par = gmsh.model.getClosestPoint(1, tag, mid)
        dx = float(closest[0]) - mid[0]
        dy = float(closest[1]) - mid[1]
        dz = float(closest[2]) - mid[2]
        if dx * dx + dy * dy + dz * dz <= tol2:
            return True
    return False


def _fillet_joints(occ, owner_map, radius: float):
    """Fillet coil–beam joints and return an updated volume owner map.

    Fillet-only volumes are labelled support ``(-1, -1, -1)``.  Conductor
    and beam volumes keep the labels in ``owner_map``.
    """
    occ.synchronize()
    junctions = _junction_curves(owner_map)
    if not junctions:
        print("to_full_body: no coil-beam joint to fillet")
        return owner_map

    vol_tags = [int(t) for t in owner_map]
    vols = [(3, t) for t in vol_tags]
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for tag in vol_tags:
        bb = gmsh.model.getBoundingBox(3, tag)
        for i in range(3):
            lo[i] = min(lo[i], bb[i])
            hi[i] = max(hi[i], bb[i + 3])
    extent = max(hi[i] - lo[i] for i in range(3))
    tol = 1e-4 * extent

    jc_copies = list(occ.copy([(1, c) for c in junctions]))
    keep = list(occ.copy(vols))
    keep_owners = [owner_map[t] for t in vol_tags]
    # ponytail: one fuse of every solid; cost and boolean fragility grow
    # with the solid count. Upgrade path: fuse each connected coil-beam
    # cluster on its own.
    fused, _fused_map = occ.fuse(vols[:1], vols[1:])
    occ.synchronize()

    jc_tags = [int(t) for d, t in jc_copies if d == 1]
    to_fillet = []
    to_drop = []
    for dim, tag in fused:
        if dim != 3:
            continue
        tag = int(tag)
        edges = [
            c for c in _boundary_curves(3, tag)
            if _near_junction(_curve_midpoint(c), jc_tags, tol)
        ]
        if edges:
            to_fillet.append((tag, edges))
        else:
            to_drop.append((3, tag))
    if not to_fillet:
        raise RuntimeError(
            "to_full_body: coil-beam junction curves did not match any "
            "edge of the fused solid"
        )

    if jc_copies:
        occ.remove(jc_copies, recursive=True)
    if to_drop:
        occ.remove(to_drop, recursive=True)

    filleted = []
    n_edges = 0
    for tag, edges in to_fillet:
        n_edges += len(edges)
        try:
            out = occ.fillet([tag], edges, [radius])
        except Exception as exc:
            raise RuntimeError(
                "to_full_body: OCC fillet failed. "
                f"Lower fillet_radius (currently {radius})."
            ) from exc
        vols_out = [(d, int(t)) for d, t in out if d == 3]
        if not vols_out:
            raise RuntimeError("to_full_body: OCC fillet returned no volume")
        filleted.extend(vols_out)
    print(f"to_full_body: filleted {n_edges} joint edges, radius={radius}")

    inputs = filleted + keep
    owners = [(-1, -1, -1)] * len(filleted) + keep_owners
    try:
        _ov, ov_map = occ.fragment(inputs, [])
        occ.synchronize()
    except Exception as exc:
        raise RuntimeError(
            "to_full_body: OCC fragment of the filleted solid failed."
        ) from exc
    if len(ov_map) != len(owners):
        raise RuntimeError(
            "to_full_body: fillet fragment map length "
            f"{len(ov_map)} != {len(owners)} inputs"
        )
    return _entity_owner_map(ov_map, owners)


def to_full_body(
    Jstress,
    mesh_scale: float = 0.5,
    path: str | Path = "full_body_fields.vtu",
    beam_length_factor: float = 0.95,
    save_step: bool = False,
    fillet_options: dict | None = None,
) -> Path:
    """Build a full-device TET10 mesh and write ``full_body_fields.vtu``.

    OCC ``fragment`` imprints beam–coil (and coil–coil) contacts so gmsh
    meshes each volume once with shared interface nodes.  A cased coil is
    two nested lofts (winding pack, then casing); the overlap is labelled
    winding pack.  CellData ``owner_coil`` / ``owner_sym`` / ``material_id``
    label cells; ``-1`` is support.  FieldData ``E``, ``nu`` and ``rho``
    are arrays indexed by ``material_id``.  When fixed clamps are
    disabled, ``clamp_centers`` is omitted from FieldData; consumers must
    treat that key as optional.

    When ``save_step`` is set, the symmetry-expanded coil and beam solids
    are written to a STEP file before the fragment.  When
    ``fillet_options`` is set, coil–beam joints are filleted after the
    fragment and before meshing; the added material is labelled support.

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
    fillet_options : dict or None
        ``None`` (default) skips filleting.  Otherwise must contain
        ``fillet_type`` (only ``'fillet'`` is supported) and
        ``fillet_radius``, one positive radius applied at every
        coil–beam joint.

    Returns
    -------
    pathlib.Path
        Path written.

    Raises
    ------
    ValueError
        Rectangular meshes or an OCC solid factory are missing,
        ``beam_length_factor`` is not positive, or ``fillet_options``
        has an unrecognized ``fillet_type`` or a missing or
        non-positive ``fillet_radius``.
    RuntimeError
        The OCC fragment or fillet failed.  Lower
        ``beam_length_factor`` or ``fillet_radius``, or use the
        wildmeshing path in ``gmsh.py.old``.

    Notes
    -----
    An OCC fillet on a beam-into-loft intersection curve can fail.  The
    radius must be well below the beam cross-section size and the local
    coil width, and ``MeshSizeMax`` is not reduced to resolve it.  A
    beam end pulled back by ``beam_length_factor`` < 1 does not touch
    the coil and is not filleted.  The fillet meets the coil and beam
    tangentially, so the imprint of the fillet solid can leave sliver
    faces; check the mesh when a joint looks faceted.
    """
    if beam_length_factor <= 0.0:
        raise ValueError(
            f"beam_length_factor must be positive, got {beam_length_factor}"
        )
    radius = _fillet_radius(fillet_options)

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
        # ponytail: global cap; a gmsh size Field on the casing volumes
        # would avoid refining the winding pack
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
        gmsh.model.add("device")
        occ = gmsh.model.occ
        beam_dimtags = _beam_solids(
            occ, support, sdofs, geom, solid_fn,
            length_factor=beam_length_factor,
        )
        inputs, owners = _fragment_inputs(occ, meshes, Q_list, beam_dimtags)
        if save_step:
            occ.synchronize()
            # Default STEP units are millimetres; the model is metres.
            gmsh.option.setString("Geometry.OCCTargetUnit", "M")
            step_path = path.with_suffix(".step")
            gmsh.write(str(step_path))
            print(f"to_full_body: wrote {step_path}")
        print(
            f"to_full_body: fragment {len(inputs)} solids "
            f"(beam_length_factor={beam_length_factor})"
        )
        try:
            _ov, ov_map = occ.fragment(inputs, [])
            occ.synchronize()
        except Exception as exc:
            raise RuntimeError(
                "to_full_body: OCC fragment failed (BOPAlgo). "
                f"Lower beam_length_factor (currently {beam_length_factor}) "
                "to shrink beam solids away from the coil surface, or use "
                "the wildmeshing path in gmsh.py.old."
            ) from exc
        if len(ov_map) != len(owners):
            raise RuntimeError(
                "to_full_body: fragment map length "
                f"{len(ov_map)} != {len(owners)} inputs"
            )

        owner_map = _entity_owner_map(ov_map, owners)
        if radius is not None:
            owner_map = _fillet_joints(occ, owner_map, radius)
        groups = {
            "winding_pack": [t for t, (_c, _s, mat) in owner_map.items() if mat == 0],
            "casing": [t for t, (_c, _s, mat) in owner_map.items() if mat == 1],
            "support": [t for t, (_c, _s, mat) in owner_map.items() if mat < 0],
        }
        for name, tags in groups.items():
            if tags:
                gmsh.model.addPhysicalGroup(3, tags, name=name)

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
