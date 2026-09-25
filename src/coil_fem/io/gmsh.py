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


def _coil_solid(occ, mesh, n_slices: int = _N_SLICES):
    """Loft one rectangular coil into two half solids; return ``(dim, tag)`` list."""
    fc = mesh.framed_curve
    phi = np.linspace(0.0, 1.0, n_slices, endpoint=False)
    r0 = np.asarray(fc.curve.gamma_eval(phi))
    _, p, q = fc.rotated_frame_eval(phi)
    p, q = np.asarray(p), np.asarray(q)

    corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    wires = []
    for k in range(n_slices):
        pts = [
            occ.addPoint(*(r0[k] + 0.5 * mesh.w1_outer * u * p[k]
                                 + 0.5 * mesh.w2_outer * v * q[k]))
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
    """Map fragment output volume tags to ``(owner_coil, owner_sym)``.

    ``owners[i]`` labels fragment input ``i``.  Each coil is several
    volumes (two half-lofts), so labels are per input volume, not per
    coil.  An output listed under several inputs (the overlap) is
    claimed by the conductor; beam labels are ``(-1, -1)``.  The first
    conductor to claim a tag keeps it.
    """
    owner_map: dict[int, tuple[int, int]] = {}
    for src, outs in zip(owners, ov_map):
        if src[0] >= 0:
            continue
        for dim, tag in outs:
            if dim == 3:
                owner_map[int(tag)] = (-1, -1)
    for src, outs in zip(owners, ov_map):
        if src[0] < 0:
            continue
        label = (int(src[0]), int(src[1]))
        for dim, tag in outs:
            if dim != 3:
                continue
            tag = int(tag)
            prev = owner_map.get(tag)
            if prev is not None and prev[0] >= 0:
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
    clamp_centers,
    r_clamp,
    eps_sigmoid,
    k_clamp,
    E,
    nu,
    rho,
    g_vec,
):
    # VTU contents for beam_dolfinx.py:
    #
    # FieldData — REQUIRED by load_vtu_problem / solve:
    #   r_clamp, eps_sigmoid, k_clamp, E, nu, rho, g_vec
    #   clamp_centers — omitted when there are no fixed clamps (empty
    #   arrays are rejected by PyVista FieldData)
    #
    # CellData — sanity / Paraview only (not read by beam_dolfinx; Lorentz
    # reclassifies quads from Jstress.json; Winkler uses FieldData spheres):
    #   owner_coil, owner_sym
    meshio.Mesh(
        points=points,
        cells=[("tetra10", cells)],
        cell_data={
            "owner_coil": [np.asarray(owner_coil, dtype=np.int32)],
            "owner_sym": [np.asarray(owner_sym, dtype=np.int32)],
        },
    ).write(path)
    grid = pv.read(str(path))
    clamp_centers = np.asarray(clamp_centers, dtype=np.float64).reshape(-1, 3)
    if clamp_centers.shape[0]:
        grid.field_data["clamp_centers"] = clamp_centers
    grid.field_data["r_clamp"] = np.array([r_clamp], dtype=np.float64)
    grid.field_data["eps_sigmoid"] = np.array([eps_sigmoid], dtype=np.float64)
    grid.field_data["k_clamp"] = np.array([k_clamp], dtype=np.float64)
    grid.field_data["E"] = np.array([E], dtype=np.float64)
    grid.field_data["nu"] = np.array([nu], dtype=np.float64)
    grid.field_data["rho"] = np.array([rho], dtype=np.float64)
    grid.field_data["g_vec"] = np.asarray(g_vec, dtype=np.float64)
    grid.save(str(path))


def _extract_tet10(owner_map):
    """Read the order-2 tet mesh and per-cell owners. Returns points, cells, owners."""
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    if node_tags.size == 0:
        raise RuntimeError("to_full_body: gmsh produced no mesh nodes")
    points = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)
    lut = np.full(int(node_tags.max()) + 1, -1, dtype=np.int64)
    lut[node_tags] = np.arange(node_tags.size)

    cells, owner_coil, owner_sym = [], [], []
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
        oc, osym = owner_map.get(int(tag), (-1, -1))
        cells.append(conn)
        owner_coil.append(np.full(len(conn), oc, dtype=np.int32))
        owner_sym.append(np.full(len(conn), osym, dtype=np.int32))
    if not cells:
        raise RuntimeError("to_full_body: fragment left no volumes to mesh")
    return (
        points,
        np.vstack(cells),
        np.concatenate(owner_coil),
        np.concatenate(owner_sym),
    )


def _fragment_inputs(occ, meshes, Q_list, beam_dimtags):
    """Full-device coil then beam volumes, with a parallel owner list.

    Coil owners are ``(base_coil, sym_image)``.  Beam owners are
    ``(-1, -1)``.  Each coil image may contribute more than one volume
    (the two half-lofts).
    """
    n_base = len(meshes)
    n_sym = len(Q_list)
    coil_vols = [[None] * n_base for _ in range(n_sym)]
    for i, mesh in enumerate(meshes):
        copies = _apply_sym_copies(occ, _coil_solid(occ, mesh), Q_list)
        for s, dts in enumerate(copies):
            coil_vols[s][i] = dts
    if beam_dimtags:
        beam_vols = _apply_sym_copies(occ, beam_dimtags, Q_list)
    else:
        beam_vols = [[] for _ in range(n_sym)]

    inputs, owners = [], []
    for s in range(n_sym):
        for i in range(n_base):
            for dim, tag in coil_vols[s][i]:
                if dim == 3:
                    inputs.append((3, tag))
                    owners.append((i, s))
    for s in range(n_sym):
        for dim, tag in beam_vols[s]:
            if dim == 3:
                inputs.append((3, tag))
                owners.append((-1, -1))
    return inputs, owners


def to_full_body(
    Jstress,
    mesh_scale: float = 0.5,
    path: str | Path = "full_body_fields.vtu",
    beam_length_factor: float = 0.95,
) -> Path:
    """Build a full-device TET10 mesh and write ``full_body_fields.vtu``.

    OCC ``fragment`` imprints beam–coil (and coil–coil) contacts so gmsh
    meshes each volume once with shared interface nodes.  CellData
    ``owner_coil`` / ``owner_sym`` label conductor cells; ``-1`` is
    support.  Overlap regions are labelled conductor.  When fixed clamps
    are disabled, ``clamp_centers`` is omitted from FieldData; consumers
    must treat that key as optional.

    Parameters
    ----------
    Jstress : CoilFEMObjective
        Must wrap ``CoilSupportBeams`` whose ``cross_section_type`` has a
        matching ``*_solid`` factory in
        :mod:`coil_fem.presets.cross_section_fns`, and rectangular coil
        meshes (:class:`~coil_fem.meshing.FramedCurveMeshRectangle`).
    mesh_scale : float
        Multiplier on gmsh ``MeshSizeMax``.  The size is
        ``mesh_scale * 0.5 * w1``.
    path : path-like
        Output VTU path (default ``full_body_fields.vtu``).
    beam_length_factor : float
        Beam solids are built with length ``beam_length_factor * L``
        (default 0.95) so the far end pulls back from the mating solid.
        Must be positive.

    Returns
    -------
    pathlib.Path
        Path written.

    Raises
    ------
    ValueError
        Rectangular meshes or an OCC solid factory are missing, or
        ``beam_length_factor`` is not positive.
    RuntimeError
        The OCC fragment failed.  Lower ``beam_length_factor``, or use
        the wildmeshing path in ``gmsh.py.old``.
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

    mesh_opts = fem.mesh_opts[0]
    w1 = float(mesh_opts["w1"])
    E = float(fem._E)
    nu = float(fem._nu)
    rho = float(fem._rho)
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
        conductor_tags = [t for t, (c, _s) in owner_map.items() if c >= 0]
        support_tags = [t for t, (c, _s) in owner_map.items() if c < 0]
        if conductor_tags:
            gmsh.model.addPhysicalGroup(3, conductor_tags, name="conductor")
        if support_tags:
            gmsh.model.addPhysicalGroup(3, support_tags, name="support")

        gmsh.option.setNumber("Mesh.MeshSizeMax", size_max)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        print(f"to_full_body: meshing, MeshSizeMax={size_max:.4g}")
        gmsh.model.mesh.generate(3)
        points, cells, owner_coil, owner_sym = _extract_tet10(owner_map)
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
        clamp_centers=clamp_centers,
        r_clamp=r_clamp,
        eps_sigmoid=eps_sigmoid,
        k_clamp=k_clamp,
        E=E,
        nu=nu,
        rho=rho,
        g_vec=g_vec,
    )
    return path
