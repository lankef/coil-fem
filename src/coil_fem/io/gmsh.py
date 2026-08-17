"""OCC full-body meshing for a CoilFEMObjective → ``full_body_fields.vtu``.

Beam cross-sections are resolved from ``presets.cross_section_fns`` via
``cross_section_type``; coil cross-sections must be rectangular
(:class:`~coil_fem.meshing.CoilMeshRectangle`).
"""

from __future__ import annotations

from pathlib import Path

import gmsh
import interpax
import meshio
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from coil_fem.meshing import CoilMeshRectangle
from coil_fem.presets import cross_section_fns


_N_SLICES = 96
_N_PHI = 4096
_UV_TOL = 2e-2


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
            occ.addPoint(*(r0[k] + 0.5 * mesh.w1 * u * p[k]
                                 + 0.5 * mesh.w2 * v * q[k]))
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


def _beam_solids(occ, support, sdofs, geom, solid_fn):
    """Build and place every beam OCC body; return ``(dim, tag)`` list.

    Cross-section DOFs are flattened per-group → per-beam in the same order
    as ``geom['x_start']`` / ``geom['gamma3']``.  Each body is created in the
    beam-local frame by ``solid_fn`` then mapped to global coordinates with
    ``affineTransform([gamma3 | x_start])``.
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
        dimtags = solid_fn(occ, dofs, float(L[b]))
        affine = list(np.hstack([gamma3[b], x_start[b, :, None]]).ravel())
        occ.affineTransform(dimtags, affine)
        solids.extend(dimtags)
    return solids


def _classify_nodes(X, meshes, Q_list, w1: float, w2: float):
    """Inverse-map nodes to base-coil ``(phi, u, v)``; return owner arrays."""
    N = X.shape[0]
    n_base = len(meshes)
    n_sym = Q_list.shape[0]
    dist_cut = 0.5 * np.hypot(w1, w2) * 1.05

    phi_s = np.linspace(0.0, 1.0, _N_PHI, endpoint=False)
    gamma_s, p_s, q_s = [], [], []
    for mesh in meshes:
        fc = mesh.framed_curve
        gamma_s.append(np.asarray(fc.curve.gamma_eval(phi_s)))
        _, p, q = fc.rotated_frame_eval(phi_s)
        p_s.append(np.asarray(p))
        q_s.append(np.asarray(q))

    sample_pts, sample_lab = [], []
    for s in range(n_sym):
        Q = Q_list[s]
        for i in range(n_base):
            sample_pts.append(gamma_s[i] @ Q.T)
            labs = np.empty((_N_PHI, 3), dtype=np.int32)
            labs[:, 0] = s
            labs[:, 1] = i
            labs[:, 2] = np.arange(_N_PHI)
            sample_lab.append(labs)
    tree = cKDTree(np.vstack(sample_pts))
    dist, nn = tree.query(X, k=3)
    cand_lab = np.vstack(sample_lab)[nn]

    owner_sym = np.full(N, -1, dtype=np.int8)
    owner_coil = np.full(N, -1, dtype=np.int8)

    for cand in range(3):
        unset = owner_coil < 0
        if not unset.any():
            break
        idxs = np.where(unset)[0]
        s_c = cand_lab[idxs, cand, 0]
        i_c = cand_lab[idxs, cand, 1]
        phi0 = phi_s[cand_lab[idxs, cand, 2]]
        d_c = dist[idxs, cand]
        keep_mask = d_c <= dist_cut

        for i in range(n_base):
            m = keep_mask & (i_c == i)
            if not m.any():
                continue
            sel = idxs[m]
            s_sel = s_c[m]
            phi_n = phi0[m].astype(np.float64).copy()
            Qs = Q_list[s_sel]
            y = np.einsum("ni,nij->nj", X[sel], Qs)

            curve = meshes[i].framed_curve.curve
            for _ in range(2):
                g0 = np.asarray(curve.gamma_eval(phi_n, 0))
                g1 = np.asarray(curve.gamma_eval(phi_n, 1))
                g2 = np.asarray(curve.gamma_eval(phi_n, 2))
                dvec = y - g0
                g = np.sum(dvec * g1, axis=1)
                gp = -np.sum(g1 * g1, axis=1) + np.sum(dvec * g2, axis=1)
                phi_n = phi_n - g / np.where(np.abs(gp) < 1e-30, 1e-30, gp)
            phi_n = np.mod(phi_n, 1.0)

            p_i = np.asarray(interpax.interp1d(
                phi_n, phi_s, p_s[i], method="cubic2", period=1.0,
            ))
            q_i = np.asarray(interpax.interp1d(
                phi_n, phi_s, q_s[i], method="cubic2", period=1.0,
            ))
            g0 = np.asarray(curve.gamma_eval(phi_n, 0))
            dvec = y - g0
            u = 2.0 * np.sum(dvec * p_i, axis=1) / w1
            v = 2.0 * np.sum(dvec * q_i, axis=1) / w2
            inside = (np.abs(u) <= 1.0 + _UV_TOL) & (np.abs(v) <= 1.0 + _UV_TOL)

            take = sel[inside]
            owner_sym[take] = s_sel[inside].astype(np.int8)
            owner_coil[take] = i

    return owner_coil, owner_sym


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
    # VTU contents for beam_dolfinx2.py:
    #
    # FieldData — REQUIRED by load_vtu_problem / solve:
    #   clamp_centers, r_clamp, eps_sigmoid, k_clamp, E, nu, rho, g_vec
    #
    # PointData — sanity / Paraview only (not read by beam_dolfinx2; Lorentz
    # reclassifies quads from Jstress.json; Winkler uses FieldData spheres):
    #   owner_coil, owner_sym
    meshio.Mesh(
        points=points,
        cells=[("tetra10", cells)],
        point_data={
            "owner_coil": owner_coil.astype(np.int32),
            "owner_sym": owner_sym.astype(np.int32),
        },
    ).write(path)
    grid = pv.read(str(path))
    grid.field_data["clamp_centers"] = np.asarray(clamp_centers, dtype=np.float64)
    grid.field_data["r_clamp"] = np.array([r_clamp], dtype=np.float64)
    grid.field_data["eps_sigmoid"] = np.array([eps_sigmoid], dtype=np.float64)
    grid.field_data["k_clamp"] = np.array([k_clamp], dtype=np.float64)
    grid.field_data["E"] = np.array([E], dtype=np.float64)
    grid.field_data["nu"] = np.array([nu], dtype=np.float64)
    grid.field_data["rho"] = np.array([rho], dtype=np.float64)
    grid.field_data["g_vec"] = np.asarray(g_vec, dtype=np.float64)
    grid.save(str(path))


def to_full_body(
    Jstress,
    mesh_scale: float = 0.5,
    path: str | Path = "full_body_fields.vtu",
    output_msh: bool = False,
    boolean_tol: float | None = None,
) -> Path:
    """Build a fused full-device TET10 mesh and write ``full_body_fields.vtu``.

    Parameters
    ----------
    Jstress : CoilFEMObjective
        Must wrap ``CoilSupportBeams`` whose ``cross_section_type`` has a
        matching ``*_solid`` factory in
        :mod:`coil_fem.presets.cross_section_fns`, and rectangular coil
        meshes (:class:`~coil_fem.meshing.CoilMeshRectangle`).
    mesh_scale : float
        Multiplier on gmsh ``MeshSizeMax`` / ``MeshSizeMin``.
    path : path-like
        Output VTU path (default ``full_body_fields.vtu``).
    output_msh : bool
        If True, also write ``full_mesh.msh`` next to ``path`` (for
        ``beam_dolfinx.py`` / ``gmshio.read_from_msh``).
    boolean_tol : float or None
        Fuzzy value for the OCC boolean unions (gmsh
        ``Geometry.ToleranceBoolean``), in metres.  Entities closer than
        this are treated as coincident, which stops the fuse from emitting
        sub-mesh-size sliver faces that the volume mesher rejects.  ``None``
        (default) uses ``1e-2`` times the smallest cross-section DOF.

    Returns
    -------
    pathlib.Path
        Path written.
    """
    path = Path(path)
    coil_support = Jstress._coil_support
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
    if not all(isinstance(m, CoilMeshRectangle) for m in meshes):
        raise ValueError(
            "to_full_body requires rectangular coil meshes (CoilMeshRectangle)."
        )

    mesh_opts = fem.mesh_opts[0]
    w1 = float(mesh_opts["w1"])
    w2 = float(mesh_opts["w2"])
    E = float(fem._E)
    nu = float(fem._nu)
    rho = float(fem._rho)
    g_vec = np.asarray(
        (fem.gravity_options or {}).get("g_vec", (0.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    # Fixed-sphere clamps are optional on CoilSupportBeams; when disabled,
    # _r_clamp / _sig_eps are None and support_dofs has no 'phis'.
    has_clamps = (
        coil_support._r_clamp is not None and coil_support._sig_eps is not None
    )
    r_clamp = float(coil_support._r_clamp) if has_clamps else 0.0
    eps_sigmoid = float(coil_support._sig_eps) if has_clamps else 0.0
    k_clamp = float(support.k_clamp)

    geom = support.beam_geometry(curves, sdofs)
    # MeshSizeMin tracks the smallest cross-section DOF (r_beam, t_beam, …).
    # For hollow sections this is typically the wall thickness, so the mesh
    # can become substantially finer than the outer footprint alone would
    # suggest.
    cs_vals = np.concatenate([
        np.atleast_1d(np.asarray(a, dtype=np.float64))
        for k in support._cross_section_dof_keys
        for a in sdofs[k]
    ])
    size_min = float(cs_vals.min())

    # gmsh is a process-global singleton: initialize at most once; clear if
    # a caller already owns a session so this call still starts clean.
    try:
        owned = not gmsh.isInitialized()
    except AttributeError:
        # Older gmsh builds lack isInitialized(); treat as uninitialized.
        owned = True
    if owned:
        gmsh.initialize()
    else:
        gmsh.clear()
    try:
        gmsh.model.add("device")
        occ = gmsh.model.occ

        tol_bool = 1e-2 * size_min if boolean_tol is None else boolean_tol
        gmsh.option.setNumber("Geometry.ToleranceBoolean", tol_bool)
        print(f"to_full_body: Geometry.ToleranceBoolean = {tol_bool:.3g}")

        solids = _beam_solids(occ, support, sdofs, geom, solid_fn)

        coil_solids = [_coil_solid(occ, m) for m in meshes]
        sector = solids + [dt for cs in coil_solids for dt in cs]
        fused_1fp, _ = occ.fuse(sector[:1], sector[1:])
        occ.synchronize()

        nfp, stellsym = support.nfp, support.stellsym
        Q_list = _symmetry_Qs(nfp, stellsym)
        flip_Q = np.diag([1.0, -1.0, -1.0])
        flip_list = (False, True) if stellsym else (False,)
        image_solids = []
        for k in range(nfp):
            phi = 2.0 * np.pi * k / nfp
            c, s = np.cos(phi), np.sin(phi)
            rot_Q = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            for flip in flip_list:
                if k == 0 and not flip:
                    continue
                Q = (flip_Q @ rot_Q) if flip else rot_Q
                affine = list(np.hstack([Q, np.zeros((3, 1))]).ravel())
                copies = occ.copy(fused_1fp)
                occ.affineTransform(copies, affine)
                image_solids += copies

        all_solids = fused_1fp + image_solids
        occ.fuse(all_solids[:1], all_solids[1:])
        occ.synchronize()

        vols = gmsh.model.getEntities(3)
        gmsh.model.addPhysicalGroup(3, [t for _, t in vols], name="device")

        size_max = mesh_scale * 0.5 * w1
        size_min_mesh = mesh_scale * 0.15 * size_min
        gmsh.option.setNumber("Mesh.MeshSizeMax", size_max)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size_min_mesh)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.model.mesh.generate(3)

        if output_msh:
            # Same artifact as mesh.ipynb / beam_dolfinx.MESH_PATH: one
            # physical volume "device", TET10, MSH 4.x via gmsh.write.
            msh_path = path.with_name("full_mesh.msh")
            gmsh.write(str(msh_path))

        # Nodes/cells must be read before finalize/clear.
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        X = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)
        node_tags = np.asarray(node_tags, dtype=np.int64)
        _, enodes = gmsh.model.mesh.getElementsByType(11)  # TET10
        enodes = np.asarray(enodes, dtype=np.int64)
        keep = np.isin(node_tags, np.unique(enodes))
        X, node_tags = X[keep], node_tags[keep]

        inv = np.zeros(int(node_tags.max()) + 1, dtype=np.int64)
        inv[node_tags] = np.arange(node_tags.size)
        # gmsh tet10 edge order (01)(12)(02)(03)(23)(13) → VTK (01)(12)(02)(03)(13)(23)
        cells = inv[enodes.reshape(-1, 10)][:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]]

        owner_coil, owner_sym = _classify_nodes(X, meshes, Q_list, w1, w2)

        # Mesh size summary: all / conductor (coil) / remaining (support).
        # Nodes use owner_coil from _classify_nodes; cells are coil if a
        # majority of their 4 corner vertices are coil-owned.
        n_nodes = int(X.shape[0])
        n_cells = int(cells.shape[0])
        coil_node = owner_coil >= 0
        n_coil_nodes = int(np.count_nonzero(coil_node))
        n_support_nodes = n_nodes - n_coil_nodes
        coil_cell = coil_node[cells[:, :4]].mean(axis=1) >= 0.5
        n_coil_cells = int(np.count_nonzero(coil_cell))
        n_support_cells = n_cells - n_coil_cells
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
            points=X,
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
    finally:
        if owned:
            gmsh.finalize()
        else:
            gmsh.clear()

    return path
