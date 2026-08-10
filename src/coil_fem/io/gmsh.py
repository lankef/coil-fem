"""OCC full-body meshing for a CoilFEMObjective → ``full_body_fields.vtu``.

Assumes circular beam cross-sections and rectangular coil cross-sections.
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
) -> Path:
    """Build a fused full-device TET10 mesh and write ``full_body_fields.vtu``.

    Parameters
    ----------
    Jstress : CoilFEMObjective
        Must wrap circular-beam ``CoilSupportBeams`` and rectangular coil meshes.
    mesh_scale : float
        Multiplier on gmsh ``MeshSizeMax`` / ``MeshSizeMin``.
    path : path-like
        Output VTU path (default ``full_body_fields.vtu``).

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
    if cs_type != "solid_circle":
        raise ValueError(
            f"to_full_body requires circular beams "
            f"(cross_section_type='solid_circle'); got {cs_type!r}."
        )
    if not all(isinstance(m, CoilMeshRectangle) for m in meshes):
        raise ValueError(
            "to_full_body requires rectangular coil meshes (CoilMeshRectangle)."
        )
    if coil_support._r_clamp is None or coil_support._sig_eps is None:
        raise ValueError("to_full_body requires fixed clamps on coil_support.")

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
    r_clamp = float(coil_support._r_clamp)
    eps_sigmoid = float(coil_support._sig_eps)
    k_clamp = float(support.k_clamp)

    geom = support.beam_geometry(curves, sdofs)
    x_start = np.asarray(geom["x_start"])
    x_end = np.asarray(geom["x_end"])
    A_all, *_ = support.cross_section_fn(sdofs)
    A = np.concatenate([np.atleast_1d(np.asarray(a)) for a in A_all])
    radii = np.sqrt(A / np.pi)

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

        solids = []
        for xs, xe, r in zip(x_start, x_end, radii):
            d = xe - xs
            tag = occ.addCylinder(
                xs[0], xs[1], xs[2], d[0], d[1], d[2], float(r),
            )
            solids.append((3, tag))

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

        r_min = float(radii.min())
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_scale * 0.5 * w1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_scale * 0.15 * r_min)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.model.mesh.generate(3)

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

        n_base = len(meshes)
        n_sym = Q_list.shape[0]
        phis_clamp = sdofs["phis"]
        clamp_centers = []
        for i in range(n_base):
            phi_i = np.asarray(phis_clamp[i], dtype=np.float64).ravel()
            c_base = np.asarray(curves[i].gamma_eval(phi_i), dtype=np.float64)
            for s in range(n_sym):
                clamp_centers.append(c_base @ Q_list[s].T)
        clamp_centers = np.vstack(clamp_centers)

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
