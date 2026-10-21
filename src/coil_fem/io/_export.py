"""Welded full-device STEP and TET10 export for simsopt objectives.

:func:`_export_simsopt` joins every beam to its coils with a convex weld of
casing material and meshes the whole device in one conforming pass. Call it
through :meth:`coil_fem.simsopt.objectives.CoilFEMObjective.export`.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
import warnings

import gmsh
import jax
import meshio
import numpy as np
import pyvista as pv
import vtk
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation

from coil_fem.coupling.beam_network import SupportBeams
from coil_fem.meshing import (
    FramedCurveMeshRectangle, FramedCurveMeshSection, corner_polygon,
)

_N_SLICES = 96
_TET10 = 11
# Gmsh tet10 edge mids differ from VTK on the last two slots.
_GMSH_TET10_TO_VTK = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 8], dtype=np.int64)
_QUALITY_MIN = 0.05
# A TET10 cell whose smallest node Jacobian is below this fraction of its
# straight-sided Jacobian gets its edge midpoints put back on the straight edges.
_DETJ_MIN_RATIO = 0.05
# VTK TET10 midpoint slots 4..9 sit on these vertex pairs.
_TET10_EDGES = np.array([(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)])


def _tet10_shape_grads() -> np.ndarray:
    """``(10, 10, 3)`` P2 shape-function gradients at the 10 reference nodes (VTK order)."""
    lam_v = np.eye(4)
    lam = np.vstack(
        [lam_v] + [0.5 * (lam_v[i] + lam_v[j]) for i, j in _TET10_EDGES]
    )
    dlam = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    dn = np.zeros((10, 10, 3))
    for p in range(10):
        for i in range(4):
            dn[p, i] = (4 * lam[p, i] - 1) * dlam[i]
        for k, (i, j) in enumerate(_TET10_EDGES):
            dn[p, 4 + k] = 4 * (lam[p, i] * dlam[j] + lam[p, j] * dlam[i])
    return dn


_TET10_DN = _tet10_shape_grads()


def _min_detj_ratio(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Smallest node Jacobian of each TET10 cell over its straight-sided Jacobian."""
    out = np.empty(len(cells))
    for s in range(0, len(cells), 500_000):
        x = points[cells[s:s + 500_000]]
        det = np.linalg.det(np.einsum("nic,pia->npca", x, _TET10_DN))
        out[s:s + 500_000] = det.min(axis=1) / np.linalg.det(x[:, 1:4] - x[:, :1])
    return out


def _straighten_folded(
    points: np.ndarray, cells: np.ndarray,
) -> tuple[int, int, np.ndarray]:
    """Put the edge midpoints of folded TET10 cells back on their straight edges.

    Edits ``points`` in place and returns ``(cells flagged, midpoint nodes
    reset, final per-cell min detJ ratio)``. Midpoint nodes are shared, so
    neighbours of a reset node are re-checked until no cell has a node
    Jacobian below ``_DETJ_MIN_RATIO`` of its straight-sided value. This
    terminates because a straight cell passes.
    """
    ratio = _min_detj_ratio(points, cells)
    bad = ratio < _DETJ_MIN_RATIO
    reset = np.zeros(len(points), dtype=bool)
    n_cells = 0
    while bad.any():
        c = cells[bad]
        n_cells += len(c)
        ends = c[:, _TET10_EDGES]  # (k, 6, 2)
        points[c[:, 4:]] = 0.5 * (points[ends[..., 0]] + points[ends[..., 1]])
        reset[c[:, 4:]] = True
        touched = np.flatnonzero(reset[cells].any(axis=1))
        ratio[touched] = _min_detj_ratio(points, cells[touched])
        bad = np.zeros(len(cells), dtype=bool)
        bad[touched] = ratio[touched] < _DETJ_MIN_RATIO
    return n_cells, int(reset.sum()), ratio

_JOINT_KEYS = ("delta", "h_contour", "h_hull", "edge_snap_fraction")


def _np(x) -> np.ndarray:
    """Host float64 copy of a numpy or JAX array."""
    return np.asarray(jax.device_get(x), dtype=np.float64)


def _default_delta(section: dict) -> float:
    """``min(w1, w2) / 4`` for a rectangle, ``r / 2`` for a circle (outer radius)."""
    if section["kind"] == "rect":
        return 0.25 * min(float(section["w1"]), float(section["w2"]))
    return 0.5 * float(section["r"])


def _options(joint_options, sections) -> dict:
    """Validate ``joint_options`` and fill defaults. ``h_*`` default to ``delta``.

    The default ``delta`` is the smallest :func:`_default_delta` over ``sections``.
    """
    raw = {} if joint_options is None else dict(joint_options)
    unknown = set(raw) - set(_JOINT_KEYS)
    if unknown:
        raise ValueError(f"unrecognized joint_options keys: {sorted(unknown)}")
    sections = list(sections)
    if raw.get("delta") is not None:
        delta = float(raw["delta"])
    elif sections:
        delta = min(_default_delta(s) for s in sections)
    else:
        raise ValueError("default delta needs at least one beam section")
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    h_contour = delta if raw.get("h_contour") is None else float(raw["h_contour"])
    h_hull = delta if raw.get("h_hull") is None else float(raw["h_hull"])
    if h_contour <= 0.0 or h_hull <= 0.0:
        raise ValueError("h_contour and h_hull must be positive")
    snap = 0.5 if raw.get("edge_snap_fraction") is None else float(raw["edge_snap_fraction"])
    if not 0.0 < snap <= 1.0:
        raise ValueError("edge_snap_fraction must be in (0, 1]")
    return {
        "delta": delta, "h_contour": h_contour, "h_hull": h_hull,
        "edge_snap_fraction": snap,
    }


# ============================================================================
# Topology
# ============================================================================


def _sym_maps(nfp: int, stellsym: bool) -> np.ndarray:
    """Orthogonal maps, identity first, in ``(k, flip)`` order. Shape ``(n, 3, 3)``."""
    flip_Q = np.diag([1.0, -1.0, -1.0])
    flips = (False, True) if stellsym else (False,)
    out = []
    for k in range(int(nfp)):
        phi = 2.0 * np.pi * k / int(nfp)
        c, s = np.cos(phi), np.sin(phi)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        for flip in flips:
            out.append(flip_Q @ rot if flip else rot)
    return np.asarray(out, dtype=np.float64)


def _image_of(Q_list: np.ndarray, Q: np.ndarray) -> int:
    """Index of ``Q`` in ``Q_list``. Raises if the symmetry is not in the list."""
    Q = np.asarray(Q, dtype=np.float64)
    for s, Qi in enumerate(Q_list):
        if np.allclose(Qi, Q, atol=1e-8):
            return int(s)
    raise ValueError("symmetry map is not one of the field-period images")


def _section(dofs: dict, n_sides_circle: int) -> dict:
    """Outer beam section from one beam's scalar cross-section dofs.

    A circular section is an ``n_sides_circle``-gon whose vertices lie on the
    circle. Vertex 0 is on local +y, and vertex k is at angle
    ``2 π k / n`` from +y toward +z.
    """
    if "w1_beam" in dofs:
        w1 = float(dofs["w1_beam"])
        w2 = float(dofs["w2_beam"])
        t = float(dofs["t_beam"]) if "t_beam" in dofs else None
        return {"kind": "rect", "w1": w1, "w2": w2, "t": t, "w_min": min(w1, w2)}
    n = int(n_sides_circle)
    if "r_beam" in dofs:
        r = float(dofs["r_beam"])
        return {
            "kind": "circle", "r": r, "ri": 0.0, "n_sides": n,
            "w_min": _polygon_width(r, n),
        }
    if "r_1_beam" in dofs:
        ro = max(float(dofs["r_1_beam"]), float(dofs["r_2_beam"]))
        ri = min(float(dofs["r_1_beam"]), float(dofs["r_2_beam"]))
        return {
            "kind": "circle", "r": ro, "ri": ri, "n_sides": n,
            "w_min": _polygon_width(ro, n),
        }
    raise ValueError(f"unrecognized beam section dofs {sorted(dofs)}")


def _polygon_width(r: float, n: int) -> float:
    """Smallest width of a regular n-gon with circumradius ``r``."""
    apothem = float(r) * np.cos(np.pi / int(n))
    if int(n) % 2 == 0:
        return 2.0 * apothem
    return float(r) + apothem


def _side_planes(section: dict) -> tuple[np.ndarray, np.ndarray]:
    """Outward side planes in the beam frame, ``n · (y, z) = offset``.

    Rectangle planes match ``addBox(0, -w2/2, -w1/2, L, w2, w1)``. Polygon
    planes are the sides of :func:`_section`'s n-gon.
    """
    if section["kind"] == "rect":
        hy = 0.5 * float(section["w2"])
        hz = 0.5 * float(section["w1"])
        normals = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        return normals, np.array([hy, hy, hz, hz])
    n = int(section["n_sides"])
    phi = np.pi / n + 2.0 * np.pi * np.arange(n) / n
    normals = np.stack([np.cos(phi), np.sin(phi)], axis=1)
    return normals, np.full(n, float(section["r"]) * np.cos(np.pi / n))


def _planes_in_frame(section: dict, frame, gamma3) -> tuple[np.ndarray, np.ndarray]:
    """Side planes of ``section`` written in a joint frame.

    The end joint's frame mirrors beam y, which turns an odd-sided polygon
    over, so the planes cannot be copied from the beam frame.
    """
    normals, offsets = _side_planes(section)
    turn = np.asarray(frame, float)[:, 1:].T @ np.asarray(gamma3, float)[:, 1:]
    return normals @ turn.T, offsets


def _master_beams(support, sdofs, geom, Q_list, n_sides_circle: int) -> list[dict]:
    """One record per beam, in ``beam_geometry`` order, with a joint at each coil end."""
    x_start = _np(geom["x_start"])
    x_end = _np(geom["x_end"])
    gamma3 = _np(geom["gamma3"])
    length = _np(geom["L"])
    flat = {
        k: np.concatenate([_np(a).reshape(-1) for a in sdofs[k]])
        for k in support._cross_section_dof_keys
    }
    flip_xy = np.diag([-1.0, -1.0, 1.0])
    recs: list[dict] = []
    b = 0

    def _add(i0, i1, tfm, n_add, cf):
        nonlocal b
        s_end = 0 if cf else _image_of(Q_list, support._tfm_Q[tfm])
        for _j in range(int(n_add)):
            dofs = {k: float(flat[k][b]) for k in flat}
            sec = _section(dofs, n_sides_circle)
            g = gamma3[b]
            L = float(length[b])
            start_frame = g.copy()
            joints = [{
                "coil": int(i0),
                "image": 0,
                "origin": x_start[b].copy(),
                "frame": start_frame,
                "planes": _planes_in_frame(sec, start_frame, g),
                "side": "start",
                "beam": b,
                "section": sec,
            }]
            if not cf:
                end_frame = g @ flip_xy
                joints.append({
                    "coil": int(i1),
                    "image": int(s_end),
                    "origin": x_end[b].copy(),
                    "frame": end_frame,
                    "planes": _planes_in_frame(sec, end_frame, g),
                    "side": "end",
                    "beam": b,
                    "section": sec,
                })
            recs.append({
                "index": b,
                "section": sec,
                "x_start": x_start[b].copy(),
                "x_end": x_end[b].copy(),
                "gamma3": g.copy(),
                "L": L,
                "cf": bool(cf),
                "joints": joints,
            })
            b += 1

    n_base = int(support.n_base)
    for i in range(n_base):
        i0, i1, tfm = support.cc_groups[i]
        _add(i0, i1, tfm, support.n_beam_cc[i], False)
        _add(i, i, "none", support.n_beam_cf[i], True)
    if support.stellsym:
        i0, i1, tfm = support.cc_groups[n_base]
        _add(i0, i1, tfm, support.n_beam_cc[n_base], False)
    for (i0, i1), side in zip(support.i_beam_cs, support.s_beam_cs):
        _add(i0, i1, "flip" if side else "flip_half", 1, False)
    if b != x_start.shape[0]:
        raise RuntimeError(
            f"beam records {b} != beam_geometry length {x_start.shape[0]}"
        )
    return recs


# ============================================================================
# Contours
# ============================================================================


class _Hull:
    """Convex hull with plane equations and a triangle mesh for outside distance."""

    def __init__(self, pts):
        # Rounding only finds duplicates. Rounded coordinates would tilt facets
        # through nearly collinear ring points off the beam's side planes.
        pts = np.asarray(pts, dtype=float)
        _keys, first = np.unique(np.round(pts, 9), axis=0, return_index=True)
        pts = pts[np.sort(first)]
        if len(pts) < 4:
            raise RuntimeError(f"convex hull needs 4 points, got {len(pts)}")
        try:
            hull = ConvexHull(pts)
        except Exception as exc:
            raise RuntimeError(
                f"convex hull failed on {len(pts)} points"
            ) from exc
        self.pts = pts
        self.hull = hull
        self.equations = np.asarray(hull.equations, dtype=float)
        tri = np.asarray(hull.simplices, dtype=np.int64)
        faces = np.hstack([
            np.full((len(tri), 1), 3, dtype=np.int64), tri,
        ]).ravel()
        self.mesh = pv.PolyData(pts, faces)

    def distance(self, points) -> np.ndarray:
        """Signed distance. Negative inside; outside is the distance to the surface."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        plane = points @ self.equations[:, :3].T + self.equations[:, 3]
        inside = np.all(plane <= 0.0, axis=1)
        dist = np.empty(len(points))
        if np.any(inside):
            dist[inside] = plane[inside].max(axis=1)
        if np.any(~inside):
            _cid, closest = self.mesh.find_closest_cell(
                points[~inside], return_closest_point=True,
            )
            dist[~inside] = np.linalg.norm(points[~inside] - closest, axis=1)
        return dist


def _section_uv(w1: float, w2: float, h: float):
    """Coil-section samples. Corners are samples, so coil edges are grid lines."""
    ha, hb = 0.5 * w1, 0.5 * w2
    corners = [(-ha, -hb), (ha, -hb), (ha, hb), (-ha, hb)]
    lengths = [w1, w2, w1, w2]
    uv, corner = [], []
    for k in range(4):
        corner.append(len(uv))
        n = max(1, int(np.ceil(lengths[k] / h)))
        a = np.asarray(corners[k], dtype=float)
        b = np.asarray(corners[(k + 1) % 4], dtype=float)
        for i in range(n):
            t = i / n
            uv.append((1.0 - t) * a + t * b)
    return np.asarray(uv, dtype=float), corner, ha, hb


def _centerline_frames(mesh, h: float):
    """Base-coil centerline samples at spacing ``h``, plus ``(p, q)`` and the length."""
    fc = mesh.framed_curve
    rough = np.linspace(0.0, 1.0, 400, endpoint=False)
    rr = _np(fc.curve.gamma_eval(rough))
    length = float(np.linalg.norm(np.diff(np.vstack([rr, rr[:1]]), axis=0), axis=1).sum())
    n_phi = max(8, int(np.ceil(length / h)))
    phi = np.linspace(0.0, 1.0, n_phi, endpoint=False)
    r = _np(fc.curve.gamma_eval(phi))
    _t, p, q = fc.rotated_frame_eval(phi)
    return phi, r, _np(p), _np(q), length, fc


def _phi_window(near: np.ndarray):
    """Smallest arc of samples covering ``near``. The flag is true for the whole coil."""
    n = int(near.size)
    if not np.any(near):
        raise RuntimeError("hull window does not meet the coil")
    if np.all(near):
        return np.arange(n), True
    false = np.concatenate([~near, ~near])
    best_len, best_at, i = 0, 0, 0
    while i < 2 * n:
        if not false[i]:
            i += 1
            continue
        j = i
        while j < 2 * n and false[j]:
            j += 1
        if j - i > best_len:
            best_len, best_at = j - i, i
        i = j
    best_len = min(best_len, n)
    start = (best_at + best_len) % n
    return (start + np.arange(n - best_len)) % n, False


def _dilate(mask: np.ndarray, k: int) -> np.ndarray:
    """Widen a periodic boolean mask by ``k`` samples each way."""
    out = np.asarray(mask, dtype=bool).copy()
    for i in range(1, k + 1):
        out |= np.roll(mask, i) | np.roll(mask, -i)
    return out


def _loops_of(poly) -> list[np.ndarray]:
    """Point arrays of each polyline in a vtk/pyvista line mesh."""
    lines = np.asarray(poly.lines)
    out, i = [], 0
    while i < len(lines):
        n = int(lines[i])
        ids = lines[i + 1:i + 1 + n]
        out.append(np.asarray(poly.points[ids], dtype=float))
        i += n + 1
    return out


def _coil_contour(
    mesh, Q, hull: _Hull, radius: float, w1: float, w2: float,
    h_contour: float, where: str, snap_tol: float, edge_clear: float = 0.0,
) -> dict:
    """Closed curve on a coil section ``(w1, w2)`` where the distance to ``hull`` is ``radius``.

    Returns ``{"pts": (n, 3), "n_loops": 1, "resid_max": float}``. Grid lines pass through the casing corners, so a crossing of a coil edge
    has a node on that edge. Each node is then moved onto the exact surface
    and the exact level set. A contiguous run within ``snap_tol`` of one
    casing corner moves onto that corner, and the loop is resampled on the
    framed surface. Unless it is on a corner, no node lies closer than
    ``edge_clear`` to a coil edge, so hull edges never run nearly along one.
    """
    Q = np.asarray(Q, dtype=float)
    radius = float(radius)
    phi, r, p, q, _length, fc = _centerline_frames(mesh, h_contour)
    uv, corner_s, ha, hb = _section_uv(float(w1), float(w2), h_contour)
    r_real = r @ Q.T
    center = hull.pts.mean(axis=0)
    reach = float(np.linalg.norm(hull.pts - center, axis=1).max())
    reach += radius + 0.5 * float(np.hypot(w1, w2))
    near = _dilate(np.linalg.norm(r_real - center, axis=1) <= reach, 2)
    ip, periodic = _phi_window(near)
    nr, ns = len(ip), len(uv)
    base = (
        r[ip][:, None, :]
        + uv[None, :, 0, None] * p[ip][:, None, :]
        + uv[None, :, 1, None] * q[ip][:, None, :]
    )
    real = base @ Q.T
    dist = hull.distance(real.reshape(-1, 3)).reshape(nr, ns)
    f = dist - radius
    f = np.where(np.abs(f) < 1e-15, 1e-15, f)

    crossings = []

    def _add(i, j, i2, j2, along_phi, on_window, on_corner):
        if f[i, j] * f[i2, j2] >= 0.0:
            return
        phi_a = float(phi[ip[i]])
        phi_b = float(phi[ip[i2]])
        if phi_b < phi_a - 0.5:
            phi_b += 1.0
        crossings.append({
            "phi": (phi_a, phi_b),
            "uv": (uv[j], uv[j2]),
            "f": (float(f[i, j]), float(f[i2, j2])),
            "xyz": (real[i, j], real[i2, j2]),
            "along_phi": along_phi,
            "on_window": on_window,
            "on_corner": on_corner,
        })

    for i in range(nr):
        on_window = (not periodic) and (i == 0 or i == nr - 1)
        for j in range(ns):
            _add(i, j, i, (j + 1) % ns, False, on_window, False)
    n_phi_edges = nr if periodic else nr - 1
    corner = set(corner_s)
    for i in range(n_phi_edges):
        i2 = (i + 1) % nr
        for j in range(ns):
            _add(i, j, i2, j, True, False, j in corner)
    if not crossings:
        raise RuntimeError(f"{where}: coil contour has no level-set crossing")

    def _surface(t, item_phi, item_uv):
        t = np.asarray(t, dtype=float)
        phi_t = (item_phi[:, 0] + t * (item_phi[:, 1] - item_phi[:, 0])) % 1.0
        u = (1.0 - t) * item_uv[:, 0, 0] + t * item_uv[:, 1, 0]
        v = (1.0 - t) * item_uv[:, 0, 1] + t * item_uv[:, 1, 1]
        r0 = _np(fc.curve.gamma_eval(phi_t))
        _tt, pp, qq = fc.rotated_frame_eval(phi_t)
        pp, qq = _np(pp), _np(qq)
        return (r0 + u[:, None] * pp + v[:, None] * qq) @ Q.T, u, v, pp, qq, phi_t

    phi_ab = np.array([c["phi"] for c in crossings])
    uv_ab = np.array([c["uv"] for c in crossings])
    f0 = np.array([c["f"][0] for c in crossings])
    f1 = np.array([c["f"][1] for c in crossings])
    a = np.zeros(len(crossings))
    b = np.ones(len(crossings))
    fa, fb = f0.copy(), f1.copy()
    for _it in range(20):
        denom = fb - fa
        t = np.where(
            np.abs(denom) < 1e-15,
            0.5 * (a + b),
            a - fa * (b - a) / np.where(np.abs(denom) < 1e-15, 1.0, denom),
        )
        t = np.where((t <= a) | (t >= b), 0.5 * (a + b), t)
        xyz, _u, _v, _pp, _qq, _phi = _surface(t, phi_ab, uv_ab)
        ft = hull.distance(xyz) - radius
        left = fa * ft <= 0.0
        b = np.where(left, t, b)
        fb = np.where(left, ft, fb)
        a = np.where(~left, t, a)
        fa = np.where(~left, ft, fa)
    t_ref = 0.5 * (a + b)
    xyz, u, v, pp, qq, phi_t = _surface(t_ref, phi_ab, uv_ab)
    t_lin = f0 / (f0 - f1)
    lin = np.array([
        (1.0 - tl) * c["xyz"][0] + tl * c["xyz"][1]
        for tl, c in zip(t_lin, crossings)
    ])

    g = np.concatenate([real, real[:, :1, :]], axis=1)
    dgrid = np.concatenate([dist, dist[:, :1]], axis=1)
    if periodic:
        g = np.concatenate([g, g[:1]], axis=0)
        dgrid = np.concatenate([dgrid, dgrid[:1]], axis=0)
    grid = pv.StructuredGrid()
    grid.dimensions = (g.shape[1], g.shape[0], 1)
    grid.points = g.reshape(-1, 3)
    grid.point_data["dist"] = dgrid.reshape(-1)
    iso = grid.contour([radius])
    if iso.n_points == 0:
        raise RuntimeError(f"{where}: coil contour is empty")
    iso = iso.clean(tolerance=1e-9)
    stripper = vtk.vtkStripper()
    stripper.SetInputData(iso)
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()
    poly = pv.wrap(stripper.GetOutput())
    loops = _loops_of(poly)
    closed = []
    for loop in loops:
        gap = np.linalg.norm(loop[0] - loop[-1])
        closed.append(bool(gap < 1e-8))
    if len(loops) != 1 or not closed[0]:
        raise RuntimeError(
            f"{where}: expected one closed coil contour, "
            f"got {len(loops)} loops ({int(sum(closed))} closed)"
        )
    vtk_pts = loops[0]
    if np.linalg.norm(vtk_pts[0] - vtk_pts[-1]) < 1e-8:
        vtk_pts = vtk_pts[:-1]
    used: set[int] = set()
    order = []
    for pnt in vtk_pts:
        dall = np.linalg.norm(lin - pnt, axis=1)
        j_near = int(np.argmin(dall))
        if dall[j_near] < 1e-5 and j_near in used:
            continue
        free = dall.copy()
        if used:
            free[list(used)] = np.inf
        j = int(np.argmin(free))
        if not np.isfinite(free[j]) or free[j] > 1e-4:
            raise RuntimeError(f"{where}: coil contour point is off the grid edges")
        used.add(j)
        order.append(j)
    if len(order) < 3:
        raise RuntimeError(f"{where}: coil contour has fewer than 3 nodes")
    if any(crossings[j]["on_window"] for j in order):
        raise RuntimeError(f"{where}: coil contour touches the phi-window edge")
    phi_s = np.asarray(phi_t[order], dtype=float)
    uu = np.asarray(u[order], dtype=float)
    vv = np.asarray(v[order], dtype=float)
    anchor = np.array([bool(crossings[j]["on_corner"]) for j in order])

    def _eval_loop(phi_loop, u_loop, v_loop):
        """Framed casing points for one ordered loop."""
        phi_mod = np.mod(np.asarray(phi_loop, dtype=float), 1.0)
        r0_base = _np(fc.curve.gamma_eval(phi_mod))
        _tt, pp, qq = fc.rotated_frame_eval(phi_mod)
        pp, qq = _np(pp), _np(qq)
        u_loop = np.asarray(u_loop, dtype=float)
        v_loop = np.asarray(v_loop, dtype=float)
        xyz_loop = (r0_base + u_loop[:, None] * pp + v_loop[:, None] * qq) @ Q.T
        return xyz_loop

    def _corner_labels(u_loop, v_loop):
        """Corner identity of each point strictly inside ``snap_tol``, else None."""
        labels = []
        for u_s, v_s in zip(u_loop, v_loop):
            du = abs(abs(u_s) - ha)
            dv = abs(abs(v_s) - hb)
            if du <= dv:
                dist = hb - abs(v_s)
                ambiguous = abs(v_s) <= 1e-15 * max(hb, 1.0)
            else:
                dist = ha - abs(u_s)
                ambiguous = abs(u_s) <= 1e-15 * max(ha, 1.0)
            if ambiguous or not dist < float(snap_tol):
                labels.append(None)
                continue
            labels.append((1 if u_s >= 0.0 else -1, 1 if v_s >= 0.0 else -1))
        return labels

    def _single_runs(labels):
        """Maximal runs of one corner. A corner split into two runs is left out."""
        n_lab = len(labels)
        breaks = [
            i for i in range(n_lab)
            if labels[i] is None or labels[i] != labels[(i - 1) % n_lab]
        ]
        runs = []
        if not breaks:
            if labels and labels[0] is not None:
                runs.append((labels[0], list(range(n_lab))))
        else:
            for b0, b1 in zip(breaks, breaks[1:] + [breaks[0] + n_lab]):
                idxs = [i % n_lab for i in range(b0, b1)]
                lab = labels[idxs[0]]
                if lab is not None and all(labels[i] == lab for i in idxs):
                    runs.append((lab, idxs))
        counts = {}
        for lab, _idxs in runs:
            counts[lab] = counts.get(lab, 0) + 1
        return [(lab, idxs) for lab, idxs in runs if counts[lab] == 1]

    done = set()
    while True:
        runs = [
            (lab, idxs) for lab, idxs in _single_runs(_corner_labels(uu, vv))
            if lab not in done
        ]
        if not runs:
            break
        lab, idxs = runs[0]
        saved = (phi_s.copy(), uu.copy(), vv.copy(), anchor.copy())
        su, sv = lab
        for i in idxs:
            uu[i] = su * ha
            vv[i] = sv * hb
            anchor[i] = True
        xyz_now = _eval_loop(phi_s, uu, vv)
        keep = []
        tol_collapse = 1e-8 * float(h_contour)
        for i in range(len(phi_s)):
            if keep and np.linalg.norm(xyz_now[i] - xyz_now[keep[-1]]) < tol_collapse:
                if anchor[i] and not anchor[keep[-1]]:
                    keep[-1] = i
                continue
            keep.append(i)
        if (
            len(keep) >= 2
            and np.linalg.norm(xyz_now[keep[0]] - xyz_now[keep[-1]]) < tol_collapse
        ):
            if anchor[keep[0]]:
                keep.pop()
            else:
                keep.pop(0)
        done.add(lab)
        if len(keep) < 3:
            phi_s, uu, vv, anchor = saved
            continue
        phi_s, uu, vv, anchor = phi_s[keep], uu[keep], vv[keep], anchor[keep]

    m = len(phi_s)
    anchor_idx = [int(i) for i in np.flatnonzero(anchor)] or [0]
    out_phi, out_u, out_v, out_corner = [], [], [], []
    for a, b in zip(anchor_idx, anchor_idx[1:] + [anchor_idx[0] + m]):
        idx = [i % m for i in range(a, b + 1)]
        seg_phi = np.asarray(phi_s[idx], dtype=float).copy()
        for k in range(1, len(idx)):
            while seg_phi[k] - seg_phi[k - 1] > 0.5:
                seg_phi[k] -= 1.0
            while seg_phi[k] - seg_phi[k - 1] < -0.5:
                seg_phi[k] += 1.0
        seg_u, seg_v = uu[idx], vv[idx]
        xyz_seg = _eval_loop(seg_phi, seg_u, seg_v)
        step = np.linalg.norm(np.diff(xyz_seg, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(step)])
        total = float(arc[-1])
        n_div = 1 if total < 1e-15 else max(1, int(np.ceil(total / float(h_contour))))
        targets = np.linspace(0.0, total, n_div + 1)
        for t in targets[:-1]:
            j = int(np.searchsorted(arc, t, side="right") - 1)
            j = min(max(j, 0), len(seg_phi) - 2)
            span = arc[j + 1] - arc[j]
            f = 0.0 if span < 1e-15 else (t - arc[j]) / span
            out_phi.append((1.0 - f) * seg_phi[j] + f * seg_phi[j + 1])
            out_u.append((1.0 - f) * seg_u[j] + f * seg_u[j + 1])
            out_v.append((1.0 - f) * seg_v[j] + f * seg_v[j + 1])
            out_corner.append(t == 0.0 and bool(anchor[a]))
    phi_s = np.asarray(out_phi, dtype=float)
    uu = np.asarray(out_u, dtype=float)
    vv = np.asarray(out_v, dtype=float)
    # Drop nodes that hug a coil edge without sitting on a corner.
    to_edge = np.hypot(ha - np.abs(uu), hb - np.abs(vv))
    keep = np.asarray(out_corner) | (to_edge >= float(edge_clear))
    if keep.sum() < 3:
        raise RuntimeError(f"{where}: fewer than 3 coil contour nodes after the edge clearance")
    phi_s, uu, vv = phi_s[keep], uu[keep], vv[keep]
    xyz = _eval_loop(phi_s, uu, vv)
    resid = np.abs(hull.distance(xyz) - radius)
    return {
        "pts": xyz,
        "n_loops": 1,
        "resid_max": float(resid.max()) if len(resid) else 0.0,
    }


def _casing_sdf(mesh, Q, h: float, k: int = 12):
    """Return ``sdf(P)``, the signed distance from points ``(n, 3)`` to the sharp casing.

    The casing is a union of short rectangular slabs, one per centreline
    sample (spacing about ``h``), placed under ``Q``. Negative inside.
    """
    _phi, r, p, q, _length, fc = _centerline_frames(mesh, h)
    # _centerline_frames drops the tangent; the slab length needs it.
    t = _np(fc.rotated_frame_eval(_phi)[0])
    t = t / np.linalg.norm(t, axis=1, keepdims=True)
    r, p, q, t = (a @ np.asarray(Q, dtype=float).T for a in (r, p, q, t))
    ha, hb = 0.5 * float(mesh.w1_outer), 0.5 * float(mesh.w2_outer)
    # Slabs overlap by 20% so no gap opens on a bend.
    hl = 0.6 * float(np.linalg.norm(np.roll(r, -1, axis=0) - r, axis=1).max())
    tree = cKDTree(r)
    k = min(int(k), len(r))

    def sdf(P):
        P = np.atleast_2d(np.asarray(P, dtype=float))
        _d, idx = tree.query(P, k=k)
        rel = P[:, None, :] - r[idx]
        u = np.einsum("nkj,nkj->nk", rel, p[idx])
        v = np.einsum("nkj,nkj->nk", rel, q[idx])
        s = np.einsum("nkj,nkj->nk", rel, t[idx])
        # Signed box distance of every candidate slab; the union is the minimum.
        e = np.stack([np.abs(u) - ha, np.abs(v) - hb, np.abs(s) - hl], axis=-1)
        box = np.linalg.norm(np.maximum(e, 0.0), axis=-1) + np.minimum(e.max(-1), 0.0)
        return box.min(axis=1)

    return sdf


def _beam_loops(
    sdf, joint: dict, length: float, delta: float, shrink: float, h: float,
    edge_clear: float = 0.0,
):
    """Ray-march the two beam loops of one joint. Returns ``(ring1, ring2)``.

    Each ring is ``(n, 3)`` in the joint frame, ``(x, y, z)`` with ``x`` along
    the beam away from the endpoint. ``ring1`` is where the beam sides meet
    the coil grown by ``delta``. ``ring2`` is the same on the beam shrunk
    inward by ``shrink`` and the coil grown by ``delta + shrink``. Ring points
    sit exactly on the side planes (corners on two), so no snapping is needed.
    Unless it is a corner, no ring point lies closer than ``edge_clear`` to a
    beam edge, so hull edges never run nearly along one.
    ``sdf`` maps real points ``(n, 3)`` to the signed distance to the casing.
    """
    origin = np.asarray(joint["origin"], dtype=float)
    g = np.asarray(joint["frame"], dtype=float)
    normals, offsets = (np.asarray(a, dtype=float) for a in joint["planes"])
    where = f"beam {joint['beam']} {joint['side']}"
    # Marching step: fine enough not to step over a thin crossing, capped so a
    # long beam costs at most ~400 distance queries per sample.
    ds = max(0.5 * float(delta), float(length) / 400.0)
    xs = np.linspace(0.0, float(length), int(np.ceil(length / ds)) + 1)

    rings = []
    for radius, off in ((delta, offsets), (delta + shrink, offsets - shrink)):
        if np.any(off <= 0.0):
            raise RuntimeError(f"{where}: shrink {shrink:.3e} exceeds the beam half-width")

        # Section polygon from the side planes. Sorting the normals by angle
        # makes consecutive planes neighbours, so vertex k is the corner
        # between plane k and plane k + 1.
        order = np.argsort(np.arctan2(normals[:, 1], normals[:, 0]))
        nrm, off = normals[order], off[order]
        m = len(nrm)
        verts = np.array([
            np.linalg.solve(nrm[[k, (k + 1) % m]], off[[k, (k + 1) % m]])
            for k in range(m)
        ])

        # Perimeter samples with every corner included. Plane k runs from
        # vertex k - 1 to vertex k. Linear interpolation keeps the points on it.
        yz = []
        for k in range(m):
            a, b = verts[k - 1], verts[k]
            side = float(np.linalg.norm(b - a))
            n = max(1, int(np.ceil(side / h)))
            t = np.linspace(0.0, 1.0, n, endpoint=False)
            keep = (t == 0.0) | ((t * side >= edge_clear) & ((1.0 - t) * side >= edge_clear))
            yz.append(a + t[keep, None] * (b - a))
        yz = np.vstack(yz)
        ns, nx = len(yz), len(xs)
        base = origin + yz[:, 0:1] * g[:, 1] + yz[:, 1:2] * g[:, 2]

        # f <= 0 inside the grown coil. The beam starts there, so every ray
        # must change sign once; the first sign change is the cut.
        P = base[:, None, :] + xs[None, :, None] * g[:, 0]
        f = sdf(P.reshape(-1, 3)).reshape(ns, nx) - radius
        if np.any(f[:, 0] > 0.0):
            raise RuntimeError(
                f"{where}: the beam endpoint is outside the coil grown by "
                f"{radius:.3e} ({int(np.sum(f[:, 0] > 0.0))} of {ns} samples)"
            )
        out = f > 0.0
        if not np.all(out.any(axis=1)):
            raise RuntimeError(
                f"{where}: {int(np.sum(~out.any(axis=1)))} of {ns} rays never leave "
                f"the coil grown by {radius:.3e}"
            )
        first = out.argmax(axis=1)
        lo, hi = xs[first - 1], xs[first]

        # Bisect each ray between its last inside and first outside sample.
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            fm = sdf(base + mid[:, None] * g[:, 0]) - radius
            inside = fm <= 0.0
            lo = np.where(inside, mid, lo)
            hi = np.where(inside, hi, mid)
        rings.append(np.column_stack([0.5 * (lo + hi), yz]))
    return rings[0], rings[1]


# ============================================================================
# OCC solids
# ============================================================================


def _rigid(occ, dimtags, R, t=None) -> None:
    """Apply ``x -> R x + t`` with a proper rotation ``R``.

    Uses rotate / translate, which keep planes analytic.
    ``occ.affineTransform`` and ``occ.mirror`` turn every face into a
    B-spline.
    """
    R = np.asarray(R, dtype=float)
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-8) or np.linalg.det(R) < 0.0:
        raise ValueError("rigid placement needs a proper rotation")
    rv = Rotation.from_matrix(R).as_rotvec()
    angle = float(np.linalg.norm(rv))
    if angle > 1e-14:
        axis = rv / angle
        occ.rotate(dimtags, 0.0, 0.0, 0.0, float(axis[0]), float(axis[1]), float(axis[2]), angle)
    if t is not None and np.any(np.asarray(t, dtype=float) != 0.0):
        t = np.asarray(t, dtype=float)
        occ.translate(dimtags, float(t[0]), float(t[1]), float(t[2]))


def _vols(dimtags) -> list[tuple[int, int]]:
    """Volume dim-tags from an OCC return list."""
    return [(3, int(t)) for d, t in dimtags if int(d) == 3]


def _drop_free_curves(occ) -> None:
    """Remove curves that belong to no volume.

    ``addThruSections`` copies its input wires, and removing the wire tags
    themselves does nothing: a wire is not a curve. The copies stay on the
    solid; the originals are free and are removed here, points included.
    """
    occ.synchronize()
    used: set[int] = set()
    for _dim, tag in gmsh.model.getEntities(3):
        faces = gmsh.model.getBoundary([(3, int(tag))], oriented=False, recursive=False)
        for dim, face in faces:
            if int(dim) != 2:
                continue
            for d, curve in gmsh.model.getBoundary(
                [(2, abs(int(face)))], oriented=False, recursive=False,
            ):
                if int(d) == 1:
                    used.add(abs(int(curve)))
    free = [
        (1, int(tag))
        for _dim, tag in gmsh.model.getEntities(1)
        if abs(int(tag)) not in used
    ]
    if free:
        occ.remove(free, recursive=True)


def _coil_loft(
    occ, mesh, w1: float, w2: float, rounding=None,
) -> list[tuple[int, int]]:
    """Coil as two thru-section halves. Returns volume dim-tags.

    ``rounding=(r, n)`` cuts the corners with the coil mesh's polygonal chamfer
    (:func:`~coil_fem.meshing.corner_polygon`); ``None`` is the sharp rectangle.
    """
    fc = mesh.framed_curve
    phi = np.linspace(0.0, 1.0, _N_SLICES, endpoint=False)
    r0 = _np(fc.curve.gamma_eval(phi))
    _t, p, q = fc.rotated_frame_eval(phi)
    p, q = _np(p), _np(q)
    if rounding is None:
        uv = 0.5 * np.array([[-w1, -w2], [w1, -w2], [w1, w2], [-w1, w2]], dtype=float)
    else:
        uv = corner_polygon(float(w1), float(w2), float(rounding[0]), int(rounding[1]))
    n_pts = len(uv)
    wires = []
    for k in range(_N_SLICES):
        xyz = r0[k] + uv[:, 0:1] * p[k] + uv[:, 1:2] * q[k]
        pts = [occ.addPoint(float(c[0]), float(c[1]), float(c[2])) for c in xyz]
        lines = [occ.addLine(pts[a], pts[(a + 1) % n_pts]) for a in range(n_pts)]
        wires.append(occ.addWire(lines))
    half = _N_SLICES // 2
    out = list(occ.addThruSections(wires[:half + 1], makeSolid=True, makeRuled=False))
    out += list(occ.addThruSections(wires[half:] + [wires[0]], makeSolid=True, makeRuled=False))
    _drop_free_curves(occ)
    return _vols(out)


def _beam_solid(occ, rec: dict) -> list[tuple[int, int]]:
    """Beam body in beam coordinates, then placed in real coordinates. Hollow bores stay open."""
    sec = rec["section"]
    L = float(rec["L"])
    if sec["kind"] == "rect":
        w1, w2 = sec["w1"], sec["w2"]
        dimtags = [(3, occ.addBox(0.0, -0.5 * w2, -0.5 * w1, L, w2, w1))]
        t = sec["t"]
        if t is not None:
            t = min(float(t), min(w1, w2) / 2.0)
            bi, hi = w1 - 2.0 * t, w2 - 2.0 * t
            if bi > 0.0 and hi > 0.0:
                eps = max(1e-6 * L, 1e-12)
                inner = [(3, occ.addBox(-eps, -0.5 * hi, -0.5 * bi, L + 2.0 * eps, hi, bi))]
                dimtags, _map = occ.cut(dimtags, inner)
                dimtags = _vols(dimtags)
    else:
        n = int(sec["n_sides"])
        dimtags = _ngon_prism(occ, float(sec["r"]), n, L)
        if sec["ri"] > 0.0:
            eps = max(1e-6 * L, 1e-12)
            inner = _ngon_prism(occ, float(sec["ri"]), n, L + 2.0 * eps, x0=-eps)
            dimtags, _map = occ.cut(dimtags, inner)
            dimtags = _vols(dimtags)
    g = np.asarray(rec["gamma3"], dtype=float)
    if np.linalg.det(g) < 0.0:
        # Rectangles and polygons with a vertex on +y are symmetric in local z
        # for every n, so flipping z keeps the solid. Flipping y would not,
        # for an odd number of sides.
        g = g @ np.diag([1.0, 1.0, -1.0])
    _rigid(occ, dimtags, g, rec["x_start"])
    return dimtags


def _ngon_prism(occ, r: float, n: int, length: float, x0: float = 0.0) -> list[tuple[int, int]]:
    """Solid n-gon prism along +x. Vertex 0 lies on local +y, circumradius ``r``."""
    tags = []
    for k in range(int(n)):
        theta = 2.0 * np.pi * k / int(n)
        tags.append(occ.addPoint(float(x0), float(r * np.cos(theta)), float(r * np.sin(theta))))
    lines = [occ.addLine(tags[k], tags[(k + 1) % int(n)]) for k in range(int(n))]
    face = occ.addPlaneSurface([occ.addCurveLoop(lines)])
    return _vols(occ.extrude([(2, face)], float(length), 0.0, 0.0))


def _oriented_simplices(hull: ConvexHull) -> np.ndarray:
    """Hull triangles wound counter-clockwise seen from outside.

    Qhull's ``simplices`` carry no consistent orientation, and the normals of
    sliver triangles (collinear samples) are noise, so winding is propagated
    across shared edges of the closed mesh rather than read off normals.
    """
    tri = np.array(hull.simplices, dtype=np.int64)
    pts = np.asarray(hull.points, dtype=float)
    owners: dict[tuple[int, int], list[int]] = defaultdict(list)
    for t, (a, b, c) in enumerate(tri):
        for i, j in ((a, b), (b, c), (c, a)):
            owners[(min(i, j), max(i, j))].append(t)
    done = np.zeros(len(tri), dtype=bool)
    done[0] = True
    stack = [0]
    while stack:
        t = stack.pop()
        a, b, c = (int(v) for v in tri[t])
        for i, j in ((a, b), (b, c), (c, a)):
            for s in owners[(min(i, j), max(i, j))]:
                if done[s]:
                    continue
                x, y, z = (int(v) for v in tri[s])
                if (i, j) in ((x, y), (y, z), (z, x)):
                    tri[s] = tri[s][::-1]
                done[s] = True
                stack.append(s)
    p0, p1, p2 = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    if np.sum(np.einsum("ij,ij->i", p0, np.cross(p1, p2))) < 0.0:
        tri = tri[:, ::-1]
    return tri


def _facet_groups(hull: ConvexHull) -> list[np.ndarray]:
    """Groups of coplanar hull triangles. Each group is an ``(n, 3)`` index array.

    A group is every unassigned triangle whose vertices lie on the plane of
    the largest remaining triangle, so slivers join their facet.
    """
    tri = _oriented_simplices(hull)
    pts = np.asarray(hull.points, dtype=float)
    eq = np.asarray(hull.equations, dtype=float)
    p0, p1, p2 = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    scale = float(np.ptp(pts, axis=0).max())
    tol = 1e-8 * max(scale, 1.0)
    used = np.zeros(len(tri), dtype=bool)
    groups = []
    for i in np.argsort(-area):
        if used[i]:
            continue
        dist = np.abs(pts @ eq[i, :3] + eq[i, 3])
        same = ~used & np.all(dist[tri] < tol, axis=1)
        used[same] = True
        groups.append(tri[same])
    return groups


def _boundary_loops(triangles: np.ndarray) -> list[list[int]]:
    """Oriented boundary loops of a consistently oriented triangle group."""
    count: dict[tuple[int, int], int] = {}
    directed = []
    for tri in triangles:
        for a, b in ((int(tri[0]), int(tri[1])), (int(tri[1]), int(tri[2])), (int(tri[2]), int(tri[0]))):
            key = (a, b) if a < b else (b, a)
            count[key] = count.get(key, 0) + 1
            directed.append((a, b))
    nxt: dict[int, list[int]] = defaultdict(list)
    boundary = []
    for a, b in directed:
        key = (a, b) if a < b else (b, a)
        if count[key] == 1:
            nxt[a].append(b)
            boundary.append((a, b))
    seen: set[tuple[int, int]] = set()
    loops = []
    for a0, b0 in boundary:
        if (a0, b0) in seen:
            continue
        loop = [a0]
        cur = a0
        while True:
            nxt_v = next((c for c in nxt[cur] if (cur, c) not in seen), None)
            if nxt_v is None:
                break
            seen.add((cur, nxt_v))
            if nxt_v == loop[0]:
                break
            loop.append(nxt_v)
            cur = nxt_v
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def _polyhedron(occ, pts, hull: ConvexHull) -> int:
    """Sew hull facets into one OCC volume. Returns the volume tag."""
    tag_of: dict[int, int] = {}
    line_of: dict[tuple[int, int], int] = {}

    def point(i: int) -> int:
        if i not in tag_of:
            tag_of[i] = occ.addPoint(float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2]))
        return tag_of[i]

    def line(i: int, j: int) -> int:
        """Shared edge curve; the reverse direction is the negated tag."""
        if (j, i) in line_of:
            return -line_of[(j, i)]
        if (i, j) not in line_of:
            line_of[(i, j)] = occ.addLine(point(i), point(j))
        return line_of[(i, j)]

    faces = []
    for group in _facet_groups(hull):
        for loop in _boundary_loops(group):
            wire = occ.addCurveLoop([line(loop[k], loop[(k + 1) % len(loop)]) for k in range(len(loop))])
            faces.append(occ.addPlaneSurface([wire]))
    shell = occ.addSurfaceLoop(faces, sewing=True)
    return int(occ.addVolume([shell]))


def _hull_solid(occ, pts) -> tuple[int, float]:
    """Convex OCC solid. The OCC volume must match ``ConvexHull.volume``."""
    cloud = _Hull(pts)
    tag = _polyhedron(occ, cloud.pts, cloud.hull)
    occ.synchronize()
    mass = float(occ.getMass(3, tag))
    ref = float(cloud.hull.volume)
    if abs(mass - ref) > 1e-6 * max(abs(ref), 1e-12):
        raise RuntimeError(
            f"hull solid volume {mass:.6e} != convex hull {ref:.6e}"
        )
    return tag, ref


# ============================================================================
# Write and mesh
# ============================================================================


def _write_step(path, vols, names) -> None:
    """Write ``vols`` to a STEP file. Other volumes stay in the model but are hidden.

    ``Geometry.OCCExportOnlyVisible`` drops the hidden volumes in gmsh 4.15.
    """
    path = Path(path)
    all_vols = [int(t) for d, t in gmsh.model.getEntities(3)]
    keep = [int(t) for t in vols]
    keep_set = set(keep)
    for tag, name in zip(keep, names):
        gmsh.model.setEntityName(3, int(tag), str(name))
    hidden = [(3, t) for t in all_vols if t not in keep_set]
    shown = [(3, t) for t in keep]
    old = gmsh.option.getNumber("Geometry.OCCExportOnlyVisible")
    try:
        if hidden:
            gmsh.model.setVisibility(hidden, 0, recursive=True)
        if shown:
            gmsh.model.setVisibility(shown, 1, recursive=True)
        gmsh.option.setNumber("Geometry.OCCExportOnlyVisible", 1)
        gmsh.write(str(path))
    finally:
        gmsh.model.setVisibility([(3, t) for t in all_vols], 1, recursive=True)
        gmsh.option.setNumber("Geometry.OCCExportOnlyVisible", old)
    print(f"_export_simsopt: wrote {path}")


def _mesh_tet10(owner, size_max: float, size_min: float):
    """Mesh every volume of this model in one pass. Shared faces share nodes.

    The mesh is generated linear and optimized (Netgen), then raised to
    second order. Edge midpoints of any folded curved cell are put back on
    its straight edges. Warns about those cells and about elements below
    ``_QUALITY_MIN``. Returns the minimum gamma
    quality, the points, the VTK TET10 cells and the per-cell owner arrays.
    """
    _drop_free_curves(gmsh.model.occ)
    present = {int(t) for _d, t in gmsh.model.getEntities(3)}
    wanted = {int(t) for t in owner}
    if present != wanted:
        raise RuntimeError(
            "_export_simsopt: model volumes "
            f"{sorted(present - wanted)} extra, {sorted(wanted - present)} missing"
        )
    gmsh.option.setNumber("Mesh.MeshSizeMax", float(size_max))
    gmsh.option.setNumber("Mesh.MeshSizeMin", float(size_min))
    print(
        f"_export_simsopt: meshing, MeshSizeMin={size_min:.4g}, "
        f"MeshSizeMax={size_max:.4g}",
        flush=True,
    )
    t0 = time.time()
    try:
        gmsh.model.mesh.generate(3)
        print(f"_export_simsopt: linear mesh ({time.time() - t0:.0f} s)", flush=True)
    except Exception as exc:
        for word in str(exc).replace("(", " ").replace(")", " ").split():
            if not word.isdigit():
                continue
            face = int(word)
            try:
                kind = gmsh.model.getType(2, face)
                area = float(gmsh.model.occ.getMass(2, face))
                com = np.asarray(gmsh.model.occ.getCenterOfMass(2, face), dtype=float)
                upward, _down = gmsh.model.getAdjacencies(2, face)
                labels = [owner.get(abs(int(v)), None) for v in upward]
            except Exception:
                continue
            print(
                f"  overlapping surface {face}: {kind}, area {area:.3e}, "
                f"com {np.round(com, 4).tolist()}, "
                f"volumes {[int(v) for v in upward]} labels {labels}",
                flush=True,
            )
        raise RuntimeError(f"_export_simsopt: mesh failed ({exc})") from exc

    # gmsh ignores dimTags for Netgen ("not interfaced yet"), so the whole model
    # is optimized. Its boundary nodes are shared by id, so the interfaces stay
    # conforming. HighOrderElastic is skipped: it cost ~15 min and, on CFQS,
    # created the only inverted cells that plain setOrder(2) did not have.
    t0 = time.time()
    gmsh.model.mesh.optimize("Netgen")
    print(f"_export_simsopt: Netgen optimize ({time.time() - t0:.0f} s)", flush=True)
    t0 = time.time()
    gmsh.model.mesh.setOrder(2)
    print(f"_export_simsopt: second order ({time.time() - t0:.0f} s)", flush=True)
    t0 = time.time()

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    if node_tags.size == 0:
        raise RuntimeError("_export_simsopt: mesh produced no nodes")
    points = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)
    lut = np.full(int(node_tags.max()) + 1, -1, dtype=np.int64)
    lut[node_tags] = np.arange(node_tags.size)

    cells = []
    owner_coil, owner_sym, material_id = [], [], []
    gamma_all, sj_all, vol_all = [], [], []
    for tag, label in owner.items():
        etypes, etags, ntags = gmsh.model.mesh.getElements(3, int(tag))
        etypes = [int(t) for t in etypes]
        if _TET10 not in etypes:
            raise RuntimeError(
                f"volume {tag} has no TET10 elements (types {etypes})"
            )
        slot = etypes.index(_TET10)
        conn = np.asarray(ntags[slot], dtype=np.int64)
        conn = lut[conn.reshape(-1, 10)[:, _GMSH_TET10_TO_VTK]]
        if (conn < 0).any():
            raise RuntimeError(f"tet node missing from getNodes (volume {tag})")
        gamma_all.append(np.asarray(
            gmsh.model.mesh.getElementQualities(etags[slot], "gamma"), dtype=float,
        ))
        sj_all.append(np.asarray(
            gmsh.model.mesh.getElementQualities(etags[slot], "minSJ"), dtype=float,
        ))
        vol_all.append(np.full(len(conn), int(tag)))
        coil, image, mat = label
        cells.append(conn)
        owner_coil.append(np.full(len(conn), int(coil), dtype=np.int32))
        owner_sym.append(np.full(len(conn), int(image), dtype=np.int32))
        material_id.append(np.full(len(conn), int(mat), dtype=np.int32))
    if not cells:
        raise RuntimeError("_export_simsopt: no volumes to mesh")
    gamma = np.concatenate(gamma_all)
    sj = np.concatenate(sj_all)
    vol = np.concatenate(vol_all)
    cells = np.vstack(cells)
    print(f"_export_simsopt: extract cells ({time.time() - t0:.0f} s)", flush=True)
    t0 = time.time()
    inv = np.flatnonzero(sj <= 0.0)
    n_fixed, n_nodes, ratio = _straighten_folded(points, cells)
    print(f"_export_simsopt: detJ check ({time.time() - t0:.0f} s)", flush=True)
    if n_fixed:
        warnings.warn(
            f"{inv.size} inverted elements (minSJ <= 0) at "
            f"{np.round(points[cells[inv][:, :4]].mean(axis=1), 3).tolist()} "
            f"in volumes {vol[inv].tolist()}; straightened {n_fixed} cells "
            f"({n_nodes} edge midpoints)"
        )
    low = np.flatnonzero(gamma < _QUALITY_MIN)
    if low.size:
        worst = low[np.argsort(gamma[low])[:5]]
        warnings.warn(
            f"{low.size} of {gamma.size} elements have gamma < {_QUALITY_MIN}; "
            f"worst {np.round(gamma[worst], 5).tolist()} at "
            f"{np.round(points[cells[worst][:, :4]].mean(axis=1), 3).tolist()} "
            f"in volumes {vol[worst].tolist()}"
        )
    # gamma is measured on the straight-sided cell, detJ ratio on the curved one
    gp = np.percentile(gamma, [0.1, 1, 50])
    rp = np.percentile(ratio, [0.1, 1, 50])
    print(
        "_export_simsopt mesh quality:\n"
        f"  gamma:      min {gamma.min():.3g}, 0.1% {gp[0]:.3g}, 1% {gp[1]:.3g}, "
        f"median {gp[2]:.3g}\n"
        f"              {int((gamma < _QUALITY_MIN).sum())} < {_QUALITY_MIN}, "
        f"{int((gamma < 0.01).sum())} < 0.01, {int((gamma < 0.001).sum())} < 0.001\n"
        f"  detJ ratio: min {ratio.min():.3g}, 0.1% {rp[0]:.3g}, 1% {rp[1]:.3g}, "
        f"median {rp[2]:.3g}\n"
        f"  inverted before straightening: {inv.size}; "
        f"straightened: {n_fixed} cells, {n_nodes} edge midpoints",
        flush=True,
    )
    return (
        float(gamma.min()),
        points,
        cells,
        np.concatenate(owner_coil),
        np.concatenate(owner_sym),
        np.concatenate(material_id),
    )


def _write_vtu(
    path, *, points, cells, owner_coil, owner_sym, material_id,
    clamp_centers, r_clamp, eps_sigmoid, k_clamp, materials, g_vec,
) -> None:
    """Write the VTU schema consumed by the full-body FEM loader."""
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
    print(f"_export_simsopt: wrote {path}")


def _clamp_centers(sdofs, curves, meshes, Q_list):
    """Clamp centres of every coil image, or an empty ``(0, 3)`` array."""
    if "phis" not in sdofs:
        return np.zeros((0, 3), dtype=np.float64)
    centers = []
    for i in range(len(meshes)):
        phi_i = _np(sdofs["phis"][i]).ravel()
        if phi_i.size == 0:
            continue
        base = _np(curves[i].gamma_eval(phi_i))
        for Q in Q_list:
            centers.append(base @ Q.T)
    if not centers:
        return np.zeros((0, 3), dtype=np.float64)
    return np.vstack(centers)


# ============================================================================
# Coil bodies
# ============================================================================


def _coil_bodies(occ, meshes, Q_list) -> dict[tuple[int, int], dict[str, list[int]]]:
    """Winding pack and full-coil solids of every ``(coil, image)``: ``{"wp": tags, "big": tags}``.

    ``coil_big`` is the full coil swept along the casing's outer outline, so it
    contains the winding pack (swept along its own outline, chamfered when the
    mesh has a rounding). Image 0 is lofted; later images are rigid copies.
    """
    bodies: dict[tuple[int, int], dict[str, list[int]]] = {}
    for i, mesh in enumerate(meshes):
        wp = _coil_loft(
            occ, mesh, float(mesh.w1), float(mesh.w2), getattr(mesh, "rounding", None),
        )
        big = _coil_loft(occ, mesh, float(mesh.w1_outer), float(mesh.w2_outer))
        occ.synchronize()
        for s, Q in enumerate(Q_list):
            tags = wp + big
            if s > 0:
                tags = list(occ.copy(tags))
                _rigid(occ, tags, Q)
            bodies[(i, s)] = {
                "wp": [int(t) for _d, t in tags[:len(wp)]],
                "big": [int(t) for _d, t in tags[len(wp):]],
            }
    occ.synchronize()
    return bodies


# ============================================================================
# Per-beam task: weld points and pass 1 of the master image
# ============================================================================

# Read-only inputs of _beam_task, set once per process by _pool_init.
_CTX: dict = {}


def _pool_init(meshes, Q_list, opt, delta, snap_tol) -> None:
    """Store the shared inputs of :func:`_beam_task` in this process."""
    _CTX.clear()
    _CTX.update(
        meshes=meshes, Q_list=Q_list, opt=opt, delta=delta, snap_tol=snap_tol, sdf={},
    )


def _beam_task(job) -> dict:
    """Weld points of one master beam and, when asked, its trimmed beam plus welds.

    ``job = (rec, brep_dir)``; ``brep_dir=None`` skips the gmsh part. Returns
    ``{"weld_pts": [per joint], "log": [lines], "brep": path or None, "roles":
    [(kind, weld index, centre of mass)]}``. The BREP holds the kept beam piece
    and every weld piece of the beam as one shape, so their shared faces stay
    shared when it is imported again. Runs in a worker process or in-process.
    """
    rec, brep_dir = job
    meshes, Q_list = _CTX["meshes"], _CTX["Q_list"]
    opt, delta, snap_tol = _CTX["opt"], _CTX["delta"], _CTX["snap_tol"]
    sdf_cache = _CTX["sdf"]

    def sdf_of(key):
        if key not in sdf_cache:
            sdf_cache[key] = _casing_sdf(meshes[key[0]], Q_list[key[1]], 0.5 * delta)
        return sdf_cache[key]

    # ------------------------------------------------------------------------
    # Weld points: four loops per joint, numpy only
    # ------------------------------------------------------------------------
    # beam_loop_1/2 come from ray-marching against the casing distance, then
    # coil_loop_1/2 are level sets of the distance to beam_loop_1's hull on the
    # casing outline and on the winding-pack outline. Both coil loops are sunk
    # half a casing thickness in section width (a quarter thickness per side)
    # under their surfaces: loops lying exactly on a surface leave micro-slivers
    # where the hull chords meet it. The weld is the convex hull of all four
    # loops; only its points are kept, in world coordinates.
    b = rec["index"]
    shrink = 0.25 * float(rec["section"]["w_min"])
    log = []
    weld_pts = []
    keys = []
    for joint in rec["joints"]:
        where = f"beam {b} {joint['side']}"
        key = (int(joint["coil"]), int(joint["image"]))
        mesh, Qc = meshes[key[0]], Q_list[key[1]]
        origin = np.asarray(joint["origin"], dtype=float)
        g = np.asarray(joint["frame"], dtype=float)
        # Joint frame (x, y, z) -> real: origin + g @ (x, y, z).
        ring1, ring2 = _beam_loops(
            sdf_of(key), joint, rec["L"], delta, shrink, opt["h_hull"], 0.5 * delta,
        )
        world1 = origin + ring1 @ g.T
        world2 = origin + ring2 @ g.T
        hull = _Hull(world1)
        t_cas = float(mesh.casing_thickness)
        coil1 = _coil_contour(
            mesh, Qc, hull, np.sqrt(2.0) * delta,
            float(mesh.w1_outer) - 0.5 * t_cas, float(mesh.w2_outer) - 0.5 * t_cas,
            opt["h_contour"], where + " casing", snap_tol, 0.5 * delta,
        )
        # coil_loop_2 lies inside the coil and is swallowed by the fuse; it only
        # has to guarantee that the weld overlaps the coil, so it is sampled at
        # the beam width instead of h_contour (fewer weld faces, faster booleans).
        coil2 = _coil_contour(
            mesh, Qc, hull, np.sqrt(2.0) * (delta + t_cas),
            float(mesh.w1) - 0.5 * t_cas, float(mesh.w2) - 0.5 * t_cas,
            float(rec["section"]["w_min"]), where + " winding pack", snap_tol, 0.5 * delta,
        )
        pts = np.vstack([world1, world2, coil1["pts"], coil2["pts"]])
        weld_pts.append(pts)
        keys.append(key)
        log.append(
            f"{where} (coil {key[0]} image {key[1]}): shrink {shrink:.4f}, "
            f"{len(pts)} weld points, |d-r| {coil1['resid_max']:.2e} {coil2['resid_max']:.2e}"
        )
    out = {"weld_pts": weld_pts, "log": log, "brep": None, "roles": None}
    if brep_dir is None:
        return out

    # ------------------------------------------------------------------------
    # Pass 1: fragment the prism with its welds
    # ------------------------------------------------------------------------
    # Fragmenting trims the beam and makes the beam-weld interface conformal.
    # Where the prism overlaps a weld the piece is weld (casing material), so
    # the material boundary lies between the weld and the beam. What remains of
    # the prism beyond the welds is a stump inside a coil and is dropped; the
    # kept beam piece is the one outside every coil it touches. A private gmsh
    # session keeps this independent of the caller's model.
    try:
        owned = not gmsh.isInitialized()
    except AttributeError:
        owned = True
    if owned:
        gmsh.initialize()
    else:
        gmsh.clear()
    try:
        gmsh.model.add(f"beam_{b}")
        gmsh.option.setNumber("General.Verbosity", 2)
        occ = gmsh.model.occ
        prism = int(_beam_solid(occ, rec)[0][1])
        welds = [int(_hull_solid(occ, pts)[0]) for pts in weld_pts]
        _out, ovmap = occ.fragment([(3, t) for t in [prism] + welds], [])
        if len(ovmap) != 1 + len(welds):
            raise RuntimeError(f"beam {b}: fragment returned an unexpected map")
        weld_of: dict[int, int] = {}
        for w, outs in enumerate(ovmap[1:]):
            for d, t in outs:
                if int(d) == 3:
                    weld_of.setdefault(int(t), w)
        pieces = [int(t) for d, t in ovmap[0] if int(d) == 3 and int(t) not in weld_of]
        depth = {}
        for t in pieces:
            com = np.asarray(occ.getCenterOfMass(3, t), dtype=float)
            depth[t] = min(float(sdf_of(key)(com)[0]) for key in keys)
        keep = [t for t in pieces if depth[t] > 0.0]
        if len(keep) != 1:
            raise RuntimeError(
                f"beam {b}: {len(keep)} of {len(pieces)} beam pieces lie outside "
                f"the coils (coil distances {[round(d, 4) for d in depth.values()]})"
            )
        occ.remove([(3, t) for t in pieces if t != keep[0]], recursive=True)
        occ.synchronize()
        roles = [("beam", -1, tuple(occ.getCenterOfMass(3, keep[0])))]
        roles += [
            ("weld", w, tuple(occ.getCenterOfMass(3, t))) for t, w in weld_of.items()
        ]
        path = Path(brep_dir) / f"beam_{b}.brep"
        gmsh.write(str(path))
    finally:
        if owned:
            gmsh.finalize()
        else:
            gmsh.clear()
    out["brep"] = str(path)
    out["roles"] = roles
    return out


def _run_beam_tasks(jobs, ctx, n_workers: int) -> list[dict]:
    """Run :func:`_beam_task` over ``jobs`` in ``n_workers`` processes, or in-process for one."""
    if n_workers <= 1:
        _pool_init(*ctx)
        try:
            return [_beam_task(job) for job in jobs]
        finally:
            _CTX.clear()
    # Spawn, not fork: JAX is already running in this process. The workers
    # use JAX on the CPU only.
    old = os.environ.get("JAX_PLATFORMS")
    os.environ["JAX_PLATFORMS"] = "cpu"
    try:
        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=_pool_init, initargs=ctx,
        ) as pool:
            return list(pool.map(_beam_task, jobs))
    finally:
        if old is None:
            os.environ.pop("JAX_PLATFORMS", None)
        else:
            os.environ["JAX_PLATFORMS"] = old


# ============================================================================
# Export
# ============================================================================


def _export_simsopt(
    Jstress,
    prefix="full_body_fields",
    mesh_scale=0.5,
    joint_options=None,
    save_step=False,
    save_vtk=True,
    n_sides_circle=12,
    n_workers=None,
) -> None:
    """Write the welded full device to STEP and/or a conforming TET10 VTU.

    Every beam end is joined to its coil by a convex weld of casing material.
    The material boundary lies between the weld and the beam. The device is
    meshed in one pass, so the welds, beams and coils share nodes.

    Parameters
    ----------
    Jstress : CoilFEMObjective
        Must wrap :class:`~coil_fem.coupling.beam_network.SupportBeams`
        (not a central-support-ring support) and rectangular coil meshes with
        a casing. A chamfered winding pack is exported chamfered, but its weld
        is built from the sharp section. Hollow beams are not supported.
    prefix : path-like
        Path without an extension. Outputs are ``<prefix>.vtu`` and, when
        ``save_step`` is set, ``<prefix>_raw.step`` and
        ``<prefix>_premerge.step``. The parent directory must exist.
    mesh_scale : float
        Multiplier on ``MeshSizeMax``. The size is ``mesh_scale * 0.5 * w1``
        of the first coil, capped at the smallest casing thickness.
        ``MeshSizeMin`` is ``min(MeshSizeMax, delta)``.
    joint_options : dict or None
        ``delta`` is the beam-to-coil gap of the inner weld ring. It defaults
        to ``min(w1, w2) / 4`` for rectangular beams and ``r / 2`` for circular
        beams, taking the smallest over all beams. The weld also reaches
        ``w_min / 4`` into the beam. ``h_contour`` (coil contour spacing) and
        ``h_hull`` (beam ring spacing) default to ``delta``. Contour runs
        within ``edge_snap_fraction * min(size_min, h_contour)`` of a casing
        corner move onto it (default ``0.5``). Unknown keys raise.
    save_step : bool
        Write ``<prefix>_raw.step`` (winding packs, ``coil_big`` solids and
        untrimmed beams) and ``<prefix>_premerge.step`` (the same plus every
        weld), all as separate, overlapping bodies. ``coil_big`` is the full
        coil swept along the casing outline and contains the winding pack.
        Default ``False``.
    save_vtk : bool
        Merge all bodies, mesh them, and write ``<prefix>.vtu``. Default
        ``True``. The linear mesh is optimized (Netgen) before it is raised to
        second order. A curved cell that folds has its edge midpoints put
        back on its straight edges, with a warning. Elements with gamma below
        ``0.05`` also give a warning with their count and positions; the VTU
        is still written.
    n_sides_circle : int
        Number of sides of the prism used for a circular beam (at least 3).
        Default 12.
    n_workers : int or None
        Processes for the per-beam weld and trim stage. ``None`` uses
        ``min(number of beams, CPU count)``; ``1`` runs in-process. Worker
        processes are spawned, so a script calling the export needs an
        ``if __name__ == "__main__":`` guard (notebooks do not).

    Raises
    ------
    ValueError
        The support, the meshes, or ``joint_options`` are not supported, or a
        coil has no casing.
    FileNotFoundError
        The parent directory of ``prefix`` does not exist.
    RuntimeError
        A contour, boolean, fragment, or meshing step failed.
    NotImplementedError
        A beam is hollow.
    """
    if int(n_sides_circle) < 3:
        raise ValueError(f"n_sides_circle must be at least 3, got {n_sides_circle}")
    n_sides_circle = int(n_sides_circle)
    if not save_step and not save_vtk:
        warnings.warn("No file exported")
        return None

    prefix = Path(prefix)
    if not prefix.parent.exists():
        raise FileNotFoundError(f"parent directory {prefix.parent} does not exist")
    if mesh_scale <= 0.0:
        raise ValueError(f"mesh_scale must be positive, got {mesh_scale}")
    try:
        coil_support = Jstress.coil_support
        fem = Jstress.fem
    except AttributeError as exc:
        raise ValueError("_export_simsopt requires a CoilFEMObjective") from exc
    support = fem.support
    if type(support) is not SupportBeams:
        raise ValueError(
            f"_export_simsopt supports SupportBeams only, got {type(support).__name__}"
        )
    meshes = fem.meshes
    if not meshes or not all(
        isinstance(m, (FramedCurveMeshRectangle, FramedCurveMeshSection)) for m in meshes
    ):
        raise ValueError(
            "_export_simsopt requires rectangular coil meshes "
            "(FramedCurveMeshRectangle or FramedCurveMeshSection)"
        )
    if any(int(m.n_casing) == 0 for m in meshes):
        raise ValueError("_export_simsopt requires a casing on every coil")
    sdofs = coil_support.support_dofs
    curves = fem.base_curves_jax
    geom = support.beam_geometry(curves, sdofs)
    Q_list = _sym_maps(support.nfp, support.stellsym)
    recs = _master_beams(support, sdofs, geom, Q_list, n_sides_circle)
    for rec in recs:
        sec = rec["section"]
        if (sec["kind"] == "rect" and sec["t"] is not None) or (
            sec["kind"] == "circle" and sec["ri"] > 0.0
        ):
            raise NotImplementedError("hollow beams are not supported")
    opt = _options(joint_options, [rec["section"] for rec in recs])
    delta = opt["delta"]
    print(f"_export_simsopt: delta={delta:.4g}", flush=True)
    size_max = float(mesh_scale) * 0.5 * float(meshes[0].w1)
    size_max = min(size_max, min(float(m.casing_thickness) for m in meshes))
    size_min = min(size_max, delta)
    snap_tol = opt["edge_snap_fraction"] * min(size_min, opt["h_contour"])

    # ========================================================================
    # Stage 1: weld points and the trimmed beam, one task per master beam
    # ========================================================================
    # Each task computes the weld points of its joints and, when meshing,
    # fragments the beam prism with its welds (pass 1). The other symmetry
    # images are placed from these results in stage 2.
    t_all = t0 = time.time()
    brep_dir = tempfile.TemporaryDirectory(prefix="coil_fem_export_") if save_vtk else None
    n_workers = min(len(recs), os.cpu_count() or 1) if n_workers is None else int(n_workers)
    results = _run_beam_tasks(
        [(rec, brep_dir.name if save_vtk else None) for rec in recs],
        (meshes, Q_list, opt, delta, snap_tol), n_workers,
    )
    by_beam = {rec["index"]: res for rec, res in zip(recs, results)}
    for rec, res in zip(recs, results):
        for joint, pts in zip(rec["joints"], res["weld_pts"]):
            joint["weld_pts"] = pts
        for line in res["log"]:
            print(line, flush=True)
    print(f"_export_simsopt: stage 1 weld points and trimmed beams, "
          f"{n_workers} workers ({time.time() - t0:.0f} s)", flush=True)

    # ========================================================================
    # Supports: every master beam under every symmetry image
    # ========================================================================
    # A support is one beam image with its welds. Weld points are rotated, not
    # the bodies, and the coil image of each weld is the image of the master
    # joint's coil image under Q. A beam image that maps onto an existing beam
    # (ends possibly swapped), as a coil-to-image beam does, is skipped.
    supports = []
    placed = []
    for s in range(len(Q_list)):
        for rec in recs:
            Q = Q_list[s]
            xs, xe = Q @ rec["x_start"], Q @ rec["x_end"]
            tol = 1e-6 * float(rec["L"])
            if s > 0 and any(
                (np.linalg.norm(xs - a) <= tol and np.linalg.norm(xe - c) <= tol)
                or (np.linalg.norm(xs - c) <= tol and np.linalg.norm(xe - a) <= tol)
                for a, c in placed
            ):
                continue
            placed.append((xs, xe))
            supports.append({
                "beam": rec["index"], "image": s, "rec": rec, "Q": Q,
                "welds": [
                    {
                        "key": (
                            int(j["coil"]),
                            _image_of(Q_list, Q @ Q_list[int(j["image"])]),
                        ),
                        "pts": j["weld_pts"] @ Q.T,
                        "side": j["side"],
                    }
                    for j in rec["joints"]
                ],
            })

    try:
        owned = not gmsh.isInitialized()
    except AttributeError:
        owned = True
    if owned:
        gmsh.initialize()
    else:
        gmsh.clear()
    occ_parallel = gmsh.option.getNumber("Geometry.OCCParallel")
    unify = gmsh.option.getNumber("Geometry.OCCUnionUnify")
    try:
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.option.setNumber("Geometry.OCCParallel", 1)

        # ====================================================================
        # STEP: whole bodies in a model of their own
        # ====================================================================
        # Winding packs, full coils, untrimmed beams and whole welds overlap, so
        # they live in a throwaway model that never reaches the merge.
        if save_step:
            t_step = time.time()
            gmsh.model.add("step")
            occ = gmsh.model.occ
            coil_bodies = _coil_bodies(occ, meshes, Q_list)
            vols, names = [], []
            for (i, s_), parts in coil_bodies.items():
                for kind, tags in (("winding_pack", parts["wp"]), ("coil_big", parts["big"])):
                    vols += tags
                    names += [f"{kind}_c{i}_s{s_}_{k}" for k in range(len(tags))]
            for sp in supports:
                rec, Q = sp["rec"], sp["Q"]
                image_rec = {
                    "section": rec["section"], "L": rec["L"],
                    "gamma3": Q @ rec["gamma3"], "x_start": Q @ rec["x_start"],
                }
                solids = _beam_solid(occ, image_rec)
                if len(solids) != 1:
                    raise RuntimeError(f"beam {sp['beam']} solid is {len(solids)} volumes")
                vols.append(int(solids[0][1]))
                names.append(f"beam_b{sp['beam']}_s{sp['image']}")
            occ.synchronize()
            _write_step(Path(f"{prefix}_raw.step"), vols, names)
            for sp in supports:
                for w in sp["welds"]:
                    vols.append(int(_hull_solid(occ, w["pts"])[0]))
                    names.append(f"weld_b{sp['beam']}_s{sp['image']}_{w['side']}")
            _write_step(Path(f"{prefix}_premerge.step"), vols, names)
            gmsh.model.remove()
            print(f"_export_simsopt: STEP files ({time.time() - t_step:.0f} s)", flush=True)
        if not save_vtk:
            return None

        gmsh.model.add("device")
        occ = gmsh.model.occ
        coil_bodies = _coil_bodies(occ, meshes, Q_list)

        # ====================================================================
        # Merge: local booleans only, never the whole device at once
        # ====================================================================
        # One fragment of every body is too heavy for OCC, so the merge runs in
        # small passes. Faces that a pass leaves untouched stay shared with the
        # bodies of the other passes. In every fragment, inputs are listed from
        # lowest to highest priority and the last owner of a piece wins.
        owner: dict[int, tuple[int, int, int]] = {}
        support_of: dict[int, tuple[int, int]] = {}  # beam piece -> support
        welds_on: dict[tuple[int, int], set[int]] = defaultdict(set)
        supports_on: dict[tuple[int, int], set] = defaultdict(set)
        # (support, coil image) -> the support's weld pieces on that coil image
        placed_with: dict[tuple, set[int]] = defaultdict(set)

        def _faces(tags):
            return {
                abs(int(f))
                for t in tags
                for d, f in gmsh.model.getBoundary([(3, int(t))], oriented=False)
                if int(d) == 2
            }

        # Stage 2: every support is the master's trimmed beam and weld pieces,
        # imported from the master's BREP (the pieces keep their shared faces)
        # and rotated into place. Weld pieces are casing material.
        t0 = time.time()
        print(f"_export_simsopt: placing {len(supports)} supports", flush=True)
        for sp in supports:
            sup = (sp["beam"], sp["image"])
            res = by_beam[sp["beam"]]
            tags = [int(t) for _d, t in _vols(occ.importShapes(res["brep"]))]
            com = np.array([occ.getCenterOfMass(3, t) for t in tags])
            # Match each imported solid to a stored role by centre of mass.
            matched = []
            for kind, w, c in res["roles"]:
                dist = np.linalg.norm(com - np.asarray(c), axis=1)
                matched.append((int(np.argmin(dist)), kind, w, float(dist.min())))
            idx = [m[0] for m in matched]
            if len(tags) != len(matched) or len(set(idx)) != len(idx) or max(
                m[3] for m in matched
            ) > 1e-6 * float(sp["rec"]["L"]):
                raise RuntimeError(f"support {sup}: imported pieces do not match the master")
            if sp["image"] > 0:
                _rigid(occ, [(3, t) for t in tags], sp["Q"])
            for i, kind, w, _d in matched:
                if kind == "beam":
                    owner[tags[i]] = (-1, -1, -1)
                    support_of[tags[i]] = sup
                else:
                    key = sp["welds"][w]["key"]
                    owner[tags[i]] = (key[0], key[1], 1)
                    welds_on[key].add(tags[i])
                    placed_with[(sup, key)].add(tags[i])
            for w in sp["welds"]:
                supports_on[w["key"]].add(sup)
        occ.synchronize()
        brep_dir.cleanup()
        # Placing the pieces must not have split the beam-weld interfaces.
        for sp in supports:
            sup = (sp["beam"], sp["image"])
            beam_faces = _faces([t for t, s_ in support_of.items() if s_ == sup])
            for key in {w["key"] for w in sp["welds"]}:
                if not beam_faces & _faces(placed_with[(sup, key)]):
                    raise RuntimeError(f"support {sup}: beam and weld lost their shared face")
        print(f"_export_simsopt: stage 2 placement ({time.time() - t0:.0f} s)", flush=True)

        # Beams of different supports can cross each other (the two images of a
        # coil-to-image beam do near the symmetry plane). Crossing beams are
        # fragmented together so the overlap becomes one shared piece.
        t_cross = time.time()
        beam_tags = list(support_of)
        box = {t: np.array(occ.getBoundingBox(3, t)) for t in beam_tags}
        group = {t: t for t in beam_tags}  # union-find over touching beams

        def _root(t):
            while group[t] != t:
                group[t] = group[group[t]]
                t = group[t]
            return t

        for i, a in enumerate(beam_tags):
            for b in beam_tags[i + 1:]:
                if (box[a][:3] > box[b][3:] + 1e-9).any() or (box[b][:3] > box[a][3:] + 1e-9).any():
                    continue
                if support_of[a] != support_of[b] and occ.getDistance(3, a, 3, b)[0] < 1e-9:
                    group[_root(a)] = _root(b)
        crossing = defaultdict(list)
        for t in beam_tags:
            crossing[_root(t)].append(t)
        for tags in crossing.values():
            if len(tags) < 2:
                continue
            _out, ovmap = occ.fragment([(3, t) for t in tags], [])
            # (Pop every old tag first: fragment may reuse them for new pieces.)
            sups = [support_of.pop(old) for old in tags]
            for old in tags:
                owner.pop(old)
            for sup, outs in zip(sups, ovmap):
                for _d, new in outs:
                    owner[int(new)] = (-1, -1, -1)
                    support_of.setdefault(int(new), sup)
        print(f"_export_simsopt: crossing beams ({time.time() - t_cross:.0f} s); "
              f"setup + pass 1 done ({time.time() - t_all:.0f} s)", flush=True)

        # Pass 2, per coil image: the welds on it are fused into coil_big (welds
        # that overlap merge), then the winding pack is fragmented out. The fuse
        # swallows every part of a weld inside the coil, so no weld face is left
        # to cut the winding pack. Unify is off so that a weld face shared with
        # a beam is never merged into a larger face.
        gmsh.option.setNumber("Geometry.OCCUnionUnify", 0)
        for key, parts in coil_bodies.items():
            print(f"  coil {key} ({time.time() - t0:.0f} s)", flush=True)
            if welds_on[key]:
                out, _map = occ.fuse(
                    [(3, t) for t in parts["big"]], [(3, t) for t in sorted(welds_on[key])],
                )
                fused = [int(t) for d, t in out if int(d) == 3]
                for t in welds_on[key]:  # the weld pieces no longer exist as bodies
                    owner.pop(t)
            else:
                fused = list(parts["big"])
            if not fused:
                raise RuntimeError(f"coil {key}: fusing the welds into the coil gave no volume")
            members = [(t, (key[0], key[1], 1)) for t in fused]
            members += [(t, (key[0], key[1], 0)) for t in parts["wp"]]
            _out, ovmap = occ.fragment([(3, t) for t, _l in members], [])
            if len(ovmap) != len(members):
                raise RuntimeError(f"coil {key}: fragment returned an unexpected map")
            for (_t, label), outs in zip(members, ovmap):
                for d, t in outs:
                    if int(d) == 3:
                        owner[int(t)] = label
        occ.synchronize()
        print(f"_export_simsopt: merged ({time.time() - t0:.0f} s)", flush=True)

        # Every beam must still share a face with the casing of each coil image
        # it is welded to.
        for key, sups in supports_on.items():
            casing = _faces([t for t, lab in owner.items() if lab == (key[0], key[1], 1)])
            for sup in sups:
                mine = [t for t, s_ in support_of.items() if s_ == sup]
                if not _faces(mine) & casing:
                    raise RuntimeError(
                        f"beam {sup[0]} image {sup[1]}: no face shared with "
                        f"coil {key[0]} image {key[1]}"
                    )

        # ====================================================================
        # Mesh
        # ====================================================================
        (
            quality, points, cells, owner_coil, owner_sym, material_id,
        ) = _mesh_tet10(owner, size_max, size_min)
    finally:
        gmsh.option.setNumber("Geometry.OCCParallel", occ_parallel)
        gmsh.option.setNumber("Geometry.OCCUnionUnify", unify)
        if owned:
            gmsh.finalize()
        else:
            gmsh.clear()

    n_cells = int(cells.shape[0])
    n_coil = int(np.count_nonzero(owner_coil >= 0))
    n_casing = int(np.count_nonzero(material_id == 1))
    print(
        "_export_simsopt mesh counts:\n"
        f"  all bodies: {int(points.shape[0])} nodes, {n_cells} cells\n"
        f"  coils+welds:{n_coil} cells\n"
        f"  casing:     {n_casing} cells\n"
        f"  beams:      {n_cells - n_coil} cells\n"
        f"  min quality {quality:.3g}"
    )
    has_clamps = (
        getattr(coil_support, "_r_clamp", None) is not None
        and getattr(coil_support, "_sig_eps", None) is not None
    )
    t_vtu = time.time()
    _write_vtu(
        Path(f"{prefix}.vtu"),
        points=points,
        cells=cells,
        owner_coil=owner_coil,
        owner_sym=owner_sym,
        material_id=material_id,
        clamp_centers=(
            _clamp_centers(sdofs, curves, meshes, Q_list)
            if has_clamps else np.zeros((0, 3))
        ),
        r_clamp=float(coil_support._r_clamp) if has_clamps else 0.0,
        eps_sigmoid=float(coil_support._sig_eps) if has_clamps else 0.0,
        k_clamp=float(support.k_clamp),
        materials=fem.materials,
        g_vec=np.asarray(
            (fem.gravity_options or {}).get("g_vec", (0.0, 0.0, 0.0)),
            dtype=np.float64,
        ),
    )
    print(f"_export_simsopt: VTU write ({time.time() - t_vtu:.0f} s)", flush=True)
    return None
