"""Hull-welded full-device STEP and TET10 export for simsopt objectives.

:func:`_export_simsopt` cuts a copy of each beam with the coils grown by
``delta`` to find the ring where the cut meets the beam sides, builds hull2
from that ring and the coil patch, trims the beam by hull2 and fuses the two.
Each welded support is copied by symmetry and cut by that image's own
coils. The cut face is a copy of the casing surface; flush copies are
imprinted back onto the coil so the two solids share that face. The
supports are then fragmented with the coil volumes. One mesh covers the
whole model, so the contact patch is shared. Optional STEP files are written too.
Call it through :meth:`coil_fem.simsopt.objectives.CoilFEMObjective.export`.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import warnings

import gmsh
import jax
import meshio
import numpy as np
import pyvista as pv
import vtk
from scipy.spatial import ConvexHull
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
# Cut spurs shorter than this are dropped so the imprint can reuse the
# coil face's edges. The sewing tolerance only accepts an exact face.
_FLUSH_EDGE = 1e-4
_FLUSH_SEW = 1e-7

_JOINT_KEYS = ("delta", "h_contour", "h_hull", "bool_inset")


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

    The grown coil is shared by every beam, so the default ``delta`` is the
    smallest :func:`_default_delta` over ``sections``.
    """
    raw = {} if joint_options is None else dict(joint_options)
    unknown = set(raw) - set(_JOINT_KEYS)
    if unknown:
        raise ValueError(
            f"unrecognized joint_options keys: {sorted(unknown)}"
        )
    if "delta" in raw and raw["delta"] is not None:
        delta = float(raw["delta"])
    else:
        sections = list(sections)
        if not sections:
            raise ValueError("default delta needs at least one beam section")
        delta = min(_default_delta(s) for s in sections)
    h_contour = raw.get("h_contour", None)
    h_hull = raw.get("h_hull", None)
    bool_inset = raw.get("bool_inset", None)
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    h_contour = delta if h_contour is None else float(h_contour)
    h_hull = delta if h_hull is None else float(h_hull)
    if h_contour <= 0.0 or h_hull <= 0.0:
        raise ValueError("h_contour and h_hull must be positive")
    if bool_inset is not None:
        bool_inset = float(bool_inset)
        if bool_inset < 0.0:
            raise ValueError("bool_inset must be >= 0")
    return {
        "delta": delta,
        "h_contour": h_contour,
        "h_hull": h_hull,
        "bool_inset": bool_inset,
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


def _resample_keep(pts, fixed, spacing, extras=()):
    """Resample a closed loop by arclength, keeping ``fixed`` nodes exactly.

    ``extras`` are arrays shaped ``(n, ...)`` interpolated with the loop.
    Returns new points, new fixed indices, and the interpolated extras.
    """
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    fixed = sorted({int(i) % n for i in fixed}) or [0]
    columns = [np.asarray(col, dtype=float) for col in extras]
    out_pts = []
    out_cols = [[] for _ in columns]
    new_fixed = []
    spans = list(zip(fixed, fixed[1:] + [fixed[0] + n]))
    for a, b in spans:
        idx = [i % n for i in range(a, b + 1)]
        seg = pts[idx]
        step = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(step)])
        total = float(arc[-1])
        n_div = 1 if total < 1e-15 else max(1, int(np.ceil(total / spacing)))
        targets = np.linspace(0.0, total, n_div + 1)
        new_fixed.append(len(out_pts))
        for t in targets[:-1]:
            j = int(np.searchsorted(arc, t, side="right") - 1)
            j = min(max(j, 0), len(seg) - 2)
            span = arc[j + 1] - arc[j]
            f = 0.0 if span < 1e-15 else (t - arc[j]) / span
            out_pts.append((1.0 - f) * seg[j] + f * seg[j + 1])
            for col, bucket in zip(columns, out_cols):
                bucket.append((1.0 - f) * col[idx[j]] + f * col[idx[j + 1]])
    stacked = [np.asarray(bucket, dtype=float) for bucket in out_cols]
    return np.asarray(out_pts, dtype=float), new_fixed, stacked


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


def _outward_normals(u, v, ha, hb, p, q) -> np.ndarray:
    """Outward unit normals of a rectangular section. ``p`` and ``q`` broadcast with ``u, v``."""
    du = np.abs(u) - ha
    dv = np.abs(v) - hb
    tol = 1e-8 * max(ha, hb, 1e-9)
    on_u = du > dv - tol
    on_v = dv > du - tol
    su = np.where(np.sign(u) == 0.0, 1.0, np.sign(u))
    sv = np.where(np.sign(v) == 0.0, 1.0, np.sign(v))
    n = on_u[..., None] * (su[..., None] * p) + on_v[..., None] * (sv[..., None] * q)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    bad = nn[..., 0] < 1e-15
    if np.any(bad):
        n = n.copy()
        n[bad] = su[bad][..., None] * p[bad] + sv[bad][..., None] * q[bad]
        nn = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.maximum(nn, 1e-15)


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


def _coil_contour(mesh, Q, hull: _Hull, delta: float, h_contour: float, where: str) -> dict:
    """Closed curve on the coil where the signed distance to ``hull`` is ``delta*sqrt(2)``.

    Grid lines pass through the casing corners, so a crossing of a coil edge
    has a node on that edge. Each node is then moved onto the exact surface
    and the exact level set.
    """
    Q = np.asarray(Q, dtype=float)
    radius = float(delta) * np.sqrt(2.0)
    phi, r, p, q, _length, fc = _centerline_frames(mesh, h_contour)
    uv, corner_s, ha, hb = _section_uv(
        float(mesh.w1_outer), float(mesh.w2_outer), h_contour,
    )
    r_real = r @ Q.T
    center = hull.pts.mean(axis=0)
    reach = float(np.linalg.norm(hull.pts - center, axis=1).max())
    reach += radius + 0.5 * float(np.hypot(mesh.w1_outer, mesh.w2_outer))
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
    normals = _outward_normals(u, v, ha, hb, pp, qq) @ Q.T
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
    xyz = xyz[order]
    normals = normals[order]
    edge_idx = [k for k, j in enumerate(order) if crossings[j]["on_corner"]]
    xyz, edge_idx, (normals,) = _resample_keep(
        xyz, edge_idx, h_contour, (normals,),
    )
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
    resid = np.abs(hull.distance(xyz) - radius)

    p_win = np.broadcast_to(p[ip][:, None, :], (nr, ns, 3))
    q_win = np.broadcast_to(q[ip][:, None, :], (nr, ns, 3))
    u_grid = np.broadcast_to(uv[None, :, 0], (nr, ns))
    v_grid = np.broadcast_to(uv[None, :, 1], (nr, ns))
    n_grid = _outward_normals(u_grid, v_grid, ha, hb, p_win, q_win) @ Q.T
    inside = dist < radius
    return {
        "pts": xyz,
        "normals": normals,
        "edge_idx": edge_idx,
        "inside_pts": real[inside],
        "inside_normals": n_grid[inside],
        "n_loops": 1,
        "resid_max": float(resid.max()) if len(resid) else 0.0,
    }


def _hull2_points(joint, ring, coil, coil_inset: float) -> np.ndarray:
    """Hull2 samples: the snapped beam ring plus the coil patch.

    ``ring`` is ``(n, 3)`` in the joint frame, already on the beam's side
    planes, so hull2 spans the beam section and can trim it.
    ``coil_inset`` pushes only the coil patch into the coil. Points that fall
    just outside a beam side plane are pulled back onto it, so hull2 does not
    leave a ledge past the beam.
    """
    origin = np.asarray(joint["origin"], dtype=float)
    g = np.asarray(joint["frame"], dtype=float)
    ring = np.asarray(ring, dtype=float)
    beam_pts = (
        origin
        + ring[:, 0:1] * g[:, 0]
        + ring[:, 1:2] * g[:, 1]
        + ring[:, 2:3] * g[:, 2]
    )
    coil_pts = coil["pts"] - float(coil_inset) * coil["normals"]
    parts = [beam_pts, coil_pts]
    if len(coil["inside_pts"]):
        parts.append(coil["inside_pts"] - float(coil_inset) * coil["inside_normals"])
    pts = np.vstack(parts)
    tol = max(1e-4 * float(joint["section"]["w_min"]), 1e-5)
    local = (pts - origin) @ g
    local[:, 1:3] = _pull_just_outside(local[:, 1:3], joint["planes"], tol)
    return origin + local @ g.T


# ============================================================================
# OCC solids
# ============================================================================


def _rigid(occ, dimtags, R, t=None) -> None:
    """Apply ``x -> R x + t`` with a proper rotation ``R``.

    Uses rotate / translate, which keep planes analytic.
    ``occ.affineTransform`` and ``occ.mirror`` turn every face into a
    B-spline, which breaks :func:`_beam_ring`'s cut-face detection.
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


def _section_points(r0, p, q, w1, w2, rounding=None) -> np.ndarray:
    """Cross-section vertices at one coil station, shape ``(n_pts, 3)``.

    ``rounding`` is ``(r, n)`` for :func:`corner_polygon`, the same polygonal
    chamfer as the coil mesh. ``None`` is the sharp rectangle.
    """
    if rounding is None:
        xy = 0.5 * np.array([[-w1, -w2], [w1, -w2], [w1, w2], [-w1, w2]], dtype=float)
    else:
        r, n = rounding
        xy = corner_polygon(float(w1), float(w2), float(r), int(n))
    return r0 + xy[:, 0:1] * p + xy[:, 1:2] * q


def _coil_loft(occ, mesh, w1: float, w2: float, rounding=None) -> list[tuple[int, int]]:
    """Two thru-section halves of a coil. Returns volume dim-tags.

    ``rounding`` chamfers the winding pack only. The casing loft stays sharp.
    """
    fc = mesh.framed_curve
    phi = np.linspace(0.0, 1.0, _N_SLICES, endpoint=False)
    r0 = _np(fc.curve.gamma_eval(phi))
    _t, p, q = fc.rotated_frame_eval(phi)
    p, q = _np(p), _np(q)
    wires = []
    for k in range(_N_SLICES):
        xyz = _section_points(r0[k], p[k], q[k], w1, w2, rounding)
        pts = [occ.addPoint(float(xyz[a, 0]), float(xyz[a, 1]), float(xyz[a, 2]))
               for a in range(len(xyz))]
        lines = [occ.addLine(pts[a], pts[(a + 1) % len(pts)]) for a in range(len(pts))]
        wires.append(occ.addWire(lines))
    half = _N_SLICES // 2
    out = list(occ.addThruSections(wires[:half + 1], makeSolid=True, makeRuled=False))
    out += list(occ.addThruSections(wires[half:] + [wires[0]], makeSolid=True, makeRuled=False))
    _drop_free_curves(occ)
    return _vols(out)


def _sym_copies(occ, dimtags, Q_list) -> list[list[tuple[int, int]]]:
    """Image 0 is ``dimtags``; later images are copies under each ``Q``."""
    copies = [list(dimtags)]
    for Q in Q_list[1:]:
        copied = list(occ.copy(dimtags))
        _rigid(occ, copied, Q)
        copies.append(copied)
    return copies


def _owner_map(ov_map, owners) -> dict[int, tuple[int, int, int]]:
    """Fragment outputs to ``(owner_coil, owner_sym, material_id)``.

    Overlap is claimed by the coil, and the lower material id wins.
    """
    owner: dict[int, tuple[int, int, int]] = {}
    for src, outs in zip(owners, ov_map):
        if src[0] >= 0:
            continue
        for dim, tag in outs:
            if int(dim) == 3:
                owner[int(tag)] = (-1, -1, -1)
    for src, outs in zip(owners, ov_map):
        if src[0] < 0:
            continue
        label = (int(src[0]), int(src[1]), int(src[2]))
        for dim, tag in outs:
            if int(dim) != 3:
                continue
            tag = int(tag)
            prev = owner.get(tag)
            if prev is not None and prev[0] >= 0 and prev[2] <= label[2]:
                continue
            owner[tag] = label
    return owner


def _coil_images(occ, meshes, Q_list) -> dict:
    """Winding-pack and casing lofts for every coil image.

    Returns ``(coil, image) -> [(volume tag, material id), ...]``.
    """
    coil_vols: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i, mesh in enumerate(meshes):
        wp = _coil_loft(
            occ, mesh, float(mesh.w1), float(mesh.w2),
            getattr(mesh, "rounding", None),
        )
        has_casing = int(mesh.n_casing) > 0
        cas = (
            _coil_loft(occ, mesh, float(mesh.w1_outer), float(mesh.w2_outer))
            if has_casing else []
        )
        occ.synchronize()
        if has_casing:
            owners = [(i, 0, 0)] * len(wp) + [(i, 0, 1)] * len(cas)
            _out, ovmap = occ.fragment(wp, cas)
            occ.synchronize()
            if len(ovmap) != len(owners):
                raise RuntimeError(
                    f"winding-pack/casing fragment failed on coil {i}"
                )
            base = [(tag, lab[2]) for tag, lab in _owner_map(ovmap, owners).items()]
        else:
            base = [(int(t), 0) for _d, t in wp]
        copies = _sym_copies(occ, [(3, t) for t, _mat in base], Q_list)
        occ.synchronize()
        for s, tags in enumerate(copies):
            coil_vols[(i, s)] = [
                (int(t), mat) for (_d, t), (_t0, mat) in zip(tags, base)
            ]
    return coil_vols


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
# Booleans
# ============================================================================


def _inside(tag: int, pt) -> bool:
    """Whether one real-coordinate point lies in an OCC volume."""
    n = gmsh.model.isInside(3, int(tag), [float(pt[0]), float(pt[1]), float(pt[2])])
    return int(n) > 0


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


def _rounded_wire(occ, r0, p, q, au: float, av: float, delta: float) -> tuple[int, list[int]]:
    """Rounded-rectangle wire: straights at half-width plus ``delta``, corner arcs of radius ``delta``.

    Returns the wire tag and the line and arc tags it is built from.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    r0 = np.asarray(r0, dtype=float)

    def add(u, v):
        x = r0 + float(u) * p + float(v) * q
        return occ.addPoint(float(x[0]), float(x[1]), float(x[2]))

    c_se, c_ne = add(au, -av), add(au, av)
    c_nw, c_sw = add(-au, av), add(-au, -av)
    s0, s1 = add(-au, -(av + delta)), add(au, -(av + delta))
    e0, e1 = add(au + delta, -av), add(au + delta, av)
    n0, n1 = add(au, av + delta), add(-au, av + delta)
    w0, w1 = add(-(au + delta), av), add(-(au + delta), -av)
    curves = [
        occ.addLine(s0, s1),
        occ.addCircleArc(s1, c_se, e0),
        occ.addLine(e0, e1),
        occ.addCircleArc(e1, c_ne, n0),
        occ.addLine(n0, n1),
        occ.addCircleArc(n1, c_nw, w0),
        occ.addLine(w0, w1),
        occ.addCircleArc(w1, c_sw, s0),
    ]
    return occ.addWire(curves), curves


def _grown_loft(occ, mesh, delta: float) -> list[tuple[int, int]]:
    """Coil casing grown by ``delta``. Two thru-section halves. Used only as a cut tool."""
    fc = mesh.framed_curve
    phi = np.linspace(0.0, 1.0, _N_SLICES, endpoint=False)
    r0 = _np(fc.curve.gamma_eval(phi))
    _t, p, q = fc.rotated_frame_eval(phi)
    p, q = _np(p), _np(q)
    au = 0.5 * float(mesh.w1_outer)
    av = 0.5 * float(mesh.w2_outer)
    built = [
        _rounded_wire(occ, r0[k], p[k], q[k], au, av, float(delta))
        for k in range(_N_SLICES)
    ]
    wires = [wire for wire, _curves in built]
    half = _N_SLICES // 2
    out = list(occ.addThruSections(wires[:half + 1], makeSolid=True, makeRuled=False))
    out += list(occ.addThruSections(wires[half:] + [wires[0]], makeSolid=True, makeRuled=False))
    _drop_free_curves(occ)
    return _vols(out)


def _place_image(occ, dimtags, Q) -> list[tuple[int, int]]:
    """Copy ``dimtags`` under ``Q``. The identity returns the same tags."""
    if np.allclose(Q, np.eye(3), atol=1e-8):
        return list(dimtags)
    copied = list(occ.copy(dimtags))
    _rigid(occ, copied, Q)
    return copied


def _grown_tools(occ, meshes, Q_list, needed, delta: float) -> dict:
    """``(coil, image) -> volume tags`` for each joint's grown coil."""
    by_coil: dict[int, set[int]] = {}
    for i, s in needed:
        by_coil.setdefault(int(i), set()).add(int(s))
    grown = {}
    for i, images in by_coil.items():
        base = _grown_loft(occ, meshes[i], delta)
        occ.synchronize()
        for s in images:
            tags = base if s == 0 else _place_image(occ, base, Q_list[s])
            grown[(i, s)] = [int(t) for d, t in tags if int(d) == 3]
        if 0 not in images:
            occ.remove(base, recursive=True)
    occ.synchronize()
    return grown


def _cut_keep(
    occ, solid: int, tools, witness, index: int, what: str = "cut",
    volume_bounds=None,
) -> tuple[int, int]:
    """Cut ``solid`` with ``tools`` and keep the volume that contains the witness.

    ``solid`` is consumed and the tools stay. Returns the kept tag and the
    number of dropped pieces. ``what`` names the cut in error messages.
    ``volume_bounds`` ``(lo, hi)`` bounds the total volume of all pieces; a
    failed boolean can return a piece that is neither empty nor split.
    """
    tools = list(dict.fromkeys(int(t) for t in tools))
    if not tools:
        return int(solid), 0
    out, _map = occ.cut(
        [(3, int(solid))], [(3, t) for t in tools],
        removeObject=True, removeTool=False,
    )
    occ.synchronize()
    vols = [int(t) for d, t in out if int(d) == 3]
    if not vols:
        raise RuntimeError(f"beam {index}: {what} removed the whole beam")
    if volume_bounds is not None:
        total = sum(float(occ.getMass(3, t)) for t in vols)
        lo, hi = volume_bounds
        if not lo <= total <= hi:
            raise RuntimeError(
                f"beam {index}: {what} pieces hold {total:.6e}, "
                f"outside [{lo:.6e}, {hi:.6e}]"
            )
    keep = [t for t in vols if _inside(t, witness)]
    if len(keep) != 1:
        raise RuntimeError(
            f"beam {index}: witness is in {len(keep)} pieces after the {what}"
        )
    drop = [t for t in vols if t != keep[0]]
    if drop:
        occ.remove([(3, t) for t in drop], recursive=True)
        occ.synchronize()
    return keep[0], len(drop)


def _curve_point(tag: int, t: float) -> np.ndarray:
    """Point on an OCC curve at one parameter."""
    val = gmsh.model.getValue(1, int(tag), [float(t)])
    return np.asarray(val, dtype=float).reshape(3)


def _sample_curve(tag: int, h: float) -> np.ndarray:
    """Curve samples spaced about ``h`` apart."""
    bounds = gmsh.model.getParametrizationBounds(1, int(tag))
    t0, t1 = float(bounds[0][0]), float(bounds[1][0])
    if t1 < t0:
        t0, t1 = t1, t0
    coarse = np.linspace(t0, t1, 17)
    pts = np.stack([_curve_point(tag, t) for t in coarse])
    length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    n = max(2, int(np.ceil(length / max(float(h), 1e-9))))
    ts = np.linspace(t0, t1, n + 1)
    return np.stack([_curve_point(tag, t) for t in ts])


_ANALYTIC_FACES = {"Plane"}


def _on_side(local, planes, tol: float) -> bool:
    """Whether one joint-frame point lies on a beam side plane."""
    normals, offsets = planes
    yz = np.array([float(local[1]), float(local[2])])
    return bool(np.any(np.abs(yz @ normals.T - offsets) <= tol))


def _snap_yz(yz, planes, tol: float) -> np.ndarray:
    """Project points onto the side planes they already nearly lie on.

    A point within ``tol`` of one plane is projected onto it. A point within
    ``tol`` of two planes, a corner, is put on both. Other points stay put.
    """
    yz = np.array(yz, dtype=float, copy=True)
    normals, offsets = planes
    signed = yz @ normals.T - offsets
    near = np.abs(signed) <= tol
    for i in range(len(yz)):
        idx = np.flatnonzero(near[i])
        if idx.size == 0:
            continue
        if idx.size >= 2:
            order = idx[np.argsort(np.abs(signed[i, idx]))][:2]
            solved = _solve_two_planes(normals[order], offsets[order])
            if solved is not None:
                yz[i] = solved
                continue
            idx = order[:1]
        k = int(idx[0])
        yz[i] -= (normals[k] @ yz[i] - offsets[k]) * normals[k]
    return yz


def _pull_just_outside(yz, planes, tol: float) -> np.ndarray:
    """Pull points that sit just outside a side plane back onto it.

    Points further outside than ``tol`` are left, so a coil patch wider than
    the beam still flares past the side. Points inside the beam are left too.
    """
    yz = np.array(yz, dtype=float, copy=True)
    normals, offsets = planes
    signed = yz @ normals.T - offsets
    outside = (signed > 0.0) & (signed <= tol)
    for i in range(len(yz)):
        idx = np.flatnonzero(outside[i])
        if idx.size == 0:
            continue
        if idx.size >= 2:
            order = idx[np.argsort(signed[i, idx])][:2]
            solved = _solve_two_planes(normals[order], offsets[order])
            if solved is not None:
                yz[i] = solved
                continue
            idx = order[:1]
        k = int(idx[0])
        yz[i] -= signed[i, k] * normals[k]
    return yz


def _solve_two_planes(normals, offsets):
    """Intersection of two lines ``n · (y, z) = offset``, or None if parallel."""
    det = float(normals[0, 0] * normals[1, 1] - normals[0, 1] * normals[1, 0])
    if abs(det) < 1e-8:
        return None
    return np.linalg.solve(normals, offsets)


def _beam_ring(piece: int, joint: dict, h_hull: float) -> np.ndarray:
    """Samples of the cut face on the beam sides, in the joint frame.

    ``piece`` must be cut by this joint's grown coil only. The beam is a box
    or an n-sided prism, and :func:`_rigid` keeps every face a plane, so every
    non-planar face on the piece came from that coil. Samples are then snapped
    onto the side planes.
    """
    origin = np.asarray(joint["origin"], dtype=float)
    frame = np.asarray(joint["frame"], dtype=float)
    section = joint["section"]
    tol_side = max(1e-4 * float(section["w_min"]), 1e-5)
    where = f"beam {joint['beam']} {joint['side']}"
    end_faces = []
    kinds = []
    for _d, t in gmsh.model.getBoundary(
        [(3, int(piece))], oriented=False, combined=False,
    ):
        face = abs(int(t))
        kind = gmsh.model.getType(2, face)
        kinds.append(kind)
        if kind not in _ANALYTIC_FACES:
            end_faces.append(face)
    if not end_faces:
        raise RuntimeError(f"{where}: no cut face (boundary face types {kinds})")
    samples = []
    seen: set[int] = set()
    for face in end_faces:
        curves = [
            abs(int(t))
            for d, t in gmsh.model.getBoundary(
                [(2, face)], oriented=False, combined=False,
            )
            if int(d) == 1
        ]
        for curve in curves:
            if curve in seen:
                continue
            bounds = gmsh.model.getParametrizationBounds(1, curve)
            mid = _curve_point(curve, 0.5 * (float(bounds[0][0]) + float(bounds[1][0])))
            local = (mid - origin) @ frame
            if not _on_side(local, joint["planes"], tol_side):
                continue
            seen.add(curve)
            samples.append(_sample_curve(curve, h_hull))
    if not samples:
        raise RuntimeError(f"{where}: end face has no curve on a beam side")
    local = (np.vstack(samples) - origin) @ frame
    local[:, 1:3] = _snap_yz(local[:, 1:3], joint["planes"], tol_side)
    return local


def _fuse_weld(occ, piece: int, hulls, index: int) -> int:
    """Fuse the kept beam with its hulls. Inputs stay until one solid comes back.

    The union must hold at least the largest input and at most their sum;
    otherwise the fuse counts as failed.
    """
    hulls = [int(h) for h in hulls]
    piece = int(piece)
    masses = [float(occ.getMass(3, t)) for t in [piece, *hulls]]
    lo = max(masses) * (1.0 - 1e-6)
    hi = sum(masses) * (1.0 + 1e-6)

    def _once():
        if not hulls:
            return [piece]
        out, _map = occ.fuse(
            [(3, piece)], [(3, h) for h in hulls],
            removeObject=False, removeTool=False,
        )
        occ.synchronize()
        vols = [int(t) for d, t in out if int(d) == 3]
        if len(vols) == 1:
            got = float(occ.getMass(3, vols[0]))
            if not lo <= got <= hi:
                _drop_extras(vols)
                raise RuntimeError(
                    f"fused volume {got:.6e} outside [{lo:.6e}, {hi:.6e}]"
                )
        return vols

    def _drop_extras(vols):
        extra = [t for t in vols if t != piece and t not in hulls]
        if extra:
            occ.remove([(3, t) for t in extra], recursive=True)
            occ.synchronize()

    first = None
    try:
        vols = _once()
    except Exception as exc:
        first = exc
        vols = []
    if len(vols) != 1:
        if first is None:
            first = RuntimeError(f"fuse returned {len(vols)} solids")
        _drop_extras(vols)
        prev = gmsh.option.getNumber("Geometry.ToleranceBoolean")
        gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-6)
        try:
            try:
                vols = _once()
            except Exception as exc:
                raise RuntimeError(
                    f"beam {index} boolean failed ({first})"
                ) from exc
        finally:
            gmsh.option.setNumber("Geometry.ToleranceBoolean", prev)
    if len(vols) != 1:
        raise RuntimeError(f"beam {index} boolean failed ({first})")
    tag = vols[0]
    drop = [t for t in [piece, *hulls] if t != tag]
    if drop:
        occ.remove([(3, t) for t in drop], recursive=True)
        occ.synchronize()
    return tag


def _trim_by_hulls(occ, solid: int, hulls, rec: dict) -> tuple[int, int]:
    """Cut the beam prism with its hull2s and keep the piece around the witness.

    The kept piece ends on hull2's faces. Each attempt cuts a copy, so the
    retry at ``Geometry.ToleranceBoolean = 1e-6`` starts from the intact
    prism. ``solid`` is removed once a piece is kept. Returns the kept tag
    and the number of dropped pieces.
    """
    index = rec["index"]
    witness = _witness_point(rec)
    hulls = [int(h) for h in hulls]
    full = float(occ.getMass(3, int(solid)))
    removed = sum(float(occ.getMass(3, h)) for h in hulls)
    bounds = (full - removed - 1e-6 * full, full * (1.0 + 1e-6))

    def _attempt():
        occ.synchronize()
        before = {int(t) for _d, t in gmsh.model.getEntities(3)}
        trial = [int(t) for d, t in occ.copy([(3, int(solid))]) if int(d) == 3][0]
        try:
            return _cut_keep(
                occ, trial, hulls, witness, index, "hull2 trim", bounds,
            )
        except Exception:
            occ.synchronize()
            extra = [
                (3, int(t)) for _d, t in gmsh.model.getEntities(3)
                if int(t) not in before
            ]
            if extra:
                occ.remove(extra, recursive=True)
                occ.synchronize()
            raise

    try:
        piece, n_drop = _attempt()
    except Exception as first:
        prev = gmsh.option.getNumber("Geometry.ToleranceBoolean")
        gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-6)
        try:
            piece, n_drop = _attempt()
        except Exception as exc:
            raise RuntimeError(f"beam {index} hull2 trim failed ({first})") from exc
        finally:
            gmsh.option.setNumber("Geometry.ToleranceBoolean", prev)
    occ.remove([(3, int(solid))], recursive=True)
    occ.synchronize()
    _check_trimmed_ends(piece, rec)
    return piece, n_drop


def _check_trimmed_ends(piece: int, rec: dict) -> None:
    """Raise if the kept piece still reaches a joint end of the prism.

    A hull2 that misses part of the section leaves the beam end attached to
    the middle through a skin along a side. One probe sits just inside each
    side plane next to each joint end, so hollow sections are probed in
    their wall.
    """
    eps = 1e-3 * float(rec["section"]["w_min"])
    for joint in rec["joints"]:
        origin = np.asarray(joint["origin"], dtype=float)
        g = np.asarray(joint["frame"], dtype=float)
        normals, offsets = joint["planes"]
        for n, off in zip(normals, offsets):
            yz = (float(off) - eps) * np.asarray(n, dtype=float)
            probe = origin + eps * g[:, 0] + yz[0] * g[:, 1] + yz[1] * g[:, 2]
            if _inside(piece, probe):
                raise RuntimeError(
                    f"beam {rec['index']} {joint['side']}: hull2 does not span "
                    "the beam section, so the trim left the beam end attached"
                )


def _witness_point(rec: dict) -> np.ndarray:
    """Mid-chord for a coil-coil beam, just inside the foundation end for a coil-foundation beam."""
    axis = rec["gamma3"][:, 0]
    if rec["cf"]:
        return rec["x_start"] + 0.99 * rec["L"] * axis
    return rec["x_start"] + 0.5 * rec["L"] * axis


# ============================================================================
# Assembly, mesh, write
# ============================================================================


def _cut_by_coils(
    occ, piece: int, rec: dict, coil_vols, hull_vol: float, Q_list, image: int,
) -> int:
    """Cut one placed support by the coil images its joints touch.

    ``image`` is the symmetry copy. The cut uses that copy's own coil
    solids. The coil-side inset makes hull2 reach into the coil, so the
    cut lands on the casing. The resulting face is a copy of that surface;
    :func:`_share_flush_contacts` makes the support use the coil's own
    face. Retries once at ``Geometry.ToleranceBoolean = 1e-6``. Returns
    the kept tag.
    """
    index = rec["index"]
    image = int(image)
    tools = []
    for joint in rec["joints"]:
        dest = _image_of(Q_list, Q_list[image] @ Q_list[int(joint["image"])])
        for tag, _mat in coil_vols[(int(joint["coil"]), int(dest))]:
            tools.append(int(tag))
    welded = float(occ.getMass(3, int(piece)))
    bounds = (welded - float(hull_vol) - 1e-6 * welded, welded * (1.0 + 1e-6))
    witness = Q_list[image] @ _witness_point(rec)

    def _attempt():
        occ.synchronize()
        before = {int(t) for _d, t in gmsh.model.getEntities(3)}
        trial = [int(t) for d, t in occ.copy([(3, int(piece))]) if int(d) == 3][0]
        try:
            return _cut_keep(
                occ, trial, tools, witness, index, "coil cut", bounds,
            )
        except Exception:
            occ.synchronize()
            extra = [
                (3, int(t)) for _d, t in gmsh.model.getEntities(3)
                if int(t) not in before
            ]
            if extra:
                occ.remove(extra, recursive=True)
                occ.synchronize()
            raise

    try:
        kept, _n_drop = _attempt()
    except Exception as first:
        prev = gmsh.option.getNumber("Geometry.ToleranceBoolean")
        gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-6)
        try:
            kept, _n_drop = _attempt()
        except Exception as exc:
            raise RuntimeError(
                f"beam {index} image {image} coil cut failed ({first})"
            ) from exc
        finally:
            gmsh.option.setNumber("Geometry.ToleranceBoolean", prev)
    occ.remove([(3, int(piece))], recursive=True)
    occ.synchronize()
    curved = sum(
        1
        for _d, face in gmsh.model.getBoundary([(3, int(kept))], oriented=False)
        if gmsh.model.getType(2, abs(int(face))) not in _ANALYTIC_FACES
    )
    need = len(rec["joints"])
    if curved < need:
        raise RuntimeError(
            f"beam {index} image {image}: coil cut left {curved} contact faces, "
            f"expected at least {need}"
        )
    print(
        f"beam {index} image {image}: coil cut dropped {_n_drop} pieces, "
        f"contact faces {curved}",
        flush=True,
    )
    return int(kept)


def _share_flush_contacts(occ, placed, coil_vols):
    """Make each flush coil-cut face a face of the coil it lies on.

    The cut builds a new B-spline with the same poles as the casing, so
    the contact sits on the casing but is a different face. Fragment does
    not imprint a contact with no volume overlap, and the mesh then stops
    on overlapping facets. Edges shorter than ``_FLUSH_EDGE`` are dropped
    first: the cut leaves spurs that the splitter would discard, and the
    support must reuse the splitter's edges. Each coil is split by the
    contact faces, and the support is rebuilt on those faces. Coils and
    supports with no such pair are left alone. ``coil_vols`` is updated
    in place. Returns the placed list with the new support tags.
    """
    if not placed:
        return placed
    try:
        from OCP.BOPAlgo import BOPAlgo_Splitter
        from OCP.BRep import BRep_Builder
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.BRepBuilderAPI import (
            BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing,
        )
        from OCP.BRepGProp import BRepGProp
        from OCP.BRepLib import BRepLib
        from OCP.BRepTools import BRepTools
        from OCP.GeomAbs import GeomAbs_BSplineSurface
        from OCP.GProp import GProp_GProps
        from OCP.ShapeFix import ShapeFix_Wireframe
        from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Shape
    except ImportError as exc:
        raise RuntimeError(
            "_export_simsopt: imprinting flush contacts needs OCP"
        ) from exc
    import os
    import tempfile

    def _mass_com(shape, volumetric):
        props = GProp_GProps()
        if volumetric:
            BRepGProp.VolumeProperties_s(shape, props)
        else:
            BRepGProp.SurfaceProperties_s(shape, props)
        centre = props.CentreOfMass()
        return abs(props.Mass()), np.array([centre.X(), centre.Y(), centre.Z()])

    def _faces(shape):
        found = []
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            found.append(TopoDS.Face_s(explorer.Current()))
            explorer.Next()
        return found

    def _solids(shape):
        found = []
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        while explorer.More():
            found.append(TopoDS.Solid_s(explorer.Current()))
            explorer.Next()
        return found

    def _pole_key(face):
        adapted = BRepAdaptor_Surface(face, True)
        if adapted.GetType() != GeomAbs_BSplineSurface:
            return None
        surface = adapted.BSpline()
        n_u, n_v = surface.NbUPoles(), surface.NbVPoles()
        poles = np.empty((n_u * n_v, 3))
        index = 0
        for i in range(1, n_u + 1):
            for j in range(1, n_v + 1):
                pole = surface.Pole(i, j)
                poles[index] = (pole.X(), pole.Y(), pole.Z())
                index += 1
        return (surface.UDegree(), surface.VDegree(), n_u, n_v, poles.tobytes())

    def _match_tag(solid, info):
        mass, centre = _mass_com(solid, True)
        hits = []
        for tag, (ref_mass, ref_centre) in info.items():
            scale = max(abs(ref_mass), 1e-16)
            if abs(mass - ref_mass) / scale > 1e-5:
                continue
            if np.linalg.norm(centre - ref_centre) > 1e-5:
                continue
            hits.append(tag)
        if len(hits) != 1:
            raise RuntimeError(
                "_export_simsopt: could not match a brep solid back to "
                f"one volume (mass {mass:.6e}, candidates {hits})"
            )
        return hits[0]

    occ.synchronize()
    info = {}
    for _dim, tag in gmsh.model.getEntities(3):
        tag = int(tag)
        info[tag] = (
            float(occ.getMass(3, tag)),
            np.asarray(occ.getCenterOfMass(3, tag), dtype=float),
        )
    fd, path = tempfile.mkstemp(suffix=".brep")
    os.close(fd)
    try:
        gmsh.write(path)
        compound = TopoDS_Shape()
        BRepTools.Read_s(compound, path, BRep_Builder())
    finally:
        os.unlink(path)
    solids = _solids(compound)
    if len(solids) != len(info):
        raise RuntimeError(
            f"_export_simsopt: brep has {len(solids)} solids, "
            f"the model has {len(info)} volumes"
        )
    shapes = {}
    remaining = dict(info)
    for solid in solids:
        tag = _match_tag(solid, remaining)
        del remaining[tag]
        shapes[tag] = solid

    coil_keys = {}
    for (coil, image), pairs in coil_vols.items():
        for tag, _mat in pairs:
            keys = set()
            for face in _faces(shapes[int(tag)]):
                key = _pole_key(face)
                if key is not None:
                    keys.add(key)
            coil_keys[int(tag)] = keys

    def _contacts_on(solid):
        found = []
        for face in _faces(solid):
            key = _pole_key(face)
            if key is None:
                continue
            hits = [tag for tag, keys in coil_keys.items() if key in keys]
            if len(hits) > 1:
                raise RuntimeError(
                    "_export_simsopt: a contact face lies on more than "
                    f"one coil solid {hits}"
                )
            if len(hits) != 1:
                continue
            coil_faces = _faces(shapes[hits[0]])
            if any(face.IsPartner(other) for other in coil_faces):
                continue
            found.append((hits[0], face))
        return found

    fixed = {}
    for tag, beam, image in placed:
        contacts = _contacts_on(shapes[int(tag)])
        if not contacts:
            continue
        wireframe = ShapeFix_Wireframe(shapes[int(tag)])
        wireframe.SetPrecision(_FLUSH_EDGE)
        wireframe.ModeDropSmallEdges = True
        wireframe.FixSmallEdges()
        repaired = wireframe.Shape()
        parts = _solids(repaired)
        if len(parts) != 1:
            raise RuntimeError(
                f"beam {beam} image {image}: dropping edges shorter than "
                f"{_FLUSH_EDGE:.0e} produced {len(parts)} solids"
            )
        before = _mass_com(shapes[int(tag)], True)[0]
        after = _mass_com(parts[0], True)[0]
        if abs(after - before) > 1e-4 * max(before, 1e-16):
            raise RuntimeError(
                f"beam {beam} image {image}: dropping short edges changed "
                f"the volume from {before:.6e} to {after:.6e}"
            )
        if len(_contacts_on(parts[0])) != len(contacts):
            raise RuntimeError(
                f"beam {beam} image {image}: dropping short edges changed "
                "the number of coil contact faces"
            )
        fixed[int(tag)] = parts[0]

    if not fixed:
        return placed

    # coil tag -> list of (support tag, tool face) that lie on it
    tools_for = defaultdict(list)
    for tag, _beam, _image in placed:
        if int(tag) not in fixed:
            continue
        for coil_tag, face in _contacts_on(fixed[int(tag)]):
            tools_for[int(coil_tag)].append((int(tag), face))

    imprinted = defaultdict(list)  # support tag -> imprint faces
    new_coils = {}
    for coil_tag, tools in tools_for.items():
        splitter = BOPAlgo_Splitter()
        splitter.AddArgument(shapes[coil_tag])
        for _support, face in tools:
            splitter.AddTool(face)
        splitter.Perform()
        if splitter.HasErrors():
            raise RuntimeError(
                f"_export_simsopt: imprinting contacts onto coil volume "
                f"{coil_tag} failed"
            )
        parts = _solids(splitter.Shape())
        if len(parts) != 1:
            raise RuntimeError(
                f"_export_simsopt: imprinting contacts onto coil volume "
                f"{coil_tag} produced {len(parts)} solids"
            )
        new_coils[coil_tag] = parts[0]
        pool = _faces(parts[0])
        for support_tag, tool in tools:
            area0, centre0 = _mass_com(tool, False)
            best, best_distance = None, None
            for face in pool:
                area, centre = _mass_com(face, False)
                if area0 <= 0 or abs(area - area0) / area0 > 1e-3:
                    continue
                distance = float(np.linalg.norm(centre - centre0))
                if distance > 1e-3:
                    continue
                if best is None or distance < best_distance:
                    best, best_distance = face, distance
            if best is None:
                beam, image = next(
                    (b, s) for t, b, s in placed if int(t) == support_tag
                )
                raise RuntimeError(
                    f"beam {beam} image {image}: imprint missed a contact "
                    f"of area {area0:.6e}"
                )
            pool.remove(best)
            imprinted[support_tag].append(best)

    new_supports = {}
    for tag, beam, image in placed:
        if int(tag) not in fixed:
            continue
        imprints = imprinted[int(tag)]
        contacts = [face for _coil, face in _contacts_on(fixed[int(tag)])]
        sewing = BRepBuilderAPI_Sewing(_FLUSH_SEW)
        for face in _faces(fixed[int(tag)]):
            if any(face.IsPartner(contact) for contact in contacts):
                continue
            sewing.Add(face)
        for face in imprints:
            sewing.Add(face)
        sewing.Perform()
        if sewing.NbFreeEdges():
            raise RuntimeError(
                f"beam {beam} image {image}: the imprinted contact left "
                f"{sewing.NbFreeEdges()} free edges"
            )
        shells = []
        explorer = TopExp_Explorer(sewing.SewedShape(), TopAbs_SHELL)
        while explorer.More():
            shells.append(TopoDS.Shell_s(explorer.Current()))
            explorer.Next()
        if len(shells) != 1:
            raise RuntimeError(
                f"beam {beam} image {image}: imprint rebuilt "
                f"{len(shells)} shells"
            )
        maker = BRepBuilderAPI_MakeSolid(shells[0])
        if not maker.IsDone():
            raise RuntimeError(
                f"beam {beam} image {image}: the imprinted support is not a solid"
            )
        solid = maker.Solid()
        BRepLib.OrientClosedSolid_s(solid)
        before = _mass_com(shapes[int(tag)], True)[0]
        after = _mass_com(solid, True)[0]
        if after <= 0 or abs(after - before) > 1e-4 * max(before, 1e-16):
            raise RuntimeError(
                f"beam {beam} image {image}: imprint changed the volume "
                f"from {before:.6e} to {after:.6e}"
            )
        owned = _faces(solid)
        for face in imprints:
            if not any(other.IsPartner(face) for other in owned):
                raise RuntimeError(
                    f"beam {beam} image {image}: the support does not use "
                    "the coil's contact face"
                )
        new_supports[int(tag)] = solid

    builder = BRep_Builder()
    exported = TopoDS_Compound()
    builder.MakeCompound(exported)
    order = []
    for tag, shape in (*new_supports.items(), *new_coils.items()):
        builder.Add(exported, shape)
        order.append((int(tag), _mass_com(shape, True)))
    fd, path = tempfile.mkstemp(suffix=".brep")
    os.close(fd)
    try:
        BRepTools.Write_s(exported, path)
        occ.remove([(3, tag) for tag, _mc in order], recursive=True)
        occ.synchronize()
        imported = [
            int(tag) for dim, tag in gmsh.model.occ.importShapes(path)
            if int(dim) == 3
        ]
        occ.synchronize()
    finally:
        os.unlink(path)
    if len(imported) != len(order):
        raise RuntimeError(
            f"_export_simsopt: imprint imported {len(imported)} volumes, "
            f"expected {len(order)}"
        )
    remap = {}
    claimed = set()
    for new_tag in imported:
        mass = float(occ.getMass(3, new_tag))
        centre = np.asarray(occ.getCenterOfMass(3, new_tag), dtype=float)
        hits = []
        for old_tag, (ref_mass, ref_centre) in order:
            if old_tag in claimed:
                continue
            scale = max(abs(ref_mass), 1e-16)
            if abs(mass - ref_mass) / scale > 1e-5:
                continue
            if np.linalg.norm(centre - ref_centre) > 1e-5:
                continue
            hits.append(old_tag)
        if len(hits) != 1:
            raise RuntimeError(
                "_export_simsopt: could not match an imprinted volume "
                f"back to its input (mass {mass:.6e}, candidates {hits})"
            )
        claimed.add(hits[0])
        remap[hits[0]] = new_tag

    placed = [
        (remap.get(int(tag), int(tag)), int(beam), int(image))
        for tag, beam, image in placed
    ]
    for pairs in coil_vols.values():
        for index, (tag, mat) in enumerate(pairs):
            if int(tag) in remap:
                pairs[index] = (remap[int(tag)], mat)
    print(
        f"_export_simsopt: imprinted {sum(len(v) for v in imprinted.values())} "
        f"flush contacts on {len(new_coils)} coil volumes",
        flush=True,
    )
    return placed


def _fragment_device(occ, copied, coil_vols, recs, Q_list):
    """Fragment every support against every coil volume.

    Supports are the objects, so supports that touch are intersected. Coils
    are the tools, so coils are not cut against each other. Returns the
    volume owner map and ``claimed``, mapping each support volume to the
    ``(beam, image)`` pairs it came from.
    """
    if not copied:
        owner = {
            int(tag): (int(i), int(s), int(mat))
            for (i, s), pairs in coil_vols.items()
            for tag, mat in pairs
        }
        return owner, {}
    by_index = {int(rec["index"]): rec for rec in recs}
    supports = [(3, int(tag)) for tag, _beam, _image in copied]
    coils, coil_labels = [], []
    for (i, s), pairs in coil_vols.items():
        for tag, mat in pairs:
            coils.append((3, int(tag)))
            coil_labels.append((int(i), int(s), int(mat)))
    inputs = supports + coils
    in_vol = sum(float(occ.getMass(3, int(t))) for _d, t in inputs)
    labels = [(-1, -1, -1)] * len(supports) + coil_labels
    try:
        _out, ov_map = occ.fragment(supports, coils)
        occ.synchronize()
    except Exception as exc:
        raise RuntimeError("_export_simsopt: fragment failed") from exc
    if len(ov_map) != len(labels):
        raise RuntimeError(
            f"_export_simsopt: fragment map length {len(ov_map)} != {len(labels)} inputs"
        )
    owner = _owner_map(ov_map, labels)
    out_vol = sum(float(occ.getMass(3, int(t))) for t in owner)
    if out_vol > in_vol * (1.0 + 1e-4):
        raise RuntimeError(
            f"_export_simsopt: fragment volume {out_vol:.6e} exceeds "
            f"the inputs {in_vol:.6e}"
        )
    print(f"_export_simsopt: fragment overlap {in_vol - out_vol:.6e}", flush=True)

    support_out = {
        int(t)
        for k in range(len(supports))
        for d, t in ov_map[k]
        if int(d) == 3
    }
    warned: set[int] = set()
    for k in range(len(coils)):
        for d, t in ov_map[len(supports) + k]:
            tag = int(t)
            if int(d) == 3 and tag in support_out and tag not in warned:
                warned.add(tag)
                warnings.warn(
                    "_export_simsopt: support still overlaps a coil by "
                    f"{float(occ.getMass(3, tag)):.6e} after the coil cut"
                )

    def faces(tag: int) -> set[int]:
        return {
            abs(int(face))
            for d, face in gmsh.model.getBoundary([(3, int(tag))], oriented=False)
            if int(d) == 2
        }

    coil_faces: dict[tuple[int, int], set[int]] = defaultdict(set)
    for tag, (coil, image, _mat) in owner.items():
        if coil >= 0:
            coil_faces[(int(coil), int(image))] |= faces(tag)

    claimed: dict[int, list] = {}
    for k, (_tag, beam, image) in enumerate(copied):
        pieces = [
            int(t) for d, t in ov_map[k]
            if int(d) == 3 and owner.get(int(t), (0, 0, 0))[0] < 0
        ]
        for tag in pieces:
            claimed.setdefault(tag, []).append((int(beam), int(image)))
        piece_faces: set[int] = set()
        for tag in pieces:
            piece_faces |= faces(tag)
        rec = by_index[int(beam)]
        for joint in rec["joints"]:
            dest = _image_of(
                Q_list, Q_list[int(image)] @ Q_list[int(joint["image"])],
            )
            key = (int(joint["coil"]), int(dest))
            if not piece_faces & coil_faces.get(key, set()):
                raise RuntimeError(
                    f"beam {beam} image {image}: no shared face with "
                    f"coil {key[0]} image {key[1]}"
                )
    return owner, claimed


def _same_ends(a0, a1, b0, b1, tol: float) -> bool:
    """Unordered endpoint match within ``tol``."""
    return (
        (np.linalg.norm(a0 - b0) <= tol and np.linalg.norm(a1 - b1) <= tol)
        or (np.linalg.norm(a0 - b1) <= tol and np.linalg.norm(a1 - b0) <= tol)
    )


def _copy_supports(occ, masters, Q_list) -> list[tuple[int, int, int]]:
    """Copy each finished support under every ``Q``. Skip a copy that lands on an existing support.

    Returns ``(volume tag, beam index, image index)``.
    """
    placed = []
    out = []
    for rec, tag in masters:
        placed.append((rec["x_start"], rec["x_end"], float(rec["L"])))
        out.append((int(tag), int(rec["index"]), 0))
    for rec, tag in masters:
        for s, Q in enumerate(Q_list):
            if s == 0:
                continue
            xs = Q @ rec["x_start"]
            xe = Q @ rec["x_end"]
            tol = 1e-6 * float(rec["L"])
            if any(_same_ends(xs, xe, a, b, tol) for a, b, _L in placed):
                continue
            copied = list(occ.copy([(3, int(tag))]))
            _rigid(occ, copied, Q)
            new_tag = [int(t) for d, t in copied if int(d) == 3][0]
            placed.append((xs, xe, float(rec["L"])))
            out.append((new_tag, int(rec["index"]), int(s)))
    occ.synchronize()
    return out


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


def _coil_names(coil_vols, touched) -> tuple[list[int], list[str]]:
    """Names for the raw STEP file: base coils, touched images, no hulls."""
    vols, names = [], []
    for (i, s), pairs in coil_vols.items():
        if s != 0 and (i, s) not in touched:
            continue
        count = {0: 0, 1: 0}
        for tag, mat in pairs:
            k = count[mat]
            count[mat] = k + 1
            kind = "winding_pack" if mat == 0 else "casing"
            vols.append(int(tag))
            names.append(f"{kind}_c{i}_s{s}_{k}")
    return vols, names


def _final_names(owner, claimed) -> tuple[list[int], list[str]]:
    """Names for the fused STEP file. Merged supports get a ``support_merged`` name."""
    vols, names = [], []
    count: dict[tuple, int] = {}
    merged = 0
    for tag, label in owner.items():
        c, s, mat = label
        if mat < 0:
            beams = claimed.get(tag, [])
            uniq = sorted(set(beams))
            if len(uniq) == 1:
                key = ("support", uniq[0][0], uniq[0][1])
                k = count.get(key, 0)
                count[key] = k + 1
                name = f"support_b{uniq[0][0]}_s{uniq[0][1]}_{k}"
            else:
                name = f"support_merged_{merged}"
                merged += 1
        else:
            kind = "winding_pack" if mat == 0 else "casing"
            key = (kind, c, s)
            k = count.get(key, 0)
            count[key] = k + 1
            name = f"{kind}_c{c}_s{s}_{k}"
        vols.append(int(tag))
        names.append(name)
    return vols, names


def _mesh_tet10(owner, size_max: float, size_min: float):
    """Mesh every volume of this model in one pass. Shared faces share nodes.

    Returns the minimum gamma quality, the volume it came from, the points,
    the VTK TET10 cells and the per-cell owner arrays.
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
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    print(
        f"_export_simsopt: meshing, MeshSizeMin={size_min:.4g}, "
        f"MeshSizeMax={size_max:.4g}",
        flush=True,
    )
    try:
        gmsh.model.mesh.generate(3)
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

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    if node_tags.size == 0:
        raise RuntimeError("_export_simsopt: mesh produced no nodes")
    points = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)
    lut = np.full(int(node_tags.max()) + 1, -1, dtype=np.int64)
    lut[node_tags] = np.arange(node_tags.size)

    cells = []
    owner_coil, owner_sym, material_id = [], [], []
    worst, worst_vol = 1.0, None
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
        quality = np.asarray(
            gmsh.model.mesh.getElementQualities(etags[slot], "gamma"), dtype=float,
        )
        if quality.size and float(quality.min()) < worst:
            worst, worst_vol = float(quality.min()), int(tag)
        coil, image, mat = label
        cells.append(conn)
        owner_coil.append(np.full(len(conn), int(coil), dtype=np.int32))
        owner_sym.append(np.full(len(conn), int(image), dtype=np.int32))
        material_id.append(np.full(len(conn), int(mat), dtype=np.int32))
    if not cells:
        raise RuntimeError("_export_simsopt: no volumes to mesh")
    return (
        worst,
        int(worst_vol) if worst_vol is not None else -1,
        points,
        np.vstack(cells),
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


def _check_rounding(meshes) -> None:
    """Reject a chamfered winding pack that is also the coil's outer surface.

    The grown coil and the coil contour use the sharp outer outline. With a
    casing that outline is the sharp casing, so the winding-pack chamfer is
    fine. Without a casing the chamfered corners would leave a gap.
    """
    bare = [
        i for i, mesh in enumerate(meshes)
        if getattr(mesh, "rounding", None) is not None and int(mesh.n_casing) == 0
    ]
    if bare:
        raise NotImplementedError(
            "winding-pack chamfer without a casing is not supported "
            f"(coils {bare}): the grown coil and the coil contour use the "
            "sharp outer outline"
        )


def _export_simsopt(
    Jstress,
    prefix="full_body_fields",
    mesh_scale=0.5,
    joint_options=None,
    save_step=False,
    save_vtu=True,
    n_sides_circle=12,
) -> None:
    """Write the hull-welded full device to STEP and/or VTU.

    Parameters
    ----------
    Jstress : CoilFEMObjective
        Must wrap :class:`~coil_fem.coupling.beam_network.SupportBeams`
        (not a central-support-ring support) and rectangular coil meshes.
    prefix : path-like
        Path without an extension. Outputs are ``<prefix>.vtu``, and when
        ``save_step`` is set, ``<prefix>_raw.step`` and ``<prefix>.step``.
        The parent directory must exist.
    mesh_scale : float
        Multiplier on ``MeshSizeMax``. The size is ``mesh_scale * 0.5 * w1``
        of the first coil, capped at the smallest casing thickness.
        ``MeshSizeMin`` is ``min(MeshSizeMax, delta)``.
    joint_options : dict or None
        ``delta`` is the grown-coil offset. It defaults to ``min(w1, w2) / 4``
        for rectangular beams and ``r / 2`` (outer radius) for circular
        beams, taking the smallest over all beams. A copy of each
        beam is cut by the grown coil only to find the ring on the beam
        sides; the beam itself is trimmed by hull2. Each symmetry copy of
        the weld is then cut by that image's coil volumes, so the support
        ends on the casing. A contact that is only a copy of the casing
        surface is imprinted onto that surface, so the support and the
        coil share the face. The coil
        contour sits at ``delta * sqrt(2)``. ``h_contour`` (contour spacing)
        and ``h_hull`` (ring sampling spacing) default to ``delta``.
        ``bool_inset`` defaults to ``0.01 * w_min`` and pushes only the coil
        patch into the coil, which is what lets the coil cut land on the
        casing; ring samples are snapped onto the beam's side planes instead
        of being inset. Unknown keys raise. A winding pack is chamfered with
        the objective's ``rounding``; a chamfer without a casing raises
        ``NotImplementedError``.
    save_step : bool
        Write the raw half-period STEP file before any cut, and the
        fragmented device afterwards. Default ``False``.
    save_vtu : bool
        Mesh every volume in one model, so the support and the coil share
        nodes on the contact patch, and write ``<prefix>.vtu``. Default
        ``True``. When ``False`` the function returns after the fragmented
        STEP file and does not mesh.
    n_sides_circle : int
        Number of sides of the prism used for a circular beam. Vertices lie
        on the circle and vertex 0 lies on local +y. Default 12. Must be
        at least 3.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        The support, the meshes, or ``joint_options`` are not supported.
    FileNotFoundError
        The parent directory of ``prefix`` does not exist.
    RuntimeError
        A contour, boolean, fragment, or element-quality check failed.
    NotImplementedError
        A winding pack is chamfered and has no casing.
    """
    if int(n_sides_circle) < 3:
        raise ValueError(f"n_sides_circle must be at least 3, got {n_sides_circle}")
    n_sides_circle = int(n_sides_circle)
    if not save_step and not save_vtu:
        warnings.warn("No file exported")
        return None

    prefix = Path(prefix)
    if not prefix.parent.exists():
        raise FileNotFoundError(
            f"parent directory {prefix.parent} does not exist"
        )
    if mesh_scale <= 0.0:
        raise ValueError(f"mesh_scale must be positive, got {mesh_scale}")
    try:
        coil_support = Jstress.coil_support
        fem = Jstress.fem
    except AttributeError as exc:
        raise ValueError(
            "_export_simsopt requires a CoilFEMObjective"
        ) from exc
    support = fem.support
    if type(support) is not SupportBeams:
        raise ValueError(
            "_export_simsopt supports SupportBeams only, "
            f"got {type(support).__name__}"
        )
    meshes = fem.meshes
    if not meshes or not all(
        isinstance(m, (FramedCurveMeshRectangle, FramedCurveMeshSection))
        for m in meshes
    ):
        raise ValueError(
            "_export_simsopt requires rectangular coil meshes "
            "(FramedCurveMeshRectangle or FramedCurveMeshSection)"
        )
    _check_rounding(meshes)

    sdofs = coil_support.support_dofs
    curves = fem.base_curves_jax
    geom = support.beam_geometry(curves, sdofs)
    Q_list = _sym_maps(support.nfp, support.stellsym)
    recs = _master_beams(support, sdofs, geom, Q_list, n_sides_circle)
    opt = _options(joint_options, [rec["section"] for rec in recs])
    print(f"_export_simsopt: delta={opt['delta']:.4g}", flush=True)
    w1 = float(meshes[0].w1)
    size_max = float(mesh_scale) * 0.5 * w1
    thicknesses = [float(m.casing_thickness) for m in meshes if int(m.n_casing) > 0]
    if thicknesses:
        size_max = min(size_max, min(thicknesses))
    size_min = min(size_max, opt["delta"])

    try:
        owned = not gmsh.isInitialized()
    except AttributeError:
        owned = True
    if owned:
        gmsh.initialize()
    else:
        gmsh.clear()
    occ_parallel = gmsh.option.getNumber("Geometry.OCCParallel")
    try:
        gmsh.model.add("device")
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.option.setNumber("Geometry.OCCParallel", 1)
        occ = gmsh.model.occ
        coil_vols = _coil_images(occ, meshes, Q_list)
        for rec in recs:
            solids = _beam_solid(occ, rec)
            occ.synchronize()
            if len(solids) != 1:
                raise RuntimeError(
                    f"beam {rec['index']} solid is {len(solids)} volumes"
                )
            rec["solid"] = solids[0][1]
        touched = {
            (j["coil"], j["image"]) for rec in recs for j in rec["joints"]
        }
        if save_step:
            vols, names = _coil_names(coil_vols, touched)
            for rec in recs:
                vols.append(int(rec["solid"]))
                names.append(f"support_b{rec['index']}_s0_0")
            _write_step(Path(f"{prefix}_raw.step"), vols, names)

        grown = _grown_tools(occ, meshes, Q_list, touched, opt["delta"])
        finished = []
        for rec in recs:
            print(f"_export_simsopt: beam {rec['index']}", flush=True)
            keys = [(joint["coil"], joint["image"]) for joint in rec["joints"]]
            if len(set(keys)) != len(keys):
                raise RuntimeError(
                    f"beam {rec['index']}: both ends touch coil {keys[0][0]} "
                    f"image {keys[0][1]}, so its cut faces cannot be told apart"
                )
            coil_inset = opt["bool_inset"]
            if coil_inset is None:
                coil_inset = 0.01 * rec["section"]["w_min"]
            hull_tags = []
            n_grown = 0
            for joint, key in zip(rec["joints"], keys):
                where = f"beam {rec['index']} {joint['side']}"
                scratch = [
                    int(t) for d, t in occ.copy([(3, int(rec["solid"]))]) if int(d) == 3
                ][0]
                cut, n_drop = _cut_keep(
                    occ, scratch, grown[key], _witness_point(rec), rec["index"],
                    f"{joint['side']} grown-coil cut",
                )
                n_grown += n_drop
                ring = _beam_ring(cut, joint, opt["h_hull"])
                occ.remove([(3, cut)], recursive=True)
                occ.synchronize()
                origin = np.asarray(joint["origin"], dtype=float)
                g = np.asarray(joint["frame"], dtype=float)
                world = (
                    origin
                    + ring[:, 0:1] * g[:, 0]
                    + ring[:, 1:2] * g[:, 1]
                    + ring[:, 2:3] * g[:, 2]
                )
                hull = _Hull(world)
                coil = _coil_contour(
                    meshes[joint["coil"]], Q_list[joint["image"]], hull,
                    opt["delta"], opt["h_contour"], where,
                )
                tag, vol = _hull_solid(occ, _hull2_points(
                    joint, ring, coil, coil_inset,
                ))
                hull_tags.append(tag)
                span = float(ring[:, 0].max() - ring[:, 0].min())
                print(
                    f"{where} (coil {joint['coil']} image {joint['image']}): "
                    f"ring span {span:.4f}, coil loops {coil['n_loops']}, "
                    f"hull2 vol {vol:.3e}, |d-r| {coil['resid_max']:.2e}",
                    flush=True,
                )
            piece, n_trim = _trim_by_hulls(occ, rec["solid"], hull_tags, rec)
            print(
                f"beam {rec['index']}: pieces dropped by the grown-coil cut "
                f"{n_grown}, by the hull2 trim {n_trim}",
                flush=True,
            )
            hull_vol = sum(float(occ.getMass(3, int(t))) for t in hull_tags)
            welded = _fuse_weld(occ, piece, hull_tags, rec["index"])
            rec["hull_vol"] = hull_vol
            finished.append((rec, welded))

        grown_tags = [tag for tags in grown.values() for tag in tags]
        if grown_tags:
            occ.remove([(3, tag) for tag in grown_tags], recursive=True)
            occ.synchronize()

        # Each copy is cut by its own coil image, not copied after a master cut:
        # the flush imprint is exact only when the contact reuses that image's faces.
        copied = _copy_supports(occ, finished, Q_list)
        by_index = {int(rec["index"]): rec for rec in recs}
        placed = []
        for tag, beam, image in copied:
            rec = by_index[int(beam)]
            placed.append((
                _cut_by_coils(
                    occ, tag, rec, coil_vols, rec["hull_vol"], Q_list, image,
                ),
                int(beam), int(image),
            ))
        placed = _share_flush_contacts(occ, placed, coil_vols)
        owner, claimed = _fragment_device(occ, placed, coil_vols, recs, Q_list)
        if save_step:
            vols, names = _final_names(owner, claimed)
            _write_step(Path(f"{prefix}.step"), vols, names)
        if not save_vtu:
            return None

        (
            quality, worst_vol, points, cells, owner_coil, owner_sym, material_id,
        ) = _mesh_tet10(owner, size_max, size_min)
    finally:
        gmsh.option.setNumber("Geometry.OCCParallel", occ_parallel)
        if owned:
            gmsh.finalize()
        else:
            gmsh.clear()

    n_nodes = int(points.shape[0])
    n_cells = int(cells.shape[0])
    n_coil = int(np.count_nonzero(owner_coil >= 0))
    n_casing = int(np.count_nonzero(material_id == 1))
    print(
        "_export_simsopt mesh counts:\n"
        f"  all bodies: {n_nodes} nodes, {n_cells} cells\n"
        f"  coils:      {n_coil} cells\n"
        f"  casing:     {n_casing} cells\n"
        f"  support:    {n_cells - n_coil} cells\n"
        f"  min quality {quality:.3g}"
    )
    has_clamps = (
        getattr(coil_support, "_r_clamp", None) is not None
        and getattr(coil_support, "_sig_eps", None) is not None
    )
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
    if quality < _QUALITY_MIN:
        raise RuntimeError(
            f"minimum element quality {quality:.3g} in volume {worst_vol} "
            f"{owner.get(int(worst_vol))} is below {_QUALITY_MIN}"
        )
    return None
