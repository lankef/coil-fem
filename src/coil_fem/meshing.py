import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from jax_fem.generate_mesh import Mesh as JAXFEMMesh



# ── Compile-time constants ───────────────────────────────────────────────────
#
#  Hex vertex layout (offsets from base cell (m, n, o)):
#
#       3 ──── 2          v0 = (m,   n,   o  )   v4 = (m,   n,   o+1)
#      /|     /|          v1 = (m+1, n,   o  )   v5 = (m+1, n,   o+1)
#     7 ──── 6 |          v2 = (m+1, n+1, o  )   v6 = (m+1, n+1, o+1)
#     | 0 ── | 1          v3 = (m,   n+1, o  )   v7 = (m,   n+1, o+1)
#     |/     |/
#     4 ──── 5
#
#  Kuhn / Freudenthal subdivision of the hex into 6 tetrahedra.  All six tets
#  share the main diagonal v0–v6 (the six monotone corner paths from (0,0,0)
#  to (1,1,1)).  Sharing one main diagonal is the property that makes the split
#  *conforming*: every pair of opposite faces receives parallel diagonals, so
#  neighbouring hexes always triangulate their shared face the same way.  Each
#  tet below is ordered for positive parametric (phi, u, v) volume.
#
#  NOTE: an earlier ad-hoc 6-tet dissection put crossing diagonals on the two
#  n(u)-faces, which made the mesh non-conforming across n-neighbours and, for
#  TET10, produced coincident-but-independent midside nodes (a hairline "crack"
#  at every interior interface).  Do not revert to a split that does not share
#  a single main diagonal.

_KUHN_6 = np.array([
    [0, 1, 2, 6],
    [0, 1, 6, 5],
    [0, 2, 3, 6],
    [0, 3, 7, 6],
    [0, 4, 5, 6],
    [0, 4, 6, 7],
], dtype=np.int32)

# VTK / JAX-FEM TET10 midside edge order (columns 4–9 of each cell):
#   [mid(c0,c1), mid(c1,c2), mid(c0,c2), mid(c0,c3), mid(c1,c3), mid(c2,c3)]
_TET10_VTK_EDGES = np.array([
    [0, 1], [1, 2], [0, 2], [0, 3], [1, 3], [2, 3],
], dtype=np.int32)

# All 6 edges of a tetrahedron as pairs of local corner indices.
# Shape (6, 2) — static compile-time constant, never traced.
_TET_EDGE_PAIRS = np.array([
    [0, 1], [0, 2], [0, 3],
    [1, 2], [1, 3], [2, 3],
], dtype=np.int32)

def _build_disk_o_grid_topology_np(n_center: int, n_radial: int):
    """
    Five-block structured O-grid for the unit disk (R = 1 in the cross-section plane).

    Returns quad connectivity (counter-clockwise in the (p, q) plane), unit-disk
    offsets ``oxy`` shape (n2d, 2), and the total node count.
    """
    Nc = int(n_center)
    Nr = int(n_radial)
    if Nc < 2 or Nr < 2:
        raise ValueError("n_center and n_radial must be >= 2")

    center_ids = np.empty((Nc, Nc), dtype=np.int32)
    nid = 0
    for i in range(Nc):
        for j in range(Nc):
            center_ids[i, j] = nid
            nid += 1

    east_ids = np.empty((Nc, Nr), dtype=np.int32)
    for i in range(Nc):
        for j in range(Nr):
            east_ids[i, j] = center_ids[i, Nc - 1] if j == 0 else nid
            if j > 0:
                nid += 1

    west_ids = np.empty((Nc, Nr), dtype=np.int32)
    for i in range(Nc):
        for j in range(Nr):
            west_ids[i, j] = center_ids[i, 0] if j == 0 else nid
            if j > 0:
                nid += 1

    north_ids = np.empty((Nc, Nr), dtype=np.int32)
    for i in range(Nc):
        for j in range(Nr):
            north_ids[i, j] = center_ids[Nc - 1, i] if j == 0 else nid
            if j > 0:
                nid += 1

    south_ids = np.empty((Nc, Nr), dtype=np.int32)
    for i in range(Nc):
        for j in range(Nr):
            south_ids[i, j] = center_ids[0, i] if j == 0 else nid
            if j > 0:
                nid += 1

    n2d = nid
    quads = []

    def add_quads(ids, ni, nj):
        for i in range(ni - 1):
            for j in range(nj - 1):
                quads.append(
                    [
                        ids[i, j],
                        ids[i + 1, j],
                        ids[i + 1, j + 1],
                        ids[i, j + 1],
                    ]
                )

    add_quads(center_ids, Nc, Nc)
    add_quads(east_ids, Nc, Nr)
    add_quads(west_ids, Nc, Nr)
    add_quads(north_ids, Nc, Nr)
    add_quads(south_ids, Nc, Nr)
    quads = np.asarray(quads, dtype=np.int32)

    s = 1.0 / np.sqrt(2.0)
    oxy = np.zeros((n2d, 2), dtype=np.float64)
    for i in range(Nc):
        for j in range(Nc):
            k = center_ids[i, j]
            n1 = -s + 2.0 * s * i / (Nc - 1)
            n2 = -s + 2.0 * s * j / (Nc - 1)
            oxy[k] = (n1, n2)
    for i in range(Nc):
        for j in range(Nr):
            k = east_ids[i, j]
            n2 = -s + 2.0 * s * i / (Nc - 1)
            ro = np.sqrt(max(1.0 - n2 * n2, 0.0))
            n1 = s + (j / (Nr - 1)) * (ro - s)
            oxy[k] = (n1, n2)
    for i in range(Nc):
        for j in range(Nr):
            k = west_ids[i, j]
            n2 = -s + 2.0 * s * i / (Nc - 1)
            ro = np.sqrt(max(1.0 - n2 * n2, 0.0))
            n1 = -s - (j / (Nr - 1)) * (ro - s)
            oxy[k] = (n1, n2)
    for i in range(Nc):
        for j in range(Nr):
            k = north_ids[i, j]
            n1 = -s + 2.0 * s * i / (Nc - 1)
            ro = np.sqrt(max(1.0 - n1 * n1, 0.0))
            n2 = s + (j / (Nr - 1)) * (ro - s)
            oxy[k] = (n1, n2)
    for i in range(Nc):
        for j in range(Nr):
            k = south_ids[i, j]
            n1 = -s + 2.0 * s * i / (Nc - 1)
            ro = np.sqrt(max(1.0 - n1 * n1, 0.0))
            n2 = -s - (j / (Nr - 1)) * (ro - s)
            oxy[k] = (n1, n2)

    n1min, n1max = oxy[:, 0].min(), oxy[:, 0].max()
    n2min, n2max = oxy[:, 1].min(), oxy[:, 1].max()
    span1 = n1max - n1min
    span2 = n2max - n2min
    # mu_log = (oxy[:, 0] - n1min) / (span1 if span1 > 1e-15 else 1.0)
    # nu_log = (oxy[:, 1] - n2min) / (span2 if span2 > 1e-15 else 1.0)
    return quads, oxy, n2d


# ─── Validation ───────────────────────────────────────────────────────────────

# JIT notes:
#   - The arithmetic (cross products, einsum) is fully JIT-safe.
#   - The *diagnostic prints* are NOT: never put Python `if traced_val` in
#     jitted code. We use jax.debug.print instead, which is JIT-safe and
#     executes at runtime (not trace time).
#   - If you want to use the returned `vols` inside a jitted caller, that is
#     fine — only the debug prints have side effects.

def validate_mesh(mesh) -> jax.Array:
    """
    Compute signed tet volumes and print a diagnostic summary.
    
    Works for both TET4 and TET10 — only the first 4 columns (corner nodes)
    are used.  Uses jax.debug.print so output appears at runtime, not trace time.

    Parameters
    ----------
    mesh : CoilMesh
        Mesh to validate.

    Returns
    -------
    vols : jax.Array
        Signed volumes, shape (num_tets,).
    """
    points = jnp.asarray(mesh.points)
    cells = jnp.asarray(mesh.cells)
    
    c = cells[:, :4]  # corners only — TET10-safe
    v0 = points[c[:, 0]]
    v1 = points[c[:, 1]]
    v2 = points[c[:, 2]]
    v3 = points[c[:, 3]]

    a = v1 - v0
    b = v2 - v0
    c_vec = v3 - v0

    # Signed volume = det([a,b,c]) / 6 via scalar triple product
    vols = jnp.einsum('ij,ij->i', a, jnp.cross(b, c_vec)) / 6.0

    n_neg = jnp.sum(vols < 0).astype(jnp.int32)
    n_zero = jnp.sum(vols == 0).astype(jnp.int32)

    # jax.debug.print is JIT-safe: deferred to runtime, not trace time
    jax.debug.print(
        "Volume check: min={mn:.3e}  max={mx:.3e}  "
        "negative={ng}  degenerate={nz}",
        mn=jnp.min(vols), mx=jnp.max(vols), ng=n_neg, nz=n_zero,
    )
    jax.debug.print(
        "  {status}",
        status=jnp.where(
            n_neg + n_zero == 0,
            jnp.int32(0),  # sentinel: 0 = all good
            jnp.int32(1),  # sentinel: 1 = problems
        )
    )

    return vols



def _rect_sweep_topology(M: int, N: int, O: int, mesh_type: str):
    """Build rectangle-sweep mesh topology in (phi, u, v) parametric space.

    Returns per-node parametric coordinates and connectivity for a
    fixed-topology mesh swept along a curve.  All output is pure
    ``numpy`` (compile-time static); the corresponding physical points
    are produced by :func:`_rect_sweep_points`.

    Parameters
    ----------
    M : int
        Number of phi slices (= ``framed_curve.curve.quadpoints.shape[0]``).
    N : int
        Number of cross-section grid points in direction 1.
    O : int
        Number of cross-section grid points in direction 2.
    mesh_type : str
        ``'TET4'`` or ``'TET10'``.

    Returns
    -------
    u_per_node : np.ndarray (num_nodes,)
        Parametric u-coordinate in ``[-1, 1]`` for each node.
    v_per_node : np.ndarray (num_nodes,)
        Parametric v-coordinate in ``[-1, 1]`` for each node.
    phi_idx : np.ndarray (num_nodes,) int32
        Index into the phi-grid of length ``K = M*stride``, where
        ``stride = 2`` for ``'TET10'`` (every odd index is a half-step
        midside) and ``1`` for ``'TET4'``.
    cells : np.ndarray (num_cells, k) int32
        Connectivity.  ``k = 4`` (TET4) or ``k = 10`` (TET10).
    """
    if mesh_type not in ('TET4', 'TET10'):
        raise ValueError(
            f"mesh_type must be 'TET4' or 'TET10', got {mesh_type!r}"
        )
    M = int(M); N = int(N); O = int(O)

    u_grid = np.linspace(-1.0, 1.0, N)                       # (N,)
    v_grid = np.linspace(-1.0, 1.0, O)                       # (O,)
    stride = 2 if mesh_type == 'TET10' else 1                # phi-grid stride

    # ── Corner nodes ──
    # Linear index nidx(m, n, o) = m * (N*O) + n * O + o.
    mm, nn, oo = np.meshgrid(
        np.arange(M), np.arange(N), np.arange(O), indexing='ij'
    )
    u_corners = u_grid[nn].ravel()                           # (M*N*O,)
    v_corners = v_grid[oo].ravel()
    phi_corners = (stride * mm).ravel().astype(np.int32)

    # ── Hex connectivity (one hex per (m, n, o), n<N-1, o<O-1) ──
    def nidx(m, n, o):
        return (m % M) * (N * O) + n * O + o

    mh, nh, oh = np.meshgrid(
        np.arange(M), np.arange(N - 1), np.arange(O - 1), indexing='ij'
    )
    mh = mh.ravel(); nh = nh.ravel(); oh = oh.ravel()
    hex_corners = np.stack([
        nidx(mh,     nh,     oh    ),
        nidx(mh + 1, nh,     oh    ),
        nidx(mh + 1, nh + 1, oh    ),
        nidx(mh,     nh + 1, oh    ),
        nidx(mh,     nh,     oh + 1),
        nidx(mh + 1, nh,     oh + 1),
        nidx(mh + 1, nh + 1, oh + 1),
        nidx(mh,     nh + 1, oh + 1),
    ], axis=1)

    if mesh_type == 'TET4':
        cells = hex_corners[:, _KUHN_6].reshape(-1, 4)
        return (
            u_corners, v_corners, phi_corners,
            cells.astype(np.int32),
        )

    # ── TET10: midside nodes by per-edge deduplication ──
    #
    # The corner tets are the Kuhn split of every hex (all six share the main
    # diagonal v0–v6), so the split is conforming: any interior triangular face
    # is shared by exactly two tets and every interior edge is shared by all the
    # tets around it.  We therefore create exactly ONE midside node per unique
    # tet edge — no per-family bookkeeping, and no coincident duplicates.
    #
    # For each of the 6 tet edges (VTK order) we form a canonical key from the
    # *global* (wrapped) corner-index pair and deduplicate.  Endpoint u/v come
    # straight from the corner arrays.  The phi index is the average of the two
    # endpoints' *unwrapped* phi levels (2*m at slice m, 2*m+2 at slice m+1),
    # taken mod 2M so the periodic seam (level 2M) wraps back to 0.  Edges that
    # traverse phi therefore land on an odd (half-step) index, giving curved
    # midsides in ``_rect_sweep_points``; in-slice edges stay on an even index.

    # Unwrapped phi level (in stride-2 units) of each of the 8 hex corners.
    # Back face (slice m): v0, v3, v4, v7.  Front face (slice m+1): v1, v2, v5, v6.
    back = (stride * mh).astype(np.int64)                    # (num_hex,)
    front = back + stride
    hex_philevel = np.stack(
        [back, front, front, back, back, front, front, back], axis=1
    )                                                        # (num_hex, 8)

    # Corner tets and their per-corner unwrapped phi levels (num_cells, 4).
    cells4 = hex_corners[:, _KUHN_6].reshape(-1, 4)
    cell_phi = hex_philevel[:, _KUHN_6].reshape(-1, 4)

    # All 6 VTK edges of every tet → global index pairs and phi-level pairs.
    edge_g = cells4[:, _TET10_VTK_EDGES].reshape(-1, 2)      # (num_cells*6, 2)
    edge_phi = cell_phi[:, _TET10_VTK_EDGES].reshape(-1, 2)  # (num_cells*6, 2)

    # Canonical (sorted) global key → one midside node per unique edge.
    key = np.sort(edge_g, axis=1)
    uniq_key, first_idx, inv = np.unique(
        key, axis=0, return_index=True, return_inverse=True
    )
    inv = inv.ravel()
    n_mid = uniq_key.shape[0]

    a = uniq_key[:, 0]
    b = uniq_key[:, 1]
    u_mid = 0.5 * (u_corners[a] + u_corners[b])
    v_mid = 0.5 * (v_corners[a] + v_corners[b])

    # Half-sum of the unwrapped phi levels, taken from a representative
    # occurrence of each unique edge; mod 2M folds the seam (2M → 0).
    edge_phi_half = (edge_phi[:, 0] + edge_phi[:, 1]) // 2    # (num_cells*6,)
    phi_mid = (edge_phi_half[first_idx] % (stride * M)).astype(np.int32)

    u_per_node = np.concatenate([u_corners, u_mid])
    v_per_node = np.concatenate([v_corners, v_mid])
    phi_idx = np.concatenate([phi_corners, phi_mid]).astype(np.int32)

    base = M * N * O
    mid_idx = (base + inv).reshape(-1, 6)                    # (num_cells, 6)
    cells_10 = np.concatenate([cells4, mid_idx], axis=1)     # (num_cells, 10)
    return u_per_node, v_per_node, phi_idx, cells_10.astype(np.int32)


@partial(jax.jit, static_argnames=('mesh_type', 'N', 'O'))
def _rect_sweep_points(
    framed_curve, w1, w2, N: int, O: int, *, mesh_type: str,
):
    r"""Curved-edge rectangle-sweep mesh points as a pure-JAX expression.

    For each node at parametric coordinates :math:`(\varphi_i, u, v)`, the
    physical position is

    .. math::

        \mathbf{x} = \boldsymbol{\gamma}(\varphi_i)
                    + (w_1/2)\,u\,\mathbf{p}(\varphi_i)
                    + (w_2/2)\,v\,\mathbf{q}(\varphi_i),

    where :math:`\boldsymbol{\gamma}` is the centerline and
    :math:`(\mathbf{p}, \mathbf{q})` is the rotated cross-section frame.
    Corner nodes use :math:`\varphi_i = m/M`; midside nodes on
    phi-traversing edges use the half-step :math:`\varphi_i = (m+\tfrac12)/M`,
    producing curved-sided TET10 elements.

    The frame is evaluated on a single uniform phi-grid of length
    ``K = stride*M`` via :meth:`FramedCurveJAX.rotated_frame_eval`
    (analytic for centroid frame, fresh-grid scan for RMF).

    Parameters
    ----------
    framed_curve : FramedCurveJAX
    w1, w2 : float
        Full widths of the rectangular cross-section.
    N, O : int (static under JIT)
        Number of cross-section grid points (= ``n_grid_1 + 1``,
        ``n_grid_2 + 1`` in :func:`rectangle_sweep`).
    mesh_type : str (static)
        ``'TET4'`` or ``'TET10'``.

    Returns
    -------
    points : jax.Array, shape (num_nodes, 3)
    """
    M = int(framed_curve.curve.quadpoints.shape[0])
    u_np, v_np, phi_idx_np, _ = _rect_sweep_topology(M, N, O, mesh_type)
    K = (2 * M) if mesh_type == 'TET10' else M

    phi_grid = jnp.linspace(0.0, 1.0, K, endpoint=False)     # (K,)
    r0 = framed_curve.curve.gamma_eval(phi_grid)             # (K, 3)
    _, p, q = framed_curve.rotated_frame_eval(phi_grid)      # each (K, 3)

    phi_idx = jnp.asarray(phi_idx_np)
    u = jnp.asarray(u_np, dtype=float)
    v = jnp.asarray(v_np, dtype=float)

    r0_n = r0[phi_idx]
    p_n  = p[phi_idx]
    q_n  = q[phi_idx]
    return (
        r0_n
        + (w1 / 2.0) * u[:, None] * p_n
        + (w2 / 2.0) * v[:, None] * q_n
    )


@partial(jax.jit, static_argnames=("mesh_type",))
def quad_sweep_points_to_mesh(
    gamma: jax.Array, quads: jax.Array, mesh_type: str = "TET4"
):
    """
    Periodic sweep of a fixed 2D quad mesh along axis 0 (phi).

    Parameters
    ----------
    gamma
        ``(M, n2d, 3)`` — same 2D node ordering at each phi slice.
    quads
        ``(Q, 4)`` int32, CCW quads in the cross-section index space.
    mesh_type
        ``'TET4'`` only (extruded-hex Freudenthal split).
    """
    if mesh_type != "TET4":
        raise NotImplementedError("quad_sweep_points_to_mesh supports TET4 only")
    M, n2d, _ = gamma.shape
    points = gamma.reshape(-1, 3)
    Q = quads.shape[0]

    def nidx(m, k):
        return (m % M) * n2d + k

    mm, qq = jnp.meshgrid(
        jnp.arange(M), jnp.arange(Q), indexing="ij"
    )
    mm = mm.ravel()
    qq = qq.ravel()
    q4 = quads[qq]
    # Match :func:`_rect_sweep_topology` brick order: v0–v3 at phi = m
    # span (q0,q1) and (q3,q2) with phi-edges v0–v1 and v3–v2; v4–v7 at phi = m+1.
    # Quad corners must be ordered CCW as (q0,q1,q2,q3) = tensor-product
    # (i,j),(i+1,j),(i+1,j+1),(i,j+1) from :func:`build_disk_o_grid_topology_np`.
    hex_corners = jnp.stack(
        [
            nidx(mm, q4[:, 0]),
            nidx(mm + 1, q4[:, 0]),
            nidx(mm + 1, q4[:, 1]),
            nidx(mm, q4[:, 1]),
            nidx(mm, q4[:, 3]),
            nidx(mm + 1, q4[:, 3]),
            nidx(mm + 1, q4[:, 2]),
            nidx(mm, q4[:, 2]),
        ],
        axis=1,
    )
    cells = hex_corners[:, _KUHN_6].reshape(-1, 4)
    return points, cells.astype(jnp.int32)


def disk_sweep(
    framed_curve,
    radius,
    *,
    n_center=None,
    n_radial=None,
    aspect_ratio=1.0,
    mesh_type: str = "TET4",
):
    """
    Circular cross-section via a 5-block structured O-grid in the (p, q) plane,
    swept periodically along the curve.

    Uses framed_curve.gamma() for centerline positions and framed_curve.rotated_frame()
    for the frame vectors (p, q) at quadrature points.

    .. note::
        ``CoilFEM``'s volumetric Lorentz-force pipeline assumes cells whose
        cross-sections are **perpendicular to the coil centerline** (this
        matches the LHA 2025 formula, which evaluates B at points
        ``r_c + (u a/2) p + (v b/2) q`` with ``(p, q)`` the cross-section
        frame).  If a non-aligned sweep is ever introduced, ``uv_quad`` must
        be computed from the actual ``(p, q)``-plane intersection of each
        cell, and the ``(phi, u, v)`` → quad-point lift in
        ``CoilFEM.__init__`` must change accordingly.

    Parameters
    ----------
    framed_curve
        Framed curve object. Can be:
        - simsopt FramedCurve (FramedCurveRMF, FramedCurveCentroid)
        - Pure JAX wrapper (FramedCurveRMFJAX, FramedCurveCentroidJAX)
        Typically use RMF frame to minimize twist for circular cross-sections.
    radius
        Disk radius.
    n_center
        O-grid center block resolution. If None, computed from aspect_ratio.
    n_radial
        O-grid radial resolution. If None, computed from aspect_ratio.
    aspect_ratio
        Target aspect ratio for mesh elements (default 1.0 for cubic elements).
        Used to compute n_center and n_radial if not provided.
    spline
        Frame interpolation method: "auto", "linear", or "cubic".
    mesh_type
        "TET4" only (currently).

    Returns
    -------
    CoilMesh
        Mesh with points and cells.
    """
    if mesh_type != "TET4":
        raise NotImplementedError("disk_sweep currently supports TET4 only")
    
    # Compute default grid sizes based on aspect ratio
    if n_center is None or n_radial is None:
        # Get characteristic length scale along the curve
        ds = framed_curve.curve.incremental_arclength()
        length_per_quadpoint = jnp.mean(ds)/ds.shape[0]
        
        # Target element size in cross-section based on aspect ratio
        target_size = length_per_quadpoint * aspect_ratio
        
        if n_center is None:
            # Center block spans roughly 2*radius/sqrt(2) in each direction
            center_span = 2.0 * radius / jnp.sqrt(2.0)
            n_center = int(jnp.ceil(center_span / target_size))
            n_center = max(2, n_center)  # Minimum of 2
        
        if n_radial is None:
            # Radial direction spans from edge of center to outer radius
            radial_span = radius * (1.0 - 1.0 / jnp.sqrt(2.0))
            n_radial = int(jnp.ceil(radial_span / target_size))
            n_radial = max(2, n_radial)  # Minimum of 2
    
    quads_np, oxy_np, _ = _build_disk_o_grid_topology_np(
        n_center + 1, n_radial + 1
    )
    quads = jnp.asarray(quads_np, dtype=jnp.int32)
    oxy = jnp.asarray(oxy_np, dtype=float)
    
    # Get centerline positions and frame at quadrature points
    r0 = framed_curve.gamma()  # shape (n_phi, 3)
    _, p, q = framed_curve.rotated_frame()  # shapes (n_phi, 3) each
    
    n_phi = r0.shape[0]
    rad = jnp.asarray(radius, dtype=float)
    
    # Broadcast to (n_phi, n2d, 3) for each cross-section point
    r0_expanded = r0[:, None, :]  # (n_phi, 1, 3)
    p_expanded = p[:, None, :]    # (n_phi, 1, 3)
    q_expanded = q[:, None, :]    # (n_phi, 1, 3)
    
    # Compute offsets: oxy has shape (n2d, 2) with normalized (p, q) coordinates
    off = rad * (
        oxy[None, :, 0:1] * p_expanded + oxy[None, :, 1:2] * q_expanded
    )  # (n_phi, n2d, 3)
    
    gamma = r0_expanded + off  # (n_phi, n2d, 3)
    pts, cells = quad_sweep_points_to_mesh(gamma, quads, mesh_type=mesh_type)
    return CoilMesh(pts, cells, ele_type=mesh_type)

def rectangle_sweep(
    framed_curve,
    w1, w2,
    *,
    n_grid_1=None,
    n_grid_2=None,
    aspect_ratio=1.0,
    mesh_type="TET4",
):
    """Tensor-product rectangle sweep → :class:`CoilMesh`.

    Sweeps a structured ``(n_grid_1 + 1) x (n_grid_2 + 1)`` cross-section
    grid along the framed curve's centerline. For ``'TET10'`` the
    midside nodes are placed at the **curved midpoint** of each edge
    (i.e. the image under the framed-curve map of the parametric
    midpoint), producing a true curve-sided isoparametric mesh.

    The same pure-JAX helper :func:`_rect_sweep_points` is used both
    here and inside ``coil_fem.CoilFEM._mesh_points_from_dofs``
    for the differentiable forward pass, so init-time and forward-pass
    meshes are bit-identical.

    .. note::
        ``CoilFEM``'s volumetric Lorentz-force pipeline assumes cells whose
        cross-sections are **perpendicular to the coil centerline** (this
        matches the LHA 2025 formula, which evaluates B at points
        ``r_c + (u a/2) p + (v b/2) q`` with ``(p, q)`` the cross-section
        frame).  If a non-aligned sweep is ever introduced, ``uv_quad`` must
        be computed from the actual ``(p, q)``-plane intersection of each
        cell, and the ``(phi, u, v)`` → quad-point lift in
        ``CoilFEM.__init__`` must change accordingly.

    Parameters
    ----------
    framed_curve : FramedCurveJAX
        Pure-JAX framed curve. Pass an RMF frame for circular-like
        cross-sections, a centroid frame otherwise.
    w1, w2 : float
        Full widths of rectangular cross-section.
    n_grid_1, n_grid_2 : int, optional
        Number of *cells* per cross-section direction.  The grid has
        ``n_grid_1 + 1`` and ``n_grid_2 + 1`` points respectively.
        If ``None``, the value is chosen from ``aspect_ratio`` and the
        mean phi-spacing.
    aspect_ratio : float
        Target cross-section element size relative to the average
        arclength per quadpoint (default 1.0 for roughly cubic elements).
    mesh_type : str
        ``'TET4'`` (straight) or ``'TET10'`` (curved isoparametric).

    Returns
    -------
    CoilMesh
    """
    if n_grid_1 is None or n_grid_2 is None:
        ds = framed_curve.curve.incremental_arclength()
        length_per_quadpoint = jnp.mean(ds) / ds.shape[0]
        target_size = length_per_quadpoint * aspect_ratio
        if n_grid_1 is None:
            n_grid_1 = max(2, int(jnp.ceil(w1 / target_size)))
        if n_grid_2 is None:
            n_grid_2 = max(2, int(jnp.ceil(w2 / target_size)))

    M = int(framed_curve.curve.quadpoints.shape[0])
    N = n_grid_1 + 1
    O = n_grid_2 + 1

    pts = _rect_sweep_points(framed_curve, w1, w2, N, O, mesh_type=mesh_type)
    _, _, _, cells = _rect_sweep_topology(M, N, O, mesh_type)
    return CoilMesh(pts, cells, ele_type=mesh_type)


class CoilMesh(JAXFEMMesh):
    """
    Tetrahedral mesh with quality metrics, inherits from JAX-FEM Mesh.
    
    This class extends ``jax_fem.generate_mesh.Mesh`` with coil-specific
    mesh generation methods (rectangle_sweep, disk_sweep) and quality metrics.

    Attributes
    ----------
    points : ndarray, (N, 3)
        Node coordinates.
    cells : ndarray, (n_tet, 4 or 10)
        Element connectivity (TET4 or TET10).
    ele_type : str
        Element type: ``'TET4'`` or ``'TET10'``.
    
    Notes
    -----
    Inherits from ``jax_fem.generate_mesh.Mesh``, which is already a registered
    JAX pytree. This means CoilMesh instances can be used with all JAX
    transformations (jit, grad, vmap, etc.) and directly with JAX-FEM solvers.
    """

    def __init__(self, points, cells, ele_type: str = "TET4"):
        """
        Initialize mesh.
        
        Parameters
        ----------
        points : array_like, (N, 3)
            Node coordinates.
        cells : array_like, (n_tet, 4 or 10)
            Element connectivity.
        ele_type : str
            Element type: 'TET4' or 'TET10'.
        """
        # Convert to JAX arrays for our methods, but store as numpy for JAX-FEM compatibility
        points_np = np.asarray(points, dtype=np.float64)
        cells_np = np.asarray(cells, dtype=np.int32)
        
        # Initialize parent class
        super().__init__(points_np, cells_np, ele_type)
        
        # Cache JAX arrays for our quality metrics
        self._points_jax = jnp.asarray(self.points)
        self._cells_jax = jnp.asarray(self.cells)
    
    @property
    def mesh_type(self):
        """Alias for ele_type for backward compatibility."""
        return self.ele_type

    @property
    def meshio_cell_type(self) -> str:
        """meshio cell-type string corresponding to this element type.

        JAX-FEM uses ``"TET4"`` / ``"TET10"`` etc., while meshio expects
        ``"tetra"`` / ``"tetra10"`` etc.  Use this property whenever
        constructing a :class:`meshio.Mesh` from a :class:`CoilMesh`::

            meshio.Mesh(
                points=np.asarray(mesh.points),
                cells=[(mesh.meshio_cell_type, np.asarray(mesh.cells))],
            ).write("out.vtu")
        """
        _MAP = {
            "TET4":  "tetra",
            "TET10": "tetra10",
            "HEX8":  "hexahedron",
            "HEX20": "hexahedron20",
        }
        try:
            return _MAP[self.ele_type]
        except KeyError:
            raise ValueError(
                f"No meshio cell-type mapping for ele_type={self.ele_type!r}. "
                f"Known: {list(_MAP)}"
            ) from None

    def to_vtu(self, path: str = "beam_mesh.vtu"):
        """
        Export mesh to VTU format for ParaView visualization.
        
        Parameters
        ----------
        path : str
            Output file path (should end in .vtu).
        """
        import meshio
        meshio_type = 'tetra' if self.ele_type == 'TET4' else 'tetra10'
        meshio.Mesh(points=self.points, cells=[(meshio_type, self.cells)]).write(path)

    def mesh_edge_length_sum(self, func=lambda x: jnp.sum(x**2)) -> jax.Array:
        """
        Sum of all tetrahedral edge lengths across the mesh.

        Interior edges shared between multiple tetrahedra are counted once per
        tet — this is intentional.  By the AM-GM inequality, minimising this sum
        drives element edges towards equal length (regular tets), making it a
        useful mesh-quality objective that is differentiable w.r.t. ``points``.

        Works for both TET4 and TET10; only the first 4 columns of ``cells``
        (corner nodes) are used.

        Parameters
        ----------
        func : callable, optional
            Function to apply to edge vectors (default: sum of squared norms).

        Returns
        -------
        scalar jax.Array
            Mesh quality metric.
        """
        points = self._points_jax
        cells = self._cells_jax
        corners = cells[:, :4]  # (N_tets, 4)
        
        # Gather edge endpoint indices
        edge_node_idx = corners[:, _TET_EDGE_PAIRS]  # (N_tets, 6, 2)
        
        # Gather XYZ coordinates
        edge_pts = points[edge_node_idx]  # (N_tets, 6, 2, 3)
        
        # Compute edge vectors
        edge_vecs = edge_pts[:, :, 1, :] - edge_pts[:, :, 0, :]  # (N_tets, 6, 3)
        
        return func(edge_vecs)

    def mesh_longest_edge_volume_ratio(self) -> jax.Array:
        r"""
        Maximum edge-length-to-volume ratio across all tetrahedra.

        For each tet with corners ``(v0, v1, v2, v3)``, let
        :math:`\mathbf{a} = \mathbf{v}_1 - \mathbf{v}_0`,
        :math:`\mathbf{b} = \mathbf{v}_2 - \mathbf{v}_0`,
        :math:`\mathbf{c} = \mathbf{v}_3 - \mathbf{v}_0`. The metric is

        .. math::

            \frac{6\,|\mathbf{a}|\,|\mathbf{b}|\,|\mathbf{c}|}
                 {|\det[\mathbf{a},\mathbf{b},\mathbf{c}]|}

        The same three edges define both the numerator and the denominator
        (via the scalar triple product), so no sorting is needed.

        Works for both TET4 and TET10; only the first 4 columns of ``cells``
        (corner nodes) are used.

        Returns
        -------
        jax.Array
            Scalar: maximum ratio across all elements.
        """
        points = self._points_jax
        cells = self._cells_jax
        corners = cells[:, :4]  # (N_tets, 4)

        v0 = points[corners[:, 0]]  # (N_tets, 3)
        a = points[corners[:, 1]] - v0
        b = points[corners[:, 2]] - v0
        c = points[corners[:, 3]] - v0

        prod3 = (jnp.linalg.norm(a, axis=-1) * 
                 jnp.linalg.norm(b, axis=-1) * 
                 jnp.linalg.norm(c, axis=-1))  # (N_tets,)
        vol = jnp.abs(jnp.einsum('ij,ij->i', a, jnp.cross(b, c))) / 6.0  # (N_tets,)

        return jnp.max(prod3 / vol)
