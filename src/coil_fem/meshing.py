"""Structured volume meshes for coil cross-sections.

Sweeps a rectangular (:class:`CoilMeshRectangle`) or disk
(:class:`CoilMeshDisk`) cross-section grid along a framed centerline curve
to produce a tetrahedral :class:`CoilMesh` (TET4 or TET10).  The
differentiable method :meth:`CoilMesh.mesh_points_from_dofs` regenerates node
positions from updated curve DOFs, enabling gradient flow through the mesh
geometry.  Use :func:`rectangle_sweep` or :func:`disk_sweep` for mesh
generation, or :meth:`CoilMesh.from_options` for dict-driven dispatch.
"""

import abc

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from jax_fem.generate_mesh import Mesh as JAXFEMMesh


# ============================================================================
# Compile-time constants
# ============================================================================
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


# ============================================================================
# Validation
# ============================================================================

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
    """Backward-compatible wrapper that builds a :class:`CoilMeshDisk`.

    The mesh-generation logic now lives in :class:`CoilMeshDisk.__init__`; this
    function is a thin shim so existing callers keep working.  See
    :class:`CoilMeshDisk` for the full parameter documentation.

    Returns
    -------
    CoilMeshDisk
    """
    return CoilMeshDisk(
        framed_curve, radius,
        n_center=n_center, n_radial=n_radial,
        aspect_ratio=aspect_ratio, mesh_type=mesh_type,
    )

def rectangle_sweep(
    framed_curve,
    w1, w2,
    *,
    n_grid_1=None,
    n_grid_2=None,
    aspect_ratio=1.0,
    mesh_type="TET4",
):
    """Backward-compatible wrapper that builds a :class:`CoilMeshRectangle`.

    The mesh-generation logic now lives in :class:`CoilMeshRectangle.__init__`;
    this function is a thin shim so existing callers keep working.  See
    :class:`CoilMeshRectangle` for the full parameter documentation.

    Returns
    -------
    CoilMeshRectangle
    """
    return CoilMeshRectangle(
        framed_curve, w1, w2,
        n_grid_1=n_grid_1, n_grid_2=n_grid_2,
        aspect_ratio=aspect_ratio, mesh_type=mesh_type,
    )


class CoilMesh(JAXFEMMesh, abc.ABC):
    """Abstract base for coil volume meshes with quality metrics.

    Concrete subclasses (:class:`CoilMeshRectangle`, :class:`CoilMeshDisk`) own
    their cross-section metadata and implement :meth:`mesh_points_from_dofs`, the
    differentiable regeneration of mesh node positions from curve DOFs.  Build
    one via :meth:`from_options` (dispatch on ``mesh_options['shape']``) or the
    subclass constructors directly.

    Attributes
    ----------
    points : ndarray, (N, 3)
        Node coordinates (initial geometry; the differentiable positions are
        produced separately by :meth:`mesh_points_from_dofs`).
    cells : ndarray, (n_tet, 4 or 10)
        Element connectivity (TET4 or TET10).
    ele_type : str
        Element type: ``'TET4'`` or ``'TET10'``.
    framed_curve : FramedCurveJAX
        Framed centerline used to generate the mesh; :meth:`mesh_points_from_dofs`
        rebuilds it from new DOFs via :meth:`FramedCurveJAX.with_dofs`.
    n_cross, n_phi, n_cells, cross_section_area, phi_cell_idx
        Static topology/geometry metadata.
    n_quads, phi_quad, uv_quad
        Reference-coordinate arrays populated by :meth:`attach_ref_coords` once
        the FEM problem (quadrature rule) is known.

    Notes
    -----
    ``CoilMesh`` is not a registered JAX pytree.  It is a mutable container
    that holds static topology data alongside a ``framed_curve`` with the
    initial DOFs; it is always accessed by closure and never passed as a JAX
    argument.  Only the traced ``dofs`` flowing through
    :meth:`mesh_points_from_dofs` participate in autodiff.
    """

    #: Cross-section family: ``'rect'`` or ``'disk'`` (set by concrete subclasses).
    shape: str | None = None

    # VTK / JAX-FEM TET10 midside edge order (columns 4-9 of each cell), used to
    # place midpoint reference coordinates in :meth:`attach_ref_coords`.
    _TET10_MID_EDGES = _TET10_VTK_EDGES

    def __init__(self, points, cells, ele_type: str = "TET4"):
        """Low-level initializer shared by subclasses.

        Parameters
        ----------
        points : array_like, (N, 3)
            Node coordinates.
        cells : array_like, (n_tet, 4 or 10)
            Element connectivity.
        ele_type : str
            Element type: 'TET4' or 'TET10'.
        """
        points_np = np.asarray(points, dtype=np.float64)
        cells_np = np.asarray(cells, dtype=np.int32)
        super().__init__(points_np, cells_np, ele_type)
        self._points_jax = jnp.asarray(self.points)
        self._cells_jax = jnp.asarray(self.cells)

    # ============================================================================
    # Shared metadata / construction helpers
    # ============================================================================

    def _set_metadata(self, framed_curve, cross_section_area, n_cross, phi_cell_idx):
        """Store the metadata common to every cross-section shape.

        Called by subclass constructors after ``super().__init__`` so that
        ``self.cells`` is already populated.  Phase-2 reference-coordinate
        fields (``n_quads``/``phi_quad``/``uv_quad``) are initialised to ``None``
        and filled later by :meth:`attach_ref_coords`.
        """
        self.framed_curve = framed_curve
        self.n_phi = int(framed_curve.curve.quadpoints.shape[0])
        self.n_cells = int(self.cells.shape[0])
        self.cross_section_area = float(cross_section_area)
        self.n_cross = int(n_cross)
        self.phi_cell_idx = np.asarray(phi_cell_idx, dtype=np.int32)
        # Phase-2 (set by attach_ref_coords once the FEM problem exists).
        self.n_quads = None
        self.phi_quad = None
        self.uv_quad = None

    @classmethod
    def from_options(cls, framed_curve, opt, mesh_type):
        """Dispatch ``mesh_options`` to the matching concrete subclass.

        Parameters
        ----------
        framed_curve : FramedCurveJAX
        opt : dict
            A single normalised ``mesh_options`` entry (must contain ``'shape'``).
        mesh_type : str
            ``'TET4'`` or ``'TET10'``.

        Returns
        -------
        CoilMeshRectangle or CoilMeshDisk
        """
        shape = opt['shape']
        if shape == 'rect':
            return CoilMeshRectangle(
                framed_curve, opt['w1'], opt['w2'],
                n_grid_1=opt.get('n_grid_1'),
                n_grid_2=opt.get('n_grid_2'),
                aspect_ratio=opt.get('aspect_ratio', 1.0),
                mesh_type=mesh_type,
            )
        elif shape == 'disk':
            return CoilMeshDisk(
                framed_curve, opt['radius'],
                n_center=opt.get('n_center'),
                n_radial=opt.get('n_radial'),
                aspect_ratio=opt.get('aspect_ratio', 1.0),
                mesh_type=mesh_type,
            )
        else:
            raise ValueError(
                f"mesh_options['shape'] must be 'rect' or 'disk', got {shape!r}."
            )

    @abc.abstractmethod
    def mesh_points_from_dofs(self, dofs_i):
        """Return ``(n_nodes, 3)`` mesh points as a differentiable JAX array.

        ``dofs_i`` is the only traced input; the frame is rebuilt from it via
        ``self.framed_curve.with_dofs(dofs_i)`` and swept using the stored static
        cross-section metadata, so init-time and forward-pass meshes are
        bit-identical.
        """
        raise NotImplementedError

    # ============================================================================
    # Reference-coordinate pre-computation (phase 2; needs the FEM problem)
    # ============================================================================

    def attach_ref_coords(self, prob) -> None:
        """Populate ``n_quads``/``phi_quad``/``uv_quad`` from a built problem.

        ``phi_quad[c, q]`` is the curve parameter phi at FEM quadrature point
        ``q`` of cell ``c`` (values exceed 1.0 for seam cells; ``interpax``
        handles this via ``period=1.0``).  ``uv_quad`` holds the cross-section
        ``(u, v)`` coordinates and is shape-specific (``None`` unless the
        subclass overrides :meth:`_compute_uv_quad`).

        Supports TET4 (4-node) and TET10 (10-node) elements; for TET10 the 6
        midpoint reference coordinates are averages of the corner pairs given by
        :data:`_TET10_MID_EDGES`.
        """
        fe = prob.fes[0]
        cells_np = np.asarray(fe.cells, dtype=np.int64)   # (n_cells, n_nodes)
        sv_np = np.asarray(fe.shape_vals)                  # (n_quads, n_nodes)
        n_cell_nodes = cells_np.shape[1]
        if n_cell_nodes not in (4, 10):
            raise ValueError(
                f"attach_ref_coords: only TET4 and TET10 meshes are supported; "
                f"found {n_cell_nodes} nodes per element."
            )

        self.n_quads = len(fe.quad_weights)
        is_tet10 = (n_cell_nodes == 10)
        corners_np = cells_np[:, :4]  # (n_cells, 4)

        phi_cell_idx_np = np.asarray(self.phi_cell_idx, dtype=np.int64)  # (n_cells,)
        n_cross = self.n_cross
        n_phi = self.n_phi

        # Per-corner-node phi integer from global node index; a node is "front"
        # (phi+1 side) if its phi-slice differs from the cell's phi-slice.
        phi_int = corners_np // n_cross          # (n_cells, 4)
        i_slice = phi_cell_idx_np[:, np.newaxis]  # (n_cells, 1)
        is_front = (phi_int != i_slice)          # (n_cells, 4)

        phi_corners = np.where(
            is_front,
            (i_slice + 1.0) / n_phi,
            i_slice / n_phi,
        ).astype(np.float64)   # (n_cells, 4)

        if is_tet10:
            e = self._TET10_MID_EDGES
            phi_mids = 0.5 * (phi_corners[:, e[:, 0]] +
                              phi_corners[:, e[:, 1]])   # (n_cells, 6)
            phi_ref_local = np.concatenate([phi_corners, phi_mids], axis=1)
        else:
            phi_ref_local = phi_corners

        phi_quad_np = np.einsum('qn, cn -> cq', sv_np, phi_ref_local)
        self.phi_quad = jnp.asarray(phi_quad_np)   # (n_cells, n_quads)

        self.uv_quad = self._compute_uv_quad(corners_np, sv_np, is_tet10)

    def _compute_uv_quad(self, corners_np, sv_np, is_tet10):
        """Cross-section ``(u, v)`` at quadrature points; ``None`` by default.

        Overridden by shapes (e.g. :class:`CoilMeshRectangle`) that carry a
        rectangular ``(u, v)`` parametrisation.
        """
        return None

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


# ============================================================================
# Concrete cross-section meshes
# ============================================================================

class CoilMeshRectangle(CoilMesh):
    """Rectangular cross-section swept along a framed curve.

    Sweeps a structured ``(n_grid_1 + 1) x (n_grid_2 + 1)`` cross-section grid
    along the framed curve's centerline.  For ``'TET10'`` the midside nodes are
    placed at the **curved midpoint** of each edge (the image under the
    framed-curve map of the parametric midpoint), producing a true curve-sided
    isoparametric mesh.  The same pure-JAX helper :func:`_rect_sweep_points` is
    used both here and in :meth:`mesh_points_from_dofs`, so init-time and
    forward-pass meshes are bit-identical.

    .. note::
        ``CoilFEM``'s volumetric Lorentz-force pipeline assumes cells whose
        cross-sections are **perpendicular to the coil centerline** (this matches
        the LHA 2025 formula, which evaluates B at points
        ``r_c + (u a/2) p + (v b/2) q`` with ``(p, q)`` the cross-section frame).

    Parameters
    ----------
    framed_curve : FramedCurveJAX
        Pure-JAX framed curve. Pass an RMF frame for circular-like cross-sections,
        a centroid frame otherwise.
    w1, w2 : float
        Full widths of the rectangular cross-section.
    n_grid_1, n_grid_2 : int, optional
        Number of *cells* per cross-section direction.  The node grid has
        ``n_grid_1 + 1`` and ``n_grid_2 + 1`` points respectively.  If ``None``
        the value is chosen from ``aspect_ratio`` and the mean phi-spacing.
    aspect_ratio : float
        Target cross-section element size relative to the average arclength per
        quadpoint (default 1.0 for roughly cubic elements).
    mesh_type : str
        ``'TET4'`` (straight) or ``'TET10'`` (curved isoparametric).
    """

    shape = 'rect'

    def __init__(
        self, framed_curve, w1, w2, *,
        n_grid_1=None, n_grid_2=None, aspect_ratio=1.0, mesh_type="TET4",
    ):
        if n_grid_1 is None or n_grid_2 is None:
            ds = framed_curve.curve.incremental_arclength()
            length_per_quadpoint = jnp.mean(ds) / ds.shape[0]
            target_size = length_per_quadpoint * aspect_ratio
            if n_grid_1 is None:
                n_grid_1 = max(1, int(jnp.round(w1 / target_size)))
            if n_grid_2 is None:
                n_grid_2 = max(1, int(jnp.round(w2 / target_size)))

        M = int(framed_curve.curve.quadpoints.shape[0])
        N = n_grid_1 + 1   # node counts per cross-section direction
        O = n_grid_2 + 1

        pts = _rect_sweep_points(framed_curve, w1, w2, N, O, mesh_type=mesh_type)
        _, _, _, cells = _rect_sweep_topology(M, N, O, mesh_type)
        super().__init__(pts, cells, ele_type=mesh_type)

        # Rectangle-specific metadata. n_grid_1/n_grid_2 store NODE counts (N, O).
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.n_grid_1 = int(N)
        self.n_grid_2 = int(O)

        n_per_phi = (N - 1) * (O - 1) * 6   # KUHN-6 tets per phi-slice
        phi_cell_idx = np.repeat(np.arange(M, dtype=np.int32), n_per_phi)
        self._set_metadata(
            framed_curve,
            cross_section_area=w1 * w2,
            n_cross=N * O,
            phi_cell_idx=phi_cell_idx,
        )

    def mesh_points_from_dofs(self, dofs_i):
        fc = self.framed_curve.with_dofs(dofs_i)
        return _rect_sweep_points(
            fc, self.w1, self.w2, self.n_grid_1, self.n_grid_2,
            mesh_type=self.ele_type,
        )

    def _compute_uv_quad(self, corners_np, sv_np, is_tet10):
        n_cross = self.n_cross
        n_g1 = self.n_grid_1
        n_g2 = self.n_grid_2

        # Cross-section node (j, k) index from global node index.
        node_j = (corners_np % n_cross) // n_g2   # (n_cells, 4)
        node_k = corners_np % n_g2

        u_corners = (2.0 * node_j / (n_g1 - 1) - 1.0).astype(np.float64)
        v_corners = (2.0 * node_k / (n_g2 - 1) - 1.0).astype(np.float64)

        if is_tet10:
            e = self._TET10_MID_EDGES
            u_mids = 0.5 * (u_corners[:, e[:, 0]] + u_corners[:, e[:, 1]])
            v_mids = 0.5 * (v_corners[:, e[:, 0]] + v_corners[:, e[:, 1]])
            u_ref = np.concatenate([u_corners, u_mids], axis=1)
            v_ref = np.concatenate([v_corners, v_mids], axis=1)
        else:
            u_ref = u_corners
            v_ref = v_corners

        uv_ref_local = np.stack([u_ref, v_ref], axis=-1)   # (n_cells, n_nodes, 2)
        uv_quad_np = np.einsum('qn, cnd -> cqd', sv_np, uv_ref_local)
        return jnp.asarray(uv_quad_np)   # (n_cells, n_quads, 2)


class CoilMeshDisk(CoilMesh):
    """Circular cross-section via a 5-block structured O-grid, swept along a curve.

    Uses ``framed_curve.gamma()`` for centerline positions and
    ``framed_curve.rotated_frame()`` for the cross-section frame ``(p, q)`` at
    quadrature points.  Currently ``'TET4'`` only.

    .. note::
        As with :class:`CoilMeshRectangle`, the Lorentz-force pipeline assumes
        cross-sections perpendicular to the centerline.

    Parameters
    ----------
    framed_curve : FramedCurveJAX
        Framed curve; typically an RMF frame to minimise twist.
    radius : float
        Disk radius.
    n_center : int, optional
        O-grid center-block resolution. If ``None``, from ``aspect_ratio``.
    n_radial : int, optional
        O-grid radial resolution. If ``None``, from ``aspect_ratio``.
    aspect_ratio : float
        Target element aspect ratio (default 1.0).
    mesh_type : str
        ``'TET4'`` only (currently).
    """

    shape = 'disk'

    def __init__(
        self, framed_curve, radius, *,
        n_center=None, n_radial=None, aspect_ratio=1.0, mesh_type="TET4",
    ):
        if mesh_type != "TET4":
            raise NotImplementedError("CoilMeshDisk currently supports TET4 only")

        if n_center is None or n_radial is None:
            ds = framed_curve.curve.incremental_arclength()
            length_per_quadpoint = jnp.mean(ds) / ds.shape[0]
            target_size = length_per_quadpoint * aspect_ratio
            if n_center is None:
                center_span = 2.0 * radius / jnp.sqrt(2.0)
                n_center = max(2, int(jnp.ceil(center_span / target_size)))
            if n_radial is None:
                radial_span = radius * (1.0 - 1.0 / jnp.sqrt(2.0))
                n_radial = max(2, int(jnp.ceil(radial_span / target_size)))

        quads_np, oxy_np, _ = _build_disk_o_grid_topology_np(
            n_center + 1, n_radial + 1
        )
        quads = jnp.asarray(quads_np, dtype=jnp.int32)
        self.oxy = jnp.asarray(oxy_np, dtype=float)   # (n2d, 2) normalized (p, q)
        self.radius = float(radius)

        M = int(framed_curve.curve.quadpoints.shape[0])
        n2d = int(oxy_np.shape[0])
        n_quads_2d = int(quads_np.shape[0])

        pts = self._points_from_framed_curve(framed_curve)          # (M*n2d, 3)
        gamma_3d = pts.reshape(M, n2d, 3)
        _, cells = quad_sweep_points_to_mesh(gamma_3d, quads, mesh_type=mesh_type)
        super().__init__(pts, cells, ele_type=mesh_type)

        self.n2d = n2d
        phi_cell_idx = np.repeat(np.arange(M, dtype=np.int32), n_quads_2d * 6)
        self._set_metadata(
            framed_curve,
            cross_section_area=np.pi * radius ** 2,
            n_cross=n2d,
            phi_cell_idx=phi_cell_idx,
        )

    def _points_from_framed_curve(self, fc):
        """Sweep the O-grid cross-section along *fc*; returns ``(n_phi*n2d, 3)``."""
        r0 = fc.gamma()                  # (n_phi, 3)
        _, p, q = fc.rotated_frame()     # (n_phi, 3) each
        off = self.radius * (
            self.oxy[None, :, 0:1] * p[:, None, :]
            + self.oxy[None, :, 1:2] * q[:, None, :]
        )                                # (n_phi, n2d, 3)
        gamma = r0[:, None, :] + off     # (n_phi, n2d, 3)
        return gamma.reshape(-1, 3)

    def mesh_points_from_dofs(self, dofs_i):
        fc = self.framed_curve.with_dofs(dofs_i)
        return self._points_from_framed_curve(fc)
