"""
Tests for the curved-sided TET10 rectangle-sweep meshes produced by
``coil_fem.meshing._rect_sweep_topology`` and
``coil_fem.meshing._rect_sweep_points``.

Properties checked
------------------
1. **Curved-edge placement on a circle.**  For a circular coil with
   zero cross-section width, the midside *M*-family node at parametric
   ``phi = (m + 1/2)/M`` lies *on* the circle, i.e. coincides with
   ``gamma_eval((m + 0.5)/M)`` to machine precision -- not the chord
   midpoint between the adjacent corner nodes.
2. **Init / forward-pass parity.**  The init-time mesh from
   ``CoilMeshRectangle`` and the forward-pass mesh from
   ``CoilMeshRectangle.mesh_points_from_dofs`` (which calls the same
   ``_rect_sweep_points`` helper) produce identical points.
3. **Topology unchanged.**  The connectivity returned by the new
   ``_rect_sweep_topology`` matches the legacy node ordering enough that
   ``CoilMesh.attach_ref_coords`` continues to work
   (i.e. corners obey ``index = m * (N*O) + n * O + o``).
4. **AD through ``curve.dofs``.**  Gradients of a mesh-based scalar flow
   through the new pipeline.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from coil_fem.geo import CurveXYZFourierJAX
from coil_fem.geo import make_centroid_frame, make_rmf_frame
from coil_fem.meshing import (
    rectangle_sweep,
    CoilMeshRectangle,
    _rect_sweep_topology,
    _rect_sweep_points,
)


jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_circle(N=32, R=1.0):
    quadpoints = jnp.linspace(0.0, 1.0, N, endpoint=False)
    dofs = jnp.array([0.0, 0.0, R,
                      0.0, R,  0.0,
                      0.0, 0.0, 0.0])
    return CurveXYZFourierJAX(quadpoints, dofs, order=1)


# ---------------------------------------------------------------------------
# 1. Curved-edge placement
# ---------------------------------------------------------------------------

class TestCurvedMidsides:
    """For zero cross-section width, every midside node must lie on the
    centerline at its parametric half-step phi."""

    @pytest.mark.parametrize("N", [16, 32, 64])
    def test_M_midsides_on_circle(self, N):
        R = 1.0
        curve = _make_circle(N=N, R=R)
        # Use centroid frame: deterministic analytic frame, no RMF closure drift.
        fc = make_centroid_frame(curve)

        N_ = 2
        O_ = 2  # minimal cross-section grid
        pts = np.asarray(_rect_sweep_points(
            fc, 0.0, 0.0, N_, O_, mesh_type="TET10",
        ))
        u, v, phi_idx, _ = _rect_sweep_topology(N, N_, O_, "TET10")

        M = N
        n_corners = M * N_ * O_

        # Midside nodes are renumbered by per-edge dedup, so we select them by
        # ``phi_idx`` parity rather than a fixed block offset: phi-traversing
        # edges carry an *odd* half-step index (2m+1) → phi = (2m+1)/(2M).
        mid = np.arange(n_corners, u.shape[0])
        odd = mid[phi_idx[mid] % 2 == 1]
        assert odd.size > 0, "no phi-traversing midside nodes found"

        # With zero cross-section width every node collapses onto the
        # centerline; a curved midside therefore sits ON the circle (radius R),
        # whereas a straight chord midpoint would fall inside it.
        radii = np.linalg.norm(pts[odd, :2], axis=-1)
        assert np.allclose(radii, R, atol=1e-12), (
            f"phi-traversing midsides not on circle: radii={radii}"
        )

    def test_midside_not_chord_midpoint(self):
        """Sanity check that the new mesher actually moves midsides away
        from the chord midpoint (otherwise the test above is trivial)."""
        R = 1.0
        N = 16          # coarse grid → noticeable curvature
        curve = _make_circle(N=N, R=R)
        fc = make_centroid_frame(curve)
        N_, O_ = 2, 2
        pts = np.asarray(_rect_sweep_points(
            fc, 0.0, 0.0, N_, O_, mesh_type="TET10",
        ))
        u, v, phi_idx, _ = _rect_sweep_topology(N, N_, O_, "TET10")

        M = N
        n_corners = M * N_ * O_
        # A midside on the first half-step edge has phi_idx == 1 → phi = 1/(2M),
        # between corners at phi = 0 and phi = 1/M.
        cand = np.where(
            (np.arange(u.shape[0]) >= n_corners) & (phi_idx == 1)
        )[0]
        assert cand.size > 0, "no phi_idx==1 midside node found"
        midside = pts[cand[0]]
        gamma0 = np.asarray(curve.gamma_eval(jnp.asarray(0.0)))
        gamma1 = np.asarray(curve.gamma_eval(jnp.asarray(1.0 / M)))
        chord_mid = 0.5 * (gamma0 + gamma1)
        # Curved midside should be strictly farther from chord_mid than tol
        diff = np.linalg.norm(midside - chord_mid)
        # For R=1 and N=16, gap ≈ R*(1 - cos(pi/(2N))) ~ 0.005
        assert diff > 1e-4, (
            f"Midside at chord midpoint (diff={diff}); curve-sided behavior lost."
        )


# ---------------------------------------------------------------------------
# 2. Init / forward-pass parity
# ---------------------------------------------------------------------------

class TestInitForwardParity:
    """The init-time ``CoilMeshRectangle`` and the forward-pass
    ``CoilMeshRectangle.mesh_points_from_dofs`` both delegate to
    ``_rect_sweep_points``; the resulting points must be bit-identical.

    We drive ``mesh.mesh_points_from_dofs(dofs)`` directly rather than build a
    full ``CoilFEM`` so the test doesn't need a FEM problem / fixed_clamp_fn /
    boundary conditions.
    """

    @pytest.mark.parametrize("mesh_type", ["TET4", "TET10"])
    @pytest.mark.parametrize(
        "make_frame", [make_centroid_frame, make_rmf_frame]
    )
    def test_init_matches_forward_pass(self, mesh_type, make_frame):
        N = 32
        curve = _make_circle(N=N)
        fc = make_frame(curve)

        n_g1, n_g2 = 3, 3
        w1, w2 = 0.05, 0.03

        mesh = CoilMeshRectangle(
            fc, w1, w2,
            n_grid_1=n_g1, n_grid_2=n_g2,
            mesh_type=mesh_type,
        )

        # Forward-pass path: regenerate mesh points straight from the stored
        # DOFs — exactly what ``CoilFEM`` calls during the differentiable solve.
        dofs0 = curve.dofs
        pts_forward = np.asarray(mesh.mesh_points_from_dofs(dofs0))

        pts_init = np.asarray(mesh.points)
        assert pts_forward.shape == pts_init.shape
        assert np.allclose(pts_forward, pts_init, atol=1e-13)


# ---------------------------------------------------------------------------
# 3. Topology / corner-index convention
# ---------------------------------------------------------------------------

class TestTopologyConvention:
    """Corner-node indexing must follow ``m * (N*O) + n * O + o`` so the
    ``CoilMesh.attach_ref_coords`` reconstruction works unchanged."""

    @pytest.mark.parametrize("mesh_type", ["TET4", "TET10"])
    def test_corner_index_layout(self, mesh_type):
        M, N, O = 8, 4, 3
        u, v, phi_idx, cells = _rect_sweep_topology(M, N, O, mesh_type)
        u_grid = np.linspace(-1.0, 1.0, N)
        v_grid = np.linspace(-1.0, 1.0, O)
        stride = 2 if mesh_type == "TET10" else 1

        # First M*N*O nodes are corners with index = m*(N*O) + n*O + o.
        for m in range(M):
            for n in range(N):
                for o in range(O):
                    idx = m * (N * O) + n * O + o
                    assert phi_idx[idx] == stride * m, (
                        f"corner idx {idx}: phi_idx={phi_idx[idx]} "
                        f"expected {stride*m}"
                    )
                    assert u[idx] == pytest.approx(u_grid[n])
                    assert v[idx] == pytest.approx(v_grid[o])

    @pytest.mark.parametrize("mesh_type", ["TET4", "TET10"])
    def test_cell_corner_indices_in_range(self, mesh_type):
        M, N, O = 8, 4, 3
        u, v, phi_idx, cells = _rect_sweep_topology(M, N, O, mesh_type)
        num_nodes = u.shape[0]
        assert cells.min() >= 0
        assert cells.max() < num_nodes

    def test_tet10_midside_count(self):
        # Midsides are now one node per *unique tet edge* (per-edge dedup),
        # so derive the expected count directly from the returned corner
        # connectivity rather than the legacy 8-family formula.
        from coil_fem.meshing import _TET10_VTK_EDGES

        M, N, O = 8, 4, 3
        u, _, _, cells = _rect_sweep_topology(M, N, O, "TET10")
        n_corners = M * N * O
        edges = np.sort(
            cells[:, :4][:, _TET10_VTK_EDGES].reshape(-1, 2), axis=1
        )
        n_unique_edges = np.unique(edges, axis=0).shape[0]
        assert u.shape[0] == n_corners + n_unique_edges

    def test_tet10_unique_midside_positions(self):
        """Every midside index in cells should be distinct -- the new
        topology shouldn't accidentally duplicate midpoint slots."""
        M, N, O = 4, 3, 3
        _, _, _, cells = _rect_sweep_topology(M, N, O, "TET10")
        # cells[:, 4:] holds the 6 midside indices for every tet.
        # We just check that there are no out-of-range indices and that
        # the per-row indices are unique (no two edges share a midside in
        # a single tet).
        for row in cells:
            mids = row[4:]
            assert len(set(mids.tolist())) == 6, (
                f"Duplicate midside in row {row}"
            )


# ---------------------------------------------------------------------------
# 3b. Mesh conformity (Kuhn split regression)
# ---------------------------------------------------------------------------

def _signed_volumes(points, cells):
    """Signed volume of the 4 corner nodes of every tet."""
    c = np.asarray(cells)[:, :4]
    p = np.asarray(points)[c]                        # (n_tet, 4, 3)
    a = p[:, 1] - p[:, 0]
    b = p[:, 2] - p[:, 0]
    d = p[:, 3] - p[:, 0]
    return np.einsum("ij,ij->i", np.cross(a, b), d) / 6.0


class TestConformity:
    """Regression tests for the conforming Kuhn split.

    A non-conforming split (crossing face diagonals) drops coincident but
    topologically independent midside nodes at shared faces and lets adjacent
    hexes triangulate an interface two different ways.  These tests would fail
    on the old ``_FREUDENTHAL_6`` dissection.
    """

    @pytest.mark.parametrize("mesh_type", ["TET4", "TET10"])
    @pytest.mark.parametrize(
        "make_frame", [make_centroid_frame, make_rmf_frame]
    )
    def test_no_coincident_nodes(self, mesh_type, make_frame):
        """No two distinct node coordinates may coincide: a conforming mesh
        has a 1:1 node set with no unmerged duplicates."""
        from scipy.spatial import cKDTree

        N = 24
        curve = _make_circle(N=N)
        fc = make_frame(curve)
        mesh = rectangle_sweep(
            fc, 0.05, 0.03, n_grid_1=3, n_grid_2=3, mesh_type=mesh_type,
        )
        pts = np.asarray(mesh.points)
        pairs = cKDTree(pts).query_pairs(r=1e-10)
        assert not pairs, (
            f"{len(pairs)} coincident node pair(s) in {mesh_type} mesh "
            f"(frame={make_frame.__name__}); split is non-conforming."
        )

    @pytest.mark.parametrize("mesh_type", ["TET4", "TET10"])
    def test_interior_faces_shared_by_two(self, mesh_type):
        """Every triangular corner-face is on the boundary (1 tet) or shared
        by exactly 2 tets — never more, and no crossing/dangling faces."""
        M, N, O = 6, 4, 3
        _, _, _, cells = _rect_sweep_topology(M, N, O, mesh_type)
        corners = cells[:, :4]
        local_faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        faces = np.sort(corners[:, local_faces].reshape(-1, 3), axis=1)
        _, counts = np.unique(faces, axis=0, return_counts=True)
        assert set(counts.tolist()) <= {1, 2}, (
            f"face shared by {counts.max()} tets — mesh is non-conforming"
        )

    @pytest.mark.parametrize("mesh_type", ["TET4", "TET10"])
    @pytest.mark.parametrize(
        "make_frame", [make_centroid_frame, make_rmf_frame]
    )
    def test_consistent_positive_volumes(self, mesh_type, make_frame):
        """All corner tets share one orientation and are non-degenerate,
        guarding the Kuhn per-tet vertex ordering."""
        N = 16
        curve = _make_circle(N=N)
        fc = make_frame(curve)
        mesh = rectangle_sweep(
            fc, 0.05, 0.03, n_grid_1=3, n_grid_2=3, mesh_type=mesh_type,
        )
        vols = _signed_volumes(mesh.points, mesh.cells)
        assert np.all(np.abs(vols) > 0), "degenerate (zero-volume) tet present"
        assert np.all(vols > 0) or np.all(vols < 0), (
            "mixed tet orientations — inverted elements from vertex ordering"
        )


# ---------------------------------------------------------------------------
# 4. AD-through-dofs sanity
# ---------------------------------------------------------------------------

class TestAutoDiff:
    """Gradients of a scalar over the curved-TET10 mesh w.r.t. the curve
    DOFs must be finite and non-trivial."""

    def test_grad_centroid_perfect_circle(self):
        """Centroid frame is analytic and AD-clean on a perfect circle."""
        N = 32
        quadpoints = jnp.linspace(0.0, 1.0, N, endpoint=False)

        def loss(dofs):
            curve = CurveXYZFourierJAX(quadpoints, dofs, order=1)
            fc = make_centroid_frame(curve)
            pts = _rect_sweep_points(
                fc, 0.05, 0.03, 3, 3, mesh_type="TET10",
            )
            return jnp.sum(pts ** 2)

        dofs0 = jnp.array([0.0, 0.0, 1.0,
                           0.0, 1.0, 0.0,
                           0.0, 0.0, 0.0])
        g = jax.grad(loss)(dofs0)
        assert g.shape == dofs0.shape
        assert jnp.all(jnp.isfinite(g))
        assert jnp.linalg.norm(g) > 0.0

    def test_grad_rmf_perfect_circle(self):
        """Regression test for the planar-RMF gradient.

        For a planar circle the RMF closes exactly so
        ``cross(n_final, n0) = 0``; without the ``_safe_norm`` guard in
        ``_rmf_normals_pure_jax`` this triggers ``d(||x||)/dx = x/||x||
        = 0/0 = NaN`` in the periodic-closure step.  The safe-norm fix
        injects a 1e-30 squared offset so the gradient is the
        mathematical limit ``0`` instead of NaN.
        """
        N = 32
        quadpoints = jnp.linspace(0.0, 1.0, N, endpoint=False)

        def loss(dofs):
            curve = CurveXYZFourierJAX(quadpoints, dofs, order=1)
            fc = make_rmf_frame(curve)
            pts = _rect_sweep_points(
                fc, 0.05, 0.03, 3, 3, mesh_type="TET10",
            )
            return jnp.sum(pts ** 2)

        dofs0 = jnp.array([0.0, 0.0, 1.0,
                           0.0, 1.0, 0.0,
                           0.0, 0.0, 0.0])
        g = jax.grad(loss)(dofs0)
        assert g.shape == dofs0.shape
        assert jnp.all(jnp.isfinite(g)), (
            "RMF gradient hit NaN on a perfect circle; "
            "_safe_norm guard in _rmf_normals_pure_jax may have regressed."
        )
        assert jnp.linalg.norm(g) > 0.0

    def test_grad_rmf_perturbed_circle(self):
        """RMF on a slightly non-planar curve (well away from the
        symmetric singularity)."""
        N = 32
        quadpoints = jnp.linspace(0.0, 1.0, N, endpoint=False)

        def loss(dofs):
            curve = CurveXYZFourierJAX(quadpoints, dofs, order=2)
            fc = make_rmf_frame(curve)
            pts = _rect_sweep_points(
                fc, 0.05, 0.03, 3, 3, mesh_type="TET10",
            )
            return jnp.sum(pts ** 2)

        dofs0 = jnp.array([
            0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.05, 0.0,
        ])
        g = jax.grad(loss)(dofs0)
        assert g.shape == dofs0.shape
        assert jnp.all(jnp.isfinite(g))
        assert jnp.linalg.norm(g) > 0.0
