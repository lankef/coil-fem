"""
Tests for TET10 support in the B_self reference-coordinate pipeline.

Validates that ``FramedCurveMesh.attach_ref_coords`` correctly builds ``phi_quad`` and
``uv_quad`` for 10-node tetrahedral elements, and that the downstream
``B_self_quadrature`` call produces consistent results.

Strategy
--------
1. Build TET4 and TET10 meshes for the same circular coil via
   ``FramedCurveMeshRectangle``, then construct JAX-FEM ``LinearElasticity3D``
   problems to obtain ``shape_vals`` and ``cells``.
2. Call ``mesh.attach_ref_coords(prob)`` and read the reference-coordinate
   arrays off the mesh; check shapes, value ranges, and mutual consistency
   (TET10 phi_quad averages should approximate TET4 phi_quad averages since
   both discretise the same geometry).
3. Feed both sets of (phi_quad, uv_quad) into ``B_self_quadrature`` and
   verify the cell-averaged self-field agrees to a reasonable tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from coil_fem.geo import CurveXYZFourierJAX
from coil_fem.geo import make_rmf_frame
from coil_fem.meshing import FramedCurveMeshRectangle
from coil_fem.magnetic import B_self_quadrature
from coil_fem.pipelines import ElasticPipeline


def _make_circle(N=32, R=1.0):
    quadpoints = jnp.linspace(0.0, 1.0, N, endpoint=False)
    dofs = jnp.array([0.0, 0.0, R,
                      0.0, R,  0.0,
                      0.0, 0.0, 0.0])
    return CurveXYZFourierJAX(quadpoints, dofs, order=1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _build_prob_dict(mesh_type, N=32, R=1.0, w1=0.05, w2=0.03):
    """Build a FramedCurveMeshRectangle + LinearElasticity3D problem dict.

    Built through :class:`~coil_fem.pipelines.ElasticPipeline`, which already
    calls ``mesh.attach_ref_coords`` to populate
    ``mesh.phi_quad``/``mesh.uv_quad`` in place, mirroring what
    :class:`~coil_fem.CoilFEM` does at construction.
    """
    curve = _make_circle(N=N, R=R)
    fc = make_rmf_frame(curve)
    mesh = FramedCurveMeshRectangle(
        fc, w1, w2,
        n_grid_1=3, n_grid_2=3,
        mesh_type=mesh_type,
    )

    pipeline = ElasticPipeline(
        mesh, 200e9, 0.3, None, (0., 0., 0.), {'solver': 'umfpack'},
    )
    prob = pipeline.problem

    return {
        'mesh_type': mesh_type,
        'mesh': mesh,
        'prob': prob,
        'fc': fc,
        'curve': curve,
        'w1': w1, 'w2': w2,
        'n_phi': mesh.n_phi,
        'n_cross': mesh.n_cross,
        'n_g1': mesh.n_grid_1, 'n_g2': mesh.n_grid_2,
        'phi_cell_idx': mesh.phi_cell_idx,
        'n_cells': mesh.n_cells,
    }


@pytest.fixture(scope="module", params=["TET4", "TET10"])
def mesh_and_prob(request):
    """Build a FramedCurveMeshRectangle and LinearElasticity3D problem."""
    return _build_prob_dict(request.param)


def _ref_coords(d):
    """Return the reference-coordinate arrays computed by attach_ref_coords."""
    mesh = d['mesh']
    return mesh.phi_quad, mesh.uv_quad


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRefCoordsShapesAndRanges:
    """Verify phi_quad and uv_quad have correct shapes and value ranges."""

    def test_phi_quad_shape(self, mesh_and_prob):
        d = mesh_and_prob
        phi_q, _ = _ref_coords(d)
        n_cells = d['n_cells']
        n_quads = d['prob'].fes[0].shape_vals.shape[0]
        assert phi_q.shape == (n_cells, n_quads), (
            f"Expected ({n_cells}, {n_quads}), got {phi_q.shape}"
        )

    def test_uv_quad_shape(self, mesh_and_prob):
        d = mesh_and_prob
        _, uv_q = _ref_coords(d)
        n_cells = d['n_cells']
        n_quads = d['prob'].fes[0].shape_vals.shape[0]
        assert uv_q.shape == (n_cells, n_quads, 2), (
            f"Expected ({n_cells}, {n_quads}, 2), got {uv_q.shape}"
        )

    def test_phi_quad_range(self, mesh_and_prob):
        """phi_quad values should be in [0, 1+1/n_phi] (seam cells wrap)."""
        d = mesh_and_prob
        phi_q, _ = _ref_coords(d)
        n_phi = d['n_phi']
        assert float(jnp.min(phi_q)) >= -1e-14, (
            f"phi_quad has negative values: min={float(jnp.min(phi_q))}"
        )
        upper = 1.0 + 1.0 / n_phi + 1e-10
        assert float(jnp.max(phi_q)) <= upper, (
            f"phi_quad exceeds upper bound: max={float(jnp.max(phi_q))}, "
            f"limit={upper}"
        )

    def test_uv_quad_range(self, mesh_and_prob):
        """uv_quad values should be in [-1, 1]."""
        d = mesh_and_prob
        _, uv_q = _ref_coords(d)
        assert float(jnp.min(uv_q)) >= -1.0 - 1e-10, (
            f"uv_quad below -1: min={float(jnp.min(uv_q))}"
        )
        assert float(jnp.max(uv_q)) <= 1.0 + 1e-10, (
            f"uv_quad above +1: max={float(jnp.max(uv_q))}"
        )


class TestBSelfTET10Consistency:
    """B_self_quadrature should accept TET10-shaped arrays without error."""

    def test_b_self_runs_without_error(self, mesh_and_prob):
        """B_self_quadrature must not crash for either TET4 or TET10 inputs."""
        d = mesh_and_prob
        phi_q, uv_q = _ref_coords(d)
        I = 1e4
        cs = {'shape': 'rect', 'w1': d['w1'], 'w2': d['w2']}
        B = B_self_quadrature(d['fc'], I, cs, phi_q, uv_q)
        n_cells = d['n_cells']
        n_quads = d['prob'].fes[0].shape_vals.shape[0]
        assert B.shape == (n_cells, n_quads, 3)
        assert not jnp.any(jnp.isnan(B)), "B_self contains NaN"
        assert not jnp.any(jnp.isinf(B)), "B_self contains Inf"

    def test_b_self_nonzero(self, mesh_and_prob):
        """The self-field should be non-trivial (not all zeros)."""
        d = mesh_and_prob
        phi_q, uv_q = _ref_coords(d)
        I = 1e4
        cs = {'shape': 'rect', 'w1': d['w1'], 'w2': d['w2']}
        B = B_self_quadrature(d['fc'], I, cs, phi_q, uv_q)
        assert float(jnp.max(jnp.abs(B))) > 1e-10, (
            "B_self is essentially zero"
        )


class TestTET4TET10CellAveragedAgreement:
    """Cell-averaged B_self from TET4 and TET10 should approximately agree."""

    @pytest.fixture(scope="class")
    def both_results(self):
        """Compute B_self for both TET4 and TET10 on the same coil."""
        w1, w2 = 0.05, 0.03
        I = 1e4
        cs = {'shape': 'rect', 'w1': w1, 'w2': w2}

        results = {}
        for mesh_type in ('TET4', 'TET10'):
            d = _build_prob_dict(mesh_type, w1=w1, w2=w2)
            phi_q, uv_q = _ref_coords(d)
            B = B_self_quadrature(d['fc'], I, cs, phi_q, uv_q)
            B_cell_avg = jnp.mean(B, axis=1)

            results[mesh_type] = {
                'B_cell_avg': B_cell_avg,
                'phi_cell_idx': d['phi_cell_idx'],
                'n_phi': d['n_phi'],
            }

        return results

    def test_phi_slice_averaged_b_self_agrees(self, both_results):
        """Per-phi-slice average B_self should agree between TET4 and TET10.

        We average over all cells in each phi slice (same geometry, different
        quadrature) and check that the RMS relative difference is small.
        """
        r4 = both_results['TET4']
        r10 = both_results['TET10']
        n_phi = r4['n_phi']

        def _phi_avg(B_cell_avg, phi_idx):
            phi_idx_np = np.asarray(phi_idx)
            avgs = []
            for k in range(n_phi):
                mask = phi_idx_np == k
                avgs.append(jnp.mean(B_cell_avg[mask], axis=0))
            return jnp.stack(avgs)

        avg4 = _phi_avg(r4['B_cell_avg'], r4['phi_cell_idx'])
        avg10 = _phi_avg(r10['B_cell_avg'], r10['phi_cell_idx'])

        rel_diff = float(
            jnp.linalg.norm(avg4 - avg10) / jnp.linalg.norm(avg4)
        )
        assert rel_diff < 0.05, (
            f"TET4 vs TET10 phi-slice-averaged B_self relative difference "
            f"is {rel_diff:.4f}, expected < 0.05"
        )
