"""Smoke tests for ElasticPipeline, Support, and the CoilFEM refactor.

Keeps the suite fast by using tiny meshes (4 phi × 1×1 cross-section).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from coil_fem.geo import CurveXYZFourierJAX, make_framed_curve
from coil_fem.meshing import CoilMesh
from coil_fem.pipelines import ElasticPipeline, ThermoElasticPipeline
from coil_fem.coupling import Support
from coil_fem.coil_fem import CoilFEM


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_circle(N: int = 4, R: float = 1.0) -> CurveXYZFourierJAX:
    """Planar circle of radius R in the xz-plane (order-1 Fourier curve)."""
    quadpoints = jnp.linspace(0.0, 1.0, N, endpoint=False)
    # order=1: [c0_x, c1_x, s1_x, c0_y, c1_y, s1_y, c0_z, c1_z, s1_z]
    # Circle in xz: x = R cos(2π φ), z = R sin(2π φ), y = 0
    dofs = jnp.array([0.0, R, 0.0,
                      0.0, 0.0, 0.0,
                      0.0, 0.0, R])
    return CurveXYZFourierJAX(quadpoints, dofs, order=1)


def _make_tiny_pipeline(R: float = 1.0) -> ElasticPipeline:
    """Build the smallest possible ElasticPipeline (4 phi × 1×1 rect mesh)."""
    curve = _make_circle(N=4, R=R)
    fc    = make_framed_curve(curve, 'rmf')
    mesh  = CoilMesh.from_options(
        fc,
        {'shape': 'rect', 'w1': 0.01, 'w2': 0.01, 'n_grid_1': 1, 'n_grid_2': 1},
        'TET4',
    )
    return ElasticPipeline(
        mesh,
        E=200e9, nu=0.3, itc=None,
        gravity_bf=(0.0, 0.0, 0.0),
        winkler_k=1e9,
        problem_options={'winkler_k': 1e9, 'solver': 'umfpack'},
    )


# ---------------------------------------------------------------------------
# 1. ElasticPipeline — uniform material forward solve
# ---------------------------------------------------------------------------

def test_elastic_pipeline_uniform_material():
    """ElasticPipeline.solve returns finite displacement of correct shape."""
    pipeline = _make_tiny_pipeline()

    mesh   = pipeline.mesh
    points = jnp.asarray(mesh.points)          # (n_nodes, 3)
    n_nodes  = points.shape[0]
    n_cells  = mesh.n_cells
    n_quads  = mesh.n_quads

    # Zero body force (no Lorentz load in this smoke test)
    body_force = jnp.zeros((n_cells, n_quads, 3))

    # Uniform unit Winkler weights (surface fully supported), at quad points
    n_sq    = pipeline.n_surface_quads
    weights = jnp.ones(n_sq)

    result = pipeline.solve(points, body_force, weights)

    u = result['u']
    assert u.shape == (n_nodes, 3), f"Expected ({n_nodes}, 3), got {u.shape}"
    assert jnp.all(jnp.isfinite(u)), "Displacement contains non-finite values"
    assert result['problem'] is pipeline.problem


# ---------------------------------------------------------------------------
# 2. Support (grounded) — always returns zero displacement
# ---------------------------------------------------------------------------

def test_support_is_not_coupled():
    """Support.is_coupled is False."""
    assert Support().is_coupled is False


def test_support_solve_returns_empty_dict():
    """Support.solve returns empty dict (no-op)."""
    state = Support().solve({'anything': 1})
    assert state == {}


def test_support_coo_raises():
    """Support.coo() raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        Support().coo()


def test_support_compute_weights_uniform():
    """Support with no fixed_clamp_fns returns all-ones weights."""
    support = Support()
    pts = jnp.ones((7, 3))
    w = support.compute_weights(0, pts, None, None)
    assert w.shape == (7,)
    assert jnp.all(w == 1.0)


def test_support_compute_weights_custom_fn():
    """Support with a custom fixed_clamp_fn calls it correctly."""
    def half_fn(surf_pts, curve_jax, dofs):
        return jnp.full(surf_pts.shape[0], 0.5)

    support = Support(fixed_clamp_fns=half_fn)
    pts = jnp.ones((6, 3))
    w = support.compute_weights(0, pts, None, None)
    assert w.shape == (6,)
    assert jnp.allclose(w, 0.5)


# ---------------------------------------------------------------------------
# 3. CoilFEM refactor — behavior-preserving smoke test
# ---------------------------------------------------------------------------

def _constant_fixed_clamp_fn(surf_pts, curve_jax, dofs):
    """Uniform weight = 1 at all surface nodes."""
    return jnp.ones(surf_pts.shape[0])


def test_coil_fem_has_pipelines_and_support():
    """CoilFEM.__init__ populates self.pipelines and self.support."""
    curve = _make_circle(N=4, R=1.0)
    support = Support(_constant_fixed_clamp_fn)
    fem = CoilFEM(
        base_curves_jax=[curve],
        base_currents_jax=jnp.array([1e4]),
        nfp=1,
        stellsym=False,
        mesh_options={'shape': 'rect', 'w1': 0.01, 'w2': 0.01,
                      'n_grid_1': 1, 'n_grid_2': 1},
        support=support,
        problem_options={'winkler_k': 1e9},
        verbose=0,
    )

    assert len(fem.pipelines) == 1
    assert isinstance(fem.support, Support)
    # meshes property shim works
    assert len(fem.meshes) == 1
    assert fem.meshes[0] is fem.pipelines[0].mesh


def test_coil_fem_default_support():
    """CoilFEM with support=None installs a Support with uniform weights."""
    curve = _make_circle(N=4, R=1.0)
    fem = CoilFEM(
        base_curves_jax=[curve],
        base_currents_jax=jnp.array([1e4]),
        nfp=1, stellsym=False,
        mesh_options={'shape': 'rect', 'w1': 0.01, 'w2': 0.01,
                      'n_grid_1': 1, 'n_grid_2': 1},
        problem_options={'winkler_k': 1e9},
        verbose=0,
    )
    assert isinstance(fem.support, Support)


def test_coil_fem_objective_finite():
    """CoilFEM.objective returns a finite max_von_mises after refactor."""
    curve = _make_circle(N=4, R=1.0)
    support = Support(_constant_fixed_clamp_fn)
    fem = CoilFEM(
        base_curves_jax=[curve],
        base_currents_jax=jnp.array([1e4]),
        nfp=1,
        stellsym=False,
        mesh_options={'shape': 'rect', 'w1': 0.01, 'w2': 0.01,
                      'n_grid_1': 1, 'n_grid_2': 1},
        support=support,
        problem_options={'winkler_k': 1e9},
        verbose=0,
    )

    dofs    = [curve.dofs]
    currents = jnp.array([1e4])
    result  = fem.objective(dofs, currents, metrics=('max_von_mises',))

    vm = result['max_von_mises']
    assert jnp.isfinite(vm), f"max_von_mises is not finite: {vm}"
    assert float(vm) >= 0.0, f"max_von_mises is negative: {vm}"


# ---------------------------------------------------------------------------
# 4. ThermoElasticPipeline stub raises NotImplementedError on solve
# ---------------------------------------------------------------------------

def test_thermo_elastic_pipeline_solve_raises():
    """ThermoElasticPipeline.solve raises NotImplementedError (stub)."""
    curve = _make_circle(N=4, R=1.0)
    fc    = make_framed_curve(curve, 'rmf')
    mesh  = CoilMesh.from_options(
        fc,
        {'shape': 'rect', 'w1': 0.01, 'w2': 0.01, 'n_grid_1': 1, 'n_grid_2': 1},
        'TET4',
    )
    stub = ThermoElasticPipeline(
        mesh, E=200e9, nu=0.3, itc=None,
        gravity_bf=(0.0, 0.0, 0.0),
        winkler_k=1e9,
        problem_options={'winkler_k': 1e9, 'solver': 'umfpack'},
    )
    points     = jnp.asarray(mesh.points)
    body_force = jnp.zeros((mesh.n_cells, mesh.n_quads, 3))

    with pytest.raises(NotImplementedError, match="not yet implemented"):
        stub.solve(points, body_force)


# ---------------------------------------------------------------------------
# 5. solve_uncoupled driver
# ---------------------------------------------------------------------------

def test_solve_uncoupled_matches_inline():
    """solve_uncoupled returns same sol_list_by_coil as direct pipeline.solve calls."""
    from coil_fem.coupling import solve_uncoupled

    p1 = _make_tiny_pipeline(R=1.0)
    p2 = _make_tiny_pipeline(R=1.5)
    support = Support()

    fe1, fe2 = p1.problem.fes[0], p2.problem.fes[0]
    pts1 = jnp.asarray(fe1.points)
    pts2 = jnp.asarray(fe2.points)
    bf1 = jnp.zeros((fe1.num_cells, fe1.num_quads, 3))
    bf2 = jnp.zeros((fe2.num_cells, fe2.num_quads, 3))

    wt1 = jnp.ones(p1.n_surface_quads)
    wt2 = jnp.ones(p2.n_surface_quads)

    params = {
        'mesh_points_by_coil': [pts1, pts2],
        'body_force_by_coil':  [bf1, bf2],
        'weights_by_coil':     [wt1, wt2],
        'curves_by_coil':      [],  # not used by solve_uncoupled
        'support_dofs':        {},
    }
    result = solve_uncoupled([p1, p2], support, params)

    # Check keys
    assert set(result.keys()) == {'sol_list_by_coil', 'u_s', 'diagnostics'}
    assert result['u_s'] is None
    assert result['diagnostics'] == {}
    assert len(result['sol_list_by_coil']) == 2

    # Verify same result as direct calls
    sol1_direct = p1.solve(pts1, bf1, wt1)['sol_list']
    sol2_direct = p2.solve(pts2, bf2, wt2)['sol_list']
    assert jnp.allclose(result['sol_list_by_coil'][0][0], sol1_direct[0], atol=1e-12)
    assert jnp.allclose(result['sol_list_by_coil'][1][0], sol2_direct[0], atol=1e-12)
