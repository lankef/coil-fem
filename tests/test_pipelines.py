"""Smoke tests for ElasticPipeline, FixedSupport, and the CoilFEM refactor.

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
from coil_fem.coupling import Support, FixedSupport
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

    # Uniform unit Winkler weights (surface fully supported)
    surf_n  = pipeline.surface_node_indices.shape[0]
    weights = jnp.ones(surf_n)

    result = pipeline.solve(points, body_force, weights)

    u = result['u']
    assert u.shape == (n_nodes, 3), f"Expected ({n_nodes}, 3), got {u.shape}"
    assert jnp.all(jnp.isfinite(u)), "Displacement contains non-finite values"
    assert result['problem'] is pipeline.problem


# ---------------------------------------------------------------------------
# 2. FixedSupport — always returns zero displacement
# ---------------------------------------------------------------------------

def test_fixed_support_is_not_coupled():
    """FixedSupport.is_coupled is False."""
    assert FixedSupport().is_coupled is False


def test_fixed_support_solve_returns_empty_dict():
    """FixedSupport.solve returns empty dict (no-op)."""
    state = FixedSupport().solve({'anything': 1})
    assert state == {}


def test_fixed_support_displacement_at_zeros():
    """FixedSupport.displacement_at returns zero array of the correct shape."""
    pts = jnp.ones((5, 3))
    u   = FixedSupport().displacement_at({}, pts)
    assert u.shape == (5, 3)
    assert jnp.all(u == 0.0)


def test_fixed_support_coo_raises():
    """FixedSupport.coo() raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        FixedSupport().coo()


# ---------------------------------------------------------------------------
# 3. CoilFEM refactor — behavior-preserving smoke test
# ---------------------------------------------------------------------------

def _constant_support_fn(surf_pts, curve_jax, dofs):
    """Uniform weight = 1 at all surface nodes."""
    return jnp.ones(surf_pts.shape[0])


def test_coil_fem_has_pipelines_and_support():
    """CoilFEM.__init__ populates self.pipelines and self.support."""
    curve = _make_circle(N=4, R=1.0)
    fem = CoilFEM(
        base_curves_jax=[curve],
        base_currents_jax=jnp.array([1e4]),
        base_support_fns=_constant_support_fn,
        base_support_dofs=None,
        nfp=1,
        stellsym=False,
        mesh_options={'shape': 'rect', 'w1': 0.01, 'w2': 0.01,
                      'n_grid_1': 1, 'n_grid_2': 1},
        problem_options={'winkler_k': 1e9},
        verbose=0,
    )

    assert len(fem.pipelines) == 1
    assert isinstance(fem.support, FixedSupport)
    # meshes property shim works
    assert len(fem.meshes) == 1
    assert fem.meshes[0] is fem.pipelines[0].mesh


def test_coil_fem_explicit_fixed_support():
    """Passing support=FixedSupport() explicitly is identical to the default."""
    curve = _make_circle(N=4, R=1.0)
    kwargs = dict(
        base_curves_jax=[curve],
        base_currents_jax=jnp.array([1e4]),
        base_support_fns=_constant_support_fn,
        base_support_dofs=None,
        nfp=1, stellsym=False,
        mesh_options={'shape': 'rect', 'w1': 0.01, 'w2': 0.01,
                      'n_grid_1': 1, 'n_grid_2': 1},
        problem_options={'winkler_k': 1e9},
        verbose=0,
    )
    fem_default  = CoilFEM(**kwargs)
    fem_explicit = CoilFEM(**kwargs, support=FixedSupport())
    # Both hold a FixedSupport
    assert isinstance(fem_default.support,  FixedSupport)
    assert isinstance(fem_explicit.support, FixedSupport)


def test_coil_fem_objective_finite():
    """CoilFEM.objective returns a finite max_von_mises after refactor."""
    curve = _make_circle(N=4, R=1.0)
    fem = CoilFEM(
        base_curves_jax=[curve],
        base_currents_jax=jnp.array([1e4]),
        base_support_fns=_constant_support_fn,
        base_support_dofs=None,
        nfp=1,
        stellsym=False,
        mesh_options={'shape': 'rect', 'w1': 0.01, 'w2': 0.01,
                      'n_grid_1': 1, 'n_grid_2': 1},
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
    pipeline = _make_tiny_pipeline.__wrapped__() if hasattr(
        _make_tiny_pipeline, '__wrapped__'
    ) else None

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
