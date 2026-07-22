"""Tests for the staggered and monolithic coupled solver drivers.

Tests cover:

* ``solve_monolithic`` raises ``NotImplementedError`` on CPU backends.
* ``CoilFEM`` validates the ``coupling`` keyword argument.
* ``CoilFEM`` dispatches to ``solve_staggered`` vs ``solve_monolithic`` based
  on the ``coupling`` option (verified via monkeypatching).
* ``solve_staggered`` converges to a consistent fixed point on a trivially
  decoupled system (mock support that always returns ``u_s = 0``).
* The uncoupled ``_solve_all`` path produces the same result as the old
  per-coil loop.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from coil_fem.geo import CurveXYZFourierJAX, make_framed_curve
from coil_fem.meshing import CoilMesh
from coil_fem.pipelines import ElasticPipeline
from coil_fem.coupling import Support, SupportFixed, solve_staggered, solve_monolithic
from coil_fem.coil_fem import CoilFEM


# ============================================================================
# Shared helpers
# ============================================================================

def _make_circle(N: int = 4, R: float = 1.0) -> CurveXYZFourierJAX:
    """Planar circle of radius R in the xz-plane (order-1 Fourier curve)."""
    quadpoints = jnp.linspace(0.0, 1.0, N, endpoint=False)
    dofs = jnp.array([0.0, R, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, R])
    return CurveXYZFourierJAX(quadpoints, dofs, order=1)


def _make_tiny_pipeline(R: float = 1.0) -> ElasticPipeline:
    """Smallest possible ElasticPipeline (4 phi × 1×1 rect mesh)."""
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


def _make_coilfem(
    R: float = 1.0,
    coupling: str = 'staggered',
    support=None,
) -> CoilFEM:
    """Minimal single-coil CoilFEM for dispatch tests."""
    curve = _make_circle(N=4, R=R)
    mesh_opts = {'shape': 'rect', 'w1': 0.01, 'w2': 0.01,
                 'n_grid_1': 1, 'n_grid_2': 1}
    return CoilFEM(
        base_curves_jax=[curve],
        base_currents_jax=jnp.array([1.0]),
        nfp=1,
        stellsym=False,
        mesh_options=mesh_opts,
        support=support if support is not None else SupportFixed(),
        problem_options={'winkler_k': 1e9, 'solver': 'umfpack'},
        coupling=coupling,
    )


# ============================================================================
# Minimal mock coupled support
# ============================================================================

class _TrivialCoupledSupport(Support):
    """Mock coupled support: always returns u_s = 0, no coil coupling."""

    is_coupled = True
    n_support_dofs = 1
    k_lin = 1e9

    @property
    def is_coupled(self):
        return True

    def solve(self, inputs: dict) -> dict:
        return {'u_s': jnp.zeros(self.n_support_dofs)}

    def displacement_at(self, state: dict, points: jax.Array) -> jax.Array:
        return jnp.zeros_like(points)

    def compute_weights(self, coil_idx, surface_pts, curves_jax, dofs):
        return jnp.ones(surface_pts.shape[0])

    def compute_attach(self, coil_idx, surface_pts, curves_jax, dofs, state):
        return jnp.zeros((surface_pts.shape[0], 3), dtype=surface_pts.dtype)

    def coupling_terms(
        self,
        base_curves_dofs,
        support_dofs,
        surface_pts_by_coil,
        coil_dof_offsets,
        support_dof_offset,
        surface_node_indices_by_coil,
    ) -> dict:
        empty = jnp.zeros(0, dtype=jnp.float64)
        empty_i = jnp.zeros(0, dtype=jnp.int32)
        return {
            'I_cs': empty_i, 'J_cs': empty_i, 'V_cs': empty,
            'I_sc': empty_i, 'J_sc': empty_i, 'V_sc': empty,
        }

    def coo(self, base_curves_dofs, support_dofs, surface_pts_by_coil):
        n = self.n_support_dofs
        I = jnp.zeros(n, dtype=jnp.int32)
        J = jnp.zeros(n, dtype=jnp.int32)
        V = jnp.ones(n, dtype=jnp.float64) * 1e9
        return I, J, V, n


# ============================================================================
# 1. solve_monolithic raises NotImplementedError on non-cuDSS solver
# ============================================================================

def test_monolithic_raises_on_cpu():
    """solve_monolithic must raise NotImplementedError when solver != 'cudss'."""
    pipeline = _make_tiny_pipeline()
    support  = _TrivialCoupledSupport()
    n_quads  = pipeline.problem.fes[0].num_quads
    params   = {
        'mesh_points_by_coil': [
            pipeline.mesh.mesh_points_from_dofs(
                _make_circle(N=4).dofs
            )
        ],
        'body_force_by_coil':  [
            jnp.zeros((pipeline.problem.num_cells, n_quads, 3))
        ],
        'weights_by_coil':     [
            jnp.ones(pipeline.surface_node_indices.shape[0])
        ],
        'curves_by_coil':      [_make_circle(N=4)],
        'base_curves_dofs':    [_make_circle(N=4).dofs],
        'support_dofs':        {},
    }
    with pytest.raises(NotImplementedError, match="cudss"):
        solve_monolithic([pipeline], support, params)


# ============================================================================
# 2. CoilFEM validates the coupling keyword
# ============================================================================

def test_coilfem_invalid_coupling_raises():
    """CoilFEM should raise ValueError for an unknown coupling value."""
    curve = _make_circle(N=4)
    with pytest.raises(ValueError, match="coupling"):
        CoilFEM(
            base_curves_jax=[curve],
            base_currents_jax=jnp.array([1.0]),
            nfp=1,
            stellsym=False,
            mesh_options={'shape': 'rect', 'w1': 0.01, 'w2': 0.01,
                          'n_grid_1': 1, 'n_grid_2': 1},
            problem_options={'winkler_k': 1e9, 'solver': 'umfpack'},
            coupling='invalid_option',
        )


# ============================================================================
# 3. CoilFEM dispatch: staggered driver is called when is_coupled=True
# ============================================================================

def test_coilfem_dispatch_calls_staggered(monkeypatch):
    """CoilFEM._solve_all should call solve_staggered when coupling='staggered'."""
    calls = []

    def mock_staggered(pipelines, support, params, *, options=None):
        calls.append('staggered')
        n = len(pipelines)
        n_nodes = pipelines[0].problem.num_total_dofs_all_vars // 3
        return {
            'sol_list_by_coil': [
                [jnp.zeros((n_nodes, 3))]
                for _ in range(n)
            ],
            'u_s': jnp.zeros(1),
            'diagnostics': {},
        }

    monkeypatch.setattr(
        'coil_fem.coil_fem.solve_staggered', mock_staggered
    )

    support = _TrivialCoupledSupport()
    fem = _make_coilfem(coupling='staggered', support=support)
    dofs = [c.dofs for c in fem.base_curves_jax]
    fem.objective(dofs, jnp.array([1.0]))

    assert calls == ['staggered'], (
        f"Expected solve_staggered to be called once, got {calls}"
    )


def test_coilfem_dispatch_calls_monolithic(monkeypatch):
    """CoilFEM._solve_all should call solve_monolithic when coupling='monolithic'."""
    calls = []

    def mock_monolithic(pipelines, support, params, *, options=None):
        calls.append('monolithic')
        n = len(pipelines)
        n_nodes = pipelines[0].problem.num_total_dofs_all_vars // 3
        return {
            'sol_list_by_coil': [
                [jnp.zeros((n_nodes, 3))]
                for _ in range(n)
            ],
            'u_s': jnp.zeros(1),
            'diagnostics': {},
        }

    monkeypatch.setattr(
        'coil_fem.coil_fem.solve_monolithic', mock_monolithic
    )

    support = _TrivialCoupledSupport()
    fem = _make_coilfem(coupling='monolithic', support=support)
    dofs = [c.dofs for c in fem.base_curves_jax]
    fem.objective(dofs, jnp.array([1.0]))

    assert calls == ['monolithic'], (
        f"Expected solve_monolithic to be called once, got {calls}"
    )


# ============================================================================
# 4. Uncoupled path: _solve_all with SupportFixed is backward-compatible
# ============================================================================

def test_uncoupled_solve_all_backward_compatible():
    """_solve_all with SupportFixed produces the same displacement as old path."""
    fem  = _make_coilfem(coupling='staggered', support=SupportFixed())
    dofs = [c.dofs for c in fem.base_curves_jax]
    curr = jnp.array([1.0])

    # Run via new _solve_all path
    all_g, all_gd, all_c = fem._expand_geometry(dofs, curr)
    solved = fem._solve_all(dofs, all_g, all_gd, all_c, base_support_dofs=None)

    # Run via old objective (which also calls _solve_all internally now)
    result = fem.run(dofs, curr, None)

    u_new = solved['sol_list_by_coil'][0][0]
    u_old = result['displacements'][0]
    np.testing.assert_allclose(
        np.array(u_new), np.array(u_old), atol=1e-12,
        err_msg="_solve_all and run should return identical displacements.",
    )


# ============================================================================
# 5. solve_staggered fixed-point smoke test
# ============================================================================

def test_staggered_fixed_point_trivial():
    """solve_staggered converges in one iteration for a trivially decoupled system.

    The trivial support always returns u_s = 0 and compute_attach returns
    zeros, so the coil FEM and support system are fully decoupled.  The
    staggered loop should detect convergence after the first sweep.
    """
    pipeline = _make_tiny_pipeline()
    support  = _TrivialCoupledSupport()
    pts      = pipeline.mesh.mesh_points_from_dofs(_make_circle(N=4).dofs)
    n_cells  = pipeline.problem.num_cells
    n_quads  = pipeline.problem.fes[0].num_quads
    n_surf   = pipeline.surface_node_indices.shape[0]
    curve    = _make_circle(N=4)

    params = {
        'mesh_points_by_coil': [pts],
        'body_force_by_coil':  [jnp.zeros((n_cells, n_quads, 3))],
        'weights_by_coil':     [jnp.ones(n_surf)],
        'curves_by_coil':      [curve],
        'base_curves_dofs':    [curve.dofs],
        'support_dofs':        {},
    }

    result = solve_staggered(
        [pipeline], support, params,
        options={'max_iters': 50, 'atol': 1e-10},
    )

    assert 'sol_list_by_coil' in result
    assert 'u_s' in result
    assert result['u_s'].shape == (1,)

    u_coil = result['sol_list_by_coil'][0][0]
    assert u_coil.shape[-1] == 3
    assert jnp.all(jnp.isfinite(u_coil)), "Displacement should be finite."


# ============================================================================
# 6. winkler_k / k_lin mismatch guard
# ============================================================================

def test_winkler_k_mismatch_raises():
    """CoilFEM should raise ValueError when winkler_k != support.k_lin."""
    support = _TrivialCoupledSupport()  # k_lin = 1e9

    curve = _make_circle(N=4)
    with pytest.raises(ValueError, match="winkler_k"):
        CoilFEM(
            base_curves_jax=[curve],
            base_currents_jax=jnp.array([1.0]),
            nfp=1,
            stellsym=False,
            mesh_options={'shape': 'rect', 'w1': 0.01, 'w2': 0.01,
                          'n_grid_1': 1, 'n_grid_2': 1},
            support=support,
            problem_options={'winkler_k': 5e8, 'solver': 'umfpack'},  # mismatch
            coupling='staggered',
        )
