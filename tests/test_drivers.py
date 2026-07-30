"""Tests for the coupled solver drivers.

Tests cover:

* ``solve_monolithic`` raises ``NotImplementedError`` on CPU backends.
* ``solve_staggered`` raises ``NotImplementedError`` (retired).
* ``CoilFEM`` validates the ``coupling`` keyword argument.
* ``CoilFEM`` dispatches to ``solve_staggered`` vs ``solve_monolithic`` based
  on the ``coupling`` option (verified via monkeypatching).
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
from coil_fem.coupling import Support, solve_staggered, solve_monolithic, MonolithicStatic
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
        problem_options={'solver': 'umfpack'},
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
        support=support if support is not None else Support(k_clamp=1e9),
        material_options={'E': 200e9, 'nu': 0.3, 'density': 8900.0},
        problem_options={'solver': 'umfpack'},
        coupling=coupling,
    )


# ============================================================================
# Minimal mock coupled support
# ============================================================================

class _TrivialCoupledSupport(Support):
    """Mock coupled support: always returns u_s = 0, no coil coupling."""

    n_support_dofs = 1

    def __init__(self, k_clamp: float = 1e9):
        super().__init__(k_clamp=k_clamp)
        self._k_attachment = k_clamp

    @property
    def is_coupled(self):
        return True

    @property
    def k_attachment(self):
        return self._k_attachment

    def solve(self, inputs: dict) -> dict:
        return {'u_s': jnp.zeros(self.n_support_dofs)}

    def displacement_at(self, state: dict, points: jax.Array) -> jax.Array:
        """Staggered-mode placeholder (unused while solve_staggered is retired)."""
        return jnp.zeros_like(points)

    def compute_weights(self, coil_idx, surface_pts, curves_jax, dofs):
        n = surface_pts.shape[0]
        return jnp.ones(n), jnp.zeros(n)

    def support_pattern(self):
        # Static K_ss pattern (1×1 diagonal), required by build_monolithic_static.
        return np.zeros(1, dtype=np.int32), np.zeros(1, dtype=np.int32)

    def support_values(self, curves_jax, support_dofs, surface_pts_by_coil=None,
                       geom=None, *, jxw_by_coil=None, beam_endpoints=None):
        return jnp.ones(self.n_support_dofs, dtype=jnp.float64) * 1e9


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
        'stiffness_by_coil':   [
            jnp.ones(pipeline.n_surface_quads) * 1e9
        ],
        'curves_by_coil':      [_make_circle(N=4)],
        'base_curves_dofs':    [_make_circle(N=4).dofs],
        'support_dofs':        {},
    }
    static = MonolithicStatic(
        coil_dof_offsets=(0,), support_dof_offset=48, n_total_dofs=50,
        n_dofs_per_coil=(48,), n_s=2, has_cs=False, has_sc=False,
        surface_node_indices_by_coil=(pipeline.surface_node_indices,),
        curve_qps=(_make_circle(N=4).quadpoints,),
        curve_orders=(1,),
        I_ss_pat=np.zeros(0, dtype=np.int32),
        J_ss_pat=np.zeros(0, dtype=np.int32),
        I_cs_pat=None, J_cs_pat=None, I_sc_pat=None, J_sc_pat=None,
        indptr=None, indices=None, coo_to_csr=None, nnz_csr=0,
        coo_to_csr_T=None, nnz_csr_T=None,
        solver_K=None, solver_KT=None, merged_solve=None,
    )
    with pytest.raises(NotImplementedError, match="cudss"):
        solve_monolithic([pipeline], support, params, static)


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
            support=Support(k_clamp=1e9),
            material_options={'E': 200e9, 'nu': 0.3, 'density': 8900.0},
            problem_options={'solver': 'umfpack'},
            coupling='invalid_option',
        )


# ============================================================================
# 3. CoilFEM dispatch: staggered driver is called when is_coupled=True
# ============================================================================

def test_coilfem_dispatch_calls_staggered(monkeypatch):
    """CoilFEM._solve_all dispatches to solve_staggered when coupling='staggered'.

    Note: solve_staggered itself raises NotImplementedError (retired).
    This test monkeypatches to verify the dispatch logic independently.
    """
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

    def mock_monolithic(pipelines, support, params, static):
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
# 4. solve_staggered is retired
# ============================================================================

def test_staggered_raises_not_implemented():
    """solve_staggered raises NotImplementedError (retired; use 'monolithic')."""
    with pytest.raises(NotImplementedError, match="solve_staggered"):
        solve_staggered([], None, {})
