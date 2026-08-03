"""Tests for the monolithic cuDSS coupling path.

The CPU-verifiable tests exercise the on-device COO assembly and the CSR
pattern machinery that back :func:`~coil_fem.coupling.drivers.solve_monolithic`
without requiring the GPU stack:

* the linearity identity ``K u == R(u) - R(0)`` for a ``gpu_assembly=True``
  problem, validating that :meth:`ElasticPipeline.assemble_coo` returns a
  consistent ``(I, J, V)`` triplet;
* the transpose CSR pattern used by the general (non-symmetric) adjoint path;
* the ``gpu_assembly`` gate and ``_fe_geom`` assemble parity.

A GPU-gated test checks that a symmetric SupportBeams + elasticity merge
skips ``solver_KT`` and that a forward monolithic solve stays finite.
"""

from __future__ import annotations

import importlib.util
import math

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from coil_fem.geo import CurveXYZFourierJAX, make_framed_curve
from coil_fem.meshing import CoilMesh
from coil_fem.pipelines import ElasticPipeline
from coil_fem.problems import LinearElasticity3D
from coil_fem.coupling import SupportBeams
from coil_fem.problems import recompute_fe_geometry
from coil_fem.solvers.cudss import build_csr_pattern, assemble_csr_values

_HAS_SPINEAX = importlib.util.find_spec("spineax") is not None
_HAS_GPU = any(d.platform == "gpu" for d in jax.devices())
_GPU_REASON = "requires spineax + a CUDA device"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_circle(N: int = 4, R: float = 1.0) -> CurveXYZFourierJAX:
    """Planar circle of radius R in the xz-plane (order-1 Fourier curve)."""
    quadpoints = jnp.linspace(0.0, 1.0, N, endpoint=False)
    dofs = jnp.array([0.0, R, 0.0,
                      0.0, 0.0, 0.0,
                      0.0, 0.0, R])
    return CurveXYZFourierJAX(quadpoints, dofs, order=1)


def _tiny_mesh(R: float = 1.0) -> CoilMesh:
    """Smallest usable coil mesh (4 phi x 1x1 rectangular cross-section)."""
    fc = make_framed_curve(_make_circle(N=4, R=R), 'rmf')
    return CoilMesh.from_options(
        fc,
        {'shape': 'rect', 'w1': 0.01, 'w2': 0.01, 'n_grid_1': 1, 'n_grid_2': 1},
        'TET4',
    )


def _make_pipeline(mesh: CoilMesh, solver: str):
    return ElasticPipeline(
        mesh,
        E=200e9, nu=0.3, itc=None,
        gravity_bf=(0.0, 0.0, 0.0),
        problem_options={'solver': solver},
    )


def _gpu_problem(mesh: CoilMesh) -> LinearElasticity3D:
    """A ``gpu_assembly=True`` problem, built without the cuDSS forward solver.

    Constructs the FEM problem directly (bypassing ``build_fwd_pred``) so the
    on-device COO assembly can be exercised on CPU without spineax.
    """
    problem = LinearElasticity3D(
        mesh, vec=3, dim=3, ele_type=mesh.ele_type,
        additional_info=(200e9, 0.3, (0.0, 0.0, 0.0), None),
        gpu_assembly=True,
    )
    mesh.attach_ref_coords(problem)
    return problem


# ---------------------------------------------------------------------------
# 1. Linearity identity for the on-device COO assembly
# ---------------------------------------------------------------------------

def test_gpu_assembly_coo_linearity():
    """K u == R(u) - R(0) for a gpu_assembly problem's COO triplet.

    Densifying ``(problem.I, problem.J, problem.V_jax)`` must reproduce the
    linear-elasticity tangent, i.e. the residual difference between any state
    ``u`` and the zero state.
    """
    mesh = _tiny_mesh()
    problem = _gpu_problem(mesh)

    n_nodes = problem.fes[0].num_total_nodes
    n_sq = problem.n_surface_quads
    params = {
        'points':          jnp.asarray(mesh.points),
        'body_force':      jnp.zeros((mesh.n_cells, mesh.n_quads, 3)),
        'support_k': jnp.ones(n_sq) * 1e9,
    }

    problem.set_params(params)
    zero_sol = [jnp.zeros((n_nodes, 3))]
    problem.compute_newton_vars(
        zero_sol, problem.internal_vars, problem.internal_vars_surfaces
    )

    I = np.asarray(problem.I)
    J = np.asarray(problem.J)
    V = problem.V_jax
    n = problem.num_total_dofs_all_vars

    assert I.shape == J.shape
    assert V.shape[0] == I.shape[0], "V_jax must align with the I/J COO pattern"

    K = jnp.zeros((n, n)).at[I, J].add(V)

    def residual(u_flat):
        u = u_flat.reshape(n_nodes, 3)
        res = problem.compute_residual_vars(
            [u], problem.internal_vars, problem.internal_vars_surfaces
        )
        return jax.flatten_util.ravel_pytree(res)[0]

    R0 = residual(jnp.zeros(n))
    u = jax.random.normal(jax.random.PRNGKey(0), (n,)) * 1e-6

    lhs = K @ u
    rhs = residual(u) - R0
    assert jnp.allclose(lhs, rhs, rtol=1e-6, atol=1e-8), (
        f"max abs err = {float(jnp.max(jnp.abs(lhs - rhs))):.3e}"
    )


# ---------------------------------------------------------------------------
# 2. Transpose CSR pattern (used by the adjoint solve)
# ---------------------------------------------------------------------------

def _csr_to_dense(indptr, indices, values, n):
    indptr = np.asarray(indptr)
    indices = np.asarray(indices)
    values = np.asarray(values)
    A = np.zeros((n, n))
    for r in range(n):
        for k in range(int(indptr[r]), int(indptr[r + 1])):
            A[r, int(indices[k])] = values[k]
    return A


def test_build_csr_pattern_transpose():
    """Scattering the same COO values through build_csr_pattern(J, I) gives Kᵀ.

    This underpins the monolithic adjoint solve, which reuses the forward COO
    value vector but assembles it into the transposed CSR layout.
    """
    n = 5
    # Every diagonal (required by build_csr_pattern) plus off-diagonals and a
    # couple of duplicate (i, j) entries whose values must sum.
    I = np.array([0, 1, 2, 3, 4, 0, 1, 3, 2, 0], dtype=np.int64)
    J = np.array([0, 1, 2, 3, 4, 2, 0, 1, 2, 0], dtype=np.int64)
    rng = np.random.default_rng(0)
    Vnp = rng.standard_normal(I.shape[0])
    V = jnp.asarray(Vnp)

    A_ref = np.zeros((n, n))
    np.add.at(A_ref, (I, J), Vnp)   # duplicates summed

    indptr, indices, coo_to_csr, _r, _d, nnz = build_csr_pattern(I, J, n)
    indptr_T, indices_T, coo_to_csr_T, _rT, _dT, nnz_T = build_csr_pattern(J, I, n)

    csr_K = assemble_csr_values(V, coo_to_csr, nnz)
    csr_KT = assemble_csr_values(V, coo_to_csr_T, nnz_T)

    K_dense = _csr_to_dense(indptr, indices, csr_K, n)
    KT_dense = _csr_to_dense(indptr_T, indices_T, csr_KT, n)

    assert np.allclose(K_dense, A_ref)
    assert np.allclose(KT_dense, A_ref.T)


# ---------------------------------------------------------------------------
# 3. gpu_assembly gate
# ---------------------------------------------------------------------------

def test_assemble_coo_requires_gpu_assembly():
    """assemble_coo raises on a CPU (gpu_assembly=False) pipeline."""
    mesh = _tiny_mesh()
    pipeline = _make_pipeline(mesh, solver='umfpack')
    n_sq = pipeline.n_surface_quads
    params = {
        'points':          jnp.asarray(mesh.points),
        'body_force':      jnp.zeros((mesh.n_cells, mesh.n_quads, 3)),
        'support_k': jnp.ones(n_sq) * 1e9,
    }
    with pytest.raises(NotImplementedError, match="gpu_assembly=True"):
        pipeline.assemble_coo(params)


def test_assemble_coo_fe_geom_parity():
    """set_params / Newton V with ``_fe_geom`` matches a fresh recompute."""
    mesh = _tiny_mesh()
    problem = _gpu_problem(mesh)
    pts = jnp.asarray(mesh.points)
    n_sq = problem.n_surface_quads
    base = {
        'points':     pts,
        'body_force': jnp.zeros((mesh.n_cells, mesh.n_quads, 3)),
        'support_k':  jnp.ones(n_sq) * 1e9,
    }
    fe_geom = recompute_fe_geometry(
        pts, problem._cells_jnp, problem._sg_ref, problem._sv, problem._qw,
    )

    def _assemble(params):
        problem.set_params(params)
        zero_sol = [jnp.zeros((problem.fes[0].num_total_nodes, 3))]
        res = problem.compute_newton_vars(
            zero_sol, problem.internal_vars, problem.internal_vars_surfaces,
        )
        load = -jax.flatten_util.ravel_pytree(res)[0]
        return problem.V_jax, load

    V0, f0 = _assemble(dict(base))
    V1, f1 = _assemble({**base, '_fe_geom': fe_geom})
    assert jnp.allclose(V0, V1, rtol=1e-12, atol=1e-14)
    assert jnp.allclose(f0, f1, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# 4. GPU-gated: symmetric merged system skips solver_KT
# ---------------------------------------------------------------------------

def _constant_section_fn(A_val=1e-4, Iy_val=1e-8, Iz_val=1e-8, J_val=2e-8):
    def fn(support_dofs):
        phi_cc = support_dofs['phis_start_cc']
        phi_cf = support_dofs['phis_start_cf']
        A, Iy, Iz, Jt = [], [], [], []
        for g in range(len(phi_cc)):
            n_cf = phi_cf[g].shape[0] if g < len(phi_cf) else 0
            n_per = phi_cc[g].shape[0] + n_cf
            A.append(jnp.full((n_per,), A_val))
            Iy.append(jnp.full((n_per,), Iy_val))
            Iz.append(jnp.full((n_per,), Iz_val))
            Jt.append(jnp.full((n_per,), J_val))
        return A, Iy, Iz, Jt
    return fn


def _uniform_clamp_fn(surface_pts_beam_frame, dofs, sign_x, constants):
    return jnp.ones(surface_pts_beam_frame.shape[0])


@pytest.mark.skipif(not (_HAS_SPINEAX and _HAS_GPU), reason=_GPU_REASON)
def test_monolithic_static_reuses_K_when_symmetric():
    """Symmetric SupportBeams + elasticity: no second cuDSS transpose workspace."""
    from coil_fem import CoilFEM

    n_base = 2
    radii = [1.0, 1.1]
    curves = [_make_circle(N=4, R=R) for R in radii]
    beam_options = {
        'n_beam_cc': 1, 'n_beam_cf': 1,
        'E': 200e9, 'nu': 0.3,
        'k_attachment': 1e8,
    }
    support = SupportBeams(
        nfp=1, stellsym=False,
        beam_options=beam_options,
        n_base=n_base,
        cross_section_fn=_constant_section_fn(),
        attachment_fn=_uniform_clamp_fn,
    )
    fem = CoilFEM(
        base_curves_jax=curves,
        base_currents_jax=jnp.ones(n_base),
        nfp=1,
        stellsym=False,
        mesh_options={
            'shape': 'rect', 'w1': 0.01, 'w2': 0.01,
            'n_grid_1': 1, 'n_grid_2': 1,
        },
        support=support,
        material_options={'E': 200e9, 'nu': 0.3, 'density': 8900.0},
        problem_options={'solver': 'cudss'},
        coupling='monolithic',
    )
    static = fem.monolithic_static
    assert static is not None
    assert static.adjoint_reuses_K is True
    assert static.solver_KT is None
    assert static.coo_to_csr_T is None

    phi = [jnp.full((1,), 0.1) for _ in range(n_base)]
    x_found = []
    for R in radii:
        xf = jnp.array([
            R * math.cos(2 * math.pi * 0.6) + 0.5, 0.0,
            R * math.sin(2 * math.pi * 0.6),
        ])
        x_found.append(xf[None, :])
    support_dofs = {
        'phis_start_cc':         list(phi),
        'phis_end_cc':           list(phi),
        'phis_start_cf':         [jnp.full((1,), 0.6) for _ in range(n_base)],
        'x_foundation':          x_found,
        'thetas_orientation_cc': [jnp.zeros((1,)) for _ in range(n_base)],
        'thetas_orientation_cf': [jnp.zeros((1,)) for _ in range(n_base)],
    }
    out = fem.run(
        base_curves_dofs=[c.dofs for c in curves],
        base_currents_dofs=jnp.ones(n_base),
        base_support_dofs=support_dofs,
    )
    assert jnp.all(jnp.isfinite(out['u_s']))
    for u in out['displacements']:
        assert jnp.all(jnp.isfinite(u))
