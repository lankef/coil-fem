"""Tests for matrix_symmetry and weakest_symmetry solver claims (Phase 3a).

Validates that LinearElasticity3D and Support declare correct symmetry,
that weakest_symmetry resolves correctly, and that build_fwd_pred derives
mtype_id from the problem rather than from a hard-coded default.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from coil_fem.geo import CurveXYZFourierJAX, make_framed_curve
from coil_fem.meshing import CoilMesh
from coil_fem.pipelines import ElasticPipeline
from coil_fem.coupling import Support, SupportBeams
from coil_fem.solvers.cudss import (
    weakest_symmetry,
    adjoint_reuses_forward_K,
    _MTYPE_ID,
    _STRENGTH,
)


# ============================================================================
# Fixtures
# ============================================================================

def _make_circle(N: int = 4, R: float = 1.0) -> CurveXYZFourierJAX:
    quadpoints = jnp.linspace(0.0, 1.0, N, endpoint=False)
    dofs = jnp.array([0.0, R, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, R])
    return CurveXYZFourierJAX(quadpoints, dofs, order=1)


def _make_tiny_pipeline(R: float = 1.0) -> ElasticPipeline:
    curve = _make_circle(N=4, R=R)
    fc = make_framed_curve(curve, 'rmf')
    mesh = CoilMesh.from_options(
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


# ============================================================================
# weakest_symmetry unit tests
# ============================================================================

def test_weakest_symmetry_single():
    assert weakest_symmetry('symmetric') == 'symmetric'
    assert weakest_symmetry('general') == 'general'
    assert weakest_symmetry('spd') == 'spd'


def test_weakest_symmetry_general_dominates():
    assert weakest_symmetry('symmetric', 'general') == 'general'
    assert weakest_symmetry('spd', 'general') == 'general'
    assert weakest_symmetry('general', 'symmetric', 'spd') == 'general'


def test_weakest_symmetry_symmetric_beats_spd():
    assert weakest_symmetry('spd', 'symmetric') == 'symmetric'


def test_mtype_id_values():
    assert _MTYPE_ID['general'] == 0
    assert _MTYPE_ID['symmetric'] == 1
    assert _MTYPE_ID['spd'] == 3


def test_strength_ordering():
    assert _STRENGTH['general'] < _STRENGTH['symmetric'] < _STRENGTH['spd']


# ============================================================================
# adjoint_reuses_forward_K
# ============================================================================

def test_adjoint_reuses_forward_K_symmetric_and_spd():
    """Symmetric / SPD mtype_id → reuse forward K (no solver_KT workspace)."""
    assert adjoint_reuses_forward_K('symmetric', _MTYPE_ID['symmetric']) is True
    assert adjoint_reuses_forward_K('spd', _MTYPE_ID['spd']) is True


def test_adjoint_reuses_forward_K_general():
    """General mtype_id → keep a separate transpose solver."""
    assert adjoint_reuses_forward_K('general', _MTYPE_ID['general']) is False


def test_adjoint_reuses_forward_K_honours_mtype_override():
    """Final mtype_id wins over the string claim (cudss_mtype_id override)."""
    # Claim says symmetric but override forces general → do not reuse.
    assert adjoint_reuses_forward_K('symmetric', 0) is False
    # Claim says general but override forces symmetric → reuse.
    assert adjoint_reuses_forward_K('general', 1) is True
    assert adjoint_reuses_forward_K('general', 3) is True


# ============================================================================
# matrix_symmetry claims
# ============================================================================

def test_linear_elasticity_matrix_symmetry():
    """LinearElasticity3D declares matrix_symmetry = 'symmetric'."""
    pipeline = _make_tiny_pipeline()
    assert pipeline.problem.matrix_symmetry == 'symmetric'


def test_support_matrix_symmetry():
    """Base Support declares matrix_symmetry = 'symmetric'."""
    assert Support(k_clamp=1e9).matrix_symmetry == 'symmetric'


def test_support_beams_matrix_symmetry_inherited():
    """SupportBeams inherits 'symmetric' from Support without an override."""
    import math
    from coil_fem.coupling import SupportBeams

    def _constant_section_fn(A_val=1e-4, Iy_val=1e-8, Iz_val=1e-8, J_val=2e-8):
        def fn(support_dofs):
            phi_cc = support_dofs['phis_start_cc']
            phi_cf = support_dofs['phis_start_cf']
            A, Iy, Iz, J = [], [], [], []
            for g in range(len(phi_cc)):
                n_cf = phi_cf[g].shape[0] if g < len(phi_cf) else 0
                n_per = phi_cc[g].shape[0] + n_cf
                A.append(jnp.full((n_per,), A_val))
                Iy.append(jnp.full((n_per,), Iy_val))
                Iz.append(jnp.full((n_per,), Iz_val))
                J.append(jnp.full((n_per,), J_val))
            return A, Iy, Iz, J
        return fn

    def _uniform_clamp_fn(surface_pts_beam_frame, dofs, sign_x, constants):
        return jnp.ones(surface_pts_beam_frame.shape[0])

    sb = SupportBeams(
        nfp=2, stellsym=False,
        beam_options={'n_beam_cc': 1, 'n_beam_cf': 1, 'E': 200e9, 'nu': 0.3, 'k_attachment': 1e8},
        n_base=2,
        cross_section_fn=_constant_section_fn(),
        attachment_fn=_uniform_clamp_fn,
    )
    assert sb.matrix_symmetry == 'symmetric'
    # Verify it's the inherited version, not an override
    assert type(sb).matrix_symmetry is type(Support(k_clamp=1e9)).matrix_symmetry or \
        SupportBeams.matrix_symmetry is Support.matrix_symmetry


# ============================================================================
# build_fwd_pred derives mtype_id (CPU path — no cuDSS needed)
# ============================================================================

# ============================================================================
# is_linear claim
# ============================================================================

def test_linear_elasticity_is_linear():
    """LinearElasticity3D declares is_linear = True."""
    pipeline = _make_tiny_pipeline()
    assert pipeline.problem.is_linear is True


def test_build_fwd_pred_rejects_nonlinear_problem_on_cudss():
    """build_fwd_pred raises NotImplementedError for non-linear problems with cudss."""
    from coil_fem.solvers import build_fwd_pred

    pipeline = _make_tiny_pipeline()
    problem = pipeline.problem
    # Temporarily mark as non-linear
    problem.is_linear = False
    try:
        with pytest.raises(NotImplementedError, match="is_linear"):
            build_fwd_pred(problem, {'solver': 'cudss'})
    finally:
        problem.is_linear = True


def test_require_cuda_for_cudss_rejects_host_backend():
    """require_cuda_for_cudss raises when no CUDA JAX device is available."""
    import unittest.mock as mock
    from coil_fem.solvers.cudss import require_cuda_for_cudss

    host = mock.Mock()
    host.platform = 'cpu'
    with mock.patch('coil_fem.solvers.cudss.jax.devices', return_value=[host]):
        with mock.patch(
            'coil_fem.solvers.cudss.jax.default_backend', return_value='cpu'
        ):
            with pytest.raises(RuntimeError, match="JAX_PLATFORMS"):
                require_cuda_for_cudss()


# ============================================================================
# build_fwd_pred warning on mtype override
# ============================================================================

def test_build_fwd_pred_warning_on_mtype_override():
    """build_fwd_pred warns when cudss_mtype_id overrides the derived value."""
    from coil_fem.solvers import build_fwd_pred

    pipeline = _make_tiny_pipeline()
    problem = pipeline.problem
    # derived is 'symmetric' → mtype_id=1; override to 0 (general) → should warn
    problem_options = {
        'solver': 'cudss',
        'cudss_mtype_id': 0,
    }
    # We can't actually build the cuDSS solver without spineax, but we can
    # trigger the warning check by patching cudss_ad_wrapper.
    import coil_fem.solvers.cudss as _cudss_mod
    original = getattr(_cudss_mod, 'cudss_ad_wrapper', None)
    if original is None:
        pytest.skip("cudss_ad_wrapper not available (spineax not installed)")

    calls = []
    def _mock_wrapper(*args, **kwargs):
        calls.append(kwargs.get('mtype_id'))
        return lambda params: None

    import unittest.mock as mock
    with mock.patch('coil_fem.solvers.cudss.cudss_ad_wrapper', _mock_wrapper):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            build_fwd_pred(problem, problem_options)
        assert len(w) == 1
        assert '0' in str(w[0].message) or 'override' in str(w[0].message).lower()
        assert calls[0] == 0  # override applied
