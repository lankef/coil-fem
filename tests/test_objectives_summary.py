"""Regression tests for ``CoilFEMObjective.summary``."""

from __future__ import annotations

import types

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from coil_fem.geo import CurveXYZFourierJAX
from coil_fem.coupling import Support
from coil_fem.coil_fem import CoilFEM
from coil_fem.simsopt import CoilFEMObjective


def _stub(casing_options=None):
    """Single-coil CoilFEM wrapped in a minimal ``summary()``-compatible stub."""
    quadpoints = jnp.linspace(0.0, 1.0, 4, endpoint=False)
    dofs = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    curve = CurveXYZFourierJAX(quadpoints, dofs, order=1)
    currents = jnp.array([1e6])
    fem = CoilFEM(
        base_curves_jax=[curve],
        base_currents_jax=currents,
        nfp=1,
        stellsym=False,
        mesh_options={'shape': 'rect', 'w1': 0.01, 'w2': 0.01,
                      'n_grid_1': 1, 'n_grid_2': 1},
        support=Support(k_clamp=1e9),
        winding_pack_options={'E': 200e9, 'nu': 0.3, 'density': 8900.0},
        casing_options=casing_options,
        problem_options={'solver': 'umfpack'},
        coupling='staggered',
    )
    run = lambda: fem.run(base_curves_dofs=[dofs], base_currents_dofs=currents)
    return types.SimpleNamespace(fem=fem, run=run), dofs, currents


def test_summary_strain_energy_matches_objective():
    """summary() strain energy uses per-quad Lamé arrays like CoilFEM.objective."""
    stub, dofs, currents = _stub()
    s = CoilFEMObjective.summary(stub)
    expected = float(stub.fem.objective(
        [dofs], currents, metrics=('strain_energy',))['strain_energy'])

    assert np.isfinite(s['strain_energy_J']) and s['strain_energy_J'] > 0.0
    np.testing.assert_allclose(s['strain_energy_J'], expected, rtol=1e-8)

    # No casing: winding pack is the whole body, no casing keys.
    np.testing.assert_allclose(s['rms_von_mises_winding_pack_Pa'], s['rms_von_mises_Pa'])
    assert s['max_von_mises_winding_pack_Pa'] == s['max_von_mises_Pa']
    assert 'rms_von_mises_casing_Pa' not in s
    assert 'max_von_mises_casing_Pa' not in s


def test_summary_von_mises_per_material_with_casing():
    """Casing adds per-material von Mises keys consistent with the global max."""
    stub, _, _ = _stub(casing_options={
        'E': 200e9, 'nu': 0.3, 'density': 8000.0, 'thickness': 0.002})
    s = CoilFEMObjective.summary(stub)

    for k in ('rms_von_mises_casing_Pa', 'max_von_mises_casing_Pa',
              'rms_von_mises_winding_pack_Pa', 'max_von_mises_winding_pack_Pa'):
        assert np.isfinite(s[k]) and s[k] > 0.0
    assert s['max_von_mises_Pa'] == max(
        s['max_von_mises_winding_pack_Pa'], s['max_von_mises_casing_Pa'])
