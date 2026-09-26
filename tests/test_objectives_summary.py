"""Regression test for ``CoilFEMObjective.summary`` strain energy."""

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


def test_summary_strain_energy_matches_objective():
    """summary() strain energy uses per-quad Lamé arrays like CoilFEM.objective."""
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
        problem_options={'solver': 'umfpack'},
        coupling='staggered',
    )
    run = lambda: fem.run(base_curves_dofs=[dofs], base_currents_dofs=currents)
    stub = types.SimpleNamespace(fem=fem, run=run)

    energy = CoilFEMObjective.summary(stub)['strain_energy_J']
    expected = float(fem.objective(
        [dofs], currents, metrics=('strain_energy',))['strain_energy'])

    assert np.isfinite(energy) and energy > 0.0
    np.testing.assert_allclose(energy, expected, rtol=1e-8)
