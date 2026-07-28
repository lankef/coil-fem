"""Regression test for CoilFEM VTU export.

Guards against the ``save_run_vtu`` regression where the per-node support
weight scatter was replaced with a call to a deleted module-level helper
(``_support_weights_full``) and lost its ``winkler_k`` binding, causing a
``NameError`` at call time.
"""

from __future__ import annotations

import jax.numpy as jnp

from coil_fem.geo import CurveXYZFourierJAX
from coil_fem.coupling import Support
from coil_fem.coil_fem import CoilFEM


def _make_coilfem() -> CoilFEM:
    quadpoints = jnp.linspace(0.0, 1.0, 4, endpoint=False)
    dofs = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    curve = CurveXYZFourierJAX(quadpoints, dofs, order=1)
    return CoilFEM(
        base_curves_jax=[curve],
        base_currents_jax=jnp.array([1.0]),
        nfp=1,
        stellsym=False,
        mesh_options={'shape': 'rect', 'w1': 0.01, 'w2': 0.01,
                      'n_grid_1': 1, 'n_grid_2': 1},
        support=Support(),
        problem_options={'winkler_k': 1e9, 'solver': 'umfpack'},
        coupling='staggered',
    )


def test_save_run_vtu_writes_files(tmp_path):
    """save_run_vtu should run end-to-end and write one VTU per coil."""
    fem = _make_coilfem()
    written = fem.save_run_vtu(str(tmp_path))
    assert len(written) == 1
    for path in written:
        assert (tmp_path / path.split('/')[-1]).exists()


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        fem = _make_coilfem()
        out = fem.save_run_vtu(d)
        assert len(out) == 1, out
        print("OK:", out)
