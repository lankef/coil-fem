"""Regression test for CoilFEM VTU export.

Guards against regressions in the per-node support field scatter that writes
``w_clamp``, ``w_attach``, ``k_clamp_Npm3``, and ``k_attach_Npm3`` to the
exported VTU files.
"""

from __future__ import annotations

import meshio
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from coil_fem.geo import CurveXYZFourierJAX
from coil_fem.coupling import Support, SupportBeams
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
        support=Support(k_clamp=1e9),
        material_options={'E': 200e9, 'nu': 0.3, 'density': 8900.0},
        problem_options={'solver': 'umfpack'},
        coupling='staggered',
    )


def test_save_run_vtu_writes_files(tmp_path):
    """save_run_vtu should run end-to-end and write one VTU per coil."""
    fem = _make_coilfem()
    written = fem.save_run_vtu(str(tmp_path))
    assert len(written) == 1
    for path in written:
        assert (tmp_path / path.split('/')[-1]).exists()

    mesh = meshio.read(tmp_path / written[0].split('/')[-1])
    for key in ('w_clamp', 'w_attach', 'k_clamp_Npm3', 'k_attach_Npm3'):
        assert key in mesh.point_data, f"missing VTU point field {key!r}"


# ============================================================================
# save_run_vtu beam-displacement file (SupportBeams only)
# ============================================================================

def _section_fn(sdofs):
    """Constant cross-section, matching the contract of SupportBeams.coo."""
    phi_cc, phi_cf = sdofs['phis_start_cc'], sdofs['phis_start_cf']
    A_l, Iy_l, Iz_l, J_l = [], [], [], []
    for g in range(len(phi_cc)):
        n_cf_g = phi_cf[g].shape[0] if g < len(phi_cf) else 0
        n_per = phi_cc[g].shape[0] + n_cf_g
        A_l.append(jnp.full((n_per,), 1e-4))
        Iy_l.append(jnp.full((n_per,), 1e-8))
        Iz_l.append(jnp.full((n_per,), 1e-8))
        J_l.append(jnp.full((n_per,), 2e-8))
    return A_l, Iy_l, Iz_l, J_l


def _uniform_clamp_fn(surface_pts_beam_frame, dofs, sign_x, beam_options):
    return jnp.ones(surface_pts_beam_frame.shape[0])


def _make_coilfem_with_beams() -> tuple[CoilFEM, CurveXYZFourierJAX, dict]:
    """Single coil, one CF beam, no CC beams — smallest SupportBeams setup."""
    quadpoints = jnp.linspace(0.0, 1.0, 8, endpoint=False)
    dofs = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    curve = CurveXYZFourierJAX(quadpoints, dofs, order=1)

    k_attachment = 1e8
    support = SupportBeams(
        nfp=1, stellsym=False,
        beam_options={'n_beam_cc': 0, 'n_beam_cf': 1, 'E': 200e9, 'nu': 0.3,
                      'k_attachment': k_attachment},
        n_base=1,
        cross_section_fn=_section_fn,
        attachment_fn=_uniform_clamp_fn,
    )
    fem = CoilFEM(
        base_curves_jax=[curve],
        base_currents_jax=jnp.array([1.0]),
        nfp=1,
        stellsym=False,
        mesh_options={'shape': 'rect', 'w1': 0.01, 'w2': 0.01,
                      'n_grid_1': 1, 'n_grid_2': 1},
        support=support,
        material_options={'E': 200e9, 'nu': 0.3, 'density': 8900.0},
        problem_options={'solver': 'umfpack'},
        coupling='staggered',
    )
    sdofs = {
        'phis_start_cc':         [jnp.zeros(0)],
        'phis_end_cc':           [jnp.zeros(0)],
        'phis_start_cf':         [jnp.array([0.0])],
        'x_foundation':          [jnp.array([[2.0, 0.0, 0.0]])],
        'thetas_orientation_cc': [jnp.zeros(0)],
        'thetas_orientation_cf': [jnp.array([0.0])],
    }
    return fem, curve, sdofs


def test_save_run_vtu_writes_beams_displacement_file(tmp_path, monkeypatch):
    """save_run_vtu writes {prefix}_beams.vtu with per-point displacement_m.

    ``SupportBeams`` is coupled and ``solve_staggered`` is retired (GPU-only
    ``solve_monolithic`` remains), so a real coupled forward solve isn't
    available on this backend. ``CoilFEM.run`` is monkeypatched to return a
    shape-correct but otherwise arbitrary result (zero mesh fields, a fixed
    nonzero ``u_s``) so this test exercises only the VTU-writing logic added
    to ``save_run_vtu`` -- ``SupportBeams.beam_displacement`` itself is
    covered directly in ``tests/test_beam_networks.py``.
    """
    fem, curve, sdofs = _make_coilfem_with_beams()
    n_sub = 5

    pipeline = fem.pipelines[0]
    n_quads  = pipeline.problem.fes[0].num_quads
    n_cells  = pipeline.problem.num_cells
    pts      = fem.meshes[0].mesh_points_from_dofs(curve.dofs)
    n_nodes  = pts.shape[0]

    support = fem.support
    u_s = jnp.arange(support.n_support_dofs, dtype=jnp.float64) * 1e-4

    fake_result = {
        'mesh_points':   [pts],
        'displacements': [jnp.zeros((n_nodes, 3))],
        'von_mises':     [jnp.zeros((n_cells, n_quads))],
        'f_vol':         [jnp.zeros((n_cells, n_quads, 3))],
        'B_self':        [jnp.zeros((n_cells, n_quads, 3))],
        'B_ext':         [jnp.zeros((n_cells, n_quads, 3))],
        'u_s':           u_s,
    }
    monkeypatch.setattr(fem, 'run', lambda **kwargs: fake_result)

    written = fem.save_run_vtu(str(tmp_path), base_support_dofs=sdofs, n_sub=n_sub)
    assert len(written) == 2
    beams_path = next(p for p in written if p.endswith('_beams.vtu'))
    assert (tmp_path / beams_path.split('/')[-1]).exists()

    n_beams = support.n_beams_total
    mesh = meshio.read(beams_path)
    assert mesh.points.shape == (n_beams * (n_sub + 1), 3)
    assert mesh.cells[0].data.shape == (n_beams * n_sub, 2)
    disp = mesh.point_data['displacement_m']
    assert disp.shape == (n_beams * (n_sub + 1), 3)

    # First/last point of the (single) beam are its node-1/node-2 translations.
    u_beam = np.asarray(u_s).reshape(n_beams, 12)
    assert np.allclose(disp[0],  u_beam[0, 0:3], atol=1e-10)
    assert np.allclose(disp[-1], u_beam[0, 6:9], atol=1e-10)

    assert mesh.cell_data['beam_type'][0].shape  == (n_beams * n_sub,)
    assert mesh.cell_data['coil_index'][0].shape == (n_beams * n_sub,)


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        fem = _make_coilfem()
        out = fem.save_run_vtu(d)
        assert len(out) == 1, out
        print("OK:", out)
