import numpy as np

from ..problems import recompute_fe_geometry
from ..metrics import total_strain_energy

def volume_weighted_summary(obj, result):
    """RMS/max displacement, RMS/max von Mises, and total strain energy.

    von Mises and the body-force magnitude are ``(n_cells, n_quads)`` quadrature
    fields, weighted directly by JxW.  Displacement magnitude is a nodal field
    interpolated to the quadrature points with the element shape functions before
    weighting.  Maxima are taken over the raw (unsmoothed) quantities.
    """
    fem = obj.fem
    lam, mu = fem._lam, fem._mu
    num_d2 = num_vm2 = num_f2 = vol = 0.0
    max_d = max_vm = max_f = 0.0
    strain_energy = 0.0
    for i in range(len(result['von_mises'])):
        prob = fem._problems[i]
        pts  = result['mesh_points'][i]
        sg, jxw_j, _, _ = recompute_fe_geometry(
            pts, prob._cells_jnp, prob._sg_ref, prob._sv, prob._qw)
        jxw   = np.asarray(jxw_j)               # (n_cells, n_quads)
        sv    = np.asarray(prob._sv)            # (n_quads, n_cell_nodes)
        cells = np.asarray(prob._cells_jnp)     # (n_cells, n_cell_nodes)

        # von Mises: quadrature field -> weight by JxW directly
        vm = np.asarray(result['von_mises'][i])
        num_vm2 += np.sum(vm**2 * jxw)
        max_vm   = max(max_vm, float(np.max(vm)))

        # |u|: nodal field -> interpolate to quadrature points, then weight
        dmag   = np.linalg.norm(np.asarray(result['displacements'][i]), axis=-1)
        dmag_q = np.einsum('qn,cn->cq', sv, dmag[cells])
        num_d2 += np.sum(dmag_q**2 * jxw)
        max_d   = max(max_d, float(np.max(dmag)))

        # body force: quadrature field magnitude -> weight by JxW directly
        fmag = np.linalg.norm(np.asarray(result['f_vol'][i]), axis=-1)
        num_f2 += np.sum(fmag**2 * jxw)
        max_f   = max(max_f, float(np.max(fmag)))

        vol += np.sum(jxw)
        strain_energy += float(total_strain_energy(
            prob, result['solutions'][i], lam, mu, shape_grads=sg, JxW=jxw_j))
    return {
        'rms_displacement_m': float(np.sqrt(num_d2 / vol)),
        'max_displacement_m': max_d,
        'rms_von_mises_Pa':   float(np.sqrt(num_vm2 / vol)),
        'max_von_mises_Pa':   max_vm,
        'rms_body_force_Npm3': float(np.sqrt(num_f2 / vol)),
        'max_body_force_Npm3': max_f,
        'strain_energy_J':    strain_energy,
    }