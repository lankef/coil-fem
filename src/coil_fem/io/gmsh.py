from coil_fem.simsopt import CoilSupportBeams
from coil_fem.simsopt import CoilFEMObjective
import gmsh
from simsopt.configs import get_data
from simsopt.mhd import Vmec
from simsopt import save
import numpy as np
import jax
import time
from simsopt.field import Coil
import gmsh


gmsh.initialize()
gmsh.model.add("device")
occ = gmsh.model.occ # OpenCASCADE

def coil_mesh_solid(mesh, n_slices=96):
    """Loft one coil into two half solids; returns their (dim, tag) list."""
    fc  = mesh.framed_curve
    phi = np.linspace(0.0, 1.0, n_slices, endpoint=False)
    r0  = np.asarray(fc.curve.gamma_eval(phi))       # (K, 3) centerline
    _, p, q = fc.rotated_frame_eval(phi)             # cross-section frame
    p, q = np.asarray(p), np.asarray(q)

    # Same corner convention as the FEM sweep; consistent CCW ordering of the
    # ring wires guarantees the loft does not twist between sections.
    corners = [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)]
    wires = []
    for k in range(n_slices):
        pts = [
            occ.addPoint(*(r0[k] + 0.5 * mesh.w1 * u * p[k]
                                 + 0.5 * mesh.w2 * v * q[k]))
            for (u, v) in corners
        ]
        lines = [occ.addLine(pts[a], pts[(a + 1) % 4]) for a in range(4)]
        wires.append(occ.addWire(lines))

    # makeRuled=True: straight facets between rings — matches the linear
    # phi-sweep of the FEM mesh and is robust (no B-spline self-intersection).
    half = n_slices // 2
    out  = []
    out += occ.addThruSections(wires[:half + 1], makeSolid=True, makeRuled=False)
    out += occ.addThruSections(wires[half:] + [wires[0]],
                               makeSolid=True, makeRuled=False)
    occ.remove([(1, w) for w in wires], recursive=True)  # construction wires
    return out


coil_solids = [add_coil_solid(m) for m in meshes]