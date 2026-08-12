"""OCC solid factories match ``A * L`` from the section-property formulas."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

gmsh = pytest.importorskip("gmsh")

from coil_fem.presets import cross_section_fns as cs


# (section_name, dofs_scalar_dict, L)
_CASES = (
    ("solid_circle", {"r_beam": 0.02}, 0.5),
    ("solid_rectangle", {"w1_beam": 0.04, "w2_beam": 0.03}, 0.5),
    ("hollow_circle", {"r_1_beam": 0.02, "r_2_beam": 0.015}, 0.5),
    ("hollow_rectangle", {"w1_beam": 0.04, "w2_beam": 0.03, "t_beam": 0.004}, 0.5),
)


def _occ_mass(solid_fn, dofs, L):
    """Build one body in a scratch gmsh model and return its OCC mass."""
    try:
        owned = not gmsh.isInitialized()
    except AttributeError:
        owned = True
    if owned:
        gmsh.initialize()
    else:
        gmsh.clear()
    try:
        gmsh.model.add("solid_mass")
        occ = gmsh.model.occ
        dimtags = solid_fn(occ, dofs, L)
        occ.synchronize()
        assert len(dimtags) >= 1
        # Sum mass over all returned volumes (cut can leave one primary solid).
        return sum(occ.getMass(d, t) for d, t in dimtags if d == 3)
    finally:
        if owned:
            gmsh.finalize()
        else:
            gmsh.clear()


@pytest.mark.parametrize("name,dofs,L", _CASES, ids=[c[0] for c in _CASES])
def test_solid_mass_matches_A_times_L(name, dofs, L):
    """``*_solid`` volume equals ``A * L`` from the matching section formula."""
    solid_fn = getattr(cs, name + "_solid")
    section_fn = getattr(cs, name)
    sdofs = {k: [jnp.array([v])] for k, v in dofs.items()}
    A_all, *_ = section_fn(sdofs)
    A = float(A_all[0][0])
    mass = _occ_mass(solid_fn, dofs, L)
    assert np.isclose(mass, A * L, rtol=1e-6, atol=1e-12)
