"""Steady-state heat conduction for JAX-FEM (stub).

Will provide :class:`HeatConduction3D`, a JAX-FEM ``Problem`` subclass that
solves ``−∇·(k ∇T) = Q`` on the coil volume.  The solution temperature field
feeds into :class:`~coil_fem.problems.LinearElasticity3D` as a spatially varying
thermal eigenstrain for thermoelastic coupling.
"""

from __future__ import annotations


class HeatConduction3D:
    """Steady-state heat conduction on a coil mesh (not yet implemented).

    Raises
    ------
    NotImplementedError
        Always — this class is a placeholder for a future implementation.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "HeatConduction3D is not yet implemented. "
            "Use ThermoElasticPipeline once this class is available."
        )
