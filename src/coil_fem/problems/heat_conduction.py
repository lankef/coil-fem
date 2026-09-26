"""Steady-state heat conduction for JAX-FEM (stub).

:class:`HeatConduction3D` is a placeholder for a JAX-FEM ``Problem`` that
would solve ``−∇·(k ∇T) = Q`` on the coil volume for thermoelastic coupling.
It is not implemented yet: instantiating it raises ``NotImplementedError``.
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
