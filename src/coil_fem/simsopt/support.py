"""
Simsopt-optimizable coil support models.

A *support* describes where a coil is structurally clamped.  In
:class:`~coil_fem.CoilFEM` this enters as a ``support_fn`` returning
per-surface-node Winkler spring weights in ``[0, 1]``::

    support_fn(surface_points, curve_jax, dofs) -> weights

This module wraps that function in a :class:`simsopt.Optimizable` so the
support parameters (``dofs``) can be co-optimised with coil geometry and
currents through a single :class:`~coil_fem.simsopt.objective.CoilFEMObjective`.

The optimisable parameters live **only** in the simsopt DOF store: the dofs
dict is flattened to the Optimizable's ``x`` at construction, and
:attr:`CoilSupport.support_dofs` reconstructs the dict on demand from
``local_full_x`` (never cached).  Fixed scalars (e.g. clamp radius) are kept
in :attr:`CoilSupport.constants` and are not optimised.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax.nn import sigmoid

from jax.flatten_util import ravel_pytree

try:
    from simsopt._core.optimizable import Optimizable
    _HAS_SIMSOPT = True
except ImportError:  # pragma: no cover
    Optimizable = object  # type: ignore[misc, assignment]
    _HAS_SIMSOPT = False


class CoilSupport(Optimizable):
    """Simsopt container for a single coil's support ``dofs`` + constants.

    The flattened ``dofs`` pytree *is* this Optimizable's degrees of freedom;
    there is no separate value attribute.  Subclasses implement the static
    :meth:`support_fn` (the core weight logic) and pass the optimisable
    ``dofs`` dict plus any fixed ``constants`` to ``__init__``.

    Parameters
    ----------
    support_dofs_jax : dict
        Optimisable support parameters (the ``dofs`` argument of
        :meth:`support_fn`).  Flattened into the simsopt DOF vector.
    constants : dict or None
        Fixed (non-optimised) scalars forwarded to :meth:`support_fn` as
        keyword arguments.
    names : list[str] or None
        Optional DOF names, length equal to the flattened ``support_dofs_jax``
        size.
    dofs : DOFs or None
        Simsopt ``DOFs`` object used to restore serialised state (passed by
        :meth:`from_dict` / :meth:`from_file`).  Leave as ``None`` when
        constructing fresh instances.
    """

    def __init__(
        self,
        support_dofs_jax: dict,
        constants: dict | None = None,
        names=None,
        dofs=None,
    ):
        if not _HAS_SIMSOPT:
            raise ImportError("simsopt is required for CoilSupport.")
        flat, self._unravel = ravel_pytree(support_dofs_jax)
        self.constants = dict(constants or {})
        # np.array (not asarray) so the DOF buffer is writable: converting a
        # JAX array via np.asarray yields a read-only view.
        if dofs is not None:
            Optimizable.__init__(self, dofs=dofs)
        else:
            Optimizable.__init__(self, x0=np.array(flat, dtype=float), names=names)

    @staticmethod
    def support_fn(surface_points, curve_jax, dofs, **constants):
        """Per-surface-node Winkler weights in ``[0, 1]`` (abstract).

        Parameters
        ----------
        surface_points : jax.Array, shape ``(n_surface_nodes, 3)``
        curve_jax : CurveXYZFourierJAX
        dofs : dict
            Optimisable support parameters (see :attr:`support_dofs`).
        **constants
            Fixed scalars from :attr:`constants`.
        """
        raise NotImplementedError

    @property
    def support_dofs(self) -> dict:
        """Current dofs as a differentiable JAX pytree, read from the DOFs."""
        return self._unravel(jnp.asarray(self.local_full_x))

    def support_callable(self):
        """Return a CoilFEM-compatible ``support_fn(sp, curve, dofs)``."""
        constants = self.constants
        fn = type(self).support_fn
        return lambda sp, curve, dofs: fn(sp, curve, dofs, **constants)

    def flatten_grad(self, grad_dofs: dict) -> np.ndarray:
        """Flatten a JAX gradient pytree into a simsopt DOF-aligned array."""
        return np.asarray(ravel_pytree(grad_dofs)[0], dtype=float)


class CoilSupportDiscrete(CoilSupport):
    """Support modelled as ``clamp_num`` clamps at optimisable arclengths.

    Each clamp is a sphere of radius ``clamp_radius`` centred on the coil at
    curve parameter ``phi``; the per-node weight is the (smooth) union of all
    clamp indicator spheres.  Only the clamp locations ``phis`` are DOFs;
    ``clamp_radius``, ``sigmoid_eps`` and ``clamp_num`` are fixed.

    Parameters
    ----------
    clamp_radius : float
        Sphere radius [m] of each clamp (region of non-zero spring weight).
    clamp_num : int
        Number of clamps (length of ``phis``).
    sigmoid_eps : float or None
        Edge sharpness of the clamp.  ``None`` (default) uses
        ``20.0 / clamp_radius``.
    phis : array-like or None
        Initial clamp locations in ``[0, 1)``.  ``None`` (default) spreads
        ``clamp_num`` clamps uniformly via ``linspace(0, 1, clamp_num,
        endpoint=False)``.
    names : list[str] or None
        Optional DOF names.
    dofs : DOFs or None
        Simsopt ``DOFs`` object for restoring serialised state
        (injected by :meth:`from_dict` / :meth:`from_file`).
    """

    def __init__(
        self,
        clamp_radius: float,
        clamp_num: int = 2,
        sigmoid_eps: float = 0.1,
        phis=None,
        names=None,
        dofs=None,
    ):

        clamp_radius = float(clamp_radius)
        clamp_num = int(clamp_num)
        if phis is None:
            phis = jnp.linspace(0.0, 1.0, clamp_num, endpoint=False)

        # Store all constructor args as instance attributes so that
        # GSONable.as_dict can introspect them for JSON serialisation.
        self.clamp_radius = clamp_radius
        self.clamp_num    = clamp_num
        self.sigmoid_eps = float(sigmoid_eps)
        self.phis         = phis
        self.names        = names

        super().__init__(
            support_dofs_jax={'phis': jnp.asarray(phis, dtype=float)},
            constants={
                'clamp_radius': clamp_radius,
                'sigmoid_eps': float(sigmoid_eps),
            },
            names=names,
            dofs=dofs,
        )

    @staticmethod
    def support_fn(surface_points, curve_jax, dofs, *, clamp_radius, sigmoid_eps):

        phis = dofs['phis']
        gamma_support = curve_jax.gamma_eval(phis)            # (n_support, 3)
        # distances = jnp.sqrt(jnp.sum(
        #     (surface_points[:, None, :] - gamma_support[None, :, :]) ** 2,
        #     axis=-1,
        # ) + 1e-10)                                            # (n_nodes, n_support)
        # w = sigmoid(sigmoid_eps * (clamp_radius - distances))
        distances = jnp.sum(
            (surface_points[:, None, :] - gamma_support[None, :, :]) ** 2,
            axis=-1,
        )                                           # (n_nodes, n_support)
        sigmoid_width = sigmoid_eps * clamp_radius
        w = sigmoid((clamp_radius**2 - distances)/(sigmoid_width**2))
        return jnp.sum(w, axis=-1)                            # union of clamps

class CoilSupportTopBottom(CoilSupport):
    """Static soft-sphere support at the top and bottom of the coil centreline.

    Has no optimisable DOFs (``support_dofs_jax={}``); ``clamp_radius`` and
    ``sigmoid_eps`` are fixed constants.
    """

    def __init__(self, clamp_radius, sigmoid_eps=0.1, dofs=None):
        self.clamp_radius = float(clamp_radius)
        self.sigmoid_eps = float(sigmoid_eps)

        super().__init__(
            support_dofs_jax={},
            constants={'clamp_radius': float(clamp_radius),
                       'sigmoid_eps': float(sigmoid_eps)},
            dofs=dofs,
        )

    @staticmethod
    def support_fn(surface_points, curve_jax, dofs, *, clamp_radius, sigmoid_eps):
        gamma  = curve_jax.gamma()                         # (n_quad, 3)
        top    = gamma[jnp.argmax(gamma[:, 2])]            # (3,) highest point
        bottom = gamma[jnp.argmin(gamma[:, 2])]            # (3,) lowest point

        # Safe norm: jnp.linalg.norm gradient is NaN at zero distance;
        # adding eps inside sqrt keeps the backward pass finite.
        d_top    = jnp.sum((surface_points - top)**2,    axis=-1)
        d_bottom = jnp.sum((surface_points - bottom)**2, axis=-1)

        # sigmoid(beta*(R-d)): ~1 inside sphere of radius clamp_radius, ~0 outside
        sigmoid_width = sigmoid_eps * clamp_radius
        w_top    = sigmoid((clamp_radius**2 - d_top)/(sigmoid_width**2))
        w_bottom = sigmoid((clamp_radius**2 - d_bottom)/(sigmoid_width**2))
        return w_top+w_bottom   # union of the two spheres
