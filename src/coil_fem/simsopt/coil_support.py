"""Simsopt base container for coil support DOFs.

:class:`CoilSupport` holds the base coils, ``nfp``, ``stellsym``, and a
pure-functional :class:`~coil_fem.coupling.Support` instance.  Optimisable
parameters live in the simsopt DOF store; :attr:`CoilSupport.support_dofs`
reconstructs them from ``local_full_x``.

Also provides shared helpers used by Fixed/Beams subclasses (angle broadcast,
``k_clamp`` defaults, and incremental ``dphis*`` encode/decode).
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from ..utils import estimate_k
from ..coupling import Support


# ============================================================================
# Sorted (incremental) angle DOFs: dphis* ↔ phis*
# ============================================================================

_DPHI_TO_PHI = {
    'dphis': 'phis',
    'dphis_start_cc': 'phis_start_cc',
    'dphis_end_cc': 'phis_end_cc',
    'dphis_start_cf': 'phis_start_cf',
}
_PHI_TO_DPHI = {v: k for k, v in _DPHI_TO_PHI.items()}
_ANGLE_UNIT_KEYS = frozenset(_DPHI_TO_PHI) | frozenset(_PHI_TO_DPHI)


def _diff_last(x):
    """Encode absolute angles as non-negative increments along the last axis."""
    return jnp.diff(x, axis=-1, prepend=jnp.zeros_like(x[..., :1]))


def _cumsum_last(x):
    """Decode increments to absolute angles along the last axis."""
    return jnp.cumsum(x, axis=-1)


def _cumsum_last_vjp(g):
    """VJP of :func:`_cumsum_last`: ``g_d[..., j] = sum_{k>=j} g_phi[..., k]``."""
    return jnp.cumsum(g[..., ::-1], axis=-1)[..., ::-1]


def _tree_diff_last(tree):
    return jax.tree_util.tree_map(_diff_last, tree)


def _tree_cumsum_last(tree):
    return jax.tree_util.tree_map(_cumsum_last, tree)


def _tree_cumsum_last_vjp(g_tree):
    return jax.tree_util.tree_map(_cumsum_last_vjp, g_tree)


def _encode_dphis(phi_dofs: dict) -> dict:
    """Functional ``phis*`` pytree → stored ``dphis*`` pytree (diff + rename)."""
    out = {}
    for k, v in phi_dofs.items():
        if k in _PHI_TO_DPHI:
            out[_PHI_TO_DPHI[k]] = _tree_diff_last(v)
        else:
            out[k] = v
    return out


def _decode_dphis(raw_dofs: dict) -> dict:
    """Stored ``dphis*`` pytree → functional ``phis*`` pytree (cumsum + rename)."""
    out = {}
    for k, v in raw_dofs.items():
        if k in _DPHI_TO_PHI:
            out[_DPHI_TO_PHI[k]] = _tree_cumsum_last(v)
        else:
            out[k] = v
    return out


def _vjp_dphis(grad_phi_dofs: dict) -> dict:
    """Grad w.r.t. ``phis*`` → grad w.r.t. ``dphis*`` (cumsum VJP + rename)."""
    out = {}
    for k, v in grad_phi_dofs.items():
        if k in _PHI_TO_DPHI:
            out[_PHI_TO_DPHI[k]] = _tree_cumsum_last_vjp(v)
        else:
            out[k] = v
    return out


class _SortedDphisMixin:
    """Decode stored ``dphis*`` for the FEM; pull grads back in ``flatten_grad``."""

    @property
    def support_dofs(self) -> dict:
        return _decode_dphis(self._unravel(jnp.asarray(self.local_full_x)))

    def flatten_grad(self, grad_dofs: dict) -> np.ndarray:
        return np.asarray(ravel_pytree(_vjp_dphis(grad_dofs))[0], dtype=float)


# ============================================================================
def _generate_k_clamp(base_coils, fixed_clamp_options):
    """ Defaults for the fixed clamp's Robin/Winkler spring coefficients.
    
    A fixed clamp's Robin/Winkler spring coefficients can be auto-generated
    based on the beam's stiffness. This method generates them from the coil's 
    stiffness. 
    """
    if 'k_clamp' in fixed_clamp_options.keys():
        return fixed_clamp_options['k_clamp']
    else:
        # The beams can bend and therefore can numerically
        # tolerate a much higher regularization factor than
        # the fixed clamps. A higher factor by at least 2^4 is 
        # needed because beams can be 2x more narrow than coils.
        # TODO: it may be possible to change this factor 
        # dynamically in an optimization based on the 
        # support beam thickness.
        eps_clamp = fixed_clamp_options.get('eps_clamp', 1e-5)
        try:
            E_coil = fixed_clamp_options['E_coil']
        except:
            raise KeyError(
                "E_coil not detected in fixed_clamp_options. "
                "When k_clamp is not provided, the Young's modulus "
                "of the coils, E_coil, must be provided so that a the value "
                "of k_clamp can be auto-generated based on the coil's stiffness."
            )
        mean_arclengths = np.mean(
            # This calculates the total length of each coil
            [jnp.mean(c.curve.incremental_arclength()) for c in base_coils]
        )
        L_coil = mean_arclengths / np.pi / 2
        k_clamp = estimate_k(L=L_coil, E=E_coil, eps=eps_clamp)
        print(
            "k_clamp is not provided. Based on the coil's stiffness, "
            f"the auto-generated value is      {k_clamp:.4e} N/m3."
        )
        return k_clamp


try:
    from simsopt._core.optimizable import Optimizable
    _HAS_SIMSOPT = True
except ImportError:  # pragma: no cover
    Optimizable = object  # type: ignore[misc, assignment]
    _HAS_SIMSOPT = False


class CoilSupport(Optimizable):
    """Simsopt container for a coilset's support DOFs, constants, and coil refs.

    Holds the base coils (curves + currents), the symmetry parameters ``nfp``
    and ``stellsym``, and any optimisable support parameters (the ``dofs``
    pytree).  The functional support model is stored in :attr:`support` and is
    passed directly to :class:`~coil_fem.CoilFEM`; subclasses do **not** need
    to inherit :class:`~coil_fem.coupling.Support` themselves.

    Parameters
    ----------
    base_coils : list
        Simsopt ``Coil`` objects (each exposing ``.curve`` and ``.current``).
        These are the *base* coils — before symmetry expansion.
    nfp : int
        Number of field periods.
    stellsym : bool
        Whether to apply stellarator symmetry during the symmetry expansion.
    support : Support
        Pure-functional support instance (e.g. a bare
        :class:`~coil_fem.coupling.Support` with ``fixed_clamp_fns``, or a
        :class:`~coil_fem.coupling.SupportBeams`).  Passed verbatim to
        :class:`~coil_fem.CoilFEM` via :attr:`support`.
    support_dofs_jax : dict
        Optimisable support parameters.  Flattened into the simsopt DOF vector.
    constants : dict or None
        Fixed (non-optimised) scalars for introspection (e.g. ``r_clamp``).
        Not forwarded to the functional support; that object owns its own
        constants.
    names : list[str] or None
        Optional DOF names, length equal to the flattened ``support_dofs_jax``
        size.
    fixed : array-like or None
        Boolean fixed-mask aligned with the flattened DOF vector.
    lower_bounds : array-like or None
        Lower bounds aligned with the flattened DOF vector (default ``-inf``).
    upper_bounds : array-like or None
        Upper bounds aligned with the flattened DOF vector (default ``+inf``).
    dofs : DOFs or None
        Simsopt ``DOFs`` object used to restore serialised state (passed by
        :meth:`from_dict` / :meth:`from_file`).  Leave as ``None`` when
        constructing fresh instances.
    """

    def __init__(
        self,
        base_coils: list,
        nfp: int,
        stellsym: bool,
        support: Support,
        support_dofs_jax: dict,
        constants: dict | None = None,
        names=None,
        fixed=None,
        lower_bounds=None,
        upper_bounds=None,
        dofs=None,
    ):
        if not _HAS_SIMSOPT:
            raise ImportError("simsopt is required for CoilSupport.")

        self._base_coils = list(base_coils)
        self.nfp = int(nfp)
        self.stellsym = bool(stellsym)
        self._support = support

        flat, self._unravel = ravel_pytree(support_dofs_jax)
        self.constants = dict(constants or {})
        if dofs is not None:
            Optimizable.__init__(self, dofs=dofs, depends_on=self._base_coils)
        else:
            if names is None:
                names = self._make_names(support_dofs_jax)
            Optimizable.__init__(
                self,
                x0=np.array(flat, dtype=float),
                names=names,
                fixed=fixed,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                depends_on=self._base_coils,
            )

    # ============================================================================
    # Functional support accessor
    # ============================================================================

    @property
    def support(self) -> Support:
        """The pure-functional :class:`~coil_fem.coupling.Support` instance."""
        return self._support

    # ============================================================================
    # Coil accessors
    # ============================================================================

    @property
    def base_curves(self) -> list:
        """Simsopt curve objects for the base coils (before symmetry expansion)."""
        return [c.curve for c in self._base_coils]

    @property
    def base_currents(self) -> list:
        """Simsopt current objects for the base coils."""
        return [c.current for c in self._base_coils]

    @property
    def n_coils(self) -> int:
        """Number of base coils."""
        return len(self._base_coils)

    # ============================================================================
    # DOF accessors
    # ============================================================================

    @property
    def support_dofs(self) -> dict:
        """Current dofs as a differentiable JAX pytree, read from the DOFs."""
        return self._unravel(jnp.asarray(self.local_full_x))

    def flatten_grad(self, grad_dofs: dict) -> np.ndarray:
        """Flatten a JAX gradient pytree into a simsopt DOF-aligned array."""
        return np.asarray(ravel_pytree(grad_dofs)[0], dtype=float)

    def _make_names(self, support_dofs_jax: dict) -> list[str]:
        """Build simsopt DOF names from a ``support_dofs`` pytree path.

        Names follow the flatten order of :func:`jax.flatten_util.ravel_pytree`.
        List/array indices are written as ``key(i,j,...)``.

        Parameters
        ----------
        support_dofs_jax : dict
            Optimisable support DOF pytree (same structure as
            :attr:`support_dofs`).

        Returns
        -------
        list of str
            One name per scalar in the flattened DOF vector.
        """
        names: list[str] = []
        paths_and_leaves, _ = jax.tree_util.tree_flatten_with_path(support_dofs_jax)
        for path, leaf in paths_and_leaves:
            key = None
            prefix: list[int] = []
            for part in path:
                if isinstance(part, jax.tree_util.DictKey):
                    key = part.key
                elif isinstance(part, jax.tree_util.SequenceKey):
                    prefix.append(part.idx)
            assert key is not None
            arr = np.asarray(leaf)
            for idx in np.ndindex(arr.shape):
                full = (*prefix, *idx)
                names.append(f"{key}({','.join(map(str, full))})")
        return names

    def _make_bounds(
        self,
        support_dofs_jax: dict,
        unit_interval_keys=(),
        nonnegative_keys=(),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build full-length lower/upper bounds for the Optimizable based on keys in support_dofs.

        Bounds are raveled in the same order as
        :func:`jax.flatten_util.ravel_pytree` / :meth:`_make_names`.

        Parameters
        ----------
        support_dofs_jax : dict
            Optimisable support DOF pytree.
        unit_interval_keys : iterable of str
            Keys bounded to ``[0, 1]``.
        nonnegative_keys : iterable of str
            Keys bounded to ``[0, +inf)``.

        Returns
        -------
        lower_bounds, upper_bounds : ndarray
            Full-length bound arrays aligned with the flattened DOF vector.
        """
        unit = set(unit_interval_keys)
        nonneg = set(nonnegative_keys)

        def _lb_leaf(leaf, key):
            shape = np.shape(leaf)
            if key in unit or key in nonneg:
                return np.zeros(shape, dtype=float)
            return np.full(shape, -np.inf, dtype=float)

        def _ub_leaf(leaf, key):
            shape = np.shape(leaf)
            if key in unit:
                return np.ones(shape, dtype=float)
            return np.full(shape, np.inf, dtype=float)

        lb_tree = {
            k: jax.tree_util.tree_map(lambda leaf, kk=k: _lb_leaf(leaf, kk), v)
            for k, v in support_dofs_jax.items()
        }
        ub_tree = {
            k: jax.tree_util.tree_map(lambda leaf, kk=k: _ub_leaf(leaf, kk), v)
            for k, v in support_dofs_jax.items()
        }
        lb, _ = ravel_pytree(lb_tree)
        ub, _ = ravel_pytree(ub_tree)
        return np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)


def _broadcast_phis(phis, n_coils, n_clamp):
    if phis is None:
        row = jnp.linspace(0.0, 1.0, n_clamp, endpoint=False)
        phis_arr = jnp.broadcast_to(row, (n_coils, n_clamp))
    else:
        phis_arr = jnp.asarray(phis, dtype=float)
        if phis_arr.ndim == 1:
            phis_arr = jnp.broadcast_to(phis_arr, (n_coils, n_clamp))
    return phis_arr


