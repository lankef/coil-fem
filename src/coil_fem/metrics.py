"""Von Mises stress and strain-energy metrics on JAX-FEM solutions.

All public functions accept optional ``shape_grads`` and ``JxW`` keyword
arguments.  When provided (as recomputed by :meth:`~coil_fem.CoilFEM.objective`
outside the ``ad_wrapper`` boundary), gradients flow through mesh geometry
correctly.  When ``None``, the function falls back to ``problem.shape_grads``
/ ``problem.JxW``, which is fine for forward-only diagnostics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def cauchy_stress_small_strain(
    u_grad: jnp.ndarray, lam: float, mu: float, *, epsilon_th=None
) -> jnp.ndarray:
    """Cauchy stress via Hooke's law: σ = λ tr(ε_m) I + 2μ ε_m.

    When ``epsilon_th`` is provided, the mechanical strain
    ``ε_m = ε − ε_th`` is used in place of the total strain ``ε``.

    Parameters
    ----------
    u_grad : jnp.ndarray, shape (..., 3, 3)
        Displacement gradient at each quadrature point.
    lam, mu : float
        Lamé parameters.
    epsilon_th : jnp.ndarray or None
        Constant thermal eigenstrain ``(3, 3)``; ``None`` for isothermal.

    Returns
    -------
    jnp.ndarray, shape (..., 3, 3)
        Cauchy stress tensor.
    """
    eps = 0.5 * (u_grad + jnp.swapaxes(u_grad, -1, -2))
    eps_m = eps - epsilon_th if epsilon_th is not None else eps
    tr = jnp.trace(eps_m, axis1=-2, axis2=-1)
    eye = jnp.eye(3, dtype=u_grad.dtype)
    return lam * tr[..., None, None] * eye + 2.0 * mu * eps_m


def _resolve_shape_grads(problem, shape_grads):
    """Return shape_grads (num_cells, num_quads, num_nodes, dim)."""
    if shape_grads is not None:
        return shape_grads
    return jnp.asarray(problem.shape_grads)


def _resolve_JxW(problem, JxW):
    """Return JxW (num_cells, num_quads)."""
    if JxW is not None:
        return JxW
    return jnp.asarray(problem.JxW[:, 0, :])


def _resolve_epsilon_th(problem, epsilon_th):
    """Return thermal eigenstrain from explicit kwarg or from ``problem.epsilon_th``.

    Returns ``None`` when neither source provides a value, meaning no thermal
    correction is applied.
    """
    if epsilon_th is not None:
        return epsilon_th
    return getattr(problem, 'epsilon_th', None)


def displacement_gradient_at_quads(
    sol: jnp.ndarray, problem, *, shape_grads=None,
) -> jnp.ndarray:
    """Displacement gradient at every FEM quadrature point.

    Parameters
    ----------
    sol : jnp.ndarray, shape (n_nodes, 3)
        Displacement field.
    problem : LinearElasticity3D
    shape_grads : jnp.ndarray or None
        Physical-space shape-function gradients ``(n_cells, n_quads,
        n_nodes, dim)``.  ``None`` falls back to ``problem.shape_grads``.

    Returns
    -------
    jnp.ndarray, shape (n_cells, n_quads, 3, 3)
        Displacement gradient ``du_i/dx_j`` at each quadrature point.
    """
    fe = problem.fes[0]
    cells = jnp.asarray(fe.cells)
    sg = _resolve_shape_grads(problem, shape_grads)
    cells_sol = sol[cells]
    return jnp.einsum("cnv,cqnd->cqvd", cells_sol, sg)


def von_mises_on_quadrature(
    problem, sol_list, lam: float, mu: float,
    *, shape_grads=None, epsilon_th=None,
) -> jnp.ndarray:
    """Von Mises stress at every FEM quadrature point.

    Parameters
    ----------
    problem : LinearElasticity3D
    sol_list : list[jnp.ndarray]
        ``ad_wrapper`` output; ``sol_list[0]`` has shape ``(n_nodes, 3)``.
    lam, mu : float
        Lamé parameters.
    shape_grads : jnp.ndarray or None
        Physical shape-function gradients; see module docstring.
    epsilon_th : jnp.ndarray or None
        Thermal eigenstrain ``(3, 3)``.  Defaults to ``problem.epsilon_th``.

    Returns
    -------
    jnp.ndarray, shape (n_cells, n_quads)
        Von Mises stress [Pa].
    """
    sol = jnp.asarray(sol_list[0])
    u_grad = displacement_gradient_at_quads(sol, problem, shape_grads=shape_grads)
    eps_th = _resolve_epsilon_th(problem, epsilon_th)
    sigma = cauchy_stress_small_strain(u_grad, lam, mu, epsilon_th=eps_th)
    # Scalar Von Mises stress from Cauchy tensor, shape (..., 3, 3) -> (...).
    tr = jnp.trace(sigma, axis1=-2, axis2=-1) / 3.0
    dev = sigma - tr[..., None, None] * jnp.eye(3, dtype=sigma.dtype)
    j2 = 0.5 * jnp.sum(dev * dev, axis=(-2, -1))
    vm_sq = 3.0 * j2
    return jnp.sqrt(vm_sq + 1e-30)


def mean_von_mises_volume_weighted(
    problem, sol_list, lam: float, mu: float,
    *, shape_grads=None, JxW=None, epsilon_th=None,
) -> jnp.ndarray:
    """Volume-weighted mean von Mises stress: ∫ σ_vm dV / ∫ dV.

    Parameters
    ----------
    problem : LinearElasticity3D
    sol_list : list[jnp.ndarray]
    lam, mu : float
    shape_grads : jnp.ndarray or None
    JxW : jnp.ndarray or None
        Quadrature weights ``(n_cells, n_quads)``.
    epsilon_th : jnp.ndarray or None

    Returns
    -------
    jnp.ndarray
        Scalar volume-weighted mean von Mises stress [Pa].
    """
    vm = von_mises_on_quadrature(
        problem, sol_list, lam, mu, shape_grads=shape_grads, epsilon_th=epsilon_th,
    )
    jxw = _resolve_JxW(problem, JxW)
    num = jnp.sum(vm * jxw)
    den = jnp.sum(jxw)
    return num / den


def max_von_mises_hard(
    problem, sol_list, lam: float, mu: float,
    *, shape_grads=None, JxW=None, epsilon_th=None,
) -> jnp.ndarray:
    """Hard maximum von Mises stress over all quadrature points.

    Parameters
    ----------
    problem : LinearElasticity3D
    sol_list : list[jnp.ndarray]
    lam, mu : float
    shape_grads : jnp.ndarray or None
    JxW : jnp.ndarray or None
    epsilon_th : jnp.ndarray or None

    Returns
    -------
    jnp.ndarray
        Scalar peak von Mises stress [Pa].
    """
    return jnp.max(von_mises_on_quadrature(
        problem, sol_list, lam, mu, shape_grads=shape_grads, epsilon_th=epsilon_th,
    ))


def max_von_mises_lse(
    problem,
    sol_list,
    lam: float,
    mu: float,
    *,
    beta: float = 20.0,
    shape_grads=None,
    JxW=None,
    epsilon_th=None,
) -> jnp.ndarray:
    r"""Smooth maximum von Mises stress via log-sum-exp.

    Computes :math:`\frac{1}{\beta}\log\sum_q e^{\beta \sigma_{vm,q}}` over
    all quadrature points.  Differentiable everywhere; approaches the hard
    maximum as ``beta`` → ∞.

    Parameters
    ----------
    problem : LinearElasticity3D
    sol_list : list[jnp.ndarray]
    lam, mu : float
    beta : float
        Smoothing parameter (default 20.0).
    shape_grads : jnp.ndarray or None
    JxW : jnp.ndarray or None
    epsilon_th : jnp.ndarray or None

    Returns
    -------
    jnp.ndarray
        Scalar smooth-maximum von Mises stress [Pa].
    """
    vm = von_mises_on_quadrature(
        problem, sol_list, lam, mu, shape_grads=shape_grads, epsilon_th=epsilon_th,
    ).ravel()
    return (1.0 / beta) * jax.nn.logsumexp(beta * vm)


def l2_von_mises(
    problem,
    sol_list,
    lam: float,
    mu: float,
    *,
    shape_grads=None,
    JxW=None,
    epsilon_th=None,
) -> jnp.ndarray:
    r"""Squared-L2 von Mises norm: :math:`\int_V \sigma_{vm}^2\,dV`.

    Parameters
    ----------
    problem : LinearElasticity3D
    sol_list : list[jnp.ndarray]
    lam, mu : float
    shape_grads : jnp.ndarray or None
    JxW : jnp.ndarray or None
    epsilon_th : jnp.ndarray or None

    Returns
    -------
    jnp.ndarray
        Scalar :math:`\int \sigma_{vm}^2\,dV` [Pa² m³].
    """
    vm = von_mises_on_quadrature(
        problem, sol_list, lam, mu, shape_grads=shape_grads, epsilon_th=epsilon_th,
    )
    jxw = _resolve_JxW(problem, JxW)
    return jnp.sum(vm**2 * jxw)


def total_strain_energy(
    problem, sol_list, lam: float, mu: float,
    *, shape_grads=None, JxW=None, epsilon_th=None,
) -> jnp.ndarray:
    r"""Total elastic strain energy :math:`\frac{1}{2}\int \boldsymbol{\sigma}:\boldsymbol{\varepsilon}_m\,dV`.

    Parameters
    ----------
    problem : LinearElasticity3D
    sol_list : list[jnp.ndarray]
    lam, mu : float
    shape_grads : jnp.ndarray or None
    JxW : jnp.ndarray or None
    epsilon_th : jnp.ndarray or None

    Returns
    -------
    jnp.ndarray
        Scalar strain energy [J].
    """
    # Lazy import to avoid a module-load cycle: coil_fem imports metrics at the
    # top level, while the strain-energy-density kernel lives on CoilFEM.
    from .coil_fem import CoilFEM

    sol = jnp.asarray(sol_list[0])
    u_grad = displacement_gradient_at_quads(sol, problem, shape_grads=shape_grads)
    eps_th = _resolve_epsilon_th(problem, epsilon_th)
    psi = CoilFEM.strain_energy_density(u_grad, lam, mu, epsilon_th=eps_th)
    jxw = _resolve_JxW(problem, JxW)
    return jnp.sum(psi * jxw)
