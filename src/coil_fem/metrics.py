"""Stress / Von Mises post-processing on JAX-FEM solutions.

Important: geometry arrays
--------------------------
Functions in this module need physical-space shape-function gradients
(``shape_grads``) and quadrature weights (``JxW``).  These must reflect
the **current** mesh geometry, not the initial mesh.

When the mesh is built from periodic curves, the initial mesh often contains
degenerate elements at the periodic seam (duplicate endpoint nodes), making
the initial ``fe.shape_grads`` / ``fe.JxW`` unusable.

More subtly, ``problem.shape_grads`` is updated as a *side-effect* of
``set_params`` which is called **inside** ``ad_wrapper``'s ``custom_vjp``
forward.  Reading those arrays after the ``ad_wrapper`` call leaks a traced
value from inside the ``custom_vjp`` scope, which causes NaN gradients
because the ``custom_vjp`` backward does not account for the leaked
dependency.

Therefore every function that needs geometry arrays accepts optional
``shape_grads`` and ``JxW`` keyword arguments.  The caller
(:meth:`CoilFEM.objective`) recomputes these **outside** the
``custom_vjp`` boundary so that JAX can differentiate through them
normally via standard AD.  When the kwargs are ``None`` (e.g. standalone
diagnostic use), the function falls back to ``problem.shape_grads`` /
``problem.JxW`` which is correct for forward evaluation but not for AD.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def cauchy_stress_small_strain(
    u_grad: jnp.ndarray, lam: float, mu: float, *, epsilon_th=None
) -> jnp.ndarray:
    """Hooke law σ = λ tr(ε_m) I + 2μ ε_m; ``u_grad`` shape (..., 3, 3).

    When ``epsilon_th`` is provided, the mechanical strain
    ``ε_m = ε − ε_th`` is used in place of the total strain ``ε``.
    """
    eps = 0.5 * (u_grad + jnp.swapaxes(u_grad, -1, -2))
    eps_m = eps - epsilon_th if epsilon_th is not None else eps
    tr = jnp.trace(eps_m, axis1=-2, axis2=-1)
    eye = jnp.eye(3, dtype=u_grad.dtype)
    return lam * tr[..., None, None] * eye + 2.0 * mu * eps_m


def von_mises_stress(sigma: jnp.ndarray) -> jnp.ndarray:
    """Scalar Von Mises stress from Cauchy tensor, shape (..., 3, 3) -> (...)."""
    tr = jnp.trace(sigma, axis1=-2, axis2=-1) / 3.0
    dev = sigma - tr[..., None, None] * jnp.eye(3, dtype=sigma.dtype)
    j2 = 0.5 * jnp.sum(dev * dev, axis=(-2, -1))
    vm_sq = 3.0 * j2
    return jnp.sqrt(vm_sq + 1e-30)


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
    """
    ``sol`` (n_nodes, 3); returns ``u_grad`` (n_cells, n_quads, 3, 3).

    Parameters
    ----------
    shape_grads : jnp.ndarray or None
        Physical-space shape-function gradients ``(n_cells, n_quads,
        n_nodes, dim)`` computed **outside** ``ad_wrapper`` so JAX can
        differentiate through them.  ``None`` falls back to
        ``problem.shape_grads`` (fine for forward-only evaluation, but
        yields NaN gradients when used inside ``jax.grad``).
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
    """Per-quadrature Von Mises, shape (n_cells, n_quads).

    ``epsilon_th`` defaults to ``problem.epsilon_th`` when the problem carries
    a thermal eigenstrain; pass explicitly to override.
    """
    sol = jnp.asarray(sol_list[0])
    u_grad = displacement_gradient_at_quads(sol, problem, shape_grads=shape_grads)
    eps_th = _resolve_epsilon_th(problem, epsilon_th)
    sig = cauchy_stress_small_strain(u_grad, lam, mu, epsilon_th=eps_th)
    return von_mises_stress(sig)


def mean_von_mises_volume_weighted(
    problem, sol_list, lam: float, mu: float,
    *, shape_grads=None, JxW=None, epsilon_th=None,
) -> jnp.ndarray:
    """Volume-weighted mean ∫ σ_vm dV / ∫ dV using FE quadrature weights."""
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
    r"""Smooth max via \(\frac{1}{\beta}\log\sum e^{\beta \sigma_{vm}}\) over quads."""
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
    r"""Squared-L2 von Mises: \(\int_V \sigma_{vm}^2 \, dV\) via FE quadrature."""
    vm = von_mises_on_quadrature(
        problem, sol_list, lam, mu, shape_grads=shape_grads, epsilon_th=epsilon_th,
    )
    jxw = _resolve_JxW(problem, JxW)
    return jnp.sum(vm**2 * jxw)


def strain_energy_density(
    u_grad: jnp.ndarray, lam: float, mu: float, *, epsilon_th=None
) -> jnp.ndarray:
    """0.5 σ : ε_m — elastic (mechanical) strain energy density per quadrature point.

    Uses mechanical strain ``ε_m = ε − ε_th`` when ``epsilon_th`` is provided,
    so thermal pre-strain does not spuriously contribute to the elastic energy.
    """
    eps = 0.5 * (u_grad + jnp.swapaxes(u_grad, -1, -2))
    eps_m = eps - epsilon_th if epsilon_th is not None else eps
    sig = cauchy_stress_small_strain(u_grad, lam, mu, epsilon_th=epsilon_th)
    return 0.5 * jnp.sum(sig * eps_m, axis=(-2, -1))


def total_strain_energy(
    problem, sol_list, lam: float, mu: float,
    *, shape_grads=None, JxW=None, epsilon_th=None,
) -> jnp.ndarray:
    sol = jnp.asarray(sol_list[0])
    u_grad = displacement_gradient_at_quads(sol, problem, shape_grads=shape_grads)
    eps_th = _resolve_epsilon_th(problem, epsilon_th)
    psi = strain_energy_density(u_grad, lam, mu, epsilon_th=eps_th)
    jxw = _resolve_JxW(problem, JxW)
    return jnp.sum(psi * jxw)
