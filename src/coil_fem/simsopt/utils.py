"""Simsopt–scipy helpers."""

from scipy.optimize import NonlinearConstraint


def constraint_from_optimizable(obj, lb, ub) -> NonlinearConstraint:
    """Build a ``NonlinearConstraint`` from an Optimizable with ``J``/``dJ``.

    Parameters
    ----------
    obj : Optimizable
        Object exposing ``x``, ``J()``, and ``dJ()``.
    lb, ub :
        Lower/upper bounds forwarded to ``NonlinearConstraint``.

    Returns
    -------
    NonlinearConstraint
    """
    def fun(x):
        obj.x = x
        return obj.J()

    def jac(x):
        obj.x = x
        return obj.dJ()

    return NonlinearConstraint(fun, lb, ub, jac=jac)
