API reference
=============

Generated from module docstrings. Optional dependencies (``jax-fem``, ``simsopt``) are mocked when building docs without those packages installed.

Each object is documented once, at its public import path (the namespace it is
re-exported into, e.g. :class:`coil_fem.problems.DeviceProblem`).  The API pages
below are intentionally **not** recursive, so a defining submodule such as
``coil_fem.problems.device_problem`` never gets its own page; autodoc registers
that internal path only as a ``:canonical:`` alias.  This relies on ``__all__``
being defined in each package's ``__init__.py``.

.. currentmodule:: coil_fem

.. autosummary::
   :toctree: generated

   coil_fem
   meshing
   metrics
   geo
   problems
   simsopt
   solvers.cudss
