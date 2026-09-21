Overall structure
=================

The ``coil-fem`` library has two main class types: 

1. ``CoilFEM``: The main runner for the FEM problem.
2. ``Support``: The container for support structure info.

Like most of JAX, ``coil-fem`` follows a functional philosophy. All information 
fed to ``CoilFEM`` and ``Support`` are constants/shapes that are known at compile
time, and kept static throughout an optimization. After initialization, ``CoilFEM``
exposes functional interfaces such as :meth:`~coil_fem.CoilFEM.objective` (for
optimisation) and :meth:`~coil_fem.CoilFEM.run` /
:meth:`~coil_fem.CoilFEM.to_vtu` (for diagnostics and visualisation).

The code performs integrated coil-support optimization via simsopt. 
``coil-fem`` provides two main simsopt classes:

1. ``CoilSupport``: A stateful ``Optimizable`` that stores the coil 
and support degrees of freedom.
2. ``CoilFEMObjective``: An objective that evaluates
:meth:`~coil_fem.CoilFEM.objective` through jitted :meth:`~coil_fem.simsopt.CoilFEMObjective.J`
/ :meth:`~coil_fem.simsopt.CoilFEMObjective.dJ`.  Diagnostics helpers such as
:meth:`~coil_fem.simsopt.CoilFEMObjective.run` call :meth:`~coil_fem.CoilFEM.run`
instead.

To implement a new type of support structure, one need to implement 
a pair of ``CoilSupport`` and ``Support`` child classes. 
