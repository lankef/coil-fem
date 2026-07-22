Overall structure
=================

The ``coil-fem`` library has two main class types: 

1. ``CoilFEM``: The main runner for the FEM problem.
2. ``Support``: The container for support structure info.

Like most of JAX, ``coil-fem`` follows a functional philosophy. All information 
fed to ``CoilFEM`` and ``Support`` are constants/shapes that are known at compile
time, and kept static throughout an optimization. After initialization, ``CoilFEM``
exposes a few functional interfaces, such as ``CoilFEM.run`` and ``CoilFEM.save_run_vtu``
that maps coil and support degrees of freedom to FEM solutions. 

The code performs integrated coil-support optimization via simsopt. 
``coil-fem`` provides two main simsopt classes:

1. ``CoilSupport``: A stateful ``Optimizable`` that stores the coil 
and support degrees of freedom.
2. ``CoilFEMObjective``: An objective function that calls ``CoilFEM.run()``
and caches the result for evaluating multiple objectives. 

To implement a new type of support structure, one need to implement 
a pait of ``CoilSupport`` and ``Support`` child classes. 