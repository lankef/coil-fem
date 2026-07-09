.. _theory-linearelasticity:

Linear elasticity
===========================================

:class:`~coil_fem.CoilFEM` solves the following linear elasticity problem 
for the small displacement of stellarator coils under gravity, Lorentz force,
and a specified uniform integral thermal contraction. The governing equations 
are as follows:

.. math::

   \begin{aligned}
       \nabla \cdot \boldsymbol{\sigma}(\mathbf{u}) + \mathbf{F}_\text{body} &= 0,
           & &\text{in } \Omega,
           & &\text{(Linear elasticity)} \\
       \boldsymbol{\sigma}(\mathbf{u})\,\mathbf{n} &= -k(\mathbf{x})\,\mathbf{u},
           & &\text{on } \partial\Omega,
           & &\text{(Robin/spring foundation BC)}
   \end{aligned}