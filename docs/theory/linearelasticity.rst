.. _theory-linearelasticity:

Linear elasticity
===========================================

:class:`~coil_fem.CoilFEM` solves the following linear elasticity problem 
for the small displacement of stellarator coils under gravity, Lorentz force,
and a specified uniform integral thermal contraction. The governing equations 
are as follows:

.. math::
   :nowrap:

   \begin{alignat}{3}
       \nabla \cdot \boldsymbol{\sigma}(\mathbf{u}) + \mathbf{F}_\text{body} &= 0,
           &&\quad \text{in } \Omega
           &&\quad \text{(Linear elasticity)} \\
       \boldsymbol{\sigma}(\mathbf{u})\,\mathbf{n} &= -k(\mathbf{x})\,\mathbf{u}.
           &&\quad \text{on } \partial\Omega
           &&\quad \text{(Robin/spring foundation BC)}
   \end{alignat}