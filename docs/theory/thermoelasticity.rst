.. _theory-thermoelasticity:

Thermoelastic Structural Analysis of Coils
===========================================

This page derives the boundary-value problem (BVP) that
:class:`~coil_fem.elasticity.LinearElasticity3D` solves, including the
treatment of thermal contraction when the coil is cooled from its initial
(stress-free) temperature to its final service temperature.

----

Governing Equation
------------------

We seek the quasi-static displacement field
:math:`\mathbf{u} : \Omega \to \mathbb{R}^3`
in the coil volume :math:`\Omega` satisfying

.. math::
   :label: eq-equilibrium

   -\nabla \cdot \boldsymbol{\sigma}(\mathbf{u}) = \mathbf{b}
   \quad \text{in } \Omega,

subject to Dirichlet conditions
:math:`\mathbf{u} = \mathbf{0}` on the clamped boundary
:math:`\Gamma_D`, and a Winkler spring-foundation condition (see
:ref:`theory-winkler`) on the remainder of the boundary.

Here :math:`\boldsymbol{\sigma}` is the Cauchy stress tensor and
:math:`\mathbf{b}` is the body-force density [N m\ :sup:`−3`].

----

Small-Strain Kinematics
-----------------------

Under the small-strain assumption the total strain tensor is

.. math::
   :label: eq-total-strain

   \boldsymbol{\varepsilon}(\mathbf{u})
   = \tfrac{1}{2}\!\left(
       \nabla\mathbf{u} + (\nabla\mathbf{u})^{\mathsf{T}}
     \right).

----

Additive Decomposition and Thermal Eigenstrain
-----------------------------------------------

When the coil is cooled from a stress-free reference temperature
:math:`T_\text{init}` to a final service temperature :math:`T_\text{final}`, an
isotropic thermal eigenstrain develops.  Under the small-strain additive
decomposition

.. math::
   :label: eq-additive

   \boldsymbol{\varepsilon}
   = \boldsymbol{\varepsilon}_m + \boldsymbol{\varepsilon}_\text{th},

where the **mechanical strain**

.. math::
   :label: eq-mech-strain

   \boldsymbol{\varepsilon}_m
   = \boldsymbol{\varepsilon}(\mathbf{u}) - \boldsymbol{\varepsilon}_\text{th}

is the strain that drives stress, and the **thermal eigenstrain** is

.. math::
   :label: eq-thermal-strain

   \boldsymbol{\varepsilon}_\text{th}
   = \alpha\,\Delta T\,\mathbf{I},
   \qquad
   \Delta T = T_\text{final} - T_\text{init}.

Here :math:`\alpha` [K\ :sup:`−1`] is the isotropic coefficient of thermal
expansion and :math:`\mathbf{I}` is the :math:`3\times 3` identity.  For
cooling (:math:`T_\text{final} < T_\text{init}`) we have :math:`\Delta T < 0`,
so :math:`\boldsymbol{\varepsilon}_\text{th}` is a contraction.

.. note::

   Both temperatures are **fixed scalar constants**, not optimisable degrees of
   freedom.  The eigenstrain :math:`\boldsymbol{\varepsilon}_\text{th}` is
   pre-computed once at construction time; no temperature adjoint is required.
   This is appropriate for a stellarator coil whose final service temperature is
   prescribed by the cryogenic system.

----

Constitutive Law
-----------------

For a linear isotropic material with Lamé parameters
:math:`\lambda` and :math:`\mu` (related to Young's modulus :math:`E` and
Poisson's ratio :math:`\nu` by
:math:`\lambda = E\nu / [(1+\nu)(1-2\nu)]`,
:math:`\mu = E / [2(1+\nu)]`),
Hooke's law applied to the **mechanical strain** gives

.. math::
   :label: eq-hooke

   \boldsymbol{\sigma}
   = \lambda\,\mathrm{tr}(\boldsymbol{\varepsilon}_m)\,\mathbf{I}
     + 2\mu\,\boldsymbol{\varepsilon}_m.

Substituting :eq:`eq-mech-strain` and :eq:`eq-thermal-strain`:

.. math::
   :label: eq-stress-full

   \boldsymbol{\sigma}
   = \lambda\,\mathrm{tr}(\boldsymbol{\varepsilon})\,\mathbf{I}
     + 2\mu\,\boldsymbol{\varepsilon}
     - \underbrace{(3\lambda + 2\mu)\,\alpha\,\Delta T}_{\kappa\,\Delta T}
       \,\mathbf{I},

where :math:`\kappa = (3\lambda + 2\mu)\,\alpha` is the thermoelastic coupling
modulus.  The last term is a uniform hydrostatic **pre-stress** whose sign is
compressive for cooling (:math:`\Delta T < 0`).

When no thermal parameters are supplied (:math:`\alpha = 0` or temperatures
are not specified), the formulation reduces to the isothermal case:

.. math::

   \boldsymbol{\sigma}
   = \lambda\,\mathrm{tr}(\boldsymbol{\varepsilon})\,\mathbf{I}
     + 2\mu\,\boldsymbol{\varepsilon}.

----

Body Forces
-----------

The body-force density :math:`\mathbf{b}` appearing in :eq:`eq-equilibrium`
combines two contributions:

.. math::
   :label: eq-body-force

   \mathbf{b} = \mathbf{f}_\text{Lorentz} + \mathbf{f}_\text{grav}.

**Lorentz body force.**  The total magnetic field
:math:`\mathbf{B} = \mathbf{B}_\text{self} + \mathbf{B}_\text{mutual}`
at each arc-length point on the coil centreline is computed from the
regularised Biot-Savart law (Landreman–Hurwitz self-field).  The resulting
line force density :math:`\boldsymbol{\kappa}` [N m\ :sup:`−1`] is converted
to a volume density by dividing by the cross-sectional area :math:`A`:

.. math::

   \mathbf{f}_\text{Lorentz}
   = \frac{I\,\mathbf{t}' \times \mathbf{B}}{A}
   \quad [\text{N m}^{-3}],

where :math:`I` is the coil current, :math:`\mathbf{t}' = d\boldsymbol{\gamma}/d\phi`
is the (unnormalised) tangent, and the assignment of the centreline force to
each finite element is topological (each cell inherits the value of its
:math:`\phi`-slice).

**Gravitational body force.**

.. math::

   \mathbf{f}_\text{grav} = \rho\,\mathbf{g}
   \quad [\text{N m}^{-3}],

where :math:`\rho` is the mass density and
:math:`\mathbf{g} = (0, 0, -g)` with :math:`g = 9.807\,\text{m s}^{-2}`.

----

Weak Form and FEM Discretisation
----------------------------------

Multiplying :eq:`eq-equilibrium` by a test function
:math:`\mathbf{v} \in H^1_0(\Omega)^3` and integrating by parts yields:

.. math::
   :label: eq-weak

   \int_\Omega \boldsymbol{\sigma}(\mathbf{u}) : \boldsymbol{\varepsilon}(\mathbf{v})
   \,\mathrm{d}\Omega
   = \int_\Omega \mathbf{b} \cdot \mathbf{v} \,\mathrm{d}\Omega
   + \int_{\Gamma_N} \mathbf{t} \cdot \mathbf{v} \,\mathrm{d}S,

where :math:`\mathbf{t}` is the prescribed surface traction on the natural
boundary :math:`\Gamma_N`.

Expanding the stress using :eq:`eq-hooke` and :eq:`eq-mech-strain`,
the left-hand side splits into a **stiffness term** that depends on
:math:`\mathbf{u}`, and a **thermal load term** that depends only on the
pre-computed eigenstrain:

.. math::
   :label: eq-split

   \underbrace{
     \int_\Omega
       \left[
         \lambda\,\mathrm{tr}(\boldsymbol{\varepsilon}(\mathbf{u}))\,\mathbf{I}
         + 2\mu\,\boldsymbol{\varepsilon}(\mathbf{u})
       \right]
       : \boldsymbol{\varepsilon}(\mathbf{v})
     \,\mathrm{d}\Omega
   }_{\mathbf{K}\,\mathbf{u}}
   =
   \int_\Omega \mathbf{b} \cdot \mathbf{v} \,\mathrm{d}\Omega
   +
   \underbrace{
     \int_\Omega
       \left(
         \lambda\,\mathrm{tr}(\boldsymbol{\varepsilon}_\text{th})\,\mathbf{I}
         + 2\mu\,\boldsymbol{\varepsilon}_\text{th}
       \right)
       : \boldsymbol{\varepsilon}(\mathbf{v})
     \,\mathrm{d}\Omega
   }_{\mathbf{f}_\text{th}}.

Because the stiffness :math:`\mathbf{K}` does not depend on
:math:`\boldsymbol{\varepsilon}_\text{th}`, the FEM tangent (and hence the
linear solver) is **identical** to the purely isothermal problem.  The thermal
contraction enters solely through the modified constitutive evaluation
(subtracting :math:`\boldsymbol{\varepsilon}_\text{th}` before applying
Hooke's law), which shifts the residual without changing the stiffness matrix.
Consequently, no additional solver infrastructure is needed.

.. _theory-winkler:

Winkler Spring-Foundation Boundary Condition
---------------------------------------------

On the spring boundary :math:`\Gamma_W` the coil is supported by a
distributed spring foundation with stiffness
:math:`k(\mathbf{x}) = k_0\,w(\mathbf{x})` [N m\ :sup:`−3`], where
:math:`k_0` is a nominal stiffness and :math:`w \in [0,1]` is a per-node
weight.  This adds the surface term

.. math::

   \int_{\Gamma_W} k(\mathbf{x})\,\mathbf{u} \cdot \mathbf{v} \,\mathrm{d}S

to the left-hand side of :eq:`eq-weak`, stiffening the system wherever
:math:`w > 0`.  In :class:`~coil_fem.elasticity.LinearElasticity3D` the
weight :math:`w` is an optimisable parameter that is interpolated from node
values to quadrature points using face shape functions and absorbed into the
Nanson-scaled integration weight, so gradients flow through it via the adjoint.

----

Post-processing: Von Mises Stress
-----------------------------------

The von Mises stress is computed from the **mechanical** (Cauchy) stress
:eq:`eq-hooke`:

.. math::

   \sigma_\text{vm}
   = \sqrt{\tfrac{3}{2}\, \mathbf{s} : \mathbf{s}},
   \qquad
   \mathbf{s} = \boldsymbol{\sigma} - \tfrac{1}{3}\mathrm{tr}(\boldsymbol{\sigma})\,\mathbf{I}.

Because :math:`\boldsymbol{\sigma}` is based on
:math:`\boldsymbol{\varepsilon}_m`, the thermal pre-stress is correctly
included: for a freely contracting coil the deviatoric stress would vanish,
and only geometric constraints (fixtures, Winkler support) raise
:math:`\sigma_\text{vm}` above zero.

----

Summary of Implementation Mapping
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Mathematical object
     - Implementation
   * - :math:`E,\,\nu,\,\rho`
     - ``material_options['E']``, ``['nu']``, ``['density']`` in
       :class:`~coil_fem.CoilFEM`
   * - :math:`\alpha,\,T_\text{init},\,T_\text{final}`
     - ``material_options['alpha']``, ``['init_temperature']``,
       ``['final_temperature']``
   * - :math:`\boldsymbol{\varepsilon}_\text{th}`
     - :func:`~coil_fem.thermal.itc_strain`;
       stored as ``problem.epsilon_th``
   * - :math:`\boldsymbol{\sigma}(\mathbf{u},\boldsymbol{\varepsilon}_\text{th})`
     - :func:`~coil_fem.thermal.cauchy_stress_with_thermal_strain`;
       returned by ``LinearElasticity3D.get_tensor_map``
   * - :math:`\mathbf{b}_\text{Lorentz}`
     - :func:`~coil_fem.forces.lorentz_line_force` divided by
       cross-section area; passed as ``params['body_force']``
   * - :math:`k(\mathbf{x})`
     - ``problem_options['winkler_k']`` :math:`\times`
       ``params['support_weights']``
   * - :math:`\sigma_\text{vm}`
     - :func:`~coil_fem.metrics.von_mises_on_quadrature` (uses
       :math:`\boldsymbol{\varepsilon}_m` automatically when
       ``problem.epsilon_th`` is set)
