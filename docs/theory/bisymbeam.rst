.. _bisymmetrical-framework-element:

==========================================================
Bisymmetrical Framework Element: Local Stiffness Relation
==========================================================

:Source: W. McGuire, R. H. Gallagher, R. D. Ziemian, *Matrix Structural
         Analysis*, 2nd ed., Wiley, 2000 -- Figure 4.6 (element and sign
         convention) and Figure 4.11 / Equation (4.34) (stiffness matrix).
         A free e-version is available at
         https://digitalcommons.bucknell.edu/books/7/
:Scope:  Local (element) coordinates only. Linear elastic, small
         displacement, prismatic member.

.. contents:: Contents
   :local:
   :depth: 2


Overview
========

Equation (4.34) is the :math:`12 \times 12` **local** stiffness relation for a
two-node prismatic space frame element with a *bisymmetrical* cross-section.
It maps the twelve nodal displacements to the twelve nodal forces:

.. math::

   \{F\} = E\,[k]\,\{d\}

The element carries six degrees of freedom per node -- three translations and
three rotations -- and is the direct superposition of four uncoupled
one-dimensional problems:

* an **axial bar** (:math:`u`),
* **St. Venant torsion** (:math:`\theta_x`),
* **Euler-Bernoulli bending in the** :math:`x`--:math:`y` **plane**
  (:math:`v,\ \theta_z`),
* **Euler-Bernoulli bending in the** :math:`x`--:math:`z` **plane**
  (:math:`w,\ \theta_y`).

The bisymmetry of the section is precisely what permits that superposition;
see `Why "bisymmetrical" matters`_.


Element geometry and sign convention
====================================

Per Figure 4.6, the element is straight and prismatic, with node 1 at the
origin of the local frame and node 2 at :math:`x = L`. The local axes
:math:`(x, y, z)` form a **right-handed** triad, with :math:`x` directed from
node 1 to node 2 along the centroidal axis.

All twelve displacement and force quantities are **positive as drawn** in
Figure 4.6: translations positive along the positive local axis directions,
and rotations and moments positive by the **right-hand rule** about the
positive local axes.

::

                     y
                     |
        My1,θy1 ↑    |    ↑ My2,θy2
         Fy1,v1 ↑    |    ↑ Fy2,v2
                     |
    Mx1,θx1 →  ┌─────┴───────────────┐  → Fx2,u2   Mx2,θx2
     Fx1,u1 →  │ 1                 2 │  ────────────────→ x
               └─────────────────────┘
        Fz1,w1 ↙            ↙ Fz2,w2
       Mz1,θz1 ↙            ↙ Mz2,θz2
      z
                |←──────── L ────────→|

        Young's modulus = E,  Shear modulus = G


The stiffness relation, Eq. (4.34)
==================================

The vertical and horizontal rules partition the array into the four
:math:`6 \times 6` blocks of Figure 4.11 (shown dashed in the original):

.. math::

   \begin{Bmatrix}
   F_{x1} \\ F_{y1} \\ F_{z1} \\ M_{x1} \\ M_{y1} \\ M_{z1} \\
   F_{x2} \\ F_{y2} \\ F_{z2} \\ M_{x2} \\ M_{y2} \\ M_{z2}
   \end{Bmatrix}
   = E
   \left[\begin{array}{cccccc|cccccc}
   \frac{A}{L} & 0 & 0 & 0 & 0 & 0
     & -\frac{A}{L} & 0 & 0 & 0 & 0 & 0 \\[4pt]
   0 & \frac{12I_z}{L^3} & 0 & 0 & 0 & \frac{6I_z}{L^2}
     & 0 & -\frac{12I_z}{L^3} & 0 & 0 & 0 & \frac{6I_z}{L^2} \\[4pt]
   0 & 0 & \frac{12I_y}{L^3} & 0 & -\frac{6I_y}{L^2} & 0
     & 0 & 0 & -\frac{12I_y}{L^3} & 0 & -\frac{6I_y}{L^2} & 0 \\[4pt]
   0 & 0 & 0 & \frac{J}{2(1+\nu)L} & 0 & 0
     & 0 & 0 & 0 & -\frac{J}{2(1+\nu)L} & 0 & 0 \\[4pt]
   0 & 0 & -\frac{6I_y}{L^2} & 0 & \frac{4I_y}{L} & 0
     & 0 & 0 & \frac{6I_y}{L^2} & 0 & \frac{2I_y}{L} & 0 \\[4pt]
   0 & \frac{6I_z}{L^2} & 0 & 0 & 0 & \frac{4I_z}{L}
     & 0 & -\frac{6I_z}{L^2} & 0 & 0 & 0 & \frac{2I_z}{L} \\[4pt]
   \hline
   -\frac{A}{L} & 0 & 0 & 0 & 0 & 0
     & \frac{A}{L} & 0 & 0 & 0 & 0 & 0 \\[4pt]
   0 & -\frac{12I_z}{L^3} & 0 & 0 & 0 & -\frac{6I_z}{L^2}
     & 0 & \frac{12I_z}{L^3} & 0 & 0 & 0 & -\frac{6I_z}{L^2} \\[4pt]
   0 & 0 & -\frac{12I_y}{L^3} & 0 & \frac{6I_y}{L^2} & 0
     & 0 & 0 & \frac{12I_y}{L^3} & 0 & \frac{6I_y}{L^2} & 0 \\[4pt]
   0 & 0 & 0 & -\frac{J}{2(1+\nu)L} & 0 & 0
     & 0 & 0 & 0 & \frac{J}{2(1+\nu)L} & 0 & 0 \\[4pt]
   0 & 0 & -\frac{6I_y}{L^2} & 0 & \frac{2I_y}{L} & 0
     & 0 & 0 & \frac{6I_y}{L^2} & 0 & \frac{4I_y}{L} & 0 \\[4pt]
   0 & \frac{6I_z}{L^2} & 0 & 0 & 0 & \frac{2I_z}{L}
     & 0 & -\frac{6I_z}{L^2} & 0 & 0 & 0 & \frac{4I_z}{L}
   \end{array}\right]
   \begin{Bmatrix}
   u_1 \\ v_1 \\ w_1 \\ \theta_{x1} \\ \theta_{y1} \\ \theta_{z1} \\
   u_2 \\ v_2 \\ w_2 \\ \theta_{x2} \\ \theta_{y2} \\ \theta_{z2}
   \end{Bmatrix}


Notation
========

Material and section properties
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 12 40 48

   * - Symbol
     - Meaning
     - Notes
   * - :math:`E`
     - Young's modulus
     - Factored out in front of the whole array, hence the form of the
       torsion term.
   * - :math:`\nu`
     - Poisson's ratio
     - Appears only through :math:`G`.
   * - :math:`G`
     - Shear modulus, :math:`G = \dfrac{E}{2(1+\nu)}`
     - Labelled in Fig. 4.6 but *eliminated* from Eq. (4.34) in favour of
       :math:`E` and :math:`\nu`.
   * - :math:`L`
     - Element length, node 1 to node 2
     - Measured along the centroidal axis.
   * - :math:`A`
     - Cross-sectional area
     - Full area; no shear area appears (see `Assumptions and limits`_).
   * - :math:`I_y`
     - Second moment of area about the local :math:`y` axis
     - Governs bending in the :math:`x`--:math:`z` plane (deflection
       :math:`w`).
   * - :math:`I_z`
     - Second moment of area about the local :math:`z` axis
     - Governs bending in the :math:`x`--:math:`y` plane (deflection
       :math:`v`).
   * - :math:`J`
     - St. Venant torsion constant
     - **Not** the polar moment :math:`I_p = I_y + I_z`, except for solid or
       hollow circular sections.

Degrees of freedom
------------------

The vectors are ordered :math:`[\,u,\ v,\ w,\ \theta_x,\ \theta_y,\
\theta_z\,]` at node 1, then the same six at node 2. Subscript :math:`n`
below denotes the node number, :math:`n = 1, 2`.

.. list-table::
   :header-rows: 1
   :widths: 14 14 36 36

   * - Displ.
     - Force
     - Kind
     - Governed by
   * - :math:`u_n`
     - :math:`F_{xn}`
     - Axial translation / axial force
     - Bar action, :math:`EA/L`
   * - :math:`v_n`
     - :math:`F_{yn}`
     - Translation along :math:`y` / shear force in :math:`y`
     - Bending about :math:`z`, :math:`EI_z`
   * - :math:`w_n`
     - :math:`F_{zn}`
     - Translation along :math:`z` / shear force in :math:`z`
     - Bending about :math:`y`, :math:`EI_y`
   * - :math:`\theta_{xn}`
     - :math:`M_{xn}`
     - Rotation about :math:`x` (twist) / torque
     - St. Venant torsion, :math:`GJ/L`
   * - :math:`\theta_{yn}`
     - :math:`M_{yn}`
     - Rotation about :math:`y` / bending moment about :math:`y`
     - Bending about :math:`y`, :math:`EI_y`
   * - :math:`\theta_{zn}`
     - :math:`M_{zn}`
     - Rotation about :math:`z` / bending moment about :math:`z`
     - Bending about :math:`z`, :math:`EI_z`


Reading the matrix
==================

Why the torsion term looks unusual
----------------------------------

Because :math:`E` is factored out in front of the array, the torsion entry
must carry the compensating factor. Substituting :math:`G = E / [2(1+\nu)]`:

.. math::

   E \cdot \frac{J}{2(1+\nu)L} \;=\; \frac{GJ}{L}

So the :math:`\theta_x` block is the familiar linear St. Venant torsion
stiffness, written without ever naming :math:`G`.

.. note::

   The :math:`\theta_x` row and column couple to nothing else. Torsion is
   fully independent of bending and of axial action in this element.

Why the two bending planes carry opposite signs
-----------------------------------------------

The :math:`I_z` coupling entries are :math:`+6I_z/L^2`, but the corresponding
:math:`I_y` entries are :math:`-6I_y/L^2`. **This is not a typo.** It follows
from the right-hand rule in a right-handed frame.

For a small rotation vector :math:`\boldsymbol{\theta}` applied to a point at
:math:`\mathbf{r} = (x, 0, 0)`, the displacement is
:math:`\mathbf{u} = \boldsymbol{\theta} \times \mathbf{r}`. Hence:

.. math::

   \boldsymbol{\theta} = (0,\,0,\,\theta_z) \;\Rightarrow\; v = +\theta_z\,x
   \qquad\text{so}\qquad \theta_z = +\frac{dv}{dx}

.. math::

   \boldsymbol{\theta} = (0,\,\theta_y,\,0) \;\Rightarrow\; w = -\theta_y\,x
   \qquad\text{so}\qquad \theta_y = -\frac{dw}{dx}

The slope-rotation relation carries a minus sign in the :math:`x`--:math:`z`
plane and a plus sign in the :math:`x`--:math:`y` plane. Every term coupling a
translation to a rotation therefore flips sign between the two planes, while
the uncoupled diagonal terms (:math:`12I/L^3`, :math:`4I/L`, :math:`2I/L`) do
not.

.. warning::

   This sign asymmetry is one of the most common implementation bugs in space
   frame codes. It is easy to write the :math:`x`--:math:`y` block correctly
   and then copy it into the :math:`x`--:math:`z` slot unchanged.

The four decoupled sub-problems
-------------------------------

Extracting the relevant rows and columns recovers four standard elements.

**Axial** -- DOF :math:`\{u_1,\, u_2\}`, rows/columns 1 and 7:

.. math::

   \frac{EA}{L}
   \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}

**Torsion** -- DOF :math:`\{\theta_{x1},\, \theta_{x2}\}`, rows/columns 4
and 10:

.. math::

   \frac{GJ}{L}
   \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}

**Bending in the** :math:`x`--:math:`y` **plane** -- DOF
:math:`\{v_1,\, \theta_{z1},\, v_2,\, \theta_{z2}\}`, rows/columns 2, 6, 8,
12:

.. math::

   \frac{EI_z}{L^3}
   \begin{bmatrix}
    12   &  6L   & -12   &  6L   \\
     6L  &  4L^2 & -6L   &  2L^2 \\
   -12   & -6L   &  12   & -6L   \\
     6L  &  2L^2 & -6L   &  4L^2
   \end{bmatrix}

**Bending in the** :math:`x`--:math:`z` **plane** -- DOF
:math:`\{w_1,\, \theta_{y1},\, w_2,\, \theta_{y2}\}`, rows/columns 3, 5, 9,
11:

.. math::

   \frac{EI_y}{L^3}
   \begin{bmatrix}
    12   & -6L   & -12   & -6L   \\
   -6L   &  4L^2 &  6L   &  2L^2 \\
   -12   &  6L   &  12   &  6L   \\
   -6L   &  2L^2 &  6L   &  4L^2
   \end{bmatrix}

The last two are identical apart from the sign of the coupling terms, exactly
as derived above.


Why "bisymmetrical" matters
===========================

A bisymmetrical section has **two axes of symmetry** -- for example solid or
hollow circular, rectangular, I- and box sections. Two consequences make the
clean block structure of Eq. (4.34) valid:

#. **The local** :math:`y` **and** :math:`z` **axes are principal centroidal
   axes**, so the product of inertia :math:`I_{yz} = 0`. Without this, the two
   bending problems couple and off-diagonal :math:`I_{yz}` terms appear
   linking :math:`v` with :math:`\theta_y` and :math:`w` with
   :math:`\theta_z`.

#. **The shear centre coincides with the centroid.** Without this, a
   transverse force applied at the centroid is statically equivalent to the
   same force at the shear centre *plus* a torque, so bending and torsion
   couple and the :math:`\theta_x` block is no longer isolated.

For a channel, angle, or tee, both assumptions fail and Eq. (4.34) does not
apply as written.

.. note::

   When assembling a model, the local :math:`y` and :math:`z` axes must be
   *aligned with the section's principal axes*. The member axis fixes only
   local :math:`x`; the roll about that axis is supplied separately (roll
   angle, reference vector, or auxiliary node) and is what makes assumption 1
   true by construction rather than by luck.


Assumptions and limits
======================

* **Euler-Bernoulli bending.** Plane sections remain plane and normal to the
  deformed centroidal axis; transverse shear deformation is neglected. Only
  :math:`A` appears, never a shear area :math:`A_s`. For a Timoshenko
  (shear-flexible) element, the bending blocks acquire the factor
  :math:`\Phi = 12EI / (G A_s L^2)`; switching to it is generally worthwhile
  below a span-to-depth ratio of roughly 10.

* **St. Venant torsion, unrestrained warping.** The :math:`GJ/L` term assumes
  free warping. For open thin-walled sections with warping restrained, this
  underestimates the torsional stiffness, and a 7th DOF per node (Vlasov
  bimoment) is required.

* **Prismatic and homogeneous.** :math:`E`, :math:`\nu`, :math:`A`,
  :math:`I_y`, :math:`I_z`, :math:`J` are all constant along the element.

* **Small displacements, no axial-flexural interaction.** There is no
  geometric stiffness contribution; a large axial force :math:`P` would add
  the term :math:`P w''` to the governing equation and destroy superposition
  in :math:`P`. See MGZ Chapter 9 for the geometric stiffness matrix.

* **Singular by construction.** The matrix has rank 6; its null space is the
  six rigid-body modes (three translations, three rotations). It cannot be
  inverted until the assembled structure is restrained. As a check, both a
  rigid translation and a rigid rotation return zero nodal forces.

* **Nodally exact.** The cubic Hermite shape functions are the exact
  homogeneous solution of :math:`EIv'''' = 0`, so one element per prismatic
  member reproduces the analytical answer at the joints. Distributed loads are
  handled with fixed-end forces as equivalent nodal loads, superposing the
  fixed-end solution afterwards, rather than by meshing. This exactness does
  **not** carry over to dynamics or buckling, where mode shapes are not cubic.


Use in a global assembly
========================

Equation (4.34) is expressed in **local** element coordinates. To assemble a
frame network:

.. math::

   [k]_{\text{global}} = [\Gamma]^{\mathsf{T}}\, [k]_{\text{local}}\, [\Gamma]

where :math:`[\Gamma]` is block-diagonal, containing four copies of the
:math:`3 \times 3` matrix of direction cosines relating the local and global
axes -- one block for each of the four triads
(:math:`u\,v\,w` and :math:`\theta_x\,\theta_y\,\theta_z`, at each of the two
nodes). See MGZ Chapter 5 for the transformation, for loads applied between
nodes, and for end releases and rigid end offsets.


References
==========

* W. McGuire, R. H. Gallagher, R. D. Ziemian, *Matrix Structural Analysis*,
  2nd ed., John Wiley & Sons, 2000. Chapter 4 (element formulation),
  Chapter 5 (transformations, fixed-end forces, releases, offsets),
  Chapter 9 (geometric stiffness). Free e-version:
  https://digitalcommons.bucknell.edu/books/7/

* J. S. Przemieniecki, *Theory of Matrix Structural Analysis*, McGraw-Hill,
  1968; Dover reprint, ISBN 0-486-64948-2. The classic source for the same
  element, given with the shear parameter :math:`\Phi` built in, so that the
  Euler-Bernoulli form follows at :math:`\Phi = 0` and the Timoshenko form at
  :math:`\Phi \neq 0`. Also the source for the consistent mass and geometric
  stiffness matrices.