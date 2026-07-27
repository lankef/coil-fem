Support Structure Model
=======================

This chapter explains how coil-fem couples a structural support to one or more
coil FEM problems, and describes the steps required to implement a new support
model.

.. contents:: Contents
   :local:
   :depth: 2

Support Structure Architecture
-------------------------------

Why a coupling layer exists
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A single high-resolution FEM solve for one coil cross-section can require
hundreds of megabytes of GPU memory just for the stiffness matrix.  For this
reason coil-fem keeps the coil and its support structure as **structurally
distinct objects**: each coil owns its own mesh, material model, and solver,
and can be placed on a separate GPU so that all coils are solved in parallel
with no inter-GPU communication.

When the support is a simple grounded boundary condition — a Winkler / Robin
spring whose far end is welded to a fixed wall — each coil problem is fully
independent and the parallelism is trivially perfect (:class:`FixedSupport
<coil_fem.coupling.FixedSupport>`).

With a realistic support structure (a beam frame, a variable-density FEM
solid, etc.) the coil displacements and the support displacements are coupled:
the Winkler springs now connect the coil surface to a *moving* support
skeleton, so all fields must be determined simultaneously.  coil-fem provides
two strategies for this coupled solve.

.. note::
   NVIDIA cuDSS can factorize one sparse matrix across multiple GPUs (MG
   mode), but `spineax <https://github.com/johnviljoen/spineax>`_ does not yet
   expose that API.  Until it does, a single monolithic solve cannot be
   distributed across GPUs, which is one reason the staggered strategy
   remains the primary production path.

Monolithic coupling
~~~~~~~~~~~~~~~~~~~

The entire coil–support system is assembled into **one linear system** and
factorized in a single step.

The global stiffness matrix is assembled by collecting COO triplets from every
pipeline (:meth:`ElasticPipeline.coo() <coil_fem.pipelines.ElasticPipeline.coo>`)
and from the support (:meth:`Support.coo() <coil_fem.coupling.supports.Support.coo>`),
inserting global DOF offsets and coupling blocks at the interface.  The merged
system is factorized once with cuDSS and solved directly — there is no
iteration.

.. math::

   \begin{bmatrix} K_{cc} & K_{cs} \\ K_{sc} & K_{ss} \end{bmatrix}
   \begin{bmatrix} u_c \\ u_s \end{bmatrix}
   =
   \begin{bmatrix} f_c \\ f_s \end{bmatrix}

Here :math:`K_{cc}` is the coil stiffness block, :math:`K_{ss}` the support
stiffness block, and :math:`K_{cs}` / :math:`K_{sc}` the interface coupling
blocks arising from the shared Winkler spring constraint at the attachment
surface.

.. mermaid::

   flowchart TD
       CoilFEM --> Driver["solve_monolithic()"]
       Driver --> MergedSolver["Single merged factorization (cuDSS)"]
       MergedSolver -->|"reads coo()"| P0["ElasticPipeline (coil 0)"]
       MergedSolver -->|"reads coo()"| P1["ElasticPipeline (coil 1)"]
       MergedSolver -->|"reads coo()"| Sup[Support]
       P0 --> M0[CoilMesh]
       P0 --> L0[LinearElasticity3D]
       P1 --> M1[CoilMesh]
       P1 --> L1[LinearElasticity3D]

**Key properties:**

- Requires cuDSS (``problem_options={'solver': 'cudss'}``).
- Per-coil parallelism is lost — all coils share one merged solve.
- The adjoint is straightforward: the same factorization is reused for the
  backward pass.
- Primary use: exact-solution baseline for validating staggered convergence on
  small problems.

Staggered coupling
~~~~~~~~~~~~~~~~~~

Each coil and the support are kept as :math:`N_{\text{coil}} + 1` independent
linear systems.  They are solved in sequence — a block Gauss–Seidel sweep —
and the sweep repeats until the interface force vanishes.

The iteration proceeds as follows:

1. Solve each coil pipeline independently, treating the support's last
   interface displacement as a prescribed boundary attachment.
2. Collect the coil-surface reaction forces and pass them to
   :meth:`support.solve() <coil_fem.coupling.supports.Support.solve>`.
3. Repeat until the interface displacement converges (or a fixed number of
   sweeps is reached).
4. The entire sweep loop is wrapped in a ``custom_vjp`` so that ``jax.grad``
   differentiates through the **converged fixed point** via the implicit
   function theorem — not through the unrolled loop.  This keeps the gradient
   cost independent of the number of sweeps.

.. mermaid::

   flowchart TD
       CoilFEM --> Driver["solve_staggered()"]
       Driver --> S1["Solver (coil 0)"] --> P0["ElasticPipeline (coil 0)"]
       Driver --> S2["Solver (coil 1)"] --> P1["ElasticPipeline (coil 1)"]
       Driver --> S3["Solver (support)"] --> Sup[Support]
       P0 --> M0[CoilMesh]
       P0 --> L0[LinearElasticity3D]
       P1 --> M1[CoilMesh]
       P1 --> L1[LinearElasticity3D]
       Driver -.->|"u_attach → support.solve()"| Sup
       Sup -.->|"u_attach from support.compute_attach()"| Driver

**Key properties:**

- Works on CPU and GPU; each unit can be placed on a separate GPU.
- Each coil and the support maintain their own solver / factorization, so the
  coil solves can run in parallel (or on separate GPUs) within each sweep.
- Convergence is guaranteed for well-conditioned Winkler-type coupling; use
  Aitken relaxation for tighter coupling ratios.
- Compatible with non-linear coil or support models: ``solve`` simply iterates
  more internally without changing the block interface.

When to use which
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * -
     - Staggered
     - Monolithic
   * - Problem size
     - Large / high-resolution
     - Small / verification
   * - Hardware
     - Multiple GPUs or CPU
     - Single GPU (cuDSS required)
   * - Parallelism
     - Per-coil solves run in parallel
     - Single merged factorization
   * - Non-linear coils/support
     - Supported (future)
     - Requires non-symmetric cuDSS path
   * - Primary use
     - Production optimization runs
     - Convergence validation

Implementation
~~~~~~~~~~~~~~

The three key objects are:

- :class:`ElasticPipeline <coil_fem.pipelines.ElasticPipeline>`
  (``src/coil_fem/pipelines.py``) — owns one coil's mesh, JAX-FEM problem, and
  differentiable forward-prediction callable.  Exposes
  :meth:`~coil_fem.pipelines.ElasticPipeline.solve` and
  :meth:`~coil_fem.pipelines.ElasticPipeline.coo`.

- :class:`Support <coil_fem.coupling.supports.Support>`
  (``src/coil_fem/coupling/supports.py``) — abstract base class for any support
  model.  Exposes :meth:`~coil_fem.coupling.supports.Support.solve`,
  :meth:`~coil_fem.coupling.supports.Support.compute_attach`, and optionally
  :meth:`~coil_fem.coupling.supports.Support.coo`.

- **Coupling drivers** (``src/coil_fem/coupling/drivers.py``) — pure functions
  ``solve_uncoupled``, ``solve_staggered``, and ``solve_monolithic`` that
  orchestrate iteration between pipelines and supports.  They are functions,
  not classes: all persistent state (factorizations, meshes) lives inside the
  pipelines and supports they receive as arguments.

The :class:`Support <coil_fem.coupling.supports.Support>` ABC standardises the
interface so that coupling strategies and support implementations are
independently swappable.

Developing a New Support Structure
------------------------------------

The following steps describe how to implement a new support model such as
``BeamNetworkSupport`` or ``DensityFieldSupport``.

Step 1 — Subclass ``Support``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a new class in ``src/coil_fem/coupling/supports.py`` (or in a separate
file imported there):

.. code-block:: python

   import abc
   from coil_fem.coupling.supports import Support

   class MySupport(Support):
       def __init__(self, ...):
           ...

Step 2 — Implement ``is_coupled``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return ``True`` to tell the coupling driver that this support has its own DOFs
that participate in the coupled solve:

.. code-block:: python

   @property
   def is_coupled(self) -> bool:
       return True

Step 3 — Implement ``solve(inputs)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``inputs`` is a dict supplied by the coupling driver.  For staggered coupling
it will contain the coil-side interface reaction forces (the spring load
:math:`k \cdot (u_{\text{coil}} - u_{\text{attach}})` scattered onto the
support attachment points).  Your implementation should:

- Accept the load and solve the support's internal FEM / beam / etc. problem.
- Return a ``state`` dict with at least ``'u_s'`` (the support DOF solution).
- Be **differentiable**: use ``jax_fem.solver.ad_wrapper`` (for a FEM-based
  support) or ``jnp.linalg.solve`` / any JAX-compatible solver so that
  ``jax.grad`` can flow through this call.

.. code-block:: python

   def solve(self, inputs: dict) -> dict:
       load = inputs["attachment_loads"]   # shape (N_attach, 3)
       # ... build RHS, call ad_wrapper / lineax ...
       return {"sol": sol}                 # sol is a differentiable jax.Array

Step 4 — Optionally implement ``coo()`` for monolithic coupling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the support is to be used with ``solve_monolithic``, override
:meth:`~coil_fem.coupling.supports.Support.coo` to return
``(I, J, V, n_dofs)`` — the COO triplets of the support stiffness matrix in
its local DOF numbering.  See
:meth:`ElasticPipeline.coo() <coil_fem.pipelines.ElasticPipeline.coo>` for the
coil-side equivalent and the docstring of
:meth:`Support.coo() <coil_fem.coupling.supports.Support.coo>` for the full
description of the block structure and COO format.

.. code-block:: python

   def coo(self):
       # Return (I, J, V, n_dofs) for the K_ss block
       return self.problem.I_jax, self.problem.J_jax, self.problem.V_jax, self.n_dofs

Step 5 — Register in ``coupling/__init__.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Export the new class from ``src/coil_fem/coupling/__init__.py`` so users can
import it with a single statement:

.. code-block:: python

   from coil_fem.coupling import MySupport

Step 6 — Write tests
~~~~~~~~~~~~~~~~~~~~~

Add tests in ``tests/test_support_<name>.py``:

- Construct the support with a small toy mesh / geometry.
- Verify that :meth:`solve` returns ``'u_s'`` with the correct shape and
  contains finite values.
- Verify that :meth:`solve` is differentiable by calling ``jax.grad`` on a
  scalar that depends on the returned ``'u_s'``.
- If :meth:`coo` is implemented, verify that the returned matrix is symmetric
  and positive semi-definite on a small example.

Step 8 — Integrate with ``CoilFEM``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass an instance to :class:`~coil_fem.CoilFEM`:

.. code-block:: python

   from coil_fem.coupling import MySupport

   support = MySupport(...)
   fem = CoilFEM(..., support=support)

``CoilFEM`` will call ``solve_staggered`` or ``solve_monolithic`` automatically
when ``support.is_coupled`` is ``True`` and the appropriate coupling mode is
selected via ``physics_options``.
