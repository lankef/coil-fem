Performance tips
================

Choosing a solver
-----------------

We strongly advise installing ``spineax`` and using ``solver="cudss"``. ``JAX-FEM``,
The FEM library that ``coil-fem`` uses, is designed to be a CPU code. It only uses
JAX for auto-differentiation. When any other backends are chosen, arrays will 
be copied within or to/from GPU, causing performance drop.  

When other JAX/CUDA sparse solvers become available, this recommendation may change.

.. _cudss-preallocation-issues:

cuDSS preallocation issues
--------------------------

``coil-fem`` is based on a dual backend of JAX and cuDSS. This causes some unusual
behaviors in memory allocation. When no other heavy JAX/XLA codes (such as DESC)
we strongly recommend running ``coil_fem.gpu_env.configure_gpu_memory()`` before 
importing ``coil_fem`` and ``jax``, which disables pre-allocation and sets a XLA
memory cap to 50%.

- Why disabling preallocation?

  XLA preallocates 75% of the device by default; cuDSS allocates
  its factorization *outside* XLA's pool and cannot borrow from it, so the
  GPU utilization may be inefficient if you are not using other jax/XLA
  codes. As the next section shows, XLA and cuDSS roughly use the same amount of 
  memory.

- Why memory capping?

  ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` makes JAX allocate as needed, but the 
  behavior is more prone to fragmentation — and crucially, the BFC allocator still
  never returns memory. Over 50 trust-constr iterations, each with forward + adjoint
  cuDSS factorizations, XLA's pool ratchets upward and progressively starves cuDSS.
  A hard cap prevents the ratchet from eating the whole device. 
  JAX Documentation

Resource use table
------------------

This section provides a reference table for the required GPU resources for calculating
the value and grad of ``CoilFEMObjective`` once on the 5 non-planar coils of W7-X
coil represented by ``simsopt.geo.CurveXYZFourier(curve_order=8)`` (51 dofs). 

Memory scales cubically with resolution.

Higher resolutions are more realistic, but require higher memory use and evaluation time. 

This table was run on an L40s using the NYU Torch cluster. It was last updated on Aug 7. 2026.

======= ========== ========== ============== =============== ============== ================== ================
``ppp`` # nodes    # cells    JAX memory     JAX memory pool cuDSS memory   Total peak memory  Eval time
======= ========== ========== ============== =============== ============== ================== ================
12      24000 x 5  11520 x 5  0.727 GB       0.826 GB        1.958 GB       2.784 GB           0.73 s
18      70560 x 5  38880 x 5  1.804 GB       2.686 GB        3.624 GB       6.310 GB           2.17 s
24      201600 x 5 123264 x 5 5.951 GB       10.373 GB       9.149 GB       19.522 GB          8.28 s
30      405600 x 5 259200 x 5 12.331 GB      18.748 GB       18.849 GB      37.597 GB          28.34 s
======= ========== ========== ============== =============== ============== ================== ================
