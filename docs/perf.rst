Performance tips
================

Choosing a solver
-----------------

We strongly advise installing ``spineax`` and using ``solver="cudss"``. ``JAX-FEM``,
The FEM library that ``coil-fem`` uses is designed to be a CPU code, and only uses
JAX for auto-differentiation. When any other backends are chosen, arrays can 
be copied within or to/from GPU, causing performance drop.  

When other JAX sparse solvers become available, this recommendation may change.

.. _cudss-preallocation-issues:

cuDSS preallocation issues
--------------------------

``coil-fem`` is based on a dual backend of JAX and cuDSS. This causes some unusual
behaviors in memory allocation. When no other heavy JAX/XLA codes (such as ``DESC``)
we strongly recommend running ``coil_fem.gpu_env.configure_gpu_memory()`` before 
importing ``coil_fem`` and ``jax``, which disables pre-allocation and sets a XLA
memory cap to 50%.

- Why disabling preallocation?

  XLA preallocates 75% of the device by default; cuDSS allocates
  its factorisation *outside* XLA's pool and cannot borrow from it, so the
  GPU utilization may be inefficient if you are not using other jax/XLA
  codes. 

- Why memory capping?

  ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` makes JAX allocate as needed, but the 
  behavior is more prone to fragmentation — and crucially, the BFC allocator still
  never returns memory. Over 50 trust-constr iterations, each with forward + adjoint
  cuDSS factorizations, XLA's pool ratchets upward and progressively starves cuDSS.
  A hard cap prevents the ratchet from eating the whole device. 
  JAX Documentation