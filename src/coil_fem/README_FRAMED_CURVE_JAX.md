# Pure JAX Framed Curves

## Overview

`framed_curve_jax.py` provides pure JAX wrappers for simsopt's FramedCurve classes. These wrappers work directly with `CurveXYZFourierJAX` without requiring `simsopt.Curve` objects, enabling fully differentiable, JIT-compatible meshing workflows.

## Quick Start

```python
from coil_fem.geo import CurveXYZFourierJAX
from coil_fem.geo import make_centroid_frame, make_rmf_frame

# Create a JAX curve
curve = CurveXYZFourierJAX(quadpoints, dofs, order)

# Create framed curve
framed_centroid = make_centroid_frame(curve)  # Centroid frame
framed_rmf = make_rmf_frame(curve)            # RMF frame

# Get frame at quadrature points
t, p, q = framed_centroid.rotated_frame()

# Evaluate frame analytically at arbitrary parameter values
# (centroid: closed-form; RMF: fresh double-reflection scan)
t_interp, p_interp, q_interp = framed_centroid.rotated_frame_eval(
    phi_values
)
```

## Classes

### `FramedCurveJAX` (Base Class)

Base class for framed curves. Provides common interface.

**Attributes**:
- `curve`: CurveXYZFourierJAX object
- `alpha`: Rotation angles at quadrature points

**Methods**:
- `gamma()`: Curve positions at quadrature points
- `gammadash()`: First derivatives at quadrature points
- `gammadashdash()`: Second derivatives at quadrature points
- `gamma_eval(phi)`: Analytic Fourier evaluation at arbitrary phi
- `alpha_eval(phi)`: Analytic Fourier evaluation of the rotation
  angle `alpha` at arbitrary phi (band-limited reconstruction)
- `rotated_frame()`: Frame (t, p, q) at quadrature points
- `rotated_frame_eval(phi)`: Frame (t, p, q) at arbitrary phi —
  closed-form for the centroid frame, fresh double-reflection scan
  on the supplied phi-grid for the RMF

### `FramedCurveCentroidJAX`

Centroid frame (Singh et al. 2020).

The **p** vector points from the curve centroid to each point, projected perpendicular to the tangent.

**Best for**: General use, stable

### `FramedCurveRMFJAX`

Rotation-minimizing frame (Bishop/parallel-transport frame).

Minimizes twist of the first transverse frame vector along the curve.

**Best for**: Circular cross-sections, minimizing twist

## Features

### Pure JAX
- No simsopt.Curve dependency
- Works with CurveXYZFourierJAX directly
- All operations are JAX arrays

### Fully Differentiable
```python
def objective(dofs):
    curve = CurveXYZFourierJAX(quadpoints, dofs, order)
    framed = make_rmf_frame(curve)
    t, p, q = framed.rotated_frame()
    return jnp.sum(jnp.linalg.norm(p, axis=1))

gradient = jax.grad(objective)(dofs)
```

### JIT Compatible
```python
@jax.jit
def fast_frame(dofs):
    curve = CurveXYZFourierJAX(quadpoints, dofs, order)
    framed = make_centroid_frame(curve)
    return framed.rotated_frame()

t, p, q = fast_frame(dofs)  # Compiled, fast
```

### Batch Processing
```python
def get_frame(dofs):
    curve = CurveXYZFourierJAX(quadpoints, dofs, order)
    framed = make_rmf_frame(curve)
    return framed.rotated_frame()

# Process batch
batch_frames = jax.vmap(get_frame)(batch_dofs)
```

### JAX Pytrees
```python
# Framed curves are pytrees - can be used in JAX transformations
framed = make_centroid_frame(curve)
framed_transformed = jax.tree_map(lambda x: x * 2, framed)
```

## Frame Rotation

Both frame types support rotation via the `alpha` parameter:

```python
# No rotation (default)
framed = make_rmf_frame(curve)

# Constant rotation
alpha = jnp.ones(n_quad) * jnp.pi / 4  # 45 degrees
framed_rotated = make_rmf_frame(curve, alpha=alpha)

# Varying rotation
alpha = jnp.linspace(0, 2*jnp.pi, n_quad)  # Full twist
framed_twisted = make_rmf_frame(curve, alpha=alpha)
```

The rotation is applied around the tangent vector:
- `p_rotated = cos(alpha) * p - sin(alpha) * q`
- `q_rotated = sin(alpha) * p + cos(alpha) * q`

## Frame Evaluation at Arbitrary phi

Both frame classes provide a pure-JAX `rotated_frame_eval(phi)` that
avoids the interpolation step entirely:

```python
framed = make_rmf_frame(curve)

# Centroid frame: closed-form (gamma_eval + alpha_eval + Gram-Schmidt)
# RMF frame: re-runs the double-reflection scan on the new phi grid,
# with alpha analytically resampled
t, p, q = framed.rotated_frame_eval(phi_values)
```

Both implementations are fully differentiable through `curve.dofs`,
`alpha`, and `phi`. For the RMF frame, pass a sorted uniform phi grid
(e.g. `jnp.linspace(0, 1, K, endpoint=False)`) for best accuracy; the
RMF is intrinsically defined by ordered discrete propagation along the
curve.

`alpha_eval(phi)` mirrors `CurveXYZFourierJAX.gamma_eval` -- it
reconstructs the band-limited Fourier series of `self.alpha` analytically
at arbitrary phi. This is the alpha value used inside
`rotated_frame_eval`.

## Comparison: Centroid vs RMF

| Feature | Centroid | RMF |
|---------|----------|-----|
| **Twist** | Can have twist | Minimizes twist |
| **Stability** | Very stable | Stable |
| **Best for** | General use | Circular cross-sections |
| **Computation** | Fast | Slightly slower |
| **Reference** | Singh+ 2020 | Wang+ 2008 |

## Integration with Meshing

Works seamlessly with meshing functions:

```python
from coil_fem.meshing import rectangle_sweep, disk_sweep

# Rectangle sweep
framed = make_centroid_frame(curve)
mesh = rectangle_sweep(framed, w_1=0.02, w_2=0.02)

# Disk sweep (use RMF for circular cross-sections)
framed = make_rmf_frame(curve)
mesh = disk_sweep(framed, radius=0.02)
```

## Examples

See:
- `examples/framed_curve_jax_example.py` - Comprehensive frame examples
- `examples/pure_jax_meshing_example.py` - End-to-end meshing workflow

## Implementation Notes

The pure-JAX frame kernels live in this module
(`_rotated_centroid_frame_pure`, `_rotated_rmf_frame_pure`,
`_rmf_normals_pure_jax`). The wrapper classes provide:

1. Convenient API matching simsopt FramedCurve
2. JAX pytree registration (so framed curves work with `jit`, `grad`, `vmap`)
3. Integration with `CurveXYZFourierJAX`
4. Pure-JAX `rotated_frame_eval` overrides:
   - Centroid: analytic closed-form via `gamma_eval` + `alpha_eval`
   - RMF: fresh double-reflection scan on the new phi-grid

## Performance

- **JIT compilation**: First call compiles, subsequent calls are fast
- **Memory**: Minimal overhead over raw JAX arrays
- **Differentiation**: Efficient reverse-mode AD through entire pipeline
- **Batch processing**: Scales linearly with vmap

## References

1. Singh et al. (2020) "Optimization of finite-build stellarator coils", J. Plasma Phys. 86
2. Wang et al. (2008) "Computation of rotation minimizing frames", ACM Trans. Graph. 27(1)
