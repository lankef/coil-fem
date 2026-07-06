#!/usr/bin/env python3
"""
Example usage of pure JAX framed curves.

Demonstrates how to use FramedCurveCentroidJAX and FramedCurveRMFJAX
without requiring simsopt.Curve objects.
"""

import jax
import jax.numpy as jnp
from coil_fem.geo import CurveXYZFourierJAX
from coil_fem.geo import (
    FramedCurveCentroidJAX,
    FramedCurveRMFJAX,
    make_centroid_frame,
    make_rmf_frame,
)


def example_basic_usage():
    """Basic usage: create a framed curve and evaluate the frame."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Create a simple circular curve
    order = 1
    n_phi = 16
    quadpoints = jnp.linspace(0.0, 1.0, n_phi, endpoint=False)
    
    # Circle: x = cos(2*pi*phi), y = sin(2*pi*phi), z = 0
    dofs = jnp.array([
        0.0, 0.0, 1.0,  # x: cos(2*pi*phi)
        0.0, 1.0, 0.0,  # y: sin(2*pi*phi)
        0.0, 0.0, 0.0   # z: 0
    ])
    
    curve = CurveXYZFourierJAX(quadpoints, dofs, order)
    
    # Create centroid frame
    framed_centroid = FramedCurveCentroidJAX(curve)
    t_c, p_c, q_c = framed_centroid.rotated_frame()
    
    print(f"Curve has {n_phi} quadrature points")
    print(f"Tangent shape: {t_c.shape}")
    print(f"p shape: {p_c.shape}")
    print(f"q shape: {q_c.shape}")
    print(f"\nFirst tangent vector: {t_c[0]}")
    print(f"First p vector: {p_c[0]}")
    print(f"First q vector: {q_c[0]}")
    
    # Verify orthonormality
    dot_tp = jnp.sum(t_c * p_c, axis=1)
    dot_tq = jnp.sum(t_c * q_c, axis=1)
    dot_pq = jnp.sum(p_c * q_c, axis=1)
    
    print(f"\nOrthogonality check (should be ~0):")
    print(f"  t·p: max={jnp.max(jnp.abs(dot_tp)):.2e}")
    print(f"  t·q: max={jnp.max(jnp.abs(dot_tq)):.2e}")
    print(f"  p·q: max={jnp.max(jnp.abs(dot_pq)):.2e}")
    
    # Verify normalization
    norm_t = jnp.linalg.norm(t_c, axis=1)
    norm_p = jnp.linalg.norm(p_c, axis=1)
    norm_q = jnp.linalg.norm(q_c, axis=1)
    
    print(f"\nNormalization check (should be ~1):")
    print(f"  |t|: mean={jnp.mean(norm_t):.6f}, std={jnp.std(norm_t):.2e}")
    print(f"  |p|: mean={jnp.mean(norm_p):.6f}, std={jnp.std(norm_p):.2e}")
    print(f"  |q|: mean={jnp.mean(norm_q):.6f}, std={jnp.std(norm_q):.2e}")


def example_rmf_vs_centroid():
    """Compare RMF and centroid frames."""
    print("\n" + "=" * 60)
    print("Example 2: RMF vs Centroid Frame")
    print("=" * 60)
    
    # Create a helical curve
    order = 2
    n_phi = 32
    quadpoints = jnp.linspace(0.0, 1.0, n_phi, endpoint=False)
    
    # Helix with some perturbation
    key = jax.random.PRNGKey(42)
    n_dof = 3 * (2 * order + 1)
    dofs = jax.random.normal(key, (n_dof,)) * 0.1
    # Add dominant circular component
    dofs = dofs.at[2].set(1.0)  # x: cos
    dofs = dofs.at[6].set(1.0)  # y: sin
    dofs = dofs.at[10].set(0.3)  # z: linear rise
    
    curve = CurveXYZFourierJAX(quadpoints, dofs, order)
    
    # Create both frame types
    framed_centroid = make_centroid_frame(curve)
    framed_rmf = make_rmf_frame(curve)
    
    # Get frames
    t_c, p_c, q_c = framed_centroid.rotated_frame()
    t_r, p_r, q_r = framed_rmf.rotated_frame()
    
    # Compare tangents (should be identical)
    tangent_diff = jnp.linalg.norm(t_c - t_r, axis=1)
    print(f"Tangent difference: max={jnp.max(tangent_diff):.2e}")
    
    # Compare p vectors (will differ)
    p_angle = jnp.arccos(jnp.clip(jnp.sum(p_c * p_r, axis=1), -1, 1))
    print(f"p-vector angle difference: mean={jnp.mean(p_angle):.4f} rad")
    print(f"                           max={jnp.max(p_angle):.4f} rad")
    
    # Compute "twist" (change in p direction along curve)
    def compute_twist(p):
        # Approximate twist as angle change between consecutive p vectors
        p_next = jnp.roll(p, -1, axis=0)
        cos_angle = jnp.sum(p * p_next, axis=1)
        return jnp.arccos(jnp.clip(cos_angle, -1, 1))
    
    twist_c = compute_twist(p_c)
    twist_r = compute_twist(p_r)
    
    print(f"\nTwist (change in p direction):")
    print(f"  Centroid: mean={jnp.mean(twist_c):.4f}, max={jnp.max(twist_c):.4f}")
    print(f"  RMF:      mean={jnp.mean(twist_r):.4f}, max={jnp.max(twist_r):.4f}")
    print(f"  (RMF minimizes twist)")


def example_frame_interpolation():
    """Demonstrate frame interpolation at arbitrary points."""
    print("\n" + "=" * 60)
    print("Example 3: Frame Interpolation")
    print("=" * 60)
    
    # Create curve
    order = 2
    n_phi = 16
    quadpoints = jnp.linspace(0.0, 1.0, n_phi, endpoint=False)
    
    key = jax.random.PRNGKey(123)
    n_dof = 3 * (2 * order + 1)
    dofs = jax.random.normal(key, (n_dof,)) * 0.1
    dofs = dofs.at[2].set(1.0)
    dofs = dofs.at[6].set(1.0)
    
    curve = CurveXYZFourierJAX(quadpoints, dofs, order)
    framed = make_rmf_frame(curve)
    
    # Evaluate at finer resolution
    phi_fine = jnp.linspace(0.0, 1.0, 64, endpoint=False)
    
    # Pure-JAX analytic / fresh-grid evaluation (no interpolation)
    t_linear, p_linear, q_linear = framed.rotated_frame_eval(phi_fine)
    
    print(f"Original quadrature points: {n_phi}")
    print(f"Interpolated points: {len(phi_fine)}")
    print(f"Interpolated frame shape: {t_linear.shape}")
    
    # Check orthonormality of interpolated frame
    dot_tp = jnp.sum(t_linear * p_linear, axis=1)
    norm_t = jnp.linalg.norm(t_linear, axis=1)
    
    print(f"\nInterpolated frame quality:")
    print(f"  Orthogonality (t·p): max={jnp.max(jnp.abs(dot_tp)):.2e}")
    print(f"  Normalization (|t|): mean={jnp.mean(norm_t):.6f}")


def example_with_rotation():
    """Demonstrate frame rotation with alpha."""
    print("\n" + "=" * 60)
    print("Example 4: Frame Rotation")
    print("=" * 60)
    
    # Create curve
    order = 1
    n_phi = 16
    quadpoints = jnp.linspace(0.0, 1.0, n_phi, endpoint=False)
    
    dofs = jnp.array([
        0.0, 0.0, 1.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0
    ])
    
    curve = CurveXYZFourierJAX(quadpoints, dofs, order)
    
    # Create frame with no rotation
    framed_0 = make_rmf_frame(curve, alpha=jnp.zeros(n_phi))
    t0, p0, q0 = framed_0.rotated_frame()
    
    # Create frame with 45-degree rotation
    alpha_45 = jnp.ones(n_phi) * jnp.pi / 4
    framed_45 = make_rmf_frame(curve, alpha=alpha_45)
    t45, p45, q45 = framed_45.rotated_frame()
    
    # Tangent should be unchanged
    tangent_diff = jnp.linalg.norm(t0 - t45, axis=1)
    print(f"Tangent difference (should be ~0): max={jnp.max(tangent_diff):.2e}")
    
    # p should be rotated by 45 degrees
    # p45 should equal cos(45°)*p0 - sin(45°)*q0
    p_expected = jnp.cos(jnp.pi/4) * p0 - jnp.sin(jnp.pi/4) * q0
    p_diff = jnp.linalg.norm(p45 - p_expected, axis=1)
    print(f"p rotation error: max={jnp.max(p_diff):.2e}")
    
    # Check angle between p0 and p45
    cos_angle = jnp.sum(p0 * p45, axis=1)
    angle = jnp.arccos(jnp.clip(cos_angle, -1, 1))
    print(f"Angle between p0 and p45: mean={jnp.mean(angle):.4f} rad")
    print(f"                          (expected: {jnp.pi/4:.4f} rad)")


def example_jax_transformations():
    """Demonstrate JAX transformations (grad, jit, vmap)."""
    print("\n" + "=" * 60)
    print("Example 5: JAX Transformations")
    print("=" * 60)
    
    order = 1
    n_phi = 16
    quadpoints = jnp.linspace(0.0, 1.0, n_phi, endpoint=False)
    
    def make_curve_and_frame(dofs):
        """Create framed curve from DOFs."""
        curve = CurveXYZFourierJAX(quadpoints, dofs, order)
        framed = make_centroid_frame(curve)
        return framed
    
    def frame_quality_metric(dofs):
        """Compute a metric based on frame orthogonality."""
        framed = make_curve_and_frame(dofs)
        t, p, q = framed.rotated_frame()
        
        # Measure deviation from orthonormality
        dot_tp = jnp.sum(t * p, axis=1)
        dot_tq = jnp.sum(t * q, axis=1)
        dot_pq = jnp.sum(p * q, axis=1)
        
        ortho_error = jnp.sum(dot_tp**2 + dot_tq**2 + dot_pq**2)
        
        norm_t = jnp.linalg.norm(t, axis=1)
        norm_p = jnp.linalg.norm(p, axis=1)
        norm_q = jnp.linalg.norm(q, axis=1)
        
        norm_error = jnp.sum((norm_t - 1)**2 + (norm_p - 1)**2 + (norm_q - 1)**2)
        
        return ortho_error + norm_error
    
    # Test DOFs
    dofs = jnp.array([
        0.0, 0.0, 1.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0
    ])
    
    # Test JIT compilation
    print("Testing JIT compilation...")
    frame_quality_jit = jax.jit(frame_quality_metric)
    quality = frame_quality_jit(dofs)
    print(f"  Frame quality metric: {quality:.2e}")
    
    # Test gradient computation
    print("\nTesting gradient computation...")
    grad_fn = jax.grad(frame_quality_metric)
    gradient = grad_fn(dofs)
    print(f"  Gradient shape: {gradient.shape}")
    print(f"  Gradient norm: {jnp.linalg.norm(gradient):.2e}")
    
    # Test vmap (batch processing)
    print("\nTesting vmap (batch processing)...")
    key = jax.random.PRNGKey(0)
    batch_dofs = jax.random.normal(key, (5, len(dofs))) * 0.1
    batch_dofs = batch_dofs.at[:, 2].set(1.0)
    batch_dofs = batch_dofs.at[:, 6].set(1.0)
    
    batch_quality = jax.vmap(frame_quality_metric)(batch_dofs)
    print(f"  Batch size: {len(batch_dofs)}")
    print(f"  Quality metrics: {batch_quality}")
    
    print("\n✅ All JAX transformations work correctly!")


def main():
    """Run all examples."""
    jax.config.update("jax_enable_x64", True)
    
    example_basic_usage()
    example_rmf_vs_centroid()
    example_frame_interpolation()
    example_with_rotation()
    example_jax_transformations()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
