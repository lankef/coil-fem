# Casing / Winding-Pack Interface Modeling

Notes from a discussion (Sep 2026) about alternatives to the hard, mesh-fitted
winding-pack (WP) / casing interface on the `casing-hard-interface` branch,
where thin casings produce poor mesh quality and inaccurate results.

## 1. Diffuse vs. sharp interfaces

The idea: extend `u, v` beyond 1 (as on the branch), place the WP/casing
boundary at `u, v = ±1`, but assign material properties and current by an
indicator (sharp or smooth) instead of cell boundaries.

### Terminology

- **Immersed interface method (IIM)** is actually a *sharp* method: it
  modifies finite-difference stencils so the jump conditions
  (`[u] = 0`, `[σ n] = 0`) hold exactly (LeVeque & Li, *SIAM J. Numer. Anal.*
  31:1019, 1994; Li & Ito, *The Immersed Interface Method*, SIAM 2006). The FE
  analogue (IFEM) modifies shape functions in cut elements (Li, Lin & Wu,
  *Numer. Math.* 96:61, 2003).
- **What is proposed here** is a diffuse / smeared / unfitted material
  interface:
  - sharp indicator at quadrature points: "voxel/image-based FEM", or the
    multi-material finite cell method (FCM) without enrichment (Parvizian,
    Düster & Rank, *Comput. Mech.* 41:121, 2007);
  - smooth transition: regularized Heaviside `H_ε(d)` with band `ε ~ 1–2h`, as
    in level-set methods (Osher & Fedkiw, *Level Set Methods and Dynamic
    Implicit Surfaces*, 2003) and diffuse immersed-boundary methods (Peskin,
    *Acta Numerica* 11:479, 2002). Review of diffuse vs. sharp families:
    Mittal & Iaccarino, *Annu. Rev. Fluid Mech.* 37:239, 2005.
- **Sharp but unfitted FEM** (same mesh, accurate interface, more machinery):
  - XFEM with weak-discontinuity enrichment (Sukumar et al., *CMAME* 190:6183,
    2001; Moës et al., *CMAME* 192:3163, 2003);
  - Nitsche/CutFEM (Hansbo & Hansbo, *CMAME* 191:5537, 2002; Burman et al.,
    *IJNME* 104:472, 2015);
  - enriched FCM (Joulaian & Düster, *Comput. Mech.* 52:741, 2013).

### Do basis functions / element types need to change?

- **Not for the diffuse version.** TET4/TET10 are fine. `LinearElasticity3D`
  already stores everything per quadrature point
  (`src/coil_fem/problems/linear_elasticity.py`, `custom_init`):
  `material_id_q`, `lam_q`, `mu_q`, `rho_q`, `current_weight_q`, `eps_th_q`.
  Today `material_id_q` is the per-cell ID broadcast over quad points; a
  diffuse model would compute it (or a blend weight) from `mesh.uv_quad`
  (`|u| > 1 or |v| > 1`).
- `(u, v)` are fixed reference coordinates, so the material pattern does not
  change during optimization; gradients w.r.t. coil DOFs are unaffected even
  with a sharp indicator.
- **Accuracy limit.** Across a material interface, displacement is
  continuous but strain jumps (weak discontinuity / kink). Polynomials inside
  one element cannot represent a kink. The solution is only `H^{3/2-ε}` near
  the interface, so standard interpolation estimates (Brenner & Scott, *The
  Mathematical Theory of Finite Element Methods*, 3rd ed., Ch. 4; Ciarlet,
  1978) give energy-norm error `O(h^{1/2})` and L2 error `O(h)`, regardless of
  polynomial order (Babuška, *Computing* 5:207, 1970). TET10 buys nothing
  asymptotically in the cut-element band, and pointwise stress in cut elements
  has an `O(1)` error.
- Recovering interface accuracy without a fitted mesh requires changing the
  basis (IFEM, XFEM ridge enrichment, CutFEM + ghost penalty). XFEM also has
  blending-element and conditioning issues (Fries & Belytschko, *IJNME*
  84:253, 2010). Not recommended here.

## 2. How material properties are stored

Per-quadrature-point storage is **not** a smooth field in the FEM basis. The
element integral is evaluated by quadrature,

```
K_e ≈ Σ_q w_q B(x_q)^T C(x_q) B(x_q) J(x_q),
```

so `C` is only ever sampled at `x_q`. Any `C(x)` with those point values gives
the same discrete system; nothing interpolates between points or enforces
continuity across elements. Currently the material is piecewise constant per
cell.

A nodal field interpolated by shape functions,
`ρ(x_q) = Σ_a N_a(x_q) ρ_a`, *is* a continuous field in the FEM space — a
different design choice (used by some topology-optimization codes).

Per-quad storage already allows varying material *within* a cell, but each
cell only sees a few samples, so a sharp jump between samples effectively moves
to wherever the samples are (see quadrature error below).

## 3. Pitfalls of the density-function (SIMP) approach

The JAX-FEM topology-optimization example is SIMP (Bendsøe & Sigmund,
*Topology Optimization*, Springer 2003). Its classic pathologies mostly come
from *optimizing* the density and largely don't apply to a prescribed density:
checkerboarding and mesh dependence (Sigmund & Petersson, *Struct. Optim.*
16:68, 1998), gray material, the stress-singularity problem (Duysinx &
Bendsøe, *IJNME* 43:1453, 1998; Bruggi, *SMO* 36:125, 2008), ill-conditioning
from near-zero void stiffness (Allaire, Jouve & Toader, *JCP* 194:363, 2004).
They become relevant if casing thickness becomes a design variable (then a
smooth `H_ε` is needed for gradients).

Issues that do apply:

1. **Casing stress is unreliable for thin casings.** If `t ≲ 2h`, almost every
   casing quad point is in a cut element with `O(1)` stress error. Casing is
   usually stress-limited, with peaks at the interface and corners. A smooth
   transition doesn't fix this: keeping modeling error small needs `ε ≪ t`,
   hence `h ≪ t`, which removes the mesh savings.
2. **Quadrature error.** Gauss rules assume smooth integrands; a sharp
   indicator gives staircasing and a wrong casing volume fraction. At low
   quadrature order this collapses to per-cell assignment. FCM fix: sub-cell
   quadrature in cut cells only (Düster et al., *CMAME* 197:3768, 2008) — cheap
   here since cut cells are known ahead of time in the structured `(u, v)`
   grid.
3. **Mixing-rule bias.** Point-wise mixing inside an element behaves like a
   Voigt (arithmetic) average: too stiff for loads normal to the interface,
   where Reuss (harmonic) is correct (Hill, *Proc. Phys. Soc. A* 65:349,
   1952). Contrast here (steel vs. smeared WP) is moderate, so secondary.
4. **Net current / force error (code-specific).** `CoilFEM` normalizes
   `J_q = (I / A_conductor) * current_weight_q * t_hat_q` with the exact
   `A_conductor = w1 * w2`. With a quad-point indicator,
   `∫ current_weight dA ≠ w1 * w2` in general, so total current and Lorentz
   load are off by the volume-fraction error. Normalize by
   `∫ current_weight dA` instead (the in-code comment already anticipates
   this).
5. **Thermal mismatch is smeared.** With per-material `itc`, cooldown stress
   concentrates at the WP/casing interface — where the diffuse model is least
   accurate.

### Cheaper fixes to try first

- Grade the WP grid toward `u, v = ±1` so cells next to the casing match the
  casing cell size.
- Thin, flat elements are fine for interpolation if no angle approaches 180°
  (Babuška & Aziz, *SINUM* 13:214, 1976; Apel, *Anisotropic Finite Elements*,
  1999). One TET10 layer through the casing is often fine; TET4 in thin
  bending-dominated layers is the real problem (Bathe, *Finite Element
  Procedures*, 2nd ed.).
- The mesh is a structured hex grid Kuhn-split into 6 tets per hex; JAX-FEM's
  HEX20/HEX27 handle thin layers much better, but switching touches the
  surface maps and curved-element code.

Recommendation: if only global displacement and WP stress are needed, the
diffuse model with sub-cell quadrature and corrected current normalization is
reasonable. If casing stress is needed, keep the fitted interface and fix the
grading.

## 4. Radius ratio 3.4 at the edge elements

- ParaView's tet radius ratio (Verdict) is `R / (3r)`; regular tet = 1;
  Verdict "acceptable" range is 1–3.
- Kuhn-splitting a perfect cube already gives ≈ 1.39, so 3.4 corresponds to
  moderately stretched hex cells, not degenerate ones.
- Accuracy degrades when angles approach 180°, not with radius ratio per se.
  Kuhn tets are orthoschemes: on an undistorted hex cell all dihedral angles
  are ≤ 90°, however stretched. Coil curvature/twist distorts this somewhat
  but won't create near-180° angles.
- Stretching mainly costs resolution along the long direction (likely `φ`).

What makes the **peak von Mises** suspect:

1. **Corner singularities.** At the WP corner embedded in the casing (a 90°/270°
   bonded bimaterial corner), stress is generally infinite for dissimilar
   stiffness or thermal contraction (Bogy, *J. Appl. Mech.* 38:377, 1971;
   Sinclair, *Appl. Mech. Rev.* 57:251, 2004). The outer 90° traction-free
   corner is not singular (Williams, *J. Appl. Mech.* 19:526, 1952), but abrupt
   Winkler support-stiffness changes can create sharp concentrations.
2. **Pointwise stress converges slowly.** TET10 stress is linear per element
   and discontinuous between elements; `max_von_mises_hard` (max over quad
   points) is the least converged quantity available (Szabó & Babuška, *Finite
   Element Analysis*, 1991).
3. **`max_von_mises_lse` is mesh-dependent**: it sums over quad points
   without volume weighting, so the same field gives a different value on a
   finer mesh.

**Check:** refine `n_grid_1`, `n_grid_2`, `n_casing`, `n_phi` together 2–3
times and track the peak value and location.

- Peak levels off → trustworthy; 3.4 is not a problem.
- Peak keeps rising ~ a power of `1/h` at the WP/casing corner → singularity.
  Use a converging measure (volume-weighted p-norm / KS aggregate,
  small-volume average, stress at a fixed distance) and treat the hard max as
  an artifact of the idealized sharp corner.
- Peak on the outer edge that wanders with `φ` refinement → `φ` resolution too
  coarse; shorten cells along the coil.

## 5. Does a smooth transition remove the singularity?

Partly.

- With a `C^1` blend of fixed width `ε`, the coefficients are smooth and
  elliptic regularity gives `H^2` displacement near the interface (Evans,
  *Partial Differential Equations*, §6.3), so stress is bounded. With `ε`
  fixed and `h ≪ ε`, the peak converges.
- A sharp quad-point indicator does **not** regularize — it is still a jump,
  placed as a staircase.
- The peak now depends on `ε`; as `ε → 0` the singular problem returns (expect
  growth roughly like `ε^{λ-1}`).
- Tying `ε` to the mesh (`ε ≈ 1–2h`) changes nothing: the peak still grows
  with refinement.
- Converging at fixed `ε` needs `h ≪ ε ≪ t`, so thin-casing mesh savings
  largely disappear.
- Other stress raisers (abrupt Winkler stiffness changes, `φ` under-resolution)
  are untouched.

Defensible when `ε` is physical: real WPs have rounded corners and ground
insulation (glass-epoxy) between conductor and case. Choose `ε` to match that
layer or corner radius, keep it fixed, refine `h` below it. A related trick:
define the interface by a rounded level set, e.g. `|u|^p + |v|^p = 1`
(superellipse) instead of `max(|u|, |v|) = 1`, which fillets the interface
corner without meshing one.

Without a physical length scale: keep either model, but judge designs on a
volume-weighted p-norm / KS aggregate or stress at a fixed distance, not the
hard max.

## 6. Thin casing under a memory limit: common practice

1. **Casing as a stiff surface layer (surface elasticity / stiff interface).**
   Collapse the thin, stiff casing into a membrane on the WP outer surface:

   ```
   Π_casing = ∫_Γ (t/2) ε_s : C_ps : ε_s dA,   ε_s = P ε P,   P = I − n⊗n
   ```

   with `C_ps` the casing plane-stress stiffness; casing membrane stress is
   `σ_c = C_ps : ε_s`. Theory: Benveniste & Miloh, *Mech. Mater.* 33:309
   (2001); Hashin, *J. Mech. Phys. Solids* 50:2509 (2002); Gurtin & Murdoch,
   *Arch. Rational Mech. Anal.* 57:291 (1975). FE implementation: Javili &
   Steinmann, *CMAME* 198:2198 (2009).
   - No new DOFs, no casing cells; it is a surface integral over the same
     exterior faces the Winkler term already uses (`_sel_face_sv`, surface
     quad points, Nanson scaling). Sparsity pattern unchanged (good for
     cuDSS).
   - Limits: membrane only (casing bending, smaller by ~`(t/w)^2`, and
     through-thickness stress are lost); no casing-corner concentrations;
     valid for `t ≪ w` and casing stiff relative to WP.
2. **Global–local analysis (submodeling).** Coarse global model (casing
   smeared/homogenized or as a surface layer) for load paths, plus a small
   refined local model of the critical region — usually a 2D generalized
   plane-strain cross-section at the most loaded `φ`, or a short 3D segment —
   driven by global displacements. The global model is what gets
   differentiated; the local model is a post-check or occasional constraint.
   Standard for large TF and stellarator coil analyses.
3. **Elements suited to one thin layer.** One layer of HEX20/HEX27 or
   p-refinement (Szabó & Babuška); solid-shell elements with assumed-strain
   formulations to avoid locking (Hauptmann & Schweizerhof, *IJNME* 42:49,
   1998; Dvorkin & Bathe, 1984, MITC shells). Big change here; the cheap
   version is `n_casing = 1` with TET10 plus WP grading toward the casing.

**Also check what uses the memory.** With a direct solver, memory is usually
dominated by factorization fill-in, not the mesh. Adding cells only through
the thin casing band may cost less than refining the whole grid; profile cuDSS
memory vs. `n_casing` separately from `n_grid_*` and `n_phi`.

**Recommendation:** implement the surface-membrane model, validate it against
the current hard-interface model at an affordable resolution (global
displacement, casing membrane stress away from corners), and use a 2D
cross-section submodel when the corner peak is needed.
