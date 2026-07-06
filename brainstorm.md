# Brainstorm: Linear parameterization satisfying BCs by construction

## The core tension

The "order-by-order" section is solving a harder problem than necessary. It seeks
a basis for the *intermediate coordinates* $(\bar\phi, \bar u, \bar v)$ with 
nice algebraic closure properties, which turns out to be impossible. But the 
nonlinearity of $\mathbf{r}$ in the intermediate coordinate approach stems entirely
from **$\bar\phi$ being a free variable** — because $\mathbf{r_0}(\bar\phi)$,
$\mathbf{n_1}(\bar\phi)$, $\mathbf{n_2}(\bar\phi)$ are nonlinear in $\bar\phi$.

If we freeze $\bar\phi = \phi$, then $\mathbf{r_0}$, $\mathbf{n_1}$, $\mathbf{n_2}$
become **known functions** of $\phi$, and $\mathbf{r}$ becomes **linear** in the
remaining unknowns. This is exactly the "Curve frame" parameterization (without $X$),
which was dismissed too hastily.

---

## Main proposal: Homogenized curve frame

Write

$$
\mathbf{r}(\phi,u,v) = \mathbf{r_0}(\phi) + Y(\phi,u,v)\,\mathbf{n_1}(\phi) + Z(\phi,u,v)\,\mathbf{n_2}(\phi)
$$

where $\mathbf{r_0}, \mathbf{n_1}, \mathbf{n_2}$ are computed **once** from the coil
centerline and treated as fixed. The BCs are:

$$
Y(\phi,\pm1,v) = \pm\tfrac{w_1}{2}, \qquad Z(\phi,u,\pm1) = \pm\tfrac{w_2}{2}.
$$

**Lift to homogeneous BCs.** Decompose:

$$
Y(\phi,u,v) = \tfrac{w_1}{2}u + \bar Y(\phi,u,v), \qquad 
Z(\phi,u,v) = \tfrac{w_2}{2}v + \bar Z(\phi,u,v),
$$

so $\bar Y$ must satisfy $\bar Y(\phi,\pm1,v) = 0$ and $\bar Z$ must satisfy
$\bar Z(\phi,u,\pm1) = 0$. Expand the corrections in a bubble basis:

$$
\bar Y(\phi,u,v) = \sum_{k,m,n} c_{kmn}\,\psi_k(\phi)\,(1-u^2)T_m(u)\,T_n(v),
$$

$$
\bar Z(\phi,u,v) = \sum_{k,m,n} d_{kmn}\,\psi_k(\phi)\,T_m(u)\,(1-v^2)T_n(v),
$$

where $\psi_k$ are Fourier modes in $\phi$ and $T_n$ are Chebyshev polynomials (or
any basis) in $u, v$. Any truncation of this sum satisfies the BCs **exactly**
and is **linear** in the coefficients $c_{kmn}, d_{kmn}$.

### Why convexity of $I_3$ (etc.) is preserved

$I_3$ is convex in $\mathbf{r}$ as an $L^2$ function (composition of norms with
linear differential operators). When $\mathbf{r}$ is itself linear in the coefficient
vector $\mathbf{a} = (c_{kmn}, d_{kmn})$, the composition
$\mathbf{a} \mapsto \mathbf{r} \mapsto I_3[\mathbf{r}]$ is convex in $\mathbf{a}$
(pre-composition with a linear map preserves convexity). The same applies to $I_1$
and $I_\text{LSE}$.

### Addressing the "perpendicularity" concern

The note in the Curve frame section says: *"removing $\mathbf{t}$ forces the mesh
coordinate to be perpendicular to the curve's tangents. This may be detrimental when
we need to deal with twists."*

This concern is overstated for two reasons:

1. **Torsion is already handled.** With $X = 0$, the $\phi$-layers are normal
   cross-sections of the centerline (perpendicular to $\mathbf{t}$). Torsion causes
   the $\mathbf{n_1}, \mathbf{n_2}$ frame to rotate around $\mathbf{t}$ as $\phi$
   advances, which is exactly the behavior we want — the mesh cross-sections rotate
   with the coil cross-section. This is already captured by the Frenet-Serret
   evolution of the frame.

2. **The physical boundary IS a normal cross-section.** The coil body's face at
   $u = +1$ consists of points $\mathbf{r_0}(\phi) + \tfrac{w_1}{2}\mathbf{n_1}(\phi)
   + Z\,\mathbf{n_2}(\phi)$, i.e. it lies exactly in the normal plane. There is no
   physical reason to allow tangential slip on the boundary faces.

---

## Optional extension: tangential interior DOF

If tangential redistribution of interior points is desired (e.g. to equalize
$|\mathbf{e}_\phi|$ along the coil), include a tangential term:

$$
\mathbf{r} = \mathbf{r_0}(\phi) + X(\phi,u,v)\,\mathbf{t}(\phi) + Y\,\mathbf{n_1}(\phi) + Z\,\mathbf{n_2}(\phi).
$$

The BC on $X$ is: $X = 0$ on all four boundary faces (since the physical coil
faces lie in normal planes, with no tangential offset). This means $X$ must vanish
at $u = \pm1$ **and** at $v = \pm1$, so it expands as a **double bubble**:

$$
X(\phi,u,v) = \sum_{k,m,n} e_{kmn}\,\psi_k(\phi)\,(1-u^2)T_m(u)\,(1-v^2)T_n(v).
$$

This is again linear in coefficients and satisfies all BCs by construction.
Linearity and convexity of $I_3$ etc. are unaffected.

---

## Summary

| Approach | Linear in params? | BCs by construction? | Notes |
|---|---|---|---|
| Curve frame (with $X$, original) | Yes | No | BC hard to encode with $X$ present |
| Intermediate coordinate | No | Yes | Nonlinear through $\bar\phi$ |
| **Homogenized curve frame (proposed)** | **Yes** | **Yes** | Freeze $\phi = \bar\phi$, lift $Y, Z$ |
| Homogenized curve frame + $X$ DOF | Yes | Yes | $X$ is double bubble, optional |

The proposed approach is essentially the "Curve frame" idea re-examined after
recognizing that: (a) $X = 0$ is the correct BC (not an arbitrary restriction),
and (b) the BCs on $Y$ and $Z$ are simple Dirichlet and can be homogenized
by subtracting the identity map $Y_0 = \tfrac{w_1}{2}u$, $Z_0 = \tfrac{w_2}{2}v$.


