---
title: "Paper Review"
subtitle: "Overcoming a Genuine Fold in Two-Phase PEM Fuel Cell Cathode Simulations via Pseudo-Arclength Continuation"
author: "신강현"
format:
  revealjs:
    theme: default
    embed-resources: true
    transition: none
    slide-number: true
---

## Research Objective

### Main Problem

Two-phase PEMFC cathode simulations become difficult under severe flooding.

- Standard continuation fails near a critical overpotential
- Is this failure caused by numerical difficulty or a genuine fold?
- Can the solution branch be traced beyond the fold?

---

### Approach

- M² two-phase mixture model
- FEniCSx / DOLFINx
- Jacobian SVD analysis
- Pseudo-arclength continuation

---

## Physical Model

### Cathode GDL

The model describes two-phase transport of gas and liquid water in the cathode GDL.

Primary unknowns:

$$\mathbf{w}
=
\left(
\hat p,\;
C_{H_2O},\;
C_{O_2,g}
\right)
$$

- $\hat p$ : mixture pressure
- $C_{H_2O}$ : mixture water mass fraction
- $C_{O_2,g}$ : gas-phase oxygen mass fraction

Liquid saturation $s$ is calculated from $C_{H_2O}$.

---

## Governing Equations

### Mixture Pressure

$$
\nabla\cdot(\kappa\nabla p)=\text{membrane source}
$$

Darcy velocity:

$$
\mathbf{u}_{Darcy}
=
-\frac{\kappa}{\rho_{mix}}\nabla p
$$

### Water Transport

$$
\nabla\cdot
\left(
D_{w,eff}\nabla C_{H_2O}
-
\mathbf{F}_{conv,w}
\right)
=
0
$$

---

## Governing Equations

### Oxygen Transport

$$
\nabla\cdot
\left[
D_{O_2,eff}\nabla C_{O_2,g}
-
C_{O_2,g}
\left(
\mathbf{W}_{O_2}+\mathbf{J}_l
\right)
\right]
=
0
$$

### Liquid Saturation

Liquid saturation is obtained from the mixture water mass fraction.

Relative permeabilities:

$$
k_{rl}=s^n
$$

$$
k_{rg}=(1-s)^n
$$

As $s$ increases, gas transport becomes increasingly restricted.

---

## Electrochemical Coupling

Local current density:

$$
i_{loc}
=
(1-s)i_0(T)
\frac{C_{O_2}}{C_{O_2,ref}}
\exp
\left(
\frac{\alpha_cF\eta}{RT}
\right)
$$

The model contains strong nonlinear coupling:

```{mermaid}
flowchart LR
    A["Overpotential η ↑"] --> B["Current density ↑"]
    B --> C["Water production ↑"]
    C --> D["Liquid saturation ↑"]
    D --> E["O₂ transport ↓"]
    E --> B
```

---

## Standard Continuation

For a prescribed overpotential:

$$
\mathbf{F}(\mathbf{w},\eta)=0
$$

Standard continuation changes $\eta$ step by step:

$$
\eta_0
\rightarrow
\eta_1
\rightarrow
\eta_2
\rightarrow
\cdots
$$

The previous solution is used as the initial guess for the next Newton solve.

### Problem

Standard continuation stops near

$$
\boxed{\eta\approx0.356\ {\rm V}}
$$

---

## Why Does Newton Fail?

Newton's method uses the Jacobian

$$
J
=
\frac{\partial\mathbf{F}}
{\partial\mathbf{w}}
$$

and solves

$$
J\Delta\mathbf{w}
=
-\mathbf{F}
$$

At a fold,

$$
J
\rightarrow
\text{singular}
$$

The overpotential $\eta$ is no longer a suitable local coordinate for tracing the solution branch.

---

## Detecting the Fold with SVD

Singular Value Decomposition:

$$
J
=
U\Sigma V^T
$$

where

$$
\Sigma
=
\operatorname{diag}
\left(
\sigma_1,\sigma_2,\ldots,\sigma_{min}
\right)
$$

Near a singular point,

$$
\boxed{
\sigma_{min}
\rightarrow
0
}
$$

Therefore the smallest singular value can be used to detect the approach to a fold.

---

## Evidence for a Genuine Fold

The paper observes a sharp collapse of $\sigma_{min}$ near the fold.

| Mesh | Fold location | $\sigma_{min}$ collapse |
|---|---:|---:|
| $24\times60$ | $\eta\approx0.356$ V | 4.6 orders |
| $36\times90$ | $\eta\approx0.357$ V | 4.8 orders |

The collapse occurs within only

$$
\Delta\eta
\approx
0.0001
-
0.0002\ {\rm V}
$$

The similar fold locations for the two meshes support the interpretation of a genuine fold rather than simple Newton failure.

---

## Pseudo-Arclength Continuation

Instead of describing the solution as $\mathbf{w}(\eta)$,

$$
\left(
\mathbf{w},\eta
\right)
=
\left(
\mathbf{w}(s),\eta(s)
\right)
$$

is used.

<div style="text-align:center;">

<svg width="760" height="400" viewBox="0 0 760 400">

  <line x1="80" y1="350" x2="650" y2="350"
        stroke="black" stroke-width="2"/>
  <line x1="80" y1="350" x2="80" y2="35"
        stroke="black" stroke-width="2"/>

  <text x="650" y="380" font-size="20">state u</text>
  <text x="35" y="45" font-size="20">η</text>

  <path d="M130 330 Q350 50 530 325"
        fill="none" stroke="#2468a2" stroke-width="4"/>

  <text x="470" y="300" font-size="17"
        fill="#2468a2">F(u,η) = 0</text>

  <circle cx="260" cy="107" r="7" fill="black"/>
  <text x="205" y="100" font-size="17">xₙ</text>

  <line x1="225" y1="143" x2="327" y2="37"
        stroke="#777777" stroke-width="2"/>
  <text x="220" y="165" font-size="16">tangent tₙ</text>

  <circle cx="327" cy="37" r="7" fill="#d1495b"/>
  <text x="338" y="32" font-size="17">x_pred</text>

  <line x1="300" y1="12" x2="395" y2="105"
        stroke="#888888" stroke-width="2"
        stroke-dasharray="7,6"/>

  <text x="405" y="110" font-size="16">normal plane</text>
  <text x="405" y="130" font-size="14">(line in 2D)</text>

  <circle cx="350" cy="61" r="8" fill="#2a9d4b"/>
  <text x="365" y="64" font-size="17">xₙ₊₁</text>

  <line x1="268" y1="99" x2="316" y2="48"
        stroke="#d1495b" stroke-width="2"/>

</svg>

</div>

---

## Predictor

At the current solution $\mathbf{x}_n$, calculate the tangent $\mathbf{t}_n$.

Move a distance $\Delta s$ along the tangent:

$$
\boxed{
\mathbf{x}_{pred}
=
\mathbf{x}_n
+
\Delta s\,\mathbf{t}_n
}
$$

However, the predictor is only an approximation.

In general,

$$
F(\mathbf{x}_{pred})
\neq
0
$$

Therefore a correction is required.

---

## Corrector

The next solution must satisfy the original governing equations:

$$
F(\mathbf{w},\eta)
=
0
$$

and the pseudo-arclength constraint:

$$
\dot{\mathbf{w}}_0\cdot
(\mathbf{w}-\mathbf{w}_0)
+
\dot{\eta}_0
(\eta-\eta_0)
-
\Delta s
=
0
$$

The corrector therefore finds the intersection between

- the solution branch $F=0$
- the plane normal to the tangent direction

The intersection becomes the next solution $\mathbf{x}_{n+1}$.

---

## Why Can It Pass the Fold?

Standard continuation uses $\eta$ as the control parameter.

Before the fold:

$$
\eta_0
<
\eta_1
<
\eta_2
<
\eta_3
$$

However, after the fold the branch turns backward:

$$
0.34
\rightarrow
0.35
\rightarrow
0.356
\rightarrow
0.35
\rightarrow
0.34
$$

Pseudo-arclength instead follows

$$
s_0
<
s_1
<
s_2
<
s_3
<
s_4
$$

Therefore $\eta$ is allowed to increase or decrease while the solution branch is continuously followed.

---

## Main Result: First Fold

Pseudo-arclength continuation successfully traces through the fold.

Fold location:

$$
\boxed{
\eta_{fold}
\approx
0.356
-
0.357\ {\rm V}
}
$$

After reaching the maximum overpotential,

$$
\frac{d\eta}{ds}
$$

changes sign and $\eta$ begins to decrease.

This post-fold branch cannot be reached by ordinary $\eta$-parameterized continuation.

---

## Flooding Across the Fold

Before the fold:

$$
s_{max}
\approx
0.01
$$

After passing the fold:

$$
s_{max}
\approx
0.55
-
0.57
$$

The fold is therefore accompanied by a large change in liquid-water saturation.

However, $s_{max}$ subsequently decreases toward approximately

$$
s_{max}
\approx
0.30
$$

The mechanism responsible for this saturation overshoot is not yet clearly identified.

---

## Second Obstruction

Continuation encounters another problem near

$$
\boxed{
\eta
\approx
0.33
-
0.34\ {\rm V}
}
$$

This point behaves differently from the first fold.

- The corrector tends to move from $s_{max}\approx0.31$ to a low-saturation solution
- The Jacobian singularity is less clear than at the first fold
- The oxygen right-null-vector component is strongly localized near the outlet
- Local flow reversal occurs at part of the outlet

Therefore the second obstruction cannot be classified as a simple fold using the same evidence.

---

## Outlet Boundary Problem

The outlet condition assumes genuine outflow:

$$
\mathbf{F}\cdot\mathbf{n}
>
0
$$

Under severe flooding, local flow reversal occurs:

$$
\mathbf{F}\cdot\mathbf{n}
<
0
$$

The paper applies an outflow-only stabilization:

$$
\boxed{
\mathbf{F}\cdot\mathbf{n}
\rightarrow
\max
\left(
\mathbf{F}\cdot\mathbf{n},
0
\right)
}
$$

This improves the numerical field quality but does not completely remove the second obstruction.

---

## Discussion

### Main Contributions

1. A genuine fold was identified using Jacobian SVD diagnostics.
2. Pseudo-arclength continuation successfully passed the fold.
3. A post-fold and strongly flooded solution branch was obtained.
4. A numerical issue associated with local outlet flow reversal was identified.

### Remaining Questions

- What causes the saturation overshoot after the first fold?
- Are two distinct fully converged solutions present at the second obstruction?
- Is the second obstruction a higher-codimension bifurcation?
- Would further mesh refinement preserve the same behavior?

---

## Conclusion

```{mermaid}
flowchart LR
    A["Standard continuation"] --> B["η ≈ 0.356 V"]
    B --> C["Jacobian singularity"]
    C --> D["Pseudo-arclength"]
    D --> E["Post-fold branch"]
    E --> F["Second obstruction"]
```

### Key Message

A solver failure in a strongly nonlinear PEMFC model does not necessarily indicate only a numerical problem.

In this model, the first failure corresponds to a genuine fold in the steady-state solution branch.

---

## References

Abdollahzadeh, M., Pascoa, J. C., Ranjbar, A. A., & Esmaili, Q. (2014).  
*Analysis of PEM (Polymer Electrolyte Membrane) Fuel Cell Cathode Two-Dimensional Modeling.* Energy, 68, 478–494.

Keller, H. B. (1977).  
*Numerical Solution of Bifurcation and Nonlinear Eigenvalue Problems.*

Baratta, I. A., Dean, J. P., Dokken, J. S., et al. (2023).  
*DOLFINx: The Next Generation FEniCS Problem Solving Environment.*
