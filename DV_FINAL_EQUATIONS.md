# Final Equations For The Current `Dv` Implementation

This note records the equations currently implemented in
[quartic_observable_model.py](/Users/vincenzobarone/centrifugal/quartic_observable_model.py)
and exposed for the linear-molecule scalar constant `Dv` through
[dv_perturbative.py](/Users/vincenzobarone/centrifugal/dv_perturbative.py).

## 1. General projected-observable ansatz

The code is organized around a projected quartic observable model

```text
O(v) = O_e + sum_i g_i w_i(v)
```

where:

- `O` is the projected quartic observable;
- `O_e` is the equilibrium value;
- `g_i` are mode-resolved perturbative coefficients;
- `w_i(v)` are state factors determined by the final projection.

For the current linear-molecule `Dv` branch, the observable has one component.

## 2. Linear-molecule projection

For a linear molecule the code uses the Aliev-style state dependence

```text
Dv = D_J - sum_n beta_n (v_n + 1/2) - sum_t beta_t (v_t + 1)
```

with:

- `n`: parallel modes,
- `t`: perpendicular modes.

Equivalently, defining

```text
w_i(v) = v_i + 1/2    for parallel modes
w_i(v) = v_i + 1      for perpendicular modes
```

the implementation evaluates

```text
Dv(v) = D_J + sum_i g_i w_i(v)
beta_i = -g_i
```

So the internal general layer stores the additive projected coefficients `g_i`,
while the public `Dv` wrapper exposes the conventional `beta_i = -g_i`.

## 3. Semi-diagonal quartic input

Only reduced quartic constants of type `Phi(iiik)` are required.

They are extracted from single-mode finite differences of analytic Hessians in
normal coordinates:

```text
H_ik(q_i) = H_ik(0) + Phi(iik) q_i + (1/2) Phi(iiik) q_i^2 + ...
```

hence

```text
Phi(iiik) = [H_ik(+dq_i) - 2 H_ik(0) + H_ik(-dq_i)] / dq_i^2
```

This is the `O(2N)` Hessian route implemented by `compute_phi_iijk()`.

## 4. Reduced perturbative decomposition currently implemented

The present reduced model splits the projected linear coefficients into three
branches:

```text
g_i = g_i^(cor) + g_i^(4) + g_i^(3x3)
```

with the following implemented formulas.

### 4.1 Coriolis-Coriolis branch

For each pair `i < j`,

```text
S_ij^(cor) = C_ij^2 / (omega_i + omega_j)
```

and

```text
g_i^(cor) += - S_ij^(cor) / (2 omega_i)
g_j^(cor) += - S_ij^(cor) / (2 omega_j)
```

### 4.2 Direct semi-diagonal quartic branch

For each ordered pair `(i, k)`,

```text
S_ik^(4) = - Phi(iiik) / [8 omega_i (omega_i + omega_k)]
```

and the mode-resolved projected coefficients are accumulated as

```text
g_i^(4) += S_ik^(4)
g_k^(4) += p S_ik^(4)     for k != i
```

where `p = partner_weight` is currently a user-level parameter, default `1/2`.

### 4.3 Cubic-mediated dressing

The cubic-mediated semi-diagonal correction uses

```text
S_ik^(3x3) =
  + [Phi(iii) Phi(iik)] / [(2 omega_i + omega_k) 8 omega_i (omega_i + omega_k)]
```

with the same redistribution pattern

```text
g_i^(3x3) += S_ik^(3x3)
g_k^(3x3) += p S_ik^(3x3)     for k != i
```

The denominator `2 omega_i + omega_k` is regularized from below by the
parameter `resonance_floor_cm`.

## 5. Final implemented `Dv` equation

Combining all current reduced terms,

```text
g_i =
  - sum_{j!=i} [C_ij^2 / (2 omega_i (omega_i + omega_j))]
  - sum_k [Phi(iiik) / (8 omega_i (omega_i + omega_k))]
  + sum_k [Phi(iii) Phi(iik) / ((2 omega_i + omega_k) 8 omega_i (omega_i + omega_k))]
  + redistributed partner terms
```

and therefore

```text
Dv(v) = D_J + sum_i g_i w_i(v)
      = D_J - sum_i beta_i w_i(v)
```

with `beta_i = -g_i`.

## 6. Status relative to the full linear-theory formulas

The equations above are the final equations of the current code.

They are:

- explicit,
- modular,
- compatible with the `O(2N)` `Phi(iiik)` strategy,
- suitable as a scaffold for a future non-linear projection layer.

They are not yet the full line-by-line transcription of Aliev's complete
linear-molecule formulas with all auxiliary `X`, `F`, `U`, `V`, and `r` terms.
Those full expressions should enter later as a refinement of the coefficient
builder, not as a change to the projected-observable architecture.
