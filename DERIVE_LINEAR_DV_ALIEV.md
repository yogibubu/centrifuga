# Deriving The Linear `Dv` Equations Instead Of Copying Aliev

This note fixes the derivation strategy that should be followed from now on.

The goal is not to transcribe Aliev's scanned formulas by hand. The goal is to
derive the linear-molecule vibrational dependence of `Dv` from the projected
quartic effective Hamiltonian and then verify that the final result is
equivalent to Aliev.

## 1. Target statement

For a linear molecule we want to recover

```text
Dv = D_J - sum_n beta_n (v_n + 1/2) - sum_t beta_t (v_t + 1),
```

where:

- `n` labels parallel modes,
- `t` labels perpendicular modes.

This is the target form. It is not the starting assumption for the derivation.

## 2. Starting point

The derivation should start from the projected quartic block of the effective
rovibrational Hamiltonian,

```text
H_eff^(4,rot) = projected coefficient of the quartic rotational operator block.
```

The observable-first point of view is:

```text
Dv = coefficient of [J(J+1)]^2 in <v | H_eff | v>.
```

For the future non-linear extension, this same logic will produce a vector of
Watson quartic constants instead of a single scalar `Dv`.

## 3. Operator decomposition to derive

The derivation must separate the quartic projected block into explicit
perturbative branches:

```text
H_eff^(4,rot) =
    H_eff^(cor-cor)
  + H_eff^(3x3)
  + H_eff^(4dir)
  + H_eff^(mixed, if needed).
```

The intended meaning is:

- `cor-cor`: second-order elimination of Coriolis couplings,
- `3x3`: cubic-cubic mediated contribution,
- `4dir`: direct quartic insertion,
- `mixed`: only if the full operator derivation shows additional terms that do
  not reduce to the previous three families.

## 4. Linear reduction

Once the projected quartic operator is available, the linear reduction should
proceed in three stages.

### 4.1 Mode partition

Split the vibrational modes into:

- parallel modes `n`,
- perpendicular modes `t`.

### 4.2 Diagonal vibrational projection

Take diagonal matrix elements on the vibrational state:

```text
<v | H_eff^(4,rot) | v>.
```

At this stage the result must be organized as a sum of:

- equilibrium contribution,
- coefficients multiplying `v_n + 1/2`,
- coefficients multiplying `v_t + 1`.

### 4.3 Quartic rotational projection

Collect the coefficient multiplying the scalar linear-molecule quartic operator
`[J(J+1)]^2` or its equivalent ` [J(J+1)-l^2]^2 ` carrier before the final
scalar reduction.

The final expression then defines:

```text
beta_n, beta_t.
```

## 5. What the current code already gives us

The current implementation already provides the software architecture that this
derivation needs:

- reduced input layer:
  - frequencies,
  - Coriolis constants,
  - cubic force constants,
  - semi-diagonal quartics `Phi(iiik)`,
- `O(2N)` extraction of `Phi(iiik)` from single-mode Hessian scans,
- observable-first projected model in
  [quartic_observable_model.py](/Users/vincenzobarone/centrifugal/quartic_observable_model.py),
- linear `Dv` wrapper in
  [dv_perturbative.py](/Users/vincenzobarone/centrifugal/dv_perturbative.py),
- symbolic scaffold in
  [linear_dv_derivation.py](/Users/vincenzobarone/centrifugal/linear_dv_derivation.py).

This means the missing piece is not the architecture. The missing piece is the
full coefficient builder derived from the operator algebra.

## 6. What is still missing for equivalence to Aliev

The current reduced builder is not yet equivalent to Aliev because the full
linear derivation introduces auxiliary objects of the schematic type

```text
X, F, U, V, r
```

and multi-index denominator structures that have not yet been derived inside
our own formalism.

Therefore the remaining tasks are:

1. derive the explicit operator elimination that generates those structures;
2. reduce them to a linearly projected scalar quartic observable;
3. identify the resulting `beta_n` and `beta_t`;
4. compare against Aliev as an external check.

## 7. Criterion of success

The linear derivation will be considered complete only when all of the
following hold:

1. the formulas for `beta_n` and `beta_t` are derived internally from the
   projected quartic Hamiltonian;
2. the current reduced builder becomes an identified truncation or disappears;
3. the final linear formulas can be shown to be algebraically equivalent to
   Aliev's expressions;
4. the same projected-observable framework remains reusable for the future
   non-linear quartic constants.
