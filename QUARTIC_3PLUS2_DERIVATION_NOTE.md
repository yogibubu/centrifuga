# Quartic 3+2 Decomposition: Current Derivation Note

## Main outcome

The current code and linear-algebra diagnostics support the following statement:

- the quartic problem does **not** admit a direct linear `3+2` splitting in the
  5D Watson parameter space;
- however, a natural set of **three spatial invariants** is already available:
  the eigenvalues of the Gaussian-consistent reduced matrix `Tau'`.
- the current fundamental tensorial description is therefore best viewed as a
  `3+3` spectral parameterization of `Tau'`, with any later `3+2` structure
  understood as a reduction or gauge choice built on top of that level.

## 1. Why the Watson 5-space is not the right place

For a fixed reduction (A or S), the forward map

```text
tau(6) -> Watson(5)
```

has rank 5 and therefore a 1D nullspace.

This excludes a literal interpretation of `3 spatial + 2 gauge` as a linear
decomposition of the Watson constants themselves.

## 2. The right object: Tau'

Given the compressed pair-pair quartic tensor

```text
tau = (tau_aaaa, tau_bbbb, tau_cccc, tau_aabb, tau_aacc, tau_bbcc),
```

the Gaussian-consistent reduced matrix is

```text
Tau' =
[ tau_aaaa      tau_aabb/2    tau_aacc/2 ]
[ tau_aabb/2    tau_bbbb      tau_bbcc/2 ]
[ tau_aacc/2    tau_bbcc/2    tau_cccc   ].
```

This is a real symmetric `3x3` matrix in `R^3`.

Under pure representation changes (axis permutations), `Tau'` transforms by
permutation similarity:

```text
Tau' -> P^T Tau' P
```

and therefore its eigenvalues are invariant.

## 3. Candidate spatial invariants

The three eigenvalues of `Tau'`,

```text
lambda_1, lambda_2, lambda_3,
```

are the strongest current candidates for the three spatial quartic invariants.

Equivalent invariant sets are:

- `(lambda_1, lambda_2, lambda_3)`,
- `(tr(Tau'), J2(Tau'), det(Tau'))`,

where

```text
J2 = 1/2 [ tr(Tau')^2 - tr(Tau'^2) ].
```

These invariants are now exposed in the workflow.

## 4. What remains unresolved

The remaining two coordinates needed for a full `3+2` decomposition are not yet
identified in closed form.

At present we only know:

- they are **not** simply the two columns `G1,G2` used in the current pair-pair
  ansatz;
- they are **not** the 1D null gauge of the fixed Watson forward map;
- they must describe how the physical `Tau'` tensor is embedded into the Watson
  parametrization used for the spectroscopic constants.

## 4a. Current concrete fallback: spectral coordinates

As an explicit tensorial parameterization of the H12H12 quartic spatial object,
the code now exposes the full spectral coordinates of `Tau'`:

- three eigenvalues,
- three ZYZ Euler angles for the eigenframe.

This gives a complete `3 + 3` description of the symmetric `3x3` tensor in
`R^3`. The open CeDiTT-side problem is to understand how this full spectral
description reduces to the intended `3 + 2` structure.

## 5. Working interpretation

The most plausible current interpretation is:

- `3 spatial`: spectral invariants of `Tau'`,
- `2 residual coordinates`: representation/gauge embedding coordinates needed to
  recover the full Watson parameter vector from the spatial quartic tensor.

This keeps the quartic problem genuinely rooted in `R^3`, while avoiding the
incorrect identification of the old Coriolis `3x3` metric with the full spatial
quartic tensor.
