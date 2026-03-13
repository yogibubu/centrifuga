# Quartic Spatial/Gauge Status Report

## Scope

This note summarizes the current status of the quartic tensor decomposition in
the codebase and clarifies what is already established, what is excluded by the
current linear algebra, and what remains to be derived.

The target conceptual structure is:

```text
5 independent quartic objects = 3 spatial invariants + 2 gauge coordinates
```

The central question is whether this `3+2` split can be realized directly in
the current linear Watson/tau machinery.

## 1. What is already established

### 1.1 Order-2 quartics are numerically validated

For `h2o`, the order-2 quartic route based on the validated Gaussian/QCent-like
pipeline reproduces the Gaussian benchmark at roundoff level in practice.

### 1.2 The compressed quartic tensor has 6 components

The pair-pair quartic tensor is represented as

```text
tau = (tau_aaaa, tau_bbbb, tau_cccc, tau_aabb, tau_aacc, tau_bbcc)
```

and the Watson projection is implemented consistently via

```text
tau -> Tau' -> T -> cylindrical T -> Watson constants.
```

### 1.3 The fixed Watson map has rank 5

For both A and S reduction, the linear forward map

```text
tau(6) -> Watson(5)
```

has rank 5. Therefore its nullspace is 1-dimensional.

This means:

- there is exactly one null gauge direction in the compressed tau space for a
  fixed reduction;
- any decomposition that treats two independent directions in `tau(6)` as pure
  gauge is inconsistent with the current linear Watson map.

### 1.4 The current tensors G1 and G2 are not both pure gauge

The two tensors currently used in the code,

```text
G1 = (3,3,3,1,1,1)
G2 = (2A,2B,2C,A+B,A+C,B+C)
```

span a convenient 2D isotropic subspace of the pair-pair tensor space.

However, they are **not** both in the nullspace of the fixed Watson forward
map. Numerically:

- `mat @ G1 != 0`
- `mat @ G2 != 0`
- the 1D null vector cannot be fitted well by a linear combination of `G1,G2`

Therefore `G1,G2` are not a 2D gauge basis in the strict linear sense of the
fixed map `tau -> Watson`.

## 2. The problem with the current Coriolis tensor M

The tensor currently printed as

```text
M_{mu nu} = sum_k G_{k,mu} G_{k,nu} / omega_k^2
```

is a symmetric `3x3` Coriolis invariant tensor in `R^3`.

Its eigenvalues are well-defined geometric invariants, but they are **not** the
eigenvalues of the quartic spatial tensor entering the compressed pair-pair
quartic decomposition.

This explains the previously observed mismatch:

- the `3x3` Coriolis tensor carries useful spatial information;
- but it does not directly reconstruct the full harmonic quartic tensor.

## 3. The natural harmonic quartic spatial object

For the harmonic `H12H12` channel, the quartic tensor is built from the
symmetric mode tensors

```text
mu_ab(k) = d(I^{-1})_{ab} / dQ_k
```

rather than from the Coriolis vectors `G_k` alone.

The natural harmonic quartic metric therefore lives in the 6D symmetric-pair
space `Sym^2(R^3)` with basis

```text
(xx, yy, zz, xy, xz, yz).
```

In the code this is now represented by

```text
quartic_pairpair_metric_au
```

constructed from

```text
u_k = (mu_xx, mu_yy, mu_zz, sqrt(3) mu_xy, sqrt(3) mu_xz, sqrt(3) mu_yz)
```

through

```text
G^(4) = sum_k u_k u_k^T / omega_k^2.
```

This object is the correct harmonic quartic analogue of the Coriolis metric.

## 4. What is excluded by the current evidence

### 4.1 No linear `3+2` decomposition in Watson 5-space

Using the current 5D Watson-space transformation matrices:

- a single representation transform (e.g. `I -> II` or `I -> III`) has a
  2-dimensional fixed subspace;
- the intersection of the fixed subspaces for different cyclic transforms is
  only 1-dimensional.

Therefore the desired `3 spatial + 2 gauge` split does **not** appear as a
linear decomposition of the Watson 5-space itself.

This is the key negative result.

## 5. What remains possible

The `3+2` structure may still be correct, but only in a **nonlinear or
covariant** sense.

The most plausible interpretation is:

- the 3 spatial quantities are nonlinear invariants (for example eigenvalues)
  of a representation-covariant spatial tensor;
- the remaining 2 coordinates describe how that tensor is embedded into the
  Watson parametrization.

In other words, the `3+2` split is likely **not** a direct linear splitting of
the 5 Watson constants, but a decomposition obtained after lifting the problem
to a higher-level tensor object.

## 6. Current best working hypothesis

The current evidence suggests the following hierarchy:

1. `tau(6)` is the natural compressed quartic tensor for projection.
2. The fixed Watson map has one true linear null direction.
3. The `3x3` Coriolis tensor is too contracted to serve as the full spatial
   quartic tensor.
4. The natural harmonic spatial object is a 6D pair-pair metric in
   `Sym^2(R^3)`.
5. If a `3+2` split exists, the 3 spatial quantities are probably nonlinear
   invariants of a representation-covariant tensor derived from that larger
   object.

## 7. Immediate next step

The next derivation should therefore not start from Watson 5-space directly.
Instead it should:

1. identify a representation-covariant quartic spatial tensor,
2. determine its invariant spectrum or equivalent set of 3 invariants,
3. derive the 2 remaining coordinates needed to reconstruct the Watson
   parametrization.

Only after that can the intended `3 spatial + 2 gauge` decomposition be
formulated in a mathematically consistent way.
