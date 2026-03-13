# Primitive Tensor Hierarchy Note

This note records the current unified picture suggested by the code.

## First level

The first primitive tensor is

```text
mu1_k = d(I^-1)/dQ_k.
```

This object transforms covariantly under representation changes up to the
standard per-mode sign freedom.

It already generates:

- the harmonic quartic tensor `tau`,
- the sextic first-level tensor `c1`,
- and, after Coriolis coupling, the sextic mode-pair tensor `c2_pair`.

So the full currently exposed sextic geometry sector is already generated from
the first-level object `mu1` together with:

- Coriolis couplings,
- rotational constants,
- frequency kernels.

## Second level

For the quartic `H22` channel, the relevant second-level tensor is not the full
coordinate array `mu2 = d^2(I^-1)/dQdQ` taken as primitive.

Instead:

```text
mu2 = B(mu1) - N
```

with

```text
B_{kl} = mu1_k I mu1_l + mu1_l I mu1_k,
N_{kl} = I^-1 (d2I_{kl}) I^-1.
```

Here:

- `B(mu1)` is generated algebraically from the first-level object,
- `N` is the genuinely new intrinsic second-level tensor.

So the quartic `H22` channel is naturally formulated from the primitive pair

```text
(mu1, N).
```

## Current division of labor

The code now supports the following hierarchy.

### Quartics

- standard harmonic quartic geometry: first level `mu1`;
- `H22`: first level `mu1` plus intrinsic second level `N`;
- heavier anharmonic quartic channels: separate VPT4 problem.

### Sextics

- currently exposed sextic geometry sector: first level `mu1` plus Coriolis
  coupling;
- explicit anharmonic sextic contribution: cubic force constants `phi3`.

## Main conceptual consequence

The presently exposed code suggests that the primitive hierarchy is:

1. first-level tensor `mu1`;
2. intrinsic second-level tensor `N`, needed already by quartic `H22`;
3. explicit anharmonic data such as `phi3`.

At the current level of evidence, there is no need to postulate an additional
intrinsic sextic tensor to explain the full sextic geometry sector currently
implemented.

## Practical consequence for writing

The cleanest current narrative is:

- quartic transport theory is built from `mu1`, and for `H22` from `(mu1, N)`;
- sextic transport theory is built from the same first-level tensor `mu1`,
  together with Coriolis coupling and explicit cubic anharmonicity;
- the full VPT4 quartic project begins when the heavier anharmonic quartic
  channels are added on top of this hierarchy.
