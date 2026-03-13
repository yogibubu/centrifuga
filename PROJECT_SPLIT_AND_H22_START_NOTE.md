# Project Split and H22 Start Note

This note fixes the current project decision and records the first concrete
step on `H22`.

## Project split

The old CeDiTT framing is no longer the right center of gravity.

The first paper should instead focus on the operational problem:

- given quartic or sextic centrifugal-distortion constants in one
  representation/reduction,
- lift them by a Moore-Penrose pseudoinverse to a canonical higher-rank tensor,
- interpret that pseudoinverse as gauge fixing,
- transform the tensor geometrically,
- reproject to the target representation/reduction.

Within that first paper:

- quartics are treated at the level of the standard Coriolis-like block plus
  `H22`;
- sextics are included in the same general framework;
- the goal is the tensorial transport/gauge-fixing theory, not the full VPT4
  anharmonic quartic closure.

The separate VPT4 project then starts when one adds the heavier quartic
anharmonic channels:

- `H21H30`,
- `H30H30`,
- and the full quartic completion built from them.

So the first paper contains:

- quartic transport theory,
- sextic transport theory,
- the `H22` extension beyond the purely Coriolis-like quartic picture.

The VPT4 project contains:

- the heavy anharmonic quartic channels and their full organization.

## Why start from H22

`H22` is not singled out because it must dominate numerically.

It is singled out because:

- it is structurally simple;
- it is the cleanest quartic place where the second-derivative inertia layer
  appears;
- the same layer is present again in the sextic problem.

So `H22` is the right laboratory for identifying the second-level geometric
object that should replace the explicit `d2I/dQ2` language.

## Current code formula

In the present code, `H22` is implemented in
[quartic_channels.py](/Users/vincenzobarone/centrifugal/quartic_channels.py#L1478)
as

```text
tau^(H22)_{ab,cd}
  = (hbar / 32) sum_{k,l} ((1 + delta_{kl}) / (omega_k omega_l))
    mu2_{ab,kl} mu2_{cd,kl}
```

where

- `mu2_{ab,kl} = d^2(I^-1)_{ab} / dQ_k dQ_l`.

So the present `H22` contribution is already a weighted Gram construction in
the mode-pair labels `(k,l)`.

This strongly suggests that the natural object is not the coordinate array
`d2I/dQ2` itself, but some second-level mode-pair tensor whose coordinate
realization is currently called `mu2`.

## First structural decomposition

From
[rovib_distortion.py](/Users/vincenzobarone/centrifugal/rovib_distortion.py#L615)
the inverse-inertia second derivative is

```text
mu2_{kl}
  = I^-1 (dI_k) I^-1 (dI_l) I^-1
  + I^-1 (dI_l) I^-1 (dI_k) I^-1
  - I^-1 (d2I_{kl}) I^-1
```

with `dI_k = dI/dQ_k` and `d2I_{kl} = d^2I/(dQ_k dQ_l)`.

Therefore `mu2` contains two qualitatively different pieces:

1. a bilinear piece generated algebraically from the first-level geometry
   `dI_k`;
2. a genuinely new second-level piece coming from `d2I_{kl}`.

It is useful to name these separately:

```text
B_{kl}
  = I^-1 (dI_k) I^-1 (dI_l) I^-1
  + I^-1 (dI_l) I^-1 (dI_k) I^-1

N_{kl}
  = I^-1 (d2I_{kl}) I^-1

mu2_{kl} = B_{kl} - N_{kl}
```

The real open problem is then:

- is there a natural tensorial definition of the second-level object `N_{kl}`
  or of the full combination `mu2_{kl}` that makes the quartic and sextic
  theories geometrically clean?

There is now a sharper algebraic conclusion:

```text
B_{kl} = mu1_k I mu1_l + mu1_l I mu1_k
```

with

```text
mu1_k = d(I^-1)/dQ_k.
```

So the bilinear part of `mu2` is not a new object at all: it is generated
entirely by the first-level tensor `mu1` and the equilibrium inertia tensor
`I`.

Therefore the only genuinely new second-level content in `mu2` is the
intrinsic piece `N_{kl}`.

The code now supports this viewpoint directly:

- `H22` can be reconstructed either from the full `mu2`,
- or from the primitive data `(mu1, N, I)` by rebuilding
  `B(mu1) = mu1 I mu1 + mu1 I mu1` internally and forming
  `mu2 = B(mu1) - N`.

These two routes agree numerically at roundoff in the current tests.

## Immediate consequence for the program

The route forward should be:

1. treat `mu2_{ab,kl}` as the current coordinate-level `H22` object;
2. separate the algebraically induced piece `B_{kl}` from the genuinely new
   second-level piece `N_{kl}`;
3. determine which of these is the right candidate for the canonical
   second-level tensor;
4. check how the same object reappears in the sextic machinery.

At minimum, this shows that the next step is not merely to "remove `d2I/dQ2`",
but to replace it by a natural mode-pair tensorial object whose current
coordinate formula is hidden inside `mu2`.
