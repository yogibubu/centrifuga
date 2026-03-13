# Intermediate Quartic Theory Outline

This note records a possible intermediate theory between the closed CeDiTT-side
`H21H21` picture and the full VPT4 quartic problem.

## Motivation

The full quartic VPT4 problem contains several channels:

- `H21H21`
- `H22`
- `H21H30`
- `H30H30`

It is not necessary to understand all of them at once.

A more productive strategy is to isolate the smallest extension of the standard
harmonic quartic picture that already reveals genuinely new geometry.

## Proposed intermediate theory

Study first the partial quartic problem

- `H21H21 + H22`

before adding `H21H30` and `H30H30`.

This is not justified because `H22` must be numerically dominant. It is
justified because `H22` is structurally simple and because it is the first
channel where the second-derivative level of the inertia geometry enters in a
clean way.

## Why `H22`

`H21H21` already has a recognizable tensorial organization.

- It is built from first-order mode geometry.
- In the present harmonic picture, the natural objects are Coriolis-like:
  vectors/tensors built linearly from the normal modes.
- Its quartic content can already be expressed in a natural pair-pair setting.

`H22` is the next clean channel because:

- it introduces the level currently written as `d2I/dQ2` or `d2(I^-1)/dQ2`;
- it is simpler than the explicitly anharmonic channels;
- the same second-derivative layer is expected to persist in the sextic
  problem.

So `H22` is not singled out by size, but by structural minimality.

## Working hypothesis

The explicit derivative arrays

- `dI/dQ`,
- `d2I/dQ2`,
- `d(I^-1)/dQ`,
- `d2(I^-1)/dQ2`,

should be viewed as coordinate descriptions, not as the final conceptual
objects.

For `H21H21`, the relevant first-level geometry is already visible in
Coriolis-like tensors.

For `H22`, there should exist a second-level geometric object, analogous in
role to the Coriolis object for `H21H21`, such that:

- `d2I/dQ2` is only its coordinate realization;
- the object has a natural transformation law under representation changes;
- it feeds a quartic tensorial construction without privileging a specific
  coordinate formula.

## Main question

Does the partial theory

- `H21H21 + H22`

already close in a natural tensor space?

Two possibilities are especially relevant:

- `H22` lives in the same pair-pair space already natural for `H21H21`;
- `H22` forces an enlargement to a richer tensor space, of which the pair-pair
  picture is only a projection.

This is the first question to settle before worrying about the full VPT4 sum.

## Relation to the new paper direction

The manuscript direction under consideration is:

- start from quartic or sextic constants in one representation/reduction;
- lift them by a Moore-Penrose pseudoinverse to a canonical higher-rank tensor;
- interpret the pseudoinverse as gauge fixing;
- transform the tensor geometrically;
- reproject to the target representation/reduction.

If that is the right framework, then the intermediate quartic tensor should be
built from geometrically natural ingredients.

This is exactly why the explicit `d2I/dQ2` language must be replaced, or at
least subordinated, to a more intrinsic second-level object.

## Sextic relevance

The interest of `H22` is strengthened by the sextic problem.

Even if the current sextic implementation does not expose the dependence in the
same direct algebraic form, the second-derivative inertia layer is still
present there in substance.

So identifying the correct second-level object is not only useful for a
quartic subtheory. It is a step toward a common geometric backend for:

- quartic `H21H21 + H22`,
- and the corresponding second-level structures appearing in the sextics.

## Immediate program

1. extract the present `H22` formula in the cleanest symbolic form;
2. isolate exactly which contraction of `d2I/dQ2` or `d2(I^-1)/dQ2` it uses;
3. determine the smallest tensorial object carrying the same information;
4. identify its natural tensor space;
5. rewrite `H22` in those terms;
6. check how the same object should enter the sextic machinery.
