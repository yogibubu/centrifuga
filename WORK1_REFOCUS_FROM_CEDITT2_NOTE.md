# Work 1 Refocus from `CeDiTT2_0`

This note records how the old manuscript

- [CeDiTT2_0.pdf](/Users/vincenzobarone/Desktop/CeDiTT2_0.pdf)

should be refocused into the new **Work 1** paper.

## Main claim

The main text should be built around one statement:

- there is a **standard linear sector** of quartic and sextic centrifugal
  distortion constants in which representation/reduction changes are described
  by linear transport of tensor representatives after pseudoinverse gauge
  fixing;
- `H22` is the first quartic term that leaves this sector;
- the first sextic analogue is the term linear in `tau(H22)` inside the
  sextic geometry functional.

So the paper should no longer read as a broad tensor theory of all quartic and
sextic structure. It should read as:

1. a theory of linear representation transport in the standard sector;
2. an identification of the first breakdown of that linearity;
3. a proposal of low-cost diagnostics for that breakdown.

## Main text vs appendices

This split should be strict.

### Main text

The main text should contain only the material needed to support the central
message:

1. Watson constants are representation-dependent coordinates, not intrinsic
   objects.
2. Pseudoinverse lifting reconstructs tensor representatives and fixes the
   gauge.
3. Standard quartics and standard sextics belong to the same linear
   representation-equivalent sector.
4. This works because the standard sextic sector can be written without taking
   second derivatives of the inertia tensor as primitive objects.
5. `H22` is the first quartic term that breaks this scheme.
6. The first sextic analogue is the term linear in `tau(H22)`.
7. These two terms are useful diagnostics of:
   - sensitivity to representation/reduction choices in fits;
   - the need to go beyond the standard perturbative sector.

### Appendices

All detailed derivations should be moved to appendices:

- tensor construction formulas;
- quartic pseudoinverse and gauge-fixing derivations;
- sextic pseudoinverse and projection formulas;
- detailed forward/backward maps between tensor objects and Watson constants;
- derivation of the standard sextic closure in the `mu1 + zeta + phi3`
  language;
- derivation of the `H22 = B(mu1) - N` decomposition;
- construction of the sextic linear-response term in `tau(H22)`;
- any harmonic decomposition or invariant-subspace material needed only to
  justify the tensor structure.

The appendices are essential, but they should support the main claim rather
than compete with it.

## What to keep from the old draft

These points remain central:

- direct vs inverse spectroscopy;
- tensor representatives behind Watson constants;
- pseudoinverse reconstruction as gauge fixing;
- quartic and sextic representation/reduction transport;
- the computational/app point of view;
- the idea that the same framework can compare fits and QM calculations.

## What to demote

These should no longer be the headline:

- CeDiTT as a broad standalone formalism;
- claims that suggest the paper already covers full VPT4;
- large structural excursions into harmonic decomposition unless they are needed
  directly for the linear-transport story;
- sextic formal developments that go beyond the standard sector plus the first
  breakdown diagnostic.

## New narrative line

The clean narrative is:

1. fitted quartic and sextic constants depend on representation and reduction;
2. the right intrinsic level is obtained by pseudoinverse lifting to tensor
   representatives;
3. in the standard quartic and sextic sectors, those representatives transform
   linearly;
4. the standard sextic sector closes on the same first-level tensorial language
   as the standard quartic sector, so `d2I` is not primitive there;
5. this defines the standard linear sector;
6. `H22` is the first quartic term that breaks that sector;
7. the first sextic analogue is the term linear in `tau(H22)`;
8. these are diagnostics, not full higher-order corrections.

## Practical manuscript structure

Recommended main-text section order:

1. Introduction
2. Direct/inverse problem and tensor representatives
3. Pseudoinverse gauge fixing and linear representation transport
4. Standard quartic sector
5. Standard sextic sector
6. Unified standard linear sector
7. First breakdown: `H22`
8. First sextic analogue: linear term in `tau(H22)`
9. Diagnostics and numerical examples
10. Conclusions

Recommended appendices:

- Appendix A. Quartic tensor construction and projections
- Appendix B. Quartic pseudoinverse slice and gauge details
- Appendix C. Sextic tensor/projection construction
- Appendix D. Elimination of primitive `d2I` from the standard sextic sector
- Appendix E. `H22` decomposition into `B(mu1)` and `N`
- Appendix F. Sextic linear response in `tau(H22)`

## Immediate editorial consequence

The old text should be rewritten so that:

- the main text is explicitly about linear transport and its breakdown;
- all long derivations are moved to appendices;
- the quartic pseudoinverse/gauge story remains, but only at the level needed
  in the main text;
- the standard sextic section is reformulated in the `mu1 + zeta + phi3`
  language;
- `H22` is introduced as the first non-standard quartic diagnostic;
- the sextic `tau(H22)` linear term is introduced as the corresponding sextic
  diagnostic.
