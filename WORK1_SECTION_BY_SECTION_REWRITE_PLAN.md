# Work 1 Section-by-Section Rewrite Plan

This note maps the current LaTeX draft onto the refocused **Work 1** paper.

Current draft logic:

- broad tensor theory of quartic and sextic distortion,
- pseudoinverse/gauge reconstruction,
- harmonic interpretation of quartic/sextic sectors,
- representation transforms,
- mixed validation/examples.

Target Work 1 logic:

- standard quartic + standard sextic = one linear representation-equivalent sector,
- pseudoinverse lifting/gauge fixing as the transport mechanism,
- elimination of primitive `d2I` from the standard sextic sector,
- `H22` as first quartic breakdown,
- sextic linear term in `tau(H22)` as first sextic analogue,
- diagnostics and practical use.

## 1. Title

Current title:

- `Reconciling Direct and Inverse Problems in Rotational Spectroscopy: A Tensor Framework for Centrifugal Distortion Constants`

Suggested direction:

- keep the direct/inverse language,
- but make the standard linear sector and breakdown visible.

Working options:

- `Linear Representation Transport and Its First Breakdown in Quartic and Sextic Centrifugal Distortion`
- `Tensorial Transport of Standard Quartic and Sextic Centrifugal Distortion Constants`
- `From Linear Representation Transport to Its First Breakdown in Centrifugal Distortion Theory`

## 2. Abstract

Status:

- current abstract is too broad and too committed to a general harmonic hierarchy.

Action:

- replace with [work1_abstract_refocused.tex](/Users/vincenzobarone/centrifugal/work1_abstract_refocused.tex)

Main points to keep:

- pseudoinverse gauge fixing,
- standard quartic + sextic common tensor language,
- no primitive `d2I` in the standard sextic sector,
- `H22` as first quartic breakdown,
- linear-in-`tau(H22)` sextic analogue,
- main text vs appendices split.

## 3. Introduction

Current status:

- good opening motivation,
- too much emphasis on a general tensor hierarchy,
- not enough emphasis on the standard linear sector and its breakdown.

Keep:

- direct vs inverse problem,
- Watson constants as representation-dependent coordinates,
- need for common tensor representatives,
- pseudoinverse/gauge logic,
- representation/reduction transport motivation.

Change:

- remove or greatly soften the old “quartic = `l=2`, sextic = `l=6` hierarchy”
  as a main-text headline;
- introduce early the statement that the present paper is about the
  **standard linear sector**;
- state explicitly that standard quartics and standard sextics share the same
  first-level tensorial language;
- announce `H22` as the first quartic breakdown and the sextic linear
  `tau(H22)` term as the corresponding sextic diagnostic;
- state the perturbative partition convention:
  - first order: `H03`, `H12`, `H30`
  - second order: `H40`, `H22`, `H13`, `H31`, `H04`
  - and clarify that this is the Watson/VPT2-style partition used here, not a
    universal one.

Move to appendix:

- any long discussion of harmonic-sector decomposition.

## 4. Theory: global restructuring

Current status:

- the theory section is overloaded and mixes:
  - perturbative setup,
  - tensor construction,
  - quartic gauge theory,
  - sextic harmonic decomposition,
  - implementation formulas.

Action:

- split into a short main-text theory and several appendices.

### New main-text theory order

1. Perturbative setup and partition convention
2. Tensor representatives and pseudoinverse lifting
3. Standard quartic sector
4. Standard sextic sector
5. Unified standard linear sector
6. First breakdown: `H22`
7. First sextic analogue: linear response in `tau(H22)`

### New appendices

- Appendix A: detailed quartic tensor construction
- Appendix B: quartic pseudoinverse/gauge derivation
- Appendix C: detailed sextic tensor construction and projections
- Appendix D: elimination of primitive `d2I` from standard sextics
- Appendix E: `H22 = B(mu1) - N`
- Appendix F: sextic linear response in `tau(H22)`
- Appendix G: implementation details / app formulas if desired

## 5. Theory subsection: Rovibrational perturbation theory

Keep:

- idea that centrifugal distortion comes from rovibrational coupling,
- role of inverse inertia derivatives / Coriolis language,
- effective Hamiltonian viewpoint.

Add:

- explicit warning that `Hnm` labels bidegree, not perturbative order by itself;
- explicit adopted partition;
- channel decomposition is exact only within the adopted partition/order and
  effective-Hamiltonian construction;
- channel coefficients include BCH / Van Vleck prefactors.

Move to appendix:

- long derivations of derivative identities if not needed in the main text.

## 6. Theory subsection: Quartic centrifugal distortion

Keep in main text:

- reduced quartic representative,
- non-invertible map to Watson constants,
- pseudoinverse gauge fixing,
- representation transport idea,
- standard quartic sector as the baseline.

Reduce:

- spectral decomposition details,
- full forward projection formulas,
- null-vector details,
- `3+2` parameterization details,
- long Yamada-convention discussion.

Move to appendix:

- explicit formulas for `tau'`,
- forward projection formulas,
- null vector / affine family,
- pseudoinverse slice equations,
- `3+2` coordinates,
- detailed Yamada comparison.

## 7. Theory subsection: Sextic centrifugal distortion

Main text should keep only:

- standard sextics are transported in the same general tensor framework;
- standard sextic geometry closes on the same first-level objects as the
  standard quartic sector;
- no primitive `d2I` is required there;
- explicit anharmonicity enters through `phi3`;
- this is why standard quartics and standard sextics belong to the same linear
  representation-equivalent sector.

Main text should add:

- the first sextic post-standard candidate is the term linear in `tau(H22)`.

Move to appendix:

- full sextic `5+2`,
- harmonic decomposition,
- `1+4`,
- `1+1+1+2`,
- long cylindrical/Watson projection formulas,
- hormone example if kept at all.

## 8. Implementation / invariant-gauge / validation

Current status:

- too much of this is written as theory.

Main text should keep:

- the app / algorithm exists,
- pseudoinverse lifting + permutation + reprojection,
- diagnostics are available before refitting.

Main text should add:

- `H22` can be included as an app diagnostic,
- and later the sextic linear-in-`tau(H22)` diagnostic can be added the same
  way.

Move to appendix:

- detailed algorithmic formulas and validation identities.

## 9. Results and discussion

Current status:

- DMSO, HBN, H2CO/H2CS are all useful, but the section is not organized around
  the new claim.

Suggested new order:

1. representation transport in the standard sector
   - one quartic example,
   - one sextic example;
2. diagnostics of conditioning in fitted constants
   - `s111`,
   - tensor invariants,
   - condition numbers;
3. first breakdown diagnostic
   - `H22` on `H2O`,
   - comparison with `H2CO`,
   - qualitative argument that the diagnostic shrinks where VPT2 and VCI are
     closer;
4. sextic analogue
   - if full `H2CO` cubic data are not yet available, present this explicitly as
     a first defined diagnostic with partial numerical support from `H2O`.

Likely demotions:

- hormone table,
- any example that does not serve either transport or breakdown.

## 10. Conclusions

Current status:

- too broad and still written as if the paper were a general tensor theory.

New conclusion should say:

- the paper identifies a standard linear sector shared by quartics and
  sextics;
- pseudoinverse lifting gives the common transport mechanism;
- standard sextics can be written without primitive `d2I`;
- `H22` is the first quartic breakdown term;
- the first sextic analogue is the term linear in `tau(H22)`;
- these are diagnostics of when one must move beyond the standard sector.

Avoid:

- broad claims about full VPT4 closure,
- overcommitting to the general harmonic hierarchy in the conclusion.

## 11. Immediate writing order

Recommended rewrite order:

1. Abstract
2. Introduction
3. Main-text theory skeleton
4. Conclusions
5. Move technical material to appendices
6. Rebuild results around transport + breakdown

## 12. Next deliverable

The next concrete step should be:

- rewrite the **Introduction** in final Work 1 form,
- then rewrite the **main-text Theory** section in final Work 1 form,
- and only after that decide the exact appendices.
