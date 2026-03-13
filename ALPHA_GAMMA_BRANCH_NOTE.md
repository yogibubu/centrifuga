# Alpha / Gamma Branch Note

This note records the current positioning of the vibration-rotation-correction
branch that runs in parallel with the centrifugal-distortion branch.

## Core observation

Alongside the sequence

- quartic centrifugal distortion,
- sextic centrifugal distortion,
- higher-order distortion constants,

there is a parallel spectroscopic sequence

- `alpha` vibration-rotation corrections to the rotational constants,
- `gamma` corrections as the next term,
- and higher-order vibration-rotation corrections beyond `gamma`.

The two branches should not be conflated, but they are likely built from a
closely related tensorial and perturbative structure. In particular, the
working hypothesis is that the `alpha` branch is to the rotational constants
what the standard sextic branch is to centrifugal distortion: same basic
harmonic/anharmonic input, different spectroscopic projection.

## Why this matters now

The current CeDiTT work has clarified the standard linear tensor sector for
quartic and standard sextic centrifugal distortion and has identified the first
controlled breakdown of that picture.

In parallel, the `alpha` branch already exists operationally in Gaussian, while
the `gamma` branch does not. The present code base already parses the printed
Gaussian `Vibro-Rot alpha Matrix`, but only as a black-box output. What is
still missing is a controlled internal reconstruction of the formula and of
its mode-resolved structure. By contrast, `gamma` must be treated from the
start as a fresh derivation in the same way that the higher-order quartic
branch was developed beyond the standard centrifugal-distortion sector. This
makes `alpha` the natural entry point if one wants to:

1. understand the perturbative/tensorial structure of the vibration-rotation
   correction branch;
2. prepare the derivation of `gamma`;
3. gain practical control over mode-resolved contributions that Gaussian does
   not expose flexibly.

## Present working hypothesis

The `alpha -> gamma -> ...` branch may stand to the rotational constants in a
way that is structurally analogous to the

- quartic -> sextic -> octic -> ...

branch for centrifugal distortion.

This does **not** mean that the formulas are identical or that the primitive
objects are the same in a trivial sense. It means:

- the same harmonic input used for standard sextics may already contain much of
  the relevant structure;
- first-level and higher-level tensor objects may reappear in different
  projections;
- the distinction between standard sector, resonant-looking intermediate
  expressions, and final nonresonant observable combinations may again be the
  key organizing principle.

## Immediate source to inspect for `alpha`

The first target is a controlled recovery of the `alpha` structure from
Gaussian. This applies only to `alpha`, not to `gamma`.

Primary source files to inspect:

- `/Users/vincenzobarone/gdv_j32p/gdv/dinautil.F`
- `/Users/vincenzobarone/gdv_j32p/gdv/l717.F`

Equivalent copies also exist under:

- `/Users/vincenzobarone/Gaussian_GDV28/gdv/`
- `/Users/vincenzobarone/Documents/gdv/`
- `/Users/vincenzobarone/Documents/gdv28/gdv/`

The goal is to determine:

1. how Gaussian constructs the mode-resolved `alpha` contributions;
2. which harmonic and anharmonic ingredients are actually required;
3. how resonance-like terms appear at the intermediate level;
4. how the final observable correction is assembled.

There is already a useful entry point on the parser side:

- `gaussian_vpt_parser.py`

which extracts the final printed `Vibro-Rot alpha Matrix`. The next step is
not more parsing, but recovering the internal algebra that leads to that
matrix.

## Important structural point about resonances

The mode-resolved `alpha` contributions may contain resonance-looking
denominators or resonant intermediate terms. However, the physically relevant
sum that defines the vibrational correction to the rotational constants can be
written in a final nonresonant form. This is an important design constraint
for the implementation: the backend should expose both the mode-resolved pieces
and the final assembled quantity, but it should treat the latter as the primary
observable.

This is strategically important:

- it means the observable branch should not be organized around the apparent
  resonant singularities of individual intermediate terms;
- it suggests that a clean tensorial or channel-based rewriting may exist for
  the final summed correction;
- it makes the branch computationally and conceptually better suited to
  selective mode analysis.

## Practical motivation beyond Gaussian

One concrete goal is to make it possible to exclude selected normal modes from
the evaluation of `alpha`.

This is useful in several contexts:

- targeted spectroscopic interpretation;
- analysis of large-amplitude or troublesome modes;
- comparison of reduced and full vibrational-correction models;
- preparation for analogous control in the future `gamma` branch.

In Gaussian this type of selective mode elimination is awkward or opaque.
Reimplementing the `alpha` branch in the present code base would make it a
natural user-level and developer-level operation.

This should be interpreted broadly. The desired controls are:

- exclusion of one or more named modes;
- exclusion of symmetry classes or manually chosen mode subsets;
- comparison between full and reduced `alpha` sums;
- propagation of the same logic later to `gamma`.

## Suggested project split

The current three-work structure should be read as follows.

### Paper 1

- standard quartic/sextic tensor transport;
- first controlled breakdown through `H22` and linear response in `tau(H22)`.

### Paper 2

- higher-order VPT quartic centrifugal-distortion theory;
- nonlinear tensor hierarchy beyond the linear sector.

### Paper 4

- higher-order spectroscopic observables required by demanding fits;
- octic centrifugal-distortion constants;
- `alpha` branch reanalysis;
- derivation of `gamma`;
- symbolic-algebra organization of these expressions.

In this picture, the `alpha/gamma` branch belongs naturally to **Paper 4**,
separate from the quartic higher-order hierarchy of Paper 2 and from the
octic-distortion branch of Paper 3.

## Immediate development plan

The natural order of work for this branch is:

1. recover the `alpha` formulas from Gaussian (`dinautil.F`, `l717.F`);
2. identify the exact input data needed by `alpha`;
3. separate intermediate resonant-looking pieces from the final nonresonant
   observable sum;
4. implement `alpha` in the present Python backend with mode-by-mode control;
5. add selective mode exclusion;
6. only then derive `gamma` from scratch in the same structural language.

Before Step 6, there is also a smaller prerequisite:

5a. verify explicitly whether the recovered `alpha` branch uses exactly the
    same information as the standard sextic branch, or only a closely related
    subset of it.

## Desired backend outputs

The implementation target should not be just a single printed number. The code
should expose at least:

- total `alpha` tensor / component correction;
- mode-resolved partial contributions;
- optional symmetry-labeled grouping of mode contributions once mode irrep
  assignment is available;
- optional grouped contributions over chosen mode subsets;
- final nonresonant assembled result;
- diagnostics showing how the summed correction relates to the intermediate
  terms.

The same philosophy should later be carried over to `gamma`, but without
assuming the existence of any Gaussian reference implementation.

## Strategic conclusion

The `alpha/gamma` branch should be treated as a major parallel development,
not as a secondary coding task.

Its scientific value is threefold:

1. it extends the tensor/perturbative program from centrifugal distortion to
   vibration-rotation corrections;
2. it provides observables that are directly useful in demanding spectroscopic
   fits;
3. it offers practical control that standard black-box implementations do not
   provide, especially with respect to selective mode elimination.

## Immediate concrete next checks

When this branch is resumed, the first practical tasks should be:

1. inspect `dinautil.F` and `l717.F` for the internal construction of the
   `alpha` matrix;
2. map the corresponding Gaussian quantities onto the present harmonic model,
   Coriolis data, and cubic-force infrastructure already used for sextics;
3. verify on a small benchmark that the reconstructed formula reproduces the
   printed Gaussian `alpha` matrix before any attempt is made to generalize it;
4. once the `alpha` branch is structurally closed, derive `gamma` from zero in
   the same tensorial language, without relying on Gaussian as a reference.
