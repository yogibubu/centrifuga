# Paper 4 Skeleton: Alpha / Gamma Tensor Branch

This note fixes the scope of the vibration-rotation-correction branch as a
separate work parallel to the centrifugal-distortion hierarchy.

The derivational strategy for this paper should be kept parallel to the one
used for the higher-order quartic branch. The common method is recorded in:

- `COMMON_STRATEGY_PAPER2_PAPER4.md`

## Positioning

The centrifugal-distortion sequence treated in CeDiTT,

- quartic,
- sextic,
- higher-order distortion,

has a parallel branch for vibration-rotation corrections to the rotational
constants,

- `alpha`,
- `gamma`,
- and higher-order corrections beyond `gamma`.

The working hypothesis is that this branch can be reorganized in the same
tensorial language used for centrifugal distortion:

- first-level geometry carried by `mu1` / `c1`,
- Coriolis structure `zeta`,
- semi-diagonal and then full cubic-force information,
- symmetry-adapted spectroscopic projection for asymmetric, symmetric, and
  linear limits.

## Why this is a separate work

This branch should be treated as **Paper 4**, not folded into the quartic
higher-order work.

Reasons:

1. `alpha` and `gamma` are different observables from centrifugal-distortion
   constants and appear directly in demanding spectroscopic fits.
2. `alpha` already exists in Gaussian, but only as an operational black box.
3. `gamma` is not available and requires a fresh derivation from zero.
4. The spectroscopic motivation is distinct: these quantities are needed when
   fitted Hamiltonians require vibration-rotation corrections beyond the usual
   `alpha` level.

## Core objectives

1. Rewrite `alpha` completely in the CeDiTT first-level tensor language,
   eliminating explicit `dI/dQ` from the working formulas.
2. Make the final observable branch explicitly nonresonant, even if
   intermediate terms contain resonance-like denominators.
3. Support symmetry-adapted output for asymmetric tops, symmetric tops, and
   linear molecules.
4. Expose mode-resolved and subset-resolved contributions, including selective
   mode elimination.
5. Derive `gamma` from zero in the same language and from the same
   perturbative hierarchy.

## Tensor hierarchy to test

The structural hypothesis to verify is:

- `alpha` belongs to the same first-level tensor sector that underlies the
  standard sextic geometry branch;
- `gamma` should then be the next observable projection of the same hierarchy,
  enlarged by the first higher-level objects only when strictly necessary.

This mirrors the logic already established for:

- standard quartic / standard sextic,
- first controlled breakdown,
- and symmetry-adapted limits.

## Immediate technical tasks

### A. Alpha branch

1. Recover the exact Gaussian algebra from:
   - `/Users/vincenzobarone/gdv_j32p/gdv/dinautil.F`
   - `/Users/vincenzobarone/gdv_j32p/gdv/l717.F`
2. Map all ingredients onto the CeDiTT harmonic model:
   - `mu1`
   - `c1`
   - `zeta`
   - semi-diagonal cubic matrix
3. Rebuild the asymmetry/symmetry/linear branches with explicit degeneracy
   metadata instead of ad hoc handling.
4. Verify numerically against Gaussian mode by mode, not only on the summed
   `A,B,C` values.
5. Add symmetry-aware mode grouping and mode exclusion.

### B. Gamma branch

1. Reanalyse the perturbative structure of `alpha` first.
2. Identify the next-order observable combination that defines `gamma`.
3. Derive the corresponding formulas from zero, rather than by porting a
   legacy implementation.
4. Decide which higher-level tensor objects are genuinely new and which are
   closures of the first-level sector.
5. Implement symbolic-algebra generation of the formulas before coding the
   numerical backend.

The current working organization of this step is recorded separately in:

- `ALPHA_SYMBOLIC_BLOCKS.md`
- `ALPHA_GAMMA_COEFFICIENT_EXTRACTION.md`
- `GAMMA_FROM_MU_EXPANSION.md`
- `GAMMA_PERTURBATIVE_STRUCTURE_NOTE.md`
- `GAMMA_CHANNEL_CANDIDATES.md`
- `GAMMA_GEOM_STANDARD_MAP.md`
- `GAMMA_GEOM_STANDARD_ANSATZ.md`
- `GAMMA_ANH_STANDARD_ANSATZ.md`
- `GAMMA_MASTER_FORMULA_AND_WORKPLAN.md`

## Expected outputs

The backend should ultimately expose:

- total `alpha` correction;
- total `gamma` correction;
- mode-resolved contributions;
- symmetry-grouped contributions;
- reduced sums after mode exclusion;
- diagnostics for resonance-prone intermediate pieces;
- symmetry-adapted outputs for special rotor limits.

## Relation to earlier works

- Paper 1 / CeDiTT:
  - standard quartic/sextic transport and first controlled breakdown.
- Paper 2:
  - higher-order quartic tensor hierarchy.
- Paper 3:
  - octics and related higher-order distortion observables.
- Paper 4:
  - `alpha/gamma` branch in the same tensorial language.

The value of Paper 4 is not only computational. It should also provide a more
interpretable bridge between quantum-chemical observables and fitted effective
Watson-Hamiltonian parameters than black-box variational workflows.
