# Gamma Master Formula and Workplan

This note collects the current `gamma` program into a single working formula
and an ordered derivation plan. It is the compact operational summary of the
more detailed notes:

- `ALPHA_SYMBOLIC_BLOCKS.md`
- `GAMMA_PERTURBATIVE_STRUCTURE_NOTE.md`
- `GAMMA_CHANNEL_CANDIDATES.md`
- `GAMMA_GEOM_STANDARD_MAP.md`
- `GAMMA_GEOM_STANDARD_ANSATZ.md`
- `GAMMA_ANH_STANDARD_ANSATZ.md`

## 1. Master formula

The current working hypothesis is that the vibration-rotation observable
`gamma` should be organized as

`gamma = gamma_geom_std + gamma_anh_std + delta_gamma_break + ...`

where:

- `gamma_geom_std` is the largest standard geometric closure built from
  first-level blocks only;
- `gamma_anh_std` is the standard explicit anharmonic extension of the same
  branch;
- `delta_gamma_break` is the first controlled breakdown term, i.e. the first
  part that requires genuinely new tensor content beyond the first-level
  sector.

This is the direct analogue, on the vibration-rotation branch, of the CeDiTT
organization already established for centrifugal distortion.

## 2. Standard geometric branch

The standard geometric branch is sought in the form

`gamma_geom_std = CC + II + CI`

with:

- `CC`: Coriolis-Coriolis closure;
- `II`: inertia-inertia closure;
- `CI`: mixed Coriolis-inertia closure.

In abstract block notation:

`gamma_ij^X(geom_std)
 = <C_i^X, C_j^X>_(X)
 + <I_i^X, I_j^X>_(X)
 + <C_i^X, I_j^X>_(X)
 + <I_i^X, C_j^X>_(X)`

where:

- `C_i^X` is the first-level Coriolis block extracted from `alpha`;
- `I_i^X` is the first-level geometric block extracted from `alpha`;
- `<.,.>_(X)` denotes the appropriate harmonic bilinear pairing for the
  rotational component `X`.

## 3. Standard anharmonic branch

The standard anharmonic branch is sought in the form

`gamma_anh_std = CA + IA + AA`

with:

- `CA`: Coriolis-cubic channels;
- `IA`: inertia-cubic channels;
- `AA`: quadratic anharmonic channels.

In abstract block notation:

`gamma_ij^X(anh_std)
 = <C_i^X, A_j^X>_(X)
 + <A_i^X, C_j^X>_(X)
 + <I_i^X, A_j^X>_(X)
 + <A_i^X, I_j^X>_(X)
 + <A_i^X, A_j^X>_(X)`

where `A_i^X` is the explicit standard anharmonic `alpha` block already
identified in the backend.

At this stage, quartic-force input `phi4` is not assumed to belong to the
minimal standard branch. It should be tested only if the `CA + IA + AA`
sector proves insufficient.

## 4. First controlled breakdown

The first controlled breakdown should be tested only after the best standard
branch

`gamma_std = gamma_geom_std + gamma_anh_std`

has been assembled.

The current first candidate is the entry of the second geometric response

`mu2 = B(mu1) - N`

so that the breakdown is organized as

`delta_gamma_break = delta_gamma[B(mu1)] + delta_gamma[N]`

with only the `N` part counted as genuinely new tensor content.

This is the exact analogue of separating closure terms from intrinsic new
tensor levels on the centrifugal-distortion branch.

## 5. Backend-to-theory dictionary

The existing `alpha` backend already provides the correct symbolic seed:

- `alpha_coriolis_cm[i,X]` -> `C_i^X`
- `alpha_inertia_cm[i,X]` -> `I_i^X`
- `alpha_anharmonic_cm[i,X]` -> `A_i^X`

Thus the first `gamma` derivation does not start from raw code, but from a
clean block decomposition already available in the present formalism.

## 6. Ordered symbolic workplan

The recommended order is:

1. derive the diagonal geometric sector:
   - `G_ii^X(CC)`
   - `G_ii^X(II)`
   - `G_ii^X(CI)`
2. verify whether the diagonal branch closes on first-level data only;
3. derive the off-diagonal geometric sector:
   - `G_ij^X(CC)`
   - `G_ij^X(II)`
   - `G_ij^X(CI)` for `i != j`;
4. assemble `gamma_geom_std`;
5. derive the diagonal standard anharmonic sector:
   - `CA`
   - `IA`
   - `AA`
6. extend the anharmonic derivation to `i != j`;
7. assemble `gamma_std = gamma_geom_std + gamma_anh_std`;
8. only then test whether `mu2`, `B(mu1)`, or `N` enter necessarily.

This order is strict for a reason:

- it keeps first-level closure separate from explicit anharmonicity;
- it keeps standard structure separate from the first breakdown;
- it prevents `gamma` from being overdesigned as a generic high-order object
  before the standard branch is understood.

## 7. Symmetry strategy

The symmetry strategy is already fixed by the `alpha` backend and should be
reused unchanged at the architectural level:

1. derive all channels in the generic rotational basis;
2. assemble mode-pair-resolved `gamma_ij^X`;
3. only then apply:
   - asymmetric-top readout,
   - axial projection for symmetric tops,
   - projector-based treatment of degenerate mode subspaces,
   - linear perpendicular/parallel reduction.

This keeps perturbative structure and spectroscopic readout cleanly separated.

## 8. Immediate next test

The next real symbolic test should therefore be:

- derive the diagonal `CC`, `II`, `CI` branch explicitly;
- check whether its structure can be written using only
  `zeta`, `c1`, frequencies, and rotational prefactors;
- if yes, that establishes the first concrete candidate for
  `gamma_geom_std`.

Only after that does it make sense to discuss `phi3`, `phi4`, or `N`.
