# Paper 2 / Paper 4 Compact Roadmap

This is the compact operational roadmap for the two branches that must now be
developed in parallel:

- Paper 2: higher-order quartic centrifugal distortion;
- Paper 4: `alpha/gamma` vibration-rotation corrections.

The purpose of this note is to replace a scattered set of preparatory notes
with a single working sequence.

## Common method

Both papers should follow the same logic:

1. start from the physical observable;
2. identify its direct perturbative/state expansion;
3. isolate the relevant coefficient or channel decomposition;
4. reduce the result to the lowest tensorial sector possible;
5. define:
   - standard sector,
   - standard explicit anharmonic extension,
   - first controlled breakdown;
6. only then organize symbolic formulas and backend outputs.

The important distinction is:

- the raw derivative level seen in the direct expansion
  is not automatically the primitive tensor content of the final observable.

That is the same lesson already established in CeDiTT.

## Paper 2

### Starting point

For Paper 2 the final VPT4 quartic equations already exist.

The observable is already explicit:

`tau^(4) = tau(H12H12) + tau(H22) + tau(H12H30) + tau(H30H30)`.

So Paper 2 does **not** need a fresh derivation from zero. It needs a
structural reinterpretation.

### Target decomposition

The working decomposition should be:

`tau^(4) = tau_std + tau_anh_std + delta_tau_break + tau_anh_high + ...`

with:

- `tau_std = tau(H12H12)`;
- `tau_anh_std` = lowest explicit anharmonic dressing of the first-level
  sector;
- `delta_tau_break` = first genuinely new geometric contribution, centered on
  `H22` and the split
  `mu2 = B(mu1) - N`;
- `tau_anh_high` = higher anharmonic channels that are important numerically
  but conceptually distinct from the first geometric breakdown.

### Immediate task

Take the existing equations and classify them explicitly into that scheme.

### Resonance rule

For Paper 2, `H30H30` must now be treated as the first channel requiring an
explicit resonance analysis with general threshold-based denominator
diagnostics. This is recorded in:

- `PAPER2_RESONANCE_STRATEGY.md`

## Paper 4

### Starting point

For Paper 4 the final equations do **not** exist yet.

The observable must therefore be built from the state-dependent rotational
tensor:

`mu^(eff)(v)
 = mu^(0)
 + sum_i M_i (n_i + 1/2)
 + sum_{ij} M_ij (n_i + 1/2)(n_j + 1/2)
 + ...`

with:

- `alpha_i = P_rot(M_i)`
- `gamma_ij = P_rot(M_ij)`.

### What is already known

The present backend already gives the `alpha` decomposition:

- `C_i^X` = Coriolis block;
- `I_i^X` = inertia/geometric block;
- `A_i^X` = explicit anharmonic block.

### Target decomposition

The working formula should be:

`gamma = gamma_geom_std + gamma_anh_std + delta_gamma_break + ...`

with:

- `gamma_geom_std = CC + II + CI`;
- `gamma_anh_std = CA + IA + AA`;
- `delta_gamma_break` = first irreducible higher-level contribution, to be
  tested against the possible entry of
  `mu2 = B(mu1) - N`.

### Immediate task

Work first at the coefficient-extraction level:

1. identify `M_i` and `M_ij`;
2. reduce them to first-level language where possible;
3. only then decide whether a true breakdown term survives.

## Strict order of work

The recommended order is:

1. Paper 2:
   - classify the existing quartic VPT4 equations into
     `std / std_anh / break / anh_high`.
2. Paper 4:
   - finish the direct coefficient-extraction logic for `alpha/gamma`;
   - derive the first standard `gamma` sector;
   - only later test the first breakdown.

## Practical rule

From now on:

- no new symbolic proliferation unless it is tied directly to one of the two
  immediate tasks above;
- for Paper 2, equations come first and notes follow them;
- for Paper 4, coefficient extraction comes first and channel labels follow it.

This is the compact roadmap to use when the actual paper writing resumes.
