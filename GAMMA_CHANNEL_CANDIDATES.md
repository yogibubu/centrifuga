# Gamma Channel Candidates

This note refines the first `gamma` structure by listing the most plausible
channel classes to test in symbolic work. It does not claim that the final
observable is already known. Its purpose is to separate:

- channels that are obvious first-level closures of the present `alpha`
  branch;
- channels that likely represent the first explicit anharmonic extension;
- channels that would signal the entry of genuinely new tensor levels.

## 1. Starting point: alpha decomposition already available

The present `alpha` backend already separates three contributions:

- `alpha_coriolis`,
- `alpha_inertia`,
- `alpha_anharmonic`.

In the current implementation these are built from:

- `zeta` and harmonic frequencies for the Coriolis part;
- `c1` for the geometric/inertia part;
- `c1` plus semi-diagonal cubic force constants `phi_iij` for the first
  explicit anharmonic part.

This gives a natural template for `gamma`.

## 2. First-level closure channels

The first group to test is the one that uses only first-level ingredients.
These are the direct analogue of the standard geometric closure on the
centrifugal-distortion side.

### G1. Coriolis-Coriolis closure

Candidate structure:

- bilinear or iterated combinations of the `alpha_coriolis` building blocks;
- dependence only on `zeta`, harmonic frequencies, and rotational factors.

This is the most obvious first-level candidate for the purely dynamical part
of `gamma`.

### G2. Inertia-Inertia closure

Candidate structure:

- bilinear or iterated combinations of the `alpha_inertia` building blocks;
- dependence only on `c1` or equivalently `mu1`.

This is the direct analogue of asking whether the next observable still closes
on the first derivative of the inverse inertia tensor.

### G3. Mixed Coriolis-Inertia closure

Candidate structure:

- cross terms between `zeta`-driven and `c1`-driven pieces.

This is probably unavoidable once the quadratic occupation dependence is
assembled, even if the pure Coriolis and pure inertia pieces are separately
identifiable.

## 3. Standard anharmonic channels

The next group is the one that should still belong to the standard branch, if
`gamma` behaves like a higher observable built on the same first-level
language.

### A1. First explicit cubic-force channels

Candidate structure:

- linear response of the geometric/inertia branch to `phi3`;
- mixed Coriolis-cubic channels if they survive in the final assembled sum.

This is the direct analogue of the explicit cubic anharmonic term already
visible in `alpha`.

### A2. Quadratic cubic-force channels

Candidate structure:

- `phi3 * phi3` terms, if the quadratic occupation dependence requires one
  further perturbative layer;
- still expressed through first-level geometric couplings if possible.

This channel is important because it may mark the difference between:

- a standard `gamma` still assembled from known inputs;
- and a first genuinely new tensorial layer.

### A3. Quartic-force channels

Candidate structure:

- explicit `phi4` dependence in the observable branch.

This is the natural analogue of the statement that standard sextics need cubic
forces but not yet new primitive inertia derivatives. For `gamma`, the role of
`phi4` has to be tested explicitly rather than assumed away.

## 4. First higher-level geometric channels

If `gamma` does not close completely on the first-level sector, the next
candidate is the second geometric response.

### B1. Bilinear closure of mu2 through B(mu1)

Candidate structure:

- terms that look like `mu2`, but reduce algebraically to `B(mu1)`.

These should be counted as enlarged closure terms, not as genuinely new
tensor content.

### B2. Intrinsic second-level tensor N

Candidate structure:

- terms involving
  `mu2 = B(mu1) - N`
  in which the `N` part survives explicitly.

This would be the vibration-rotation analogue of the first controlled
breakdown already identified on the centrifugal-distortion branch.

### B3. Mixed first-level / second-level channels

Candidate structure:

- `mu1` with `N`,
- `zeta` with `N`,
- or `c1` with second-level geometric response.

If these appear, they are strong evidence that `gamma` is already beyond the
pure first-level sector.

## 5. Recommended decomposition for symbolic work

The symbolic derivation should try to organize `gamma` as:

`gamma = gamma_geom_std + gamma_anh_std + delta_gamma_break + ...`

where:

- `gamma_geom_std` collects `G1 + G2 + G3`;
- `gamma_anh_std` collects `A1 + A2 + A3` as far as they still belong to a
  standard observable branch;
- `delta_gamma_break` collects the first terms involving `N` or the first
  irreducible second-level tensor content.

This mirrors the CeDiTT strategy:

- isolate the largest transportable or reusable standard sector;
- then identify the first controlled breakdown rather than mixing everything
  into a single high-order formula.

## 6. What to test first

The most efficient order of attack is:

1. test whether the quadratic occupation dependence can be assembled from
   `G1 + G2 + G3` alone;
2. if not, add `A1` and identify the first explicit cubic-force contribution;
3. only then test whether `A2/A3` are sufficient or whether `B2` is needed;
4. treat `B2` as the first candidate for a controlled `gamma` breakdown.

This order is important because it prevents `gamma` from being overdesigned
too early as a fully general high-order object.

## 7. Practical output target

The backend should eventually expose, at minimum:

- `gamma_total`,
- `gamma_geom_std`,
- `gamma_anh_std`,
- `gamma_break_first`,
- mode-pair-resolved partial contributions,
- symmetry-adapted readout for special rotor limits.

This is the natural extension of the current `alpha` API.
