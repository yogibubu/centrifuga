# Paper 2 Resonance Strategy

This note fixes an important structural decision for the quartic VPT4 branch.

## 1. Why resonances enter Paper 2

In the present channel decomposition,

`tau^(4) = tau(H12H12) + tau(H22) + tau(H12H30) + tau(H30H30)`,

the first three pieces behave as non-resonant corrections in the current
formulation:

- `H12H12` is the standard first-level sector;
- `H22` is the first controlled geometric breakdown;
- `H12H30` is the lowest explicit anharmonic dressing of the standard sector.

The first channel that can generate genuinely dangerous perturbative
denominators is `H30H30`.

This is therefore the first place where a resonance treatment analogous to the
one used in vibrational VPT2 becomes necessary.

By contrast, the present `H12H30` implementation uses only positive-sum
frequency kernels of the type

- `w_i^2`,
- `w_i (w_i + w_j)`,
- `(w_i + w_j)(w_j + w_k)(w_i + w_k)`,

and therefore does not show the same resonance sensitivity in its current
non-resonant form.

## 2. General principle

Resonances must not be handled case by case for individual molecules.
The code and the theory should instead identify them automatically from the
denominator structure of the channel.

The correct strategy is:

1. derive the full denominator families entering `H30H30`;
2. define general resonance indicators for those families;
3. apply numerical thresholds to detect near-resonant denominators;
4. separate:
   - non-resonant contribution,
   - resonant or quasi-resonant contribution;
5. decide whether the resonant part should be:
   - excluded from the non-resonant perturbative sum,
   - regularized,
   - or transferred to a dedicated effective treatment.

## 3. Why this matters for H2O

Water already shows a clear near-Fermi condition:

`2 nu_2 - nu_3 ~ -107.6 cm^-1`

with the current harmonic/anharmonic frequencies extracted from the Gaussian
benchmark.

This makes H2O the natural stress test for the resonance machinery, but it
must not be treated as a special exception. The same machinery should apply to
all molecules.

## 4. Working hypothesis

The current `H30H30` scaffold model uses only positive-sum frequency kernels in
its explicit implementation. This suggests that the present code is still
missing the full denominator structure of the cubic-cubic channel near
resonance, even though the numerical behavior of `H2O` already shows that the
channel is sensitive to near-resonant situations.

The working hypothesis for Paper 2 is therefore:

- `H30H30` is the first and lowest quartic VPT4 channel that requires an
  explicit resonance analysis;
- the poor behavior observed for H2O is likely a sign that the non-resonant
  cubic-cubic formula is being used outside its domain of validity;
- the resonance-aware decomposition should be developed at the channel level,
  not by ad hoc fixes on individual molecules.

## 5. Immediate technical tasks

The next steps should be:

1. extract the full denominator families entering the cubic-cubic quartic
   correction;
2. classify them by resonance type, for example:
   - `2w_i - w_j`,
   - `w_i + w_j - w_k`,
   - other small-difference combinations appearing in the BCH reduction;
3. add a diagnostic layer that reports, for each molecule:
   - the small denominators,
   - the active resonance class,
   - the scaffolds fed by those denominators;
4. only after that decide how to modify the numerical `H30H30` contribution in
   production mode.

## 6. Present evidence

The current four-molecule benchmark indicates:

- `H12H30` stays small for `H2O`, `H2S`, `H2CO`, and `H2CS`;
- `H30H30` is negligible for `H2CO` and `H2CS`,
  moderate for `H2S`,
  and dominant for `H2O`;
- in `H2O`, the dominant `H30H30` scaffolds are `diag_1_iii_iii` and
  `diag_1_iii_iij_0`.

This makes it unlikely that the immediate resonance problem belongs to the
mixed `H12H30` channel. The resonance-sensitive channel to fix first is
therefore `H30H30`.

## 7. Conceptual consequence for Paper 2

This is an important structural point of the paper:

- `H22` is the first controlled geometric breakdown;
- `H30H30` is the first channel that introduces explicit resonance
  sensitivity.

These are not the same thing and must be kept conceptually distinct in the
manuscript.
