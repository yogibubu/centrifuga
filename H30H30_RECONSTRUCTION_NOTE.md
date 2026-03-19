# H30H30 Reconstruction Note

## Current Status

The current numerical implementation of `H30H30` in
[quartic_channels.py](/Users/vincenzobarone/centrifugal/quartic_channels.py)
is still a placeholder:

```text
phi3[i,j,j] * mu1[i] * mu1[j] / (omega_i + omega_j + 1)
```

This is useful only as a diagnostic stub. It is not structurally compatible
with the symbolic BCH-derived channel.

## What Is Already Established

### 1. One-mode symbolic baseline

The symbolic extractor
[scripts/extract_h30h30_symbolic_denominators.py](/Users/vincenzobarone/centrifugal/scripts/extract_h30h30_symbolic_denominators.py)
already shows that, in the one-mode case, the genuine `H30,H30` channel scales
as

```text
tau_mu^(30,30) ~ Phi_iii^2 / omega_i^7
```

for all six quartic tensor components.

This is the first hard constraint on any reconstruction:

- the true channel is an `omega^-7` object already at one mode;
- the current placeholder kernel `omega_i + omega_j + 1` is therefore in the
  wrong structural class.

### 2. Resonance sensitivity belongs here first

The notes
[PAPER2_RESONANCE_STRATEGY.md](/Users/vincenzobarone/centrifugal/PAPER2_RESONANCE_STRATEGY.md)
and
[H2O_quartic_VPT4_diagnostic.md](/Users/vincenzobarone/centrifugal/H2O_quartic_VPT4_diagnostic.md)
already make the correct physical point:

- `H30H30` is the first quartic channel that genuinely requires resonance
  analysis;
- `H2O` is the stress test;
- the dominant scaffolds are
- `diag_0_iii_iii`
  - `diag_1_iii_iij_0`

So reconstruction should start from those families first, not from a fully
generic all-scaffold fit.

### 3. The current symbolic probe is fragile under pruning

The updated denominator extractor now supports

- `--channel-aware`
- `--max-vib-word`
- `--max-j-word`

but the channel is much more fragile than `H12H30`:

- aggressive pruning can kill even the one-mode signal;
- two-mode generic scans quickly become too heavy to be useful as blind probes.

This suggests that a successful BCH-first recovery of `H30H30` will have to be
family-targeted, not just globally pruned.

## Practical Consequences

### 4. External VPT vs PFIT/RVCI only gives an order-of-magnitude ceiling

The later comparison paper at
`/Users/vincenzobarone/Desktop/centrifugal_VCI.pdf` should not be used as a
channel benchmark, but it is still a useful sanity check on the total quartic
scale.

The script
[scripts/h30h30_vci_sanity_check.py](/Users/vincenzobarone/centrifugal/scripts/h30h30_vci_sanity_check.py)
compares the current total `H30H30` contribution against the external
`PFIT/RVCI(E) - VPT` quartic differences for `H2O`, `H2S`, `H2CO`, and
`H2CS`.

The result is unambiguous: the present `H30H30` total is still far too large.
The norm ratios are currently

- `H2O`: about `8.0e3`
- `H2S`: about `8.5e3`
- `H2CO`: about `4.6e3`
- `H2CS`: about `1.1e5`

So even allowing for reduction differences and different electronic-structure
levels, the current scaffold sum is not yet in the physically plausible range.
This is a strong indication that the operative scaffold definitions are still
incorrect or incomplete.

This now motivates the current solver split:

- `pure_diagonal_total = D^(0)_{iii,iii}`
- `trusted_total = D^(0)_{iii,iii} + D^(1)_{iii,iii}`
- `extended_total = D^(0)_{iii,iii} + D^(1)_{iii,iii} + D^(1;0)_{iii,iij} + residual`

Only `trusted_total` should be treated as the operative provisional channel.
`pure_diagonal_total` is kept explicit to monitor how much of the present
solver-facing channel already comes from the uncorrected diagonal scaffold.
The wider `extended_total` remains diagnostic-only until the remaining scaffold
families are reconstructed with a defensible normalization.

The current diagonal balance is already informative:

- `H2O`: `||D^(1)||/||D^(0)|| ≈ 1.18`, `||D^(0)+D^(1)||/||D^(0)|| ≈ 0.53`
- `H2S`: `||D^(1)||/||D^(0)|| ≈ 1.74`, `||D^(0)+D^(1)||/||D^(0)|| ≈ 0.75`
- `H2CO`: `||D^(1)||/||D^(0)|| ≈ 24.8`, `||D^(0)+D^(1)||/||D^(0)|| ≈ 23.9`
- `H2CS`: `||D^(1)||/||D^(0)|| ≈ 31.0`, `||D^(0)+D^(1)||/||D^(0)|| ≈ 30.0`

So for `H2CO/H2CS` the present `D^(1)_{iii,iii}` correction is not a mild
refinement of the diagonal scaffold: it overwhelmingly controls the operative
placeholder channel. This is a strong signal that the diagonal family must be
reconstructed before any broader scaffold promotion is meaningful.

This conclusion is also consistent with the current appendix-style prefactors.
The componentwise `D^(1)/D^(0)` coefficient ratios are highly anisotropic:

- `tau_xxxx`: about `-1.46`
- `tau_yyyy`: about `-32.8`
- `tau_zzzz`: about `-0.83`
- `tau_xxyy`: about `-11.6`
- `tau_xxzz`: about `-2.45`
- `tau_yyzz`: about `+3.45`

So the present `D^(1)` scaffold is not behaving like a mild uniform correction
to the diagonal core. Even before any molecular weighting by `mu1` and `phi3`,
it already carries a strongly component-dependent normalization pattern.

The same conclusion holds if one tries the most forgiving global repair:
fit scalar weights `alpha D^(0) + beta D^(1)` against the working-reference
`H30H30` values for `H2O/H2S/H2CO/H2CS`. The diagonal-only least-squares audit
now gives

- `corr(D^(0), D^(1)) ≈ -0.91`
- `alpha ≈ -28.37`
- `beta ≈ -11.68`

with residuals that remain macroscopically large, especially for `H2O/H2S`.
So the current issue is not a missing global scale factor. The present diagonal
basis is itself strongly collinear, but even an orthogonalized reparametrization
of the same 2D subspace leaves the residual unchanged. The diagonal family must
therefore be rebuilt structurally, not just reweighted or rebased.

A useful benchmark simplification is now available in
[scripts/h30h30_diagonal_symbolic_probe.py](/Users/vincenzobarone/centrifugal/scripts/h30h30_diagonal_symbolic_probe.py):
if one keeps only diagonal rotational couplings and the purely diagonal cubic
block `\Phi_{iii}`, the exact BCH channel collapses onto a *single* diagonal
scaffold with universal ratios

- `tau_xxxx`, `tau_yyyy`, `tau_zzzz`: `-1/1728`
- `tau_xxyy`, `tau_xxzz`, `tau_yyzz`: `-1/864`

independently of `n_modes` and of the collapsed benchmark seed.

This means that the second diagonal pivot is not generated by the trivial
`iii`-only core itself. It must originate from the broader rotational scaffold
mixing present in the full rank-7 benchmark space. This is exactly where the
next reconstruction step should focus.

This picture is now sharpened further by
[scripts/h30h30_diagonal_rotmix_probe.py](/Users/vincenzobarone/centrifugal/scripts/h30h30_diagonal_rotmix_probe.py):
for a two-mode collapsed benchmark, adding just one off-diagonal rotational
block `(ab)+(ba)` to the `iii`-only core already deforms the universal ratios.
So the missing second diagonal pivot is plausibly a *rotational-mixing pivot*
rather than a second independent `\Phi_{iii}^2/\omega_i^7` scalar core.

An even simpler sanity check is now available through
[scripts/h30h30_diagonal_subspace_probe.py](/Users/vincenzobarone/centrifugal/scripts/h30h30_diagonal_subspace_probe.py):
already in the one-mode collapsed benchmark, switching from diagonal rotational
couplings to the full rotational set changes the `xxyy`, `xxzz`, and `yyzz`
entries of the `iii`-only channel. So the deformation of the diagonal core does
not require multi-mode cubic structure; it can arise from rotational mixing
alone.

This now suggests a concrete reconstruction route for the second diagonal pivot:
define the first candidate as the *rotational-mixing correction* of the
`iii`-only core,
\[
\Delta^{\mathrm{rotmix}}_{iii}
=
\tau^{(30,30)}_{\mathrm{iii\text{-}only,full\ rot}}
-
\tau^{(30,30)}_{\mathrm{iii\text{-}only,diag\ rot}},
\]
which is inspected operationally in
[scripts/h30h30_diag1_rotmix_candidate.py](/Users/vincenzobarone/centrifugal/scripts/h30h30_diag1_rotmix_candidate.py).
In the one-mode probe this correction is supported only on the mixed tensor
components (`xxyy`, `xxzz`, `yyzz`), while the pure `xxxx`, `yyyy`, `zzzz`
entries remain unchanged.

Comparing this candidate against the current code placeholder through
[scripts/h30h30_diag1_candidate_compare.py](/Users/vincenzobarone/centrifugal/scripts/h30h30_diag1_candidate_compare.py)
gives a very weak alignment:

- one-mode probe (`seed=7`): `corr(diag_1, \Delta^{rotmix}_{iii}) \approx -2.3\times 10^{-2}`
- two-mode probe (`seed=7`): `corr(diag_1, \Delta^{rotmix}_{iii}) \approx -3.9\times 10^{-1}`

So the present `diag_1_iii_iii` placeholder is not just mis-scaled; it is not
well aligned with the rotational-mixing correction singled out by the restricted
symbolic probe. This strongly argues for replacing the current placeholder
rather than merely renormalizing it.

The next diagnostic step, recorded in
[scripts/h30h30_post_rotmix_residual_audit.py](/Users/vincenzobarone/centrifugal/scripts/h30h30_post_rotmix_residual_audit.py),
shows that the residual left after `diag_0 + rotmix` is *not* captured by a
single universal follow-up scaffold:

- `H2O`: best one-block follow-up is `diag_0_iij_iij_1_candidate`
- `H2S`: best one-block follow-up is `diag_1_iii_iij_0`
- `H2CO`: best one-block follow-up is `diag_0_iii_iij_2_regularized_preview`
- `H2CS`: best one-block follow-up is `diag_0_iij_iij_1_candidate`

So `diag_0 + rotmix` is a better starting point than the old diagonal pair, but
it still leaves a species-dependent residual structure. This again suggests
that the true second diagonal-family correction is not identical to any single
already coded semi-diagonal scaffold.

The more aggressive check in
[scripts/h30h30_post_rotmix_global_fit.py](/Users/vincenzobarone/centrifugal/scripts/h30h30_post_rotmix_global_fit.py)
confirms the same conclusion from the opposite direction: a *global* fit of the
post-rotmix residual in the span of the currently coded follow-up scaffolds does
not yield a stable universal correction block. It improves the `H2O/H2S` side
but badly degrades `H2CO/H2CS`. So the missing piece is not simply a universal
linear combination of the already coded semi-diagonal candidates.

Even in the restricted one-mode `iii`-only benchmark, the two-vector basis
`{diag_0, rotmix}` is not universally complete once different collapsed seeds
are compared. The probe
[scripts/h30h30_diag_rotmix_basis_probe.py](/Users/vincenzobarone/centrifugal/scripts/h30h30_diag_rotmix_basis_probe.py)
shows that:

- seed `7` is reproduced exactly by construction;
- seeds `11` and `13` leave non-zero residuals;
- the residual sample span already has rank `2`.

So the next missing correction is present even in the minimal `iii`-only/full
rotational benchmark family. This strongly suggests that the diagonal sector
needs at least one additional benchmark-derived basis vector beyond
`diag_0 + rotmix`.

This can now be stated more sharply. The helper
[scripts/h30h30_diag_benchmark_basis.py](/Users/vincenzobarone/centrifugal/scripts/h30h30_diag_benchmark_basis.py)
builds the minimal one-mode benchmark-derived basis obtained from the sampled
collapsed probes:

- `diag(seed=7)`
- `rotmix(seed=7)`
- `residual(seed=11)`
- `residual(seed=13)`

and that basis has rank `4` while reconstructing the sampled full-rotation
vectors for seeds `7`, `11`, and `13` exactly. So the minimal one-mode
diagonal enlarged sector is no longer consistent with a two-pivot closure.
The next reconstruction step must therefore target at least a four-vector
benchmark-derived carrier before any solver-facing reduction is attempted.

That four-vector carrier can also be tested immediately on the molecular
side. The script
[scripts/h30h30_diag_benchmark_basis_fit.py](/Users/vincenzobarone/centrifugal/scripts/h30h30_diag_benchmark_basis_fit.py)
fits the external `PFIT/RVCI(E)-VPT` quartic deltas in the basis

- `diag_0_iii_iii`
- `diag_1_iii_iii_rotmix_candidate`
- `diag_bench_resid11_candidate`
- `diag_bench_resid13_candidate`

with the following residual norms:

- `H2O`: about `3.6e1`
- `H2S`: about `2.25`
- `H2CO`: about `5.8e-1`
- `H2CS`: about `1.5e-1`

So the benchmark-derived four-vector basis is already much healthier than the
old diagonal placeholder pair, and it captures the `H2CO/H2CS` side rather
well. But `H2O` remains badly described. This strongly suggests that the next
missing ingredient is not just "more diagonal basis", but specifically a
water-like correction that is not contained in the one-mode benchmark-derived
carrier.

The follow-up audit
[scripts/h30h30_post_bench4_residual_audit.py](/Users/vincenzobarone/centrifugal/scripts/h30h30_post_bench4_residual_audit.py)
shows something more specific: after the benchmark-derived 4D diagonal fit,
the remaining `H2O` residual is *not* strongly aligned with the current signed
`iii,iij` candidate. The one-block correlations are all weak, with the best
single follow-up currently coming from `diag_0_iij_iij_2_candidate`, not from
the signed `diag_0_iii_iij_2_resonance_candidate`.

So the next missing `H2O`-like correction does not yet look like a clean
"turn on the signed near-resonant scaffold" story. The more likely reading is
that the water-side residual still mixes semidiagonal and off-diagonal
structures that are not captured by the present coded `iii,iij` / `iij,iij`
families.

This picture can now be sharpened one step further. The script
[scripts/h30h30_waterlike_residual_span.py](/Users/vincenzobarone/centrifugal/scripts/h30h30_waterlike_residual_span.py)
shows that the post-diagonal residuals of `H2O` and `H2S`

- have sampled rank `2` in Watson space;
- are contained exactly in the span of the *already coded* follow-up families
  `diag_1_iii_iij_0`, `diag_0_iii_iij_1_candidate`,
  `diag_0_iii_iij_2_(...)`, `diag_0_iij_iij_(...)`.

So the current evidence no longer points to a *missing new scaffold family* on
the water side. Instead, it points to a poor basis choice inside the existing
semi-/off-diagonal follow-up span.

One must however be careful not to overinterpret raw interpolation. The helper
[scripts/h30h30_diag4_plus_water2_fit.py](/Users/vincenzobarone/centrifugal/scripts/h30h30_diag4_plus_water2_fit.py)
shows that adding a rank-2 "water-like" residual basis to the diagonal 4D
carrier reproduces the four benchmark molecules exactly, but this is partly a
dimensionality effect in the 5D Watson parameter space. The meaningful point is
not exact interpolation; it is that the residual correction still appears to be
low-rank and already lives inside the present follow-up scaffold span.

The two orthonormal water-like directions are now also exposed operationally in
[quartic_channels.py](/Users/vincenzobarone/centrifugal/quartic_channels.py)
as

- `waterlike_basis1_candidate`
- `waterlike_basis2_candidate`

obtained as fixed linear combinations of the currently coded follow-up
scaffolds. They should be interpreted as *diagnostic basis vectors* of the
semi-/off-diagonal residual sector, not as normalized physical contributions to
be added with unit coefficient.

For interpretation, an even cleaner representation is now available through
[scripts/h30h30_waterlike_scaffold_basis.py](/Users/vincenzobarone/centrifugal/scripts/h30h30_waterlike_scaffold_basis.py).
In scaffold coefficient space, the sampled water-like residual is again rank
`2`, and its dominant directions are:

- an `iii,iij`-dominated axis led by `diag_0_iii_iij_1_candidate`, with
  subleading signed and `iij,iij` admixtures;
- an `iij,iij`-dominated axis led by
  `diag_0_iij_iij_1_candidate` / `diag_0_iij_iij_2_candidate`.

These scaffold-space directions are also exposed in
[quartic_channels.py](/Users/vincenzobarone/centrifugal/quartic_channels.py)
as

- `waterlike_scaffold_basis1_candidate`
- `waterlike_scaffold_basis2_candidate`

and are currently the most interpretable summary of the residual sector beyond
the diagonal benchmark-derived carrier.

An even cleaner reduction is now available. If the first scaffold-space
direction is projected onto the `iii,iij` family only, and the second onto the
`iij,iij` family only, the resulting reduced pivots

- `waterlike_reduced_iii_iij_candidate`
- `waterlike_reduced_iij_iij_candidate`

give the simplest interpretable summary of the residual sector. They do *not*
reproduce the `H2O/H2S` residual as accurately as the unreduced two-vector
water-like basis, but they show that the dominant content separates into:

- one reduced `iii,iij` pivot;
- one reduced `iij,iij` pivot.

This is presently the cleanest roadmap for the BCH-side closure of `H30H30`.

The last sanity check is global, not species-specific. The script
[scripts/h30h30_diag6_global_fit.py](/Users/vincenzobarone/centrifugal/scripts/h30h30_diag6_global_fit.py)
fits all four benchmark molecules simultaneously in the six-vector basis

- diagonal benchmark carrier `diag4`
- water-like residual carrier `water2`

with *shared* coefficients. That fit fails badly: the residual norms remain
large for all four systems and the fitted norms become strongly distorted.
So the current six-vector carrier is a useful diagnostic organization of the
channel, but it is not yet a solver-facing universal parameterization with
species-independent coefficients.

### 5. A first signed ``iii,iij`` family is now visible

The new candidate
`diag_0_iii_iij_2_resonance_candidate`
introduces the first explicit signed denominator factor of the form

```text
2*omega_i - omega_j
```

and is therefore the first scaffold that is genuinely Martin/2x2-ready.

For `H2O` this factor is already small for one of the mode pairs:

- `2*omega_1 - omega_2 ≈ -9 cm^-1`

which is exactly the kind of near-resonant detuning that the resonance
machinery is supposed to catch.

The regularized preview
`diag_0_iii_iij_2_regularized_preview`
shows that a smooth Martin/2x2 treatment can strongly suppress the raw
perturbative value of this signed candidate:

- `H2O`: the norm drops from about `2.35e6` to about `8.50e4`
- `H2S`: the norm drops from about `5.31e3` to about `2.37e3`

This does *not* mean that the family is now validated. It only means that the
resonance-control layer is finally attached to a denominator family where it is
physically meaningful.

The immediate next steps should be:

1. keep the current placeholder clearly marked as diagnostic-only;
2. use the one-mode `omega^-7` result as the normalization baseline;
3. reconstruct first the diagonal family corresponding to `diag_0_iii_iii` and `diag_1_iii_iii`;
4. then add the semi-diagonal family corresponding to `diag_1_iii_iij_0`;
5. only after these two are under control, reopen the resonance machinery on
   the recovered denominators.

## What Should Not Be Done Next

The following would likely waste time right now:

- broad two-mode symbolic scans without family targeting;
- fitting the current placeholder to working references;
- introducing resonance regularization before the denominator families are
  restored.

## Working Interpretation

At the current stage, the cleanest reading of `H30H30` is:

- the solver placeholder is intentionally not physical;
- the one-mode symbolic result already fixes the asymptotic structure;
- the reconstruction should proceed scaffold-first, beginning with the two
  diagonal families known to dominate `H2O`.
