# Linear Aliev Unit Audit (2026-03-25)

## Scope

Audit of the current Gaussian -> Aliev bridge used by:

- [/Users/vincenzobarone/centrifugal/scripts/build_linear_aliev_payload_from_gaussian.py](/Users/vincenzobarone/centrifugal/scripts/build_linear_aliev_payload_from_gaussian.py)
- [/Users/vincenzobarone/centrifugal/linear_dv_aliev_terms.py](/Users/vincenzobarone/centrifugal/linear_dv_aliev_terms.py)

Reference test case:

- `C2H2`

## What is already coherent

- `B` is taken from the perpendicular rotational constants and converted to `cm^-1`.
- `D_J` is parsed from Gaussian linear Pickett output and converted from `MHz` to `cm^-1`.
- `q^e`, `q^J`, `q^K` parsing is internally consistent.
- `alpha_cm`, `phi3_reduced_cm`, `phi4_reduced_cm`, `phi3_raw_au`, `phi4_raw_au` are parsed correctly as printed by Gaussian.

These are parser-level statements only. They do not imply compatibility with the Aliev convention.

## Main quantitative problem

The current bridge is not yet unit-clean for quantitative Aliev use.

## Current recommended baseline for `zeta_nt`

The scanned definition identifies

- `zeta_nt = zeta^x_(n,tb)`

so the scalar `zeta_nt` should be built from one explicit Coriolis component,
not from a norm over the full degenerate pair.

Working conclusion from the current `C2H2` audit:

- `zeta_reduction = component_tb` is the only defensible default so far
- `zeta_reduction = component_ua` is numerically equivalent to `component_tb`
- `zeta_reduction = norm` is not compatible with the paper definition
- `zeta_reduction = maxabs` is only a rough sensitivity check

Systematic doublet scan on `C2H2`:

- small branch:
  - `component_tb`
  - `component_ua`
- large branch:
  - `component_ta`
  - `component_ub`
  - `maxabs`
- worst branch:
  - `norm`

Therefore the current operational rule is:

- keep `component_tb` as the default
- accept `component_ua` as an equivalent sensitivity branch
- treat `component_ta`, `component_ub`, `maxabs`, and especially `norm` as
  non-primary diagnostics only

In the current implementation this equivalence is not just qualitative.
For the `C2H2` reference case, `component_tb` and `component_ua` give the same
results to machine precision for:

- `beta_parallel`
- `beta_perpendicular`
- `uv_parallel`
- `uv_perpendicular`
- `D_v(0)`
- `L`

The only visible difference is in the sign/layout of the raw selected
`zeta_nt` components before they enter the quadratic combinations used by the
Aliev expressions. Operationally, the repo can therefore treat
`component_ua` as a strict basis-equivalent branch of `component_tb`.

Therefore the reporting policy is:

- use `component_tb` in operational reports and comparisons
- keep `component_ua` only in dedicated equivalence checks
- reserve the full four-component scan for diagnostic work

This improves the scale of the prototype significantly for the non-quadratic
blocks, but it does not close the quantitative bending gap with experiment.

## Current operational status of bending modes

The bending branch is no longer treated as a quantitative `Delta D_t`
benchmark.

Current operational policy:

- use `pair_seed_source = gaussian_qe_source` as a non-physical bridge
- interpret that bridge as a proxy for the pairwise `l`-type `J0` layer
- do **not** interpret `v4/v5` output as a quantitative validation of
  `beta_t` or `Delta D_t`

Reason:

- the pure Aliev bending seed built directly from the current Gaussian-side
  `zeta` objects is not physically trustworthy
- Gaussian `q^e` is not the same observable as the scalar vibrational
  increment `Delta D_t`
- therefore the current bending comparison is qualitative only

## Small-branch residual audit

Using the physically acceptable small branch

- `zeta_reduction = component_tb`
- or equivalently `zeta_reduction = component_ua`

and fixing the bending seed to Gaussian

- `pair_seed_source = gaussian_qe_source`

the residual dependence on the bootstrap of `B_n^(xx)` is now very limited for
the perpendicular sector.

On `C2H2`, the following three choices were compared:

- `alpha_perp_with_Bxx_equals_minus_alpha_perp`
- `alpha_perp_with_Bxx_equals_minus_half_alpha_perp`
- `didq_linear_v_iscr`

The results are:

- `max|beta_parallel| = 2.19e-04, 2.07e-04, 2.03e-04 cm^-1`
- `max|beta_perpendicular| = 5.44e-03, 5.49e-03, 5.50e-03 cm^-1`
- `max|uv_parallel| = 2.19e-04, 2.07e-04, 2.03e-04 cm^-1`
- `max|uv_perpendicular| = 2.61e-04, 2.18e-04, 2.04e-04 cm^-1`

The practical interpretation is:

- the perpendicular residual is almost insensitive to the current `B_n^(xx)`
  bootstrap
- the parallel sector remains somewhat sensitive to the geometric bridge
- `L` changes very little across the same scan

Therefore, once the analysis is restricted to the operational bridge branch,
the dominant remaining bending issue is no longer primarily in the `B_n^(xx)`
bootstrap.

The largest issue is the bootstrap of `B_n^(xx)` from Gaussian `alpha_perp`.

Current code path:

- [`build_linear_aliev_payload_from_gaussian.py`](/Users/vincenzobarone/centrifugal/scripts/build_linear_aliev_payload_from_gaussian.py)
- `bxx_parallel <- -alpha_perp` or `-alpha_perp/2`
- [`linear_dv_aliev_terms.py`](/Users/vincenzobarone/centrifugal/linear_dv_aliev_terms.py)
- `C_n = -B * B_n^(xx) / omega_n`

With `B`, `B_n^(xx)`, and `omega_n` all in `cm^-1`, the resulting `C_n` carries an extra frequency scale unless `B_n^(xx)` already matches the exact Aliev normal-coordinate convention. That identification has not been demonstrated.

## Second major problem

The force constants fed into the Aliev equations are Gaussian reduced force constants:

- `phi3_reduced_cm`
- `phi4_reduced_cm`

The current implementation uses them directly as:

- `k3_parallel`
- `k3_perp_pair`
- `k4_reduced`

This is structurally useful, but it has not been shown that these objects coincide numerically with the `k'` constants assumed in the Aliev formulas.

Therefore the current bridge should be treated as:

- structurally useful
- good for sign/ranking experiments
- not yet quantitatively validated

## Numerical evidence from C2H2

Representative payload values:

- `B = 1.1766208404697134 cm^-1`
- `D_J = 1.4489396527780562e-06 cm^-1`
- `bxx_parallel = [0.00267, 0.00213, 0.00213]`
- `C_n = [-1.50e-06, -7.28e-07, -7.08e-07]`

Current model output:

- `beta_parallel ~ -1.36e-02 cm^-1`
- `beta_perpendicular ~ +5.2e-01 ... +5.6e-01 cm^-1`

Experimental working dataset:

- `beta_exp ~ 1e-09 ... 2e-08 cm^-1`

This mismatch is far too large to be explained by a minor sign or indexing mistake. The bridge is mixing conventions, not merely suffering from a small coding bug.

## Important structural observation

For `C2H2` the current bootstrap produces:

- `k3_parallel = 0`
- one nonzero block in `k3_perp_pair`
- sizeable `k4_reduced`

Therefore the current `beta` values are dominated by the quartic and l-type sectors, not by a balanced full Aliev evaluation.

## Operational conclusion

The current Gaussian -> Aliev bridge is acceptable only as:

- a structural prototype
- a ranking/sign diagnostic
- a way to test mode bookkeeping and formula wiring

More precisely:

- stretching-side output can still be used as a quantitative internal
  diagnostic
- bending-side output is currently only a qualitative proxy through the
  pairwise `l`-type bridge

It is **not** acceptable yet as a quantitative pipeline for comparison with
experimental bending `beta_k` or `Delta H_k`.

## Next required work

1. Build the true rotational-derivative `zeta` object for linear molecules.
2. Reconstruct the exact Aliev convention for `B_n^(xx)`.
3. Determine whether the paper's `k'_{ijk}` and `k'_{ijkl}` match Gaussian reduced force constants, or require re-normalization from `phi_raw_au`.
4. Rebuild the payload using those exact conventions.
5. Only then compare bending observables against experiment.

## Frozen diagnostic state (2026-03-26)

The current frozen linear diagnostic branch is:

- `pair_seed_source = rotder_seed_gram`
- `beta_t_xf_cross_sign = -1` for the perpendicular `xf/cross` block
- `uv_block = 0` in the parallel-sector `beta_n` assembly

This is not claimed as a final derivation of the linear Aliev theory. It is the
first state that gives numerically sane linear outputs on both reference
systems now in the repo:

- `C2H2`
- `HCN`

With this frozen branch:

- `C2H2` gives
  - `v1_CH ~ -1.530e-09`
  - `v2_CC ~ -1.199e-10`
  - `v3_asym ~ -9.539e-10`
  - `v4_bend ~ +5.970e-09`
  - `v5_bend ~ +4.728e-09`
- `HCN` gives
  - `beta_parallel ~ 4.45e-17, 1.74e-15`
  - `beta_perpendicular ~ -1.79e-15`

Operational interpretation:

- the catastrophic scale error is gone
- the perpendicular sector is controlled by the rotational-derivative seed
  scaffold plus the sign-corrected `xf/cross` insertion
- the old off-diagonal parallel `uv` contribution behaved as a representation
  artifact and is disabled in the frozen diagnostic branch

So the linear branch is now usable as a **working benchmark branch**.
It is still not a literature-validated final implementation.

## Rigorous local conclusion on the remaining two conventions

The current linear branch rests on two conventions that work numerically but are
not yet proved from the literature:

- `beta_t_xf_cross_sign = -1`
- `uv_parallel = 0`

These two conventions do not have the same status.

### 1. Perpendicular `xf/cross` sign

For the current `rotder_seed_gram` branch, the bending residual is controlled by
the full perpendicular `xf + cross` block, not by one subterm alone.

On `C2H2`:

- after correcting the negative branch to use `F_tt` / `F^tt` rather than
  reusing `F_nn` / `F^nn`, the current insertion still gives `xf + cross > 0`
- flipping only `xf` gives the wrong magnitude
- flipping only `cross` gives the wrong magnitude
- flipping the **common prefactor of the full `xf + cross` insertion** gives the
  correct sign and keeps the correct order of magnitude

Therefore the current evidence supports only the following strict statement:

- the unresolved issue is a sign/convention problem of the **whole**
  perpendicular `xf/cross` insertion in `beta_t`
- it is **not** evidence that only one of the two pieces (`xf` or `cross`) was
  individually transcribed with the wrong sign
- there was also a real structural bug in the earlier implementation:
  the negative `xf/cross` branch of `beta_t` must use the perpendicular
  families `F_tt` / `F^tt`, not the parallel families `F_nn` / `F^nn`

### 2. Parallel `uv`

For the current `rotder_seed_gram` branch, the parallel discrepancy is dominated
by the parallel `uv` insertion.

On `C2H2` and `HCN`:

- the diagonal part of parallel `U_{nn}` is already negligible after the earlier
  correction
- the large residual comes from the off-diagonal `U_{nn'}` / `V_{nn'}`
  contribution
- this residual behaves as an almost mode-independent offset across active
  parallel modes

When the parallel `uv` insertion is removed:

- `C2H2` parallel coefficients drop from `~10^-3` to `~10^-10 ... 10^-9`
- `HCN` parallel coefficients drop to `~10^-17 ... 10^-15`

This strongly suggests that the present parallel `uv` insertion is not yet the
physical object intended by the Aliev formalism under the current
Gaussian-to-Aliev mapping.

However, this does **not** prove that the true theoretical value is exactly
zero. It proves only:

- the catastrophic `uv_parallel` blowup was not caused by the mere presence of
  the `U_(n n')/V_(n n')` block
- it was caused by feeding that operator block with the wrong degenerate-pair
  scalar Coriolis reduction

The decisive discriminator is now numerical and structural:

- with `principal_direction`, the off-diagonal `U_(n n')/V_(n n')` term blows
  up to about `10^-3` on `C2H2` and `10^-7` on `HCN`
- with `pair_offdiag`, the same block drops to about `10^-8` on `C2H2` and
  `10^-12 ... 10^-14` on `HCN`

So the real issue was not ``uv_parallel must be zero''. The real issue was:

- the operator sector `X/F/U/V` was being fed with a degenerate-subspace
  reduction (`principal_direction`) that is not admissible for the parallel
  `U_(n n')/V_(n n')` block

Under the current working branch, `uv_parallel` is therefore active again, but
the operator terms use `pair_offdiag` as the Coriolis reduction.

### Operational meaning

So the correct rigorous reading of the current linear branch is:

- perpendicular branch:
  - sign of the full `xf/cross` insertion is fixed operationally at `-1`
- parallel branch:
  - `uv_parallel` is retained
  - the operator sector uses `pair_offdiag`, because `principal_direction`
    produces a non-physical off-diagonal `U_(n n')/V_(n n')` blowup

This closes the implementation as a **working branch**.
It does not yet close the linear theory as a publishable derivation.
