# Linear Aliev: What Remains To Prove

This note records the two points that still prevent the current linear branch
from being treated as a fully derived, publishable implementation.

It is intentionally shorter and stricter than the full audit in
[`LINEAR_ALIEV_UNIT_AUDIT_2026-03-25.md`](/Users/vincenzobarone/centrifugal/LINEAR_ALIEV_UNIT_AUDIT_2026-03-25.md).

## Status Already Reached

The current linear branch is no longer in exploratory debugging.

What is already established:

- the Gaussian bootstrap is representation-consistent enough to produce stable
  linear diagnostics
- the `X` families match Eq. (4)
- `F^(nn')` is consistent with Eq. (6)
- `F_(nn')` had a real extra factor and has been corrected against Eq. (7)
- the negative `X/F` block of `beta_t` must use perpendicular families
  `F_(tt')` and `F^(tt')`, not `F_(nn')` and `F^(nn')`
- the bending seed built directly from Gaussian Coriolis data is not a valid
  physical `zeta` replacement
- the current working bending branch is numerically sane only with the
  rotational-derivative diagnostic seed and the frozen sign convention on the
  perpendicular `xf/cross` block
- the parallel `uv` problem is now localized more sharply: the off-diagonal
  `U_{nn'}/V_{nn'}` block becomes non-physical only when the operator sector is
  fed with the wrong degenerate-pair scalar Coriolis reduction

So the current branch is a strong working branch, but not yet a closed theory
branch.

## Residual Point 1: Global Sign Of The Perpendicular `xf/cross` Block

### What is known

With all structural corrections already applied, the residual sign problem in
the bending sector is local:

- it is not caused by `D_v`
- it is not caused by the comparison layer
- it is not caused by `X`
- it is not caused by the already-corrected `F_(nn')`
- it is not caused by the earlier wrong use of `F_(nn')/F^(nn')` in the
  negative `beta_t` block

The remaining issue is the global sign of the full perpendicular
`xf/cross` insertion in `beta_t`.

### What is still missing

One of the following must be proved:

1. the global sign in the scanned `beta_t` formula is effectively opposite to
   the literal transcription now frozen in the note;
2. the perpendicular upper/lower convention of the `F_(tt')` / `F^(tt')`
   families still differs from the convention used in the code;
3. the sign is fixed by a mode-axis convention that is not stated explicitly in
   the scanned equations but is implicit in the underlying derivation.

### What would close this point

This point is closed only if one of these is produced:

- a text-level derivation from the original Aliev formalism;
- an independent reference derivation reproducing the sign;
- a mathematically explicit convention map showing why the code sign must be
  the opposite of the literal scan sign.

Without that, the sign remains a frozen working convention:

- `beta_t_xf_cross_sign = -1`

not a proved theorem.

## Residual Point 2: Physical Status Of `uv_parallel`

### What is known

`check.jpg` closes the textual ambiguity about the last line of `beta_n`: the
parallel `uv` block is an `U_{nn'}/V_{nn'}` term, not a `U_{tt'}/V_{tt'}`
term.

So the unresolved issue is no longer the placement of the `uv` term. It is the
object entering that term.

There is now a direct benchmark-level discriminator:

- with `principal_direction`, the off-diagonal `U_{nn'}/V_{nn'}` contribution
  blows up:
  - `C2H2`: about `1.6\times 10^{-3}` on two parallel modes
  - `HCN`: about `1.56\times 10^{-7}` on both parallel modes
- with `pair_offdiag`, the same `U_{nn'}/V_{nn'}` block drops to a physically
  sane scale:
  - `C2H2`: about `10^{-8}`
  - `HCN`: about `10^{-12}\dots 10^{-14}`

So the real unresolved point is now precise:

- the parallel `uv` sector exists
- but its operator input is extremely sensitive to the degenerate-subspace
  scalar used for the Coriolis proxy
- `principal_direction` is not admissible there
- `pair_offdiag` is the first candidate reduction that keeps the parallel
  branch numerically sane without deleting `uv_parallel`

### What is still missing

One of the following must be proved:

1. the scanned `beta_n` formula really does contain this common
   `U_{nn'}/V_{nn'}` insertion, but the operator proxy entering it must be
   built with a specific degenerate-subspace reduction such as `pair_offdiag`;
2. the present Gaussian-to-Aliev mapping is still feeding the wrong object into
   the `U/V` sector for the parallel branch.

### What would close this point

This point is closed only if one of these is produced:

- a derivation showing that the common `U_{tt'}/V_{tt'}` term belongs outside
  the mode-resolved `beta_n` coefficients;
- a derivation showing that `pair_offdiag` is the correct operator reduction
  for the `U_{nn'}/V_{nn'}` sector;
- or a derivation of the true parallel-sector object replacing the current
  scalar Coriolis proxy.

Until then, the working branch uses:

- `uv_parallel` active
- `pair_offdiag` as the frozen operational reduction for the operator sector

## Publication Threshold

The linear branch becomes publishable only after both residual points are
promoted from working conventions to derived statements.

That means:

- the sign of the perpendicular `xf/cross` insertion must be justified;
- the use of `pair_offdiag` in the parallel `U_{nn'}/V_{nn'}` sector must be
  derived, not only frozen operationally.

Everything else is now in a state good enough to support that final theoretical
closure.
