# H12H30 S3 Formal Derivation Note

## Scope

This note tracks the formal status of the genuine three-index block of the
mixed quartic channel `H12H30`.

The current operational solver distinguishes:

- `S1`: diagonal cubic block `Phi_iii`
- `S2`: two-index cubic block `Phi_iij` / `Phi_ijj`
- `S3`: appendix-style three-index block based on the pair-sum denominator
  family
- `S3_addition`: extra three-index family retained in the solver because it is
  often comparable to, or larger than, `S3`

The present note concerns only the formal derivation of the `Phi_ijk` sector.

## Current Solver Identities

In the current code path in [quartic_channels.py](/Users/vincenzobarone/centrifugal/quartic_channels.py):

- `S3 = S3_pair = tri_ijk_over_pair_sums`
- `S3_addition = S3_wpair = tri_ijk_over_w_pair_sums`
- `three_index = S3 + S3_addition`

This is an operational decomposition, not yet a formally closed derivation.

## What Is Already Established

The following points are now clear.

1. The appendix form in [paper2.tex](/Users/vincenzobarone/centrifugal/paper2.tex) is consistent with taking
   `S3` as the pair-sum family
   `1 / [ (w_i+w_j)(w_j+w_k)(w_i+w_k) ]`.

2. In the current solver, the additional family `S3_addition` is often
   numerically dominant or comparable in norm.

3. `S3` and `S3_addition` are almost perfectly anti-collinear in the active
   test cases:
   - `HFO`
   - `H2CO`
   - `H2CS`

4. Therefore the total three-index block is currently obtained by a strong
   cancellation:
   `three_index = S3 + S3_addition`.

## Numerical Evidence

The scripts
[scripts/h12h30_s3_appendix_vs_solver.py](/Users/vincenzobarone/centrifugal/scripts/h12h30_s3_appendix_vs_solver.py)
and
[scripts/h12h30_appendix_basis_audit.py](/Users/vincenzobarone/centrifugal/scripts/h12h30_appendix_basis_audit.py)
show:

- `corr(S3, S3_addition) ~ -1`
- `||S3_addition|| / ||S3||`
  - `HFO ~ 1.65`
  - `H2CO ~ 0.94`
  - `H2CS ~ 1.05`

So the additive term cannot be dismissed as a tiny correction.

## What Failed Formally

The following symbolic probes did not recover a nonzero autonomous
three-index `H12,H30` block from the current BCH pipeline:

- [scripts/h12h30_s3_symbolic_probe.py](/Users/vincenzobarone/centrifugal/scripts/h12h30_s3_symbolic_probe.py)
  - collapsed, `tri-index only`
- [scripts/h12h30_s3_noncollapsed_probe.py](/Users/vincenzobarone/centrifugal/scripts/h12h30_s3_noncollapsed_probe.py)
  - non-collapsed, `tri-index only`
- [scripts/h12h30_s3_singletri_probe.py](/Users/vincenzobarone/centrifugal/scripts/h12h30_s3_singletri_probe.py)
  - non-collapsed, single explicit triad family `Phi_012`

Conclusion:

- the current BCH workflow does not yet expose the `Phi_ijk` block as a clean
  standalone symbolic channel;
- therefore the present `S3_addition` is not yet formally justified by the BCH
  derivation path.

## Correct Interpretation For The Manuscript

At the current stage:

- the appendix may safely present `S3` in its reduced pair-sum form;
- the extra three-index term used in the solver must be described as an
  operational additive extension still requiring formal derivation;
- it must not be presented as a fully established perturbative identity.

## Immediate Formal Goal

The next formal target is not to fit more coefficients, but to derive the
three-index contribution in a semianalytic way outside the current BCH black
box.

The minimal target is:

```text
tau_mu^(12,30;ijk) = hbar * [ A_mu * F_pair(i,j,k) + B_mu * G_extra(i,j,k) ]
```

with:

- `F_pair(i,j,k)` reducing to the appendix `S3` denominator family
- `G_extra(i,j,k)` accounting for the dominant additive extension now called
  `S3_addition`

## Recommended Derivation Path

1. Start from the perturbative commutator structure of `H12` with `H30`
   restricted to a single triad `(i,j,k)`.
2. Keep only the `J_x^4` / quartic rotational projection first.
3. Derive denominator families before full tensor reconstruction.
4. Only after the denominator families are clean, lift from `xxxx` to the full
   six-component tensor basis.
5. Then compare the derived families against:
   - `S3`
   - `S3_addition`
   - `HFO`, `H2CO`, `H2CS`

## Practical Rule For Code Until Formal Closure

Until the derivation is closed:

- keep `S3` as the appendix reference block;
- keep `S3_addition` explicit and separate in diagnostics;
- do not silently fold `S3_addition` into `S3` in the manuscript.

## Solver-Faithful Triad Form

The current solver can now be rewritten exactly on unordered triads
`{i,j,k}`. Define

```text
D_pair(i,j,k) = (w_i+w_j)(w_j+w_k)(w_i+w_k)
M_mu(a,b)     = the H12/H30 rotational mixing factor used by the solver
```

with `M_mu(a,b)` obtained from the component-level `mu1 * mu2` contraction for
the rotational tensor component `mu`.

For a single unordered triad `{i,j,k}`, it is cleaner to introduce the
symmetrized numerators

```text
N_mu^(0)({i,j,k}) = sum_{(a,b,c) in Perm(i,j,k)} M_mu(a,b)
N_mu^(1)({i,j,k}) = sum_{(a,b,c) in Perm(i,j,k)} M_mu(a,b) / w_a
```

The appendix block is then

```text
tau_mu^S3({i,j,k})
  = c_pair * Phi_ijk / D_pair(i,j,k) * N_mu^(0)({i,j,k})
```

while the weighted solver deformation is

```text
tau_mu^S3,wt({i,j,k})
  = c_add * Phi_ijk / D_pair(i,j,k) * N_mu^(1)({i,j,k})
```

and the current three-index solver block is

```text
tau_mu^three_index({i,j,k}) =
tau_mu^S3({i,j,k}) + tau_mu^S3,wt({i,j,k}).
```

This is not yet a BCH-first derivation. It is, however, a semianalytic
rewriting of the operational solver in which:

- the unordered triad is the primitive object;
- the denominator family of `S3` is isolated cleanly;
- the second piece appears as a weighted deformation of the same symmetrized
  triadic numerator.

This form is the correct bridge between the appendix notation and the current
solver implementation.

## Minimal `xxxx` Derivation For One Triad

For the component `mu = xxxx`, the rotational mixing used by the present
solver is especially simple:

```text
M_xxxx(a,b) = mu1_xx(a) * mu2_xx(bb)
```

where `mu2_xx(bb)` means the diagonal second-order inertia response attached to
mode `b`.

For a single unordered triad `{i,j,k}`, define

```text
A_i = mu1_xx(i)
B_i = mu2_xx(ii)
```

Then the appendix numerator is

```text
N_xxxx^(0)({i,j,k})
  = sum_perm M_xxxx(a,b)
  = A_i (B_j + B_k) + A_j (B_i + B_k) + A_k (B_i + B_j).
```

The weighted solver deformation uses the same triadic mixing, but weighted by
the frequency of the first mode in the ordered pair:

```text
N_xxxx^(1)({i,j,k})
  = sum_perm M_xxxx(a,b) / w_a
  = A_i (B_j + B_k) / w_i
    + A_j (B_i + B_k) / w_j
    + A_k (B_i + B_j) / w_k.
```

Therefore the current solver writes the single-triad `xxxx` contribution as

```text
tau_xxxx^three_index({i,j,k})
  = Phi_ijk / D_pair(i,j,k)
    * [ c_pair N_xxxx^(0)({i,j,k}) + c_add N_xxxx^(1)({i,j,k}) ].
```

This is the first explicit derivation-level statement that can be made without
claiming a full BCH-first closure. It shows that, for `xxxx`, the difference
between the appendix block and the weighted solver deformation is not a new
pair-sum denominator family, but a frequency-weighted deformation of the same
triadic numerator.

## Extension To The Full Six-Component Tensor

The same structural statement holds for all six quartic tensor components

```text
mu in {xxxx, yyyy, zzzz, xxyy, xxzz, yyzz}.
```

For each component `mu`, define the ordered-pair mixing

```text
M_mu(a,b)
```

from the component-level `mu1 * mu2` contraction used by the solver. Then for
every unordered triad `{i,j,k}`:

```text
N_mu^(0)({i,j,k}) = sum_{(a,b,c) in Perm(i,j,k)} M_mu(a,b)
N_mu^(1)({i,j,k}) = sum_{(a,b,c) in Perm(i,j,k)} M_mu(a,b) / w_a
```

and the solver writes

```text
tau_mu^three_index({i,j,k})
  = Phi_ijk / D_pair(i,j,k)
    * [ c_pair N_mu^(0)({i,j,k}) + c_add N_mu^(1)({i,j,k}) ].
```

So the formal pattern is uniform across the tensor:

- the appendix block and the weighted solver deformation share the same triadic
  denominator `D_pair`;
- the second term is a frequency-weighted deformation of the numerator;
- the component dependence sits only in the mixing map `M_mu(a,b)`.

An even cleaner rewriting is obtained by introducing the effective inverse
frequency weight

```text
rho_mu({i,j,k}) = N_mu^(1)({i,j,k}) / N_mu^(0)({i,j,k}),
```

whenever `N_mu^(0)` does not vanish. Then the whole three-index block can be
written as

```text
tau_mu^three_index({i,j,k})
  = Phi_ijk / D_pair(i,j,k)
    * [ c_pair + c_add rho_mu({i,j,k}) ] * N_mu^(0)({i,j,k}).
```

This is the most economical solver-faithful form currently available:

- the appendix object remains `N_mu^(0)` on the pair-sum denominator;
- the solver correction is absorbed into a single effective triadic weight
  `rho_mu`;
- the open formal problem becomes the perturbative meaning of `rho_mu`, not the
  existence of a separate extra denominator family.

There is also a more transparent way to read the same quantity. For a triad
`{i,j,k}`, define the three leading-mode partial sums

```text
C_mu^(i)({i,j,k}) = M_mu(i,j) + M_mu(i,k)
C_mu^(j)({i,j,k}) = M_mu(j,i) + M_mu(j,k)
C_mu^(k)({i,j,k}) = M_mu(k,i) + M_mu(k,j).
```

Then

```text
N_mu^(0) = C_mu^(i) + C_mu^(j) + C_mu^(k)
N_mu^(1) = C_mu^(i)/w_i + C_mu^(j)/w_j + C_mu^(k)/w_k
```

and therefore

```text
rho_mu({i,j,k})
  = [C_mu^(i)/w_i + C_mu^(j)/w_j + C_mu^(k)/w_k]
    / [C_mu^(i) + C_mu^(j) + C_mu^(k)].
```

So `rho_mu` is a solver-defined weighted mean of the inverse frequencies of the
three modes active in the triad, with weights given by the three leading-mode
partial contractions. This is the most compact interpretation available at the
current stage.

For the diagonal components `yyyy` and `zzzz`, the `xxxx` derivation carries
over verbatim by replacing `(xx,xx)` with `(yy,yy)` or `(zz,zz)`.

For the mixed components, the same denominator structure remains valid, but the
mixing map becomes a six-term bilinear contraction. For example,

```text
M_xxyy(a,b) =
  mu1_xx(a) mu2_yy(bb)
  + mu1_xy(a) mu2_xy(bb)
  + mu1_xy(a) mu2_yx(bb)
  + mu1_yx(a) mu2_xy(bb)
  + mu1_yx(a) mu2_yx(bb)
  + mu1_yy(a) mu2_xx(bb),
```

and similarly for `xxzz` and `yyzz`.

This closes the semianalytic structural derivation of the current solver in a
cleaner form: the open formal problem is no longer the denominator family of a
separate `S3_addition`, but the perturbative origin of the weighted numerators
`N_mu^(1)`.
