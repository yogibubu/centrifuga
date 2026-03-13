# Gamma Standard Geometry Map

This note records the first formal map of the candidate standard geometric
sector of `gamma`, denoted here as `gamma_geom_std`. The purpose is not to
claim a finished formula, but to define the most likely closure pattern that
should be targeted in symbolic derivations.

The key idea is simple:

- the present `alpha` backend already separates the observable into
  Coriolis, inertia, and anharmonic pieces;
- `gamma_geom_std` should be built first from the quadratic combinations of
  the *geometric* pieces before any explicit higher-level breakdown is tested.

## 1. Observable pattern to reproduce

For each rotational constant component `X in {A,B,C}` or for the corresponding
symmetry-adapted readout, the current branch has

`alpha_i^X = alpha_i^X(cor) + alpha_i^X(in) + alpha_i^X(anh)`.

The first standard candidate for `gamma` should then have the pair-resolved
structure

`gamma_ij^X = gamma_ij^X(geom_std) + gamma_ij^X(anh_std) + ...`

with

`gamma_ij^X(geom_std)`

built from the first-level geometric closures alone.

## 2. First-level building blocks

At the level already present in the code, the geometric pieces are:

- `C_i^X` : Coriolis contribution carried by `zeta` and harmonic frequencies;
- `I_i^X` : inertia/geometric contribution carried by `c1` or equivalently
  `mu1`.

The anharmonic part is:

- `A_i^X` : explicit cubic-force contribution driven by `phi3`.

The standard geometric `gamma` map should initially ignore `A_i^X` and test
whether the quadratic occupation dependence can be assembled from `C_i^X` and
`I_i^X` only.

## 3. Formal decomposition of gamma_geom_std

The most natural decomposition is

`gamma_ij^X(geom_std) = G_ij^X(CC) + G_ij^X(II) + G_ij^X(CI)`

where:

- `G_ij^X(CC)` is the Coriolis-Coriolis closure;
- `G_ij^X(II)` is the inertia-inertia closure;
- `G_ij^X(CI)` is the mixed Coriolis-inertia closure.

This is exactly the `G1 + G2 + G3` sector introduced in
`GAMMA_CHANNEL_CANDIDATES.md`.

## 4. Formal tensor character of each term

### 4.1 Coriolis-Coriolis closure

`G_ij^X(CC)` should be built from objects of the form

- `zeta_X,ik zeta_X,jk`
- `zeta_X,ij zeta_X,kl`
- or equivalent sums over intermediate modes with harmonic denominators.

This term belongs entirely to the dynamical first-level sector and should not
require new inertia derivatives.

### 4.2 Inertia-Inertia closure

`G_ij^X(II)` should be built from objects of the form

- `c1_i^X : c1_j^X`
- or equivalently bilinear contractions in `mu1_i` and `mu1_j`
  after projection on the rotational component `X`.

This is the direct analogue of the first-level geometric closure already used
throughout CeDiTT.

### 4.3 Mixed Coriolis-Inertia closure

`G_ij^X(CI)` should be built from cross terms coupling

- `zeta`-driven mode mixing,
- with `c1`-driven geometric response.

This term is expected on general grounds because the final quadratic
occupation dependence is unlikely to separate exactly into a purely Coriolis
part and a purely inertia part once the effective observable is assembled.

## 5. Mode-pair structure

The first useful formal distinction is between:

- diagonal terms `gamma_ii^X`,
- off-diagonal terms `gamma_ij^X` with `i != j`.

The recommended organization is:

`gamma_geom_std^X = gamma_diag^X + gamma_offdiag^X`

with each of these then split into:

- `CC`,
- `II`,
- `CI`.

This is important because:

- diagonal pieces are the direct analogue of the single-mode `alpha_i^X`
  contributions;
- off-diagonal pieces are where new pair structure first becomes explicit;
- special rotor limits and degeneracies will act primarily on the pair
  structure.

## 6. Symmetry-adapted interpretation

The same special-limit machinery already built for `alpha` should act only
after the geometric channels have been assembled in generic form.

Thus the logic should be:

1. build `gamma_ij^X(CC)`, `gamma_ij^X(II)`, `gamma_ij^X(CI)` in the generic
   rotational basis;
2. sum them into `gamma_geom_std`;
3. only then apply:
   - axial projection for symmetric tops,
   - projector-based pair canonization for degenerate mode subspaces,
   - linear perpendicular/parallel reduction.

This separation is important to avoid mixing the perturbative derivation with
the rotor-limit readout.

## 7. Expected standard anharmonic extension

Once the pure geometric closure is tested, the next layer should be added as

`gamma_std = gamma_geom_std + gamma_anh_std`

where `gamma_anh_std` is first organized as:

- `CA` : Coriolis-cubic channels,
- `IA` : inertia-cubic channels,
- possibly `AA` : quadratic cubic channels if needed.

This makes the standard branch of `gamma` the direct analogue of:

- standard geometric closure,
- plus explicit standard anharmonic input.

## 8. First breakdown candidate

Only after the standard map above is tested should one ask whether the first
controlled breakdown appears. In the present language that means testing
whether terms of the form

- `mu2`,
- `B(mu1)`,
- `N`,
- or mixed `mu1/N`

enter necessarily.

If so, the first breakdown should be written as

`delta_gamma_break = delta_gamma[B(mu1)] + delta_gamma[N]`

with only the `N` part counted as genuinely new tensor content.

## 9. Immediate symbolic target

The first symbolic target is therefore not the full `gamma`, but the map

`{C_i^X, I_i^X}  -->  {G_ij^X(CC), G_ij^X(II), G_ij^X(CI)}`

followed by the assembly

`gamma_geom_std = CC + II + CI`.

This is the cleanest next step because it asks only one question:

- does the quadratic occupation dependence already close on the same
  first-level data as the present `alpha` branch?

Everything else should come only after this test is structurally clear.
