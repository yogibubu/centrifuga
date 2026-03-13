# Gamma Standard Anharmonic Ansatz

This note records the first explicit ansatz for the standard anharmonic branch
of `gamma`, denoted here as `gamma_anh_std`. It is meant to complement
`GAMMA_GEOM_STANDARD_ANSATZ.md`.

The logic is the same:

- isolate the largest standard branch first;
- keep the tensor content separate from the perturbative weighting;
- postpone exact denominators and coefficients until the symbolic derivation.

## 1. Starting point

The current `alpha` backend already contains an explicit anharmonic block

- `A_i^X = alpha_i^X(anh)`

built from:

- first-level geometric quantities through `c1`,
- semi-diagonal cubic force constants through `phi3`.

The natural next question is how this block enters the pair-resolved
observable `gamma_ij^X`.

The minimal standard anharmonic candidate is

`gamma_ij^X(anh_std) = G_ij^X(CA) + G_ij^X(IA) + G_ij^X(AA)`

where:

- `CA` means Coriolis-cubic channels;
- `IA` means inertia-cubic channels;
- `AA` means quadratic anharmonic channels.

## 2. Coriolis-cubic ansatz

The first plausible mixed anharmonic contribution couples:

- the Coriolis first-level block `C_i^X`,
- with the cubic-force driven block `A_j^X`.

The corresponding ansatz is

`G_ij^X(CA) = <C_i^X, A_j^X>_(X) + <A_i^X, C_j^X>_(X)`

or more explicitly,

`G_ij^X(CA) = sum_k F_X^(CA)(nu_i,nu_j,nu_k)
              [ zeta_X,ik V_X(j;k) + zeta_X,jk V_X(i;k) ]`

where `V_X(i;k)` denotes a cubic-force-driven first-level factor still built
from `c1` and `phi3`.

This is the natural anharmonic extension of the mixed `CI` geometric channel.

## 3. Inertia-cubic ansatz

The second plausible standard anharmonic contribution couples:

- the pure geometric block `I_i^X`,
- with the cubic-force driven block `A_j^X`.

The corresponding ansatz is

`G_ij^X(IA) = <I_i^X, A_j^X>_(X) + <A_i^X, I_j^X>_(X)`.

At the more explicit tensor level, the natural structure is

`G_ij^X(IA) = F_X^(IA)(nu_i,nu_j) W_ij^X(c1,phi3)`

where `W_ij^X(c1,phi3)` is bilinear in:

- one first-level geometric factor,
- one cubic-force-driven first-level factor.

This should be treated as the most likely dominant standard anharmonic
extension if `gamma` behaves like a higher observable still anchored in the
same first-level language.

## 4. Quadratic anharmonic ansatz

The third candidate is the genuinely quadratic cubic-force contribution:

`G_ij^X(AA) = <A_i^X, A_j^X>_(X)`.

The minimal symbolic form is

`G_ij^X(AA) = F_X^(AA)(nu_i,nu_j) R_ij^X(phi3; c1)`

where `R_ij^X` collects terms quadratic in the cubic-force-driven standard
block.

This channel is especially important because it may be the first place where
the standard branch becomes visibly more expensive while still not requiring a
new geometric tensor level.

## 5. Role of quartic-force information

There is an unresolved structural question:

- do quartic-force constants `phi4` enter already in the standard anharmonic
  observable branch of `gamma`?

At the current level, the safest ansatz is:

- `gamma_anh_std` should first be tested with `CA + IA + AA` built only from
  `c1`, `zeta`, and `phi3`;
- if this fails structurally, a second standard extension
  `gamma_anh_std(phi4)` should be added separately.

This avoids building `phi4` into the standard branch prematurely.

## 6. Compact block form

Suppressing explicit kernels, the first standard anharmonic branch can be
written abstractly as

`gamma_ij^X(anh_std)
 = <C_i^X, A_j^X>_(X) + <A_i^X, C_j^X>_(X)
 + <I_i^X, A_j^X>_(X) + <A_i^X, I_j^X>_(X)
 + <A_i^X, A_j^X>_(X)`.

This is the most useful summary for the actual derivation.

## 7. Diagonal vs off-diagonal sectors

As for the geometric branch, the standard anharmonic branch should be split
into:

- diagonal terms `gamma_ii^X(anh_std)`,
- off-diagonal terms `gamma_ij^X(anh_std)` with `i != j`.

The recommended strategy is:

1. derive `IA` first on the diagonal sector;
2. then test whether `CA` is required already there;
3. only then derive off-diagonal `CA/IA`;
4. postpone `AA` until it is clear whether the first standard branch can be
   closed without it.

This keeps the complexity under control.

## 8. Relation to the first breakdown

The channels `CA`, `IA`, and `AA` should still be counted as belonging to the
standard observable branch as long as they can be written using:

- `mu1/c1`,
- `zeta`,
- `phi3`,
- and, if necessary but still standard, `phi4`.

They should **not** be confused with the first controlled breakdown.

The first breakdown should be tested only after the best standard branch

`gamma_std = gamma_geom_std + gamma_anh_std`

has been assembled.

## 9. Immediate symbolic target

The next practical symbolic target after the geometric ansatz is:

`{C_i^X, I_i^X, A_i^X}
 -> {G_ij^X(CA), G_ij^X(IA), G_ij^X(AA)}`

with diagonal sectors treated first.

This is the cleanest next step because it preserves the CeDiTT logic:

- first maximal standard closure,
- then explicit standard anharmonic extension,
- only then first controlled breakdown.
