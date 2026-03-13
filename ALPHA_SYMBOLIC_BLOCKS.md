# Alpha Symbolic Blocks

This note translates the presently implemented `alpha` backend into a uniform
symbolic notation. Its role is to provide a direct dictionary between:

- backend quantities in `vibrot_alpha.py`,
- first-level CeDiTT tensor objects,
- and the formal blocks that should later be lifted to the `gamma` branch.

It deliberately focuses on the two geometric blocks already needed for
`gamma_geom_std`:

- the Coriolis block,
- the inertia/geometric block.

The explicit anharmonic block is included only as a reference at the end.

## 1. Common notation

Mode indices:

- `i,j,k,...`

Rotational axes:

- `X,Y,Z` for the working Cartesian rotational basis used internally,
- `a,b,c` for the final spectroscopic principal-axis readout.

Harmonic frequencies:

- `nu_i`

Rotational constants in wavenumber units:

- `B_X`

First-level geometric tensor:

- `c1_i,XY`

Coriolis couplings:

- `zeta_X,ij`

The current code also uses the axis-prefactor

`A_X = 2 B_X^2`

which is exactly the quantity called `aa[ix]` in the backend.

## 2. Coriolis block in the backend

In `vibrot_alpha.py`, the Coriolis piece is built as

`alpha_cor[i,X] += (A_X / nu_i) zeta_X,ij^2 C_ij^(i)`

with a corresponding contribution to mode `j`,

`alpha_cor[j,X] += (A_X / nu_j) zeta_X,ij^2 C_ij^(j)`.

Here the code-dependent coefficients `C_ij^(i)` and `C_ij^(j)` are the
frequency-dependent rational factors written in the backend as:

- the nonresonant form
  `(3 nu_i^2 + nu_j^2)/(nu_i^2 - nu_j^2)`
  and its partner with `i <-> j`,
- or the near-degenerate regularized form used when `|nu_i - nu_j|` is below
  the numerical threshold.

### Symbolic definition

It is convenient to define the pair kernel

`K_X^cor(i,j | i) = (A_X / nu_i) zeta_X,ij^2 C_ij^(i)`

and similarly

`K_X^cor(i,j | j) = (A_X / nu_j) zeta_X,ij^2 C_ij^(j)`.

Then the Coriolis block becomes

`alpha_i^X(cor) = - sum_{j != i} K_X^cor(i,j | i)`.

This is the cleanest symbolic summary of the currently implemented code.

### Structural classification

This block depends only on:

- `zeta`,
- harmonic frequencies,
- rotational prefactors.

Therefore it is a pure first-level dynamical closure. In the `gamma` work it
should be promoted to the symbolic building block

- `C_i^X`.

## 3. Inertia/geometric block in the backend

In the backend, the inertia piece is built through the intermediate quantity

`D_i,XY^(X) = 2 I_X I_Y x_i c1_i,XY`

where:

- `I_X` denotes the principal moments entering the current representation,
- `x_i` is the frequency scale factor called `sqrt((FACTG * nu_i)^3)` in the
  code.

The actual backend contraction is

`alpha_inertia[i,X] = (3/4) (A_X / nu_i) sum_Y (D_i,XY^(X))^2 / I_Y`

followed by the global minus sign applied at the end.

### Symbolic definition

Define the geometric kernel

`G_i,XY^(X) = 2 I_X I_Y x_i c1_i,XY`

Then the inertia block is

`alpha_i^X(in) = - (3/4) (A_X / nu_i) sum_Y [G_i,XY^(X)]^2 / I_Y`.

Equivalently, one may define the quadratic contraction

`Q_i^X(c1) = sum_Y [G_i,XY^(X)]^2 / I_Y`

so that

`alpha_i^X(in) = - (3/4) (A_X / nu_i) Q_i^X(c1)`.

### Structural classification

This block depends only on:

- `c1`, hence on `mu1`,
- harmonic frequencies,
- equilibrium moments / rotational prefactors.

Therefore it is the pure first-level geometric block. In the `gamma` work it
should be promoted to the symbolic building block

- `I_i^X`.

## 4. Practical dictionary: backend to theory

The direct translation is:

- `alpha_coriolis_cm[i,X]`
  -> `alpha_i^X(cor)`
  -> first-level block `C_i^X`

- `alpha_inertia_cm[i,X]`
  -> `alpha_i^X(in)`
  -> first-level block `I_i^X`

- `alpha_anharmonic_cm[i,X]`
  -> `alpha_i^X(anh)`
  -> explicit standard anharmonic block `A_i^X`

Thus the current backend already provides the decomposition needed by the
formal map

`{C_i^X, I_i^X} -> {G_ij^X(CC), G_ij^X(II), G_ij^X(CI)}`

introduced in `GAMMA_GEOM_STANDARD_MAP.md`.

## 5. Explicit anharmonic block for reference

The present code writes the first explicit anharmonic contribution as

`alpha_anh[i,X] = - P_X (A_X / nu_i)
                  sum_j G_j,XX^(X) F_ij(phi3)`

where:

- `P_X` collects the numerical prefactor called `PICH12`,
- `G_j,XX^(X)` is the diagonal geometric factor built from `c1`,
- `F_ij(phi3)` is the cubic-force term currently represented in code as
  `phi3[i,i,j] nu_i sqrt(nu_j) / nu_j^2`.

This block is not needed yet for `gamma_geom_std`, but it fixes the natural
symbolic ancestor of:

- `CA`,
- `IA`,
- and related standard anharmonic channels in the future `gamma_std`.

## 6. Immediate consequence for gamma

The crucial point is that the present `alpha` backend is already in the form
needed for the first symbolic `gamma` step.

The correct next target is therefore:

1. treat `C_i^X` and `I_i^X` as abstract first-level building blocks;
2. derive the pair-resolved closures
   - `G_ij^X(CC)`,
   - `G_ij^X(II)`,
   - `G_ij^X(CI)`;
3. only later reinsert the explicit backend realizations in terms of
   `zeta`, `c1`, frequencies, and moments.

This keeps the `gamma` derivation aligned with the CeDiTT style:

- first identify the tensorial block structure,
- then specialize to explicit formulas.
