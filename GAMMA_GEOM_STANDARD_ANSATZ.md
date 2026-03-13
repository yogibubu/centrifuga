# Gamma Standard Geometry Ansatz

This note records the first explicit ansatz for the standard geometric branch
of `gamma`. It is intentionally written at the level of tensor kernels rather
than finished closed formulas. The aim is to make the next symbolic step
precise without pretending that the derivation is already complete.

The guiding principle is:

- use the already defined first-level `alpha` blocks
  `C_i^X` and `I_i^X`;
- write the most general pair-resolved closures compatible with the CeDiTT
  hierarchy;
- postpone the exact harmonic denominators and prefactors to the actual
  derivation.

## 1. Generic pair-resolved form

For each rotational component `X`, define

`gamma_ij^X(geom_std) = G_ij^X(CC) + G_ij^X(II) + G_ij^X(CI)`.

The ansatz is then:

`G_ij^X(CC) = sum_k  K_X^(CC)(i,j;k)  T_X^(CC)(i,j;k)`

`G_ij^X(II) =         K_X^(II)(i,j)    T_X^(II)(i,j)`

`G_ij^X(CI) = sum_k [ K_X^(CI,1)(i,j;k) T_X^(CI,1)(i,j;k)
                    + K_X^(CI,2)(i,j;k) T_X^(CI,2)(i,j;k) ]`

where:

- the `K` are scalar frequency/rotational kernels to be derived;
- the `T` are tensor monomials built from first-level blocks only.

This separates immediately:

- tensor content,
- from perturbative weighting.

## 2. Coriolis-Coriolis ansatz

The natural Coriolis building block already present in `alpha` is

`K_X^cor(i,j|i) = (A_X / nu_i) zeta_X,ij^2 C_ij^(i)`.

For `gamma`, the pair structure should therefore be organized through
intermediate-mode sums. The minimal ansatz is:

`T_X^(CC)(i,j;k) = zeta_X,ik zeta_X,jk`

with kernels that depend on:

- `nu_i`,
- `nu_j`,
- `nu_k`,
- and the rotational prefactor `A_X`.

Thus the first explicit symbolic form is

`G_ij^X(CC) = sum_k F_X^(CC)(nu_i,nu_j,nu_k) zeta_X,ik zeta_X,jk`.

For diagonal terms `i=j`, this becomes

`G_ii^X(CC) = sum_k F_X^(CC)(nu_i,nu_i,nu_k) zeta_X,ik^2`.

This is the cleanest direct generalization of the current Coriolis `alpha`
block.

## 3. Inertia-Inertia ansatz

The first-level geometric block already defined is

`I_i^X = - (3/4) (A_X / nu_i) Q_i^X(c1)`.

The corresponding pair closure should therefore be bilinear in the underlying
`c1` content before scalar contraction.

Define the modewise projected geometric kernel

`Q_ij^X(c1) = sum_Y G_i,XY^(X) G_j,XY^(X) / I_Y`

where `G_i,XY^(X)` is the first-level object introduced in
`ALPHA_SYMBOLIC_BLOCKS.md`.

The minimal ansatz is then

`G_ij^X(II) = F_X^(II)(nu_i,nu_j) Q_ij^X(c1)`.

For `i=j`, this reduces to the already familiar quadratic contraction behind
`alpha_i^X(in)`.

This makes `G_ij^X(II)` the direct pair-lift of the geometric `alpha` block.

## 4. Mixed Coriolis-Inertia ansatz

The mixed term should couple the two first-level sectors without introducing a
new primitive object. The most natural tensor forms are:

`T_X^(CI,1)(i,j;k) = zeta_X,ik M_X(j;k)`

`T_X^(CI,2)(i,j;k) = zeta_X,jk M_X(i;k)`

where `M_X(i;k)` denotes a projected geometric factor linear in the
first-level object for mode `i` and carrying the intermediate-mode label `k`
through the perturbative contraction.

At the purely symbolic level, the cleanest notation is

`G_ij^X(CI) = sum_k F_X^(CI)(nu_i,nu_j,nu_k)
              [ zeta_X,ik U_X(j;k) + zeta_X,jk U_X(i;k) ]`

with `U_X(i;k)` some first-level geometric factor to be identified in the
derivation.

This keeps the mixed channel broad enough to cover the plausible closures
without freezing the exact intermediate representation too early.

## 5. Compact ansatz in block language

Once the explicit kernels are suppressed, the whole standard geometric branch
may be summarized as

`gamma_ij^X(geom_std)
 = <C_i^X, C_j^X>_(X)
 + <I_i^X, I_j^X>_(X)
 + <C_i^X, I_j^X>_(X) + <I_i^X, C_j^X>_(X)`

where `<.,.>_(X)` denotes a channel-dependent bilinear pairing with the
appropriate harmonic denominators and rotational prefactors.

This is probably the most useful abstract form for the actual derivation.

## 6. Diagonal vs off-diagonal consequences

The ansatz predicts a sharp difference between:

- diagonal `gamma_ii^X`, where the pair closures collapse onto self-kernels;
- off-diagonal `gamma_ij^X`, where intermediate-mode sums and symmetry of the
  bilinear pairing become essential.

This matters because:

- diagonal pieces are likely to be derivationally simpler and should be
  attacked first;
- off-diagonal pieces are the real test of whether the first-level closure is
  sufficient.

## 7. What remains unknown

This ansatz does **not** yet determine:

- the exact frequency denominators;
- the exact combinatorial coefficients;
- whether the mixed block `CI` survives in the final assembled observable with
  the naive symmetry suggested above;
- whether additional standard geometric kernels, still first-level in content,
  appear beyond the three blocks written here.

These are derivational questions, not conceptual gaps in the organization.

## 8. Immediate symbolic program

The next symbolic steps should therefore be:

1. derive the diagonal branch:
   - `G_ii^X(CC)`
   - `G_ii^X(II)`
   - `G_ii^X(CI)`
2. test whether those three blocks already reproduce the expected quadratic
   occupation structure for the self-mode sector;
3. only then derive the off-diagonal branch `i != j`;
4. after that, decide whether the first explicit anharmonic extension can be
   added without leaving the first-level closure.

## 9. Working conclusion

The standard geometric `gamma` branch should be sought first in the form

`gamma_geom_std = CC + II + CI`

with:

- `CC` built from bilinear Coriolis pairings,
- `II` built from bilinear first-level geometric pairings,
- `CI` built from mixed Coriolis-geometric pairings.

This is the closest exact analogue, on the vibration-rotation side, of the
CeDiTT strategy that first isolates the largest first-level closure before
testing the first controlled breakdown.
