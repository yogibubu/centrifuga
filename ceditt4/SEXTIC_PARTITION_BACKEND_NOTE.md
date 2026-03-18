# Sextic Partition Note

This note records the backend-faithful partition used in the current CeDiTT4
code base for the standard sextic tensor. It is intended as a working bridge
between the paper notation and the implementation in
`compare_gaussian_sextic.py`.

## Scope

The validated backend does not build the sextic tensor from a single compact
rank-6 formula. Instead, it constructs the Cartesian sextic components
`Phi_aaa`, `Phi_aab`, `Phi_abc`, ... directly, splitting them into:

- geometry sector;
- semi-diagonal cubic sector;
- genuine three-index cubic sector.

The cubic split is performed on the full reduced cubic tensor `phi3[i,j,k]`
by separating:

- semi-diagonal: at least two equal indices;
- three-index: all three indices distinct.

See:

- `split_cubic_force_constants(...)` in `compare_gaussian_sextic.py`
- `sextic_cubic_hierarchy_hz(...)` in `compare_gaussian_sextic.py`

## Notation

The notation must distinguish clearly between:

- vibrational-mode indices: `i,j,k,...`
- rotational/cartesian axes in the principal-axis frame: `a,b,c`

In particular:

- `phi3_{ijk}` always refers to reduced cubic force constants in the normal-mode
  basis;
- `Phi_{aaa}`, `Phi_{aab}`, `Phi_{abc}`, ... denote Cartesian sextic tensor
  components after projection onto the principal-axis frame.

These two index families must never be mixed.

## Index bookkeeping

The backend sums over ordered triples `(i,j,k)`.

Therefore:

- if the paper uses unrestricted sums over `i,j,k`, the code and the notation
  are directly aligned;
- if the paper uses restricted sums such as `i < j < k`, the kernel must be
  explicitly symmetrized over the six permutations;
- if the paper uses two-index sums, it must distinguish the `iii` and `iij`
  patterns explicitly, or use a symmetrized unordered-pair kernel.

Mixing an unrestricted two-index sum with a restricted three-index sum is not
safe unless the corresponding permutation factors are written explicitly.

## Backend-faithful Cartesian formulas

Let `Phi_{aaa}` denote a diagonal sextic Cartesian component in the chosen
principal-axis representation, with the same axis repeated three times. The
backend builds

```text
Phi_{aaa} = X1 - X2 + X3 + X4
```

with

```text
X1 = (3/16) sum_x tau_{aaax}^2 / B_x
X2 = (1/2) sum_i nu_i c2_{i,aaa}^2
X3 = (1/6) sum_{i,j,k} phi3_{ijk}
     c1_{i,aa} c1_{j,aa} c1_{k,aa}
X4 = (1/4) sum_{x != a} tau_{xaaa}^2 / (B_a - B_x)
```

For the mixed class `Phi_{aab}` with `a != b`, the backend uses

```text
Phi_{aab} = Y1 - Y2 + Y3 + Y4 + Y5 + Y6
```

where

```text
Y1 = (3/32) sum_x [ T_abx^2 + 2 tau_{aaax} U_abx ] / B_x
T_abx = tau_{aabx} + 2 tau_{abax}
U_abx = tau_{bbax} + 2 tau_{abbx}

Y2 = (3/4) sum_i nu_i [
       3 c2_{i,aab}^2
       + 2 c2_{i,aaa} c2_{i,abb}
     ]

Y3 = (1/4) sum_{i,j,k} phi3_{ijk} c1_{k,aa}
     [ c1_{i,aa} c1_{j,bb}
       + 4 c1_{i,ab} c1_{j,ab} ]

Y4 = tau_{aaab}
     [ 4 tau_{bbba} - 3 tau_{aaab} ]
     / [ 8 (B_a - B_b) ]

Y5 = sum_{x != a,b}
     tau_{aaax}
     [ (B_a - B_x)
       (tau_{bbax} + 2 tau_{babx})
       + (B_a - B_b)
       (tau_{xxxa} - tau_{aaax}) ]
     / [ 4 (B_a - B_x)^2 ]

Y6 = sum_{x != a,b}
     (tau_{aabx} + 2 tau_{abax})
     [ (B_b - B_x)
       (tau_{aabx} + 2 tau_{abax})
       + 2 (B_a - B_b)
       (tau_{bbbx} - tau_{xxxb}) ]
     / [ 8 (B_b - B_x)^2 ]
```

For the fully mixed class `Phi_{abc}` with the three principal axes all
different, the backend uses

```text
Phi_{abc} = Z1 - Z2 + Z3 + Z4
```

with

```text
Z1 = (3/16) sum_x [ 2 V_x^2 + W_x^{(ab)} W_x^{(ac)}
                    + W_x^{(bc)} W_x^{(ba)} + W_x^{(ca)} W_x^{(cb)} ] / B_x

V_x       = tau_{x a b c} + tau_{x b c a} + tau_{x c a b}
W_x^{(ab)} = tau_{x a b b} + 2 tau_{x b a b}
W_x^{(ac)} = tau_{x a c c} + 2 tau_{x c a c}
W_x^{(bc)} = tau_{x b c c} + 2 tau_{x c b c}
W_x^{(ba)} = tau_{x b a a} + 2 tau_{x a b a}
W_x^{(ca)} = tau_{x c a a} + 2 tau_{x a c a}
W_x^{(cb)} = tau_{x c b b} + 2 tau_{x b c b}

Z2 = (9/2) sum_i nu_i [
       2 c2_{i,abc}^2
       + c2_{i,bba} c2_{i,cca}
       + c2_{i,ccb} c2_{i,aab}
       + c2_{i,aac} c2_{i,bbc}
     ]

Z3 = (1/2) sum_{i,j,k} phi3_{ijk} [
       c1_{i,aa} c1_{j,bb} c1_{k,cc}
       + 2 c1_{i,aa} c1_{j,bc} c1_{k,bc}
       + 2 c1_{i,bb} c1_{j,ca} c1_{k,ca}
       + 2 c1_{i,cc} c1_{j,ab} c1_{k,ab}
       + 8 c1_{i,bc} c1_{j,ca} c1_{k,ab}
     ]

Z4 = (3/2) sum_{x,y,z \; pairwise\ distinct}
     [ tau_{y y y z} - tau_{z z z y} ]
     [ (B_z - B_x) tau_{y y y z} + (B_x - B_y) tau_{z z z y} ]
     / [ 4 (B_y - B_z)^2 ]
```

These are the formulas that are actually partitioned in the current backend.

## Consequence for paper notation

If the paper wants explicit formulas without `c1` and `c2`, the correct route
is not to invent new compact rank-6 expressions, but to eliminate `c1` and
`c2` through their definitions in terms of:

- first derivatives of the inverse inertia tensor;
- rotational constants;
- Coriolis couplings;
- vibrational frequencies.

Only after that elimination can one safely rewrite the cubic sectors using
restricted sums such as `i < j < k` or `i != j`, provided the permutation
symmetrization is made explicit.
