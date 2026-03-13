# Sextic Harmonic Structure Note

## Starting Point

Let `P_6^even` denote the space of homogeneous degree-six rotational polynomials
in `J_a, J_b, J_c` containing only even powers.  This space has dimension 10.

Its standard harmonic decomposition is

`P_6^even = H_6^even ⊕ r^2 H_4^even ⊕ r^4 H_2^even ⊕ r^6 H_0`

with dimensions

- `dim H_6^even = 4`
- `dim H_4^even = 3`
- `dim H_2^even = 2`
- `dim H_0 = 1`

and therefore

`10 = 4 + 3 + 2 + 1`.

Here `r^2 = J_a^2 + J_b^2 + J_c^2`.

For any `p ∈ P_6^even`, the decomposition

`p = h_6 + r^2 h_4 + r^4 h_2 + r^6 h_0`

with

- `h_6 ∈ H_6^even`,
- `h_4 ∈ H_4^even`,
- `h_2 ∈ H_2^even`,
- `h_0 ∈ H_0`,

is unique. If `Δ` denotes the Laplacian in `(J_a,J_b,J_c)`, the corresponding
harmonic components are extracted canonically by

`h_0 = Δ^3 p / 5040`

`h_2 = Δ^2 p / 504 - r^2 Δ^3 p / 3024`

`h_4 = Δ p / 22 - r^2 Δ^2 p / 308 + r^4 Δ^3 p / 6160`

`h_6 = p - r^2 Δ p / 22 + r^4 Δ^2 p / 792 - r^6 Δ^3 p / 33264`

Thus the pair `(h_0,h_6)` is basis-independently determined by any sextic
even polynomial.

Moreover, `H_6^even` is not just an abstract 4D space: it is precisely the
`D_2`-even sector of the irreducible `\ell=6` representation of `SO(3)`.
Equivalently, it may be spanned by four real tesseral harmonic polynomials
corresponding to `m=0,2,4,6`. One convenient choice is

`Y_6^{(6c)} = (Ja-Jb)(Ja+Jb)(Ja^2-4 Ja Jb+Jb^2)(Ja^2+4 Ja Jb+Jb^2)`

`Y_6^{(4c)} = -(Ja^2+Jb^2-10 Jc^2)(Ja^2-2 Ja Jb-Jb^2)(Ja^2+2 Ja Jb-Jb^2)/11`

`Y_6^{(2c)} = (Ja-Jb)(Ja+Jb)(Ja^4+2 Ja^2 Jb^2-16 Ja^2 Jc^2+Jb^4-16 Jb^2 Jc^2+16 Jc^4)/33`

`Y_6^{(0)} = -5 Ja^6 - 15 Ja^4 Jb^2 + 90 Ja^4 Jc^2 - 15 Ja^2 Jb^4
             + 180 Ja^2 Jb^2 Jc^2 - 120 Ja^2 Jc^4 - 5 Jb^6
             + 90 Jb^4 Jc^2 - 120 Jb^2 Jc^4 + 16 Jc^6`

all of which satisfy `ΔY_6=0`.  The physical sextic content may therefore be
regarded as

- one isotropic scalar amplitude in `H_0`,
- one irreducible rank-6 even component in the `D_2`-even sector of `\ell=6`.

In particular, if

`h_6 = c_0 Y_6^(0) + c_2 Y_6^(2c) + c_4 Y_6^(4c) + c_6 Y_6^(6c),`

then the canonical CeDiTT coordinates `sigma` are linearly equivalent to the
harmonic amplitudes

`(h_0,c_0,c_2,c_4,c_6).`

The map between the operational `sigma` coordinates and these harmonic
coordinates is invertible.  Therefore the sextic physical sector is not only
canonically isomorphic to `H_0 ⊕ H_6^even`; it may also be described directly
by one scalar amplitude and four real tesseral amplitudes of the irreducible
`\ell=6` component.

## Canonical Sextic Subspace

The CeDiTT sextic algorithm uses a canonical 5D subspace `S_6 ⊂ P_6^even`,
generated symbolically in `derive_sextic_transform.py`.

The key observation from the present analysis is:

1. `dim S_6 = 5`.
2. The projection of `S_6` onto `H_6^even ⊕ H_0` has full rank `5`.
3. Since `dim(H_6^even ⊕ H_0)=5`, this projection is an isomorphism.
4. Therefore `S_6` is the graph of a linear map

`K : H_6^even ⊕ H_0  ->  r^2 H_4^even ⊕ r^4 H_2^even`.

Equivalently,

`S_6 = { x + Kx : x in H_6^even ⊕ H_0 }`.

This is a much stronger statement than saying only that the sextic algorithm
uses a "practical 5D coordinate system". In particular, the physical sextic
content may be parameterized equivalently by:

- one scalar coordinate in `H_0`,
- four amplitudes in the even sextic harmonic sector `H_6^even`.

Thus the canonical sextic subspace admits a natural `1+4` interpretation.

## Interpretation

The canonical sextic subspace is therefore not arbitrary:

- it is not just a numerical reduction,
- it is not merely a convenient coordinate chart,
- it is a five-dimensional graph subspace inside the natural harmonic
  decomposition of the even sextic operator space.

What makes the sextic case different from the quartic one is that the physical
subspace does **not** collapse to a single symmetric second-rank tensor.
Instead, the canonical sextic sector couples:

- a 4D even harmonic degree-6 block `H_6^even`,
- a 1D isotropic scalar block `H_0`,

to the lower harmonic sectors

- `r^2 H_4^even` (3D),
- `r^4 H_2^even` (2D),

through a fixed linear graph relation.

The key simplification is that the lower-order harmonic blocks do not carry
independent physical coordinates: they are slaved to the `H_6^even ⊕ H_0`
sector by the graph map `K`.

Equivalently, the canonical physical sextic content is exactly the pair
`(h_0,h_6)`, whereas `(h_2,h_4)` are induced from it by the fixed graph
relation defining `S_6`.

## Consequence for the Manuscript

If this formulation is adopted, the sextic section can be strengthened as follows:

- the present `5+2` decomposition remains valid operationally,
- but its theoretical interpretation becomes:

  "the physical sextic content is a canonical five-dimensional graph subspace
   inside the harmonic decomposition of the even sextic rotational polynomial
   space"

rather than only

  "a practical coordinate system".

This remains weaker than a complete invariant classification analogous to the
quartic `Tau'` tensor, but it is substantially stronger than a purely practical
statement.  The present sextic model is canonically equivalent to a
`scalar + irreducible rank-6` harmonic parameterization, i.e. a `1+4`
description in the `D_2`-even sector.

## Status

This observation is currently based on symbolic linear-algebra analysis of the
canonical sigma basis.  It is mathematically consistent and already strong
enough to justify a `1+4` interpretation of the sextic physical sector.
What is still missing, if desired for final publication, is a polished
basis-independent presentation of the four `H_6^even` amplitudes in a
physically motivated irreducible basis.
