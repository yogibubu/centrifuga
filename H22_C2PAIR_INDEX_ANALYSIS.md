# H22 / c2_pair Index Analysis

This note records the first direct comparison between the quartic `H22` object
and the unsummed sextic second-level object.

## Quartic H22 object

The present quartic `H22` block uses

```text
mu2_{ab,kl} = d^2(I^-1)_{ab} / dQ_k dQ_l.
```

Its visible index properties are:

- symmetric in the spatial pair `(a,b)`,
- symmetric in the mode pair `(k,l)`.

So `mu2` is naturally a tensor in

```text
Sym^2(R^3) tensor Sym^2(V_modes).
```

## Sextic second-level object

The unsummed sextic object now exposed in code is

```text
c2_pair_{ij,abc}.
```

From its definition:

```text
c2_pair_{ij,abc}
  = K_{ij} [
      A_a zeta_{a,ij} c1_{j,bc}
    + A_b zeta_{b,ij} c1_{j,ca}
    + A_c zeta_{c,ij} c1_{j,ab}
    ]
```

the first direct structural facts are:

- it is cyclic in the spatial indices `(a,b,c)`,
- it is not symmetric in the mode pair `(i,j)`,
- it is not antisymmetric in `(i,j)` either.

So `c2_pair` does not live in the same raw tensor space as `mu2`.

## Interpretation

This matters because it rules out the naive guess

- "`c2_pair` is just `mu2` with one more spatial index".

That is too simple.

The difference in index behavior strongly suggests:

- `mu2` is a symmetric mode-pair object of second-level inertia type;
- `c2_pair` is a mixed mode-pair object obtained after coupling a first-level
  tensor to a Coriolis leg and weighting by a frequency kernel.

So the sextic second-level layer appears to be a *twisted* mode-pair object,
not a plain copy of the quartic `H22` tensor.

## Consequence

The next candidate common structure should probably not be searched directly at
the level of `mu2` versus `c2_pair`.

Instead one should look for a larger object from which:

- `mu2` arises by symmetric mode-pair projection,
- `c2_pair` arises by coupling one mode leg to `zeta` and one spatial pair to
  the first-level tensor `c1`.

That points toward a mixed mode-pair/spatial tensor space rather than a purely
pair-pair quartic space.

## New concrete bridge

On the local `h2o` harmonic model, the sextic first-level tensor `c1[i,a,b]`
is modewise proportional to the inverse-inertia first derivative

```text
mu1_{i,ab} = d(I^-1)_{ab} / dQ_i.
```

This is now checked explicitly in
[test_sextic_c2_mu1_bridge.py](/Users/vincenzobarone/centrifugal/test_sextic_c2_mu1_bridge.py),
which gives roundoff-level agreement for

```text
c1_i = s_i mu1_i
```

with mode-dependent scalars `s_i`.

As a consequence, the sextic unsummed object `c2_pair` can be rewritten exactly
on that model as a linear image of `mu1` coupled to Coriolis data:

```text
c2_pair = L(zeta, omega, A,B,C) [mu1].
```

So the current evidence is:

- `H22` introduces a genuinely second-level object through `mu2 = B - N`;
- the sextic `c2_pair` does not point directly to `mu2`, but to a Coriolis-
  coupled image of the first-level object `mu1`.

There is now a further sharpening:

```text
B_{kl} = mu1_k I mu1_l + mu1_l I mu1_k
```

so the bilinear part of `mu2` is generated algebraically by `mu1` and the
equilibrium inertia tensor `I`.

This means that the only genuinely new second-level ingredient entering `H22`
is the intrinsic remainder `N_{kl}`.

This suggests that the common geometric backend may have at least two distinct
layers:

- a first-level object `mu1` whose Coriolis-coupled image feeds the sextics,
- an intrinsic second-level object `N` whose completion with the algebraic
  closure `B(mu1)` gives `mu2` and hence feeds `H22`.

The remaining question is whether these two layers themselves belong to a
single larger tensorial hierarchy.

## Representation covariance of the intrinsic tensor

This point is now checked directly on the local `h2o.fchk` harmonic model in
[test_intrinsic_representation_covariance.py](/Users/vincenzobarone/centrifugal/test_intrinsic_representation_covariance.py).

Using the representation changes `I -> II` and `I -> III`, and allowing only
the standard per-mode sign freedom already present in `mu1`, the following
objects transform covariantly at roundoff:

- `mu1`,
- the bilinear closure `B(mu1)`,
- the intrinsic tensor `N`,
- the total `mu2 = B - N`.

So the intrinsic tensor `N` does have the expected natural transformation law
under representation changes.

This is strong evidence that `N`, rather than the full coordinate formula
`d2I/dQ2`, is the correct primitive second-level object to isolate.
