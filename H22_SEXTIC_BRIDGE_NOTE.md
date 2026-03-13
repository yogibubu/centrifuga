# H22 / Sextic Bridge Note

This note records the first concrete bridge between the current `H22`
implementation and the sextic machinery.

## 1. Present H22 object

The current quartic `H22` block uses

- `mu2_{ab,kl} = d^2(I^-1)_{ab} / dQ_k dQ_l`

and contracts it in a weighted Gram form over the mode-pair labels `(k,l)`.

A backend split is now available in
[rovib_distortion.py](/Users/vincenzobarone/centrifugal/rovib_distortion.py)
through `inverse_inertia_second_derivative_components(...)`, which writes

```text
mu2_{kl} = B_{kl} - N_{kl}
```

with

- `B_{kl}` the bilinear piece generated from first derivatives `dI_k`,
- `N_{kl} = I^-1 (d2I_{kl}) I^-1` the genuinely new second-level piece.

So the quartic `H22` question can now be stated more sharply:

- is the canonical second-level object the full `mu2_{kl}`,
- or only the intrinsic part `N_{kl}`,
- or some equivalent mode-pair tensor built from them?

## 2. Sextic side

In the current sextic implementation, the explicit array `mu2` does not appear.

Instead the code exposes:

- `c1`, built from the first-level quartic object (`didq`),
- `c2`, built from `c1`, Coriolis couplings `zeta`, frequencies, and
  rotational constants.

Operationally, the sextic terms are organized through:

- quartic tensors `tau`, `tau'`,
- the first-level tensor `c1`,
- the next-level tensor `c2`,
- cubic force constants.

So on the sextic side the “second layer” is currently visible not as `mu2`,
but as a derived object of `c2` type.

More explicitly, from
[compare_gaussian_sextic.py](/Users/vincenzobarone/centrifugal/compare_gaussian_sextic.py#L159),

```text
c2_{i,abc}
  = sum_j K_{ij} [
      A_a zeta_{a,ij} c1_{j,bc}
    + A_b zeta_{b,ij} c1_{j,ca}
    + A_c zeta_{c,ij} c1_{j,ab}
    ]
```

for an explicit frequency kernel `K_{ij}`.

So `c2` is already a contracted image of a richer mode-pair object: it carries
one external mode index `i`, but it is built by summing over an internal mode
`j` against Coriolis data and first-level tensors.

This richer object is now exposed explicitly in code as

- `c2_pair[i,j,a,b,c]`

through
[compare_gaussian_sextic.py](/Users/vincenzobarone/centrifugal/compare_gaussian_sextic.py),
with exact reconstruction

```text
c2_{i,abc} = sum_j c2_pair_{ij,abc}.
```

## 3. Working interpretation

At this stage the cleanest interpretation is:

- `c1` is the exposed first-level tensorial object;
- `mu2` is the exposed second-level object on the quartic `H22` side;
- `c2` is evidence that the sextic code is already using a second-level layer,
  but in a mixed derived form rather than in a canonical intrinsic form.

This suggests that the next conceptual task is not merely to study `H22`
internally, but to ask whether:

- `mu2`,
- its intrinsic part `N`,
- or a closely related mode-pair tensor

is the object from which the sextic second-level structure should also be
built.

## 4. Immediate technical question

The next useful step is to compare the index structure of:

- quartic `H22`: mode-pair object `mu2_{ab,kl}`,
- sextic `c2_pair`: mixed mode-pair object `c2_pair_{ij,abc}`,
- sextic `c2`: single-mode object obtained only after contraction over the
  internal mode.

This strongly suggests that the sextic `c2` is not the most primitive
second-level object itself, but a contracted image of a richer mode-pair
tensorial object.

That is the concrete bridge to investigate next.
