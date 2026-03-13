# H22 Primitive Formulation Note

This note records the current preferred formulation of the quartic `H22`
channel.

## Old viewpoint

The previous coordinate-level presentation was:

```text
H22 is built from

    mu2_{ab,kl} = d^2(I^-1)_{ab} / dQ_k dQ_l

through a weighted Gram contraction over the mode-pair labels `(k,l)`.
```

This is still correct, but it is no longer the cleanest conceptual starting
point.

## New viewpoint

The preferred primitive data are now:

- the first-level tensor

  ```text
  mu1_k = d(I^-1)/dQ_k,
  ```

- the equilibrium inertia tensor

  ```text
  I,
  ```

- and the intrinsic second-level tensor

  ```text
  N_{kl} = I^-1 (d2I_{kl}) I^-1.
  ```

From these, the bilinear second-level closure is not independent. It is
generated algebraically by

```text
B_{kl} = mu1_k I mu1_l + mu1_l I mu1_k.
```

The old object `mu2` is therefore derived:

```text
mu2_{kl} = B_{kl} - N_{kl}.
```

## H22 in primitive form

With this notation, the present quartic `H22` contribution should be viewed as

```text
H22 = H22[B(mu1) - N, B(mu1) - N]
```

where `H22[left,right]` denotes the weighted mode-pair quadratic form used in
the current code.

Expanding:

```text
H22 = H22[B,B] - H22[B,N] - H22[N,B] + H22[N,N].
```

So:

- `H22[B,B]` is induced entirely by the first-level object `mu1`;
- `H22[B,N] + H22[N,B]` is the interference between first- and second-level
  geometry;
- `H22[N,N]` is the purely intrinsic second-level contribution.

## Why this is the right formulation

The code now supports and checks the following facts:

1. `B(mu1)` is reconstructed exactly from `mu1` and `I`.
2. `mu2 = B(mu1) - N`.
3. `H22(mu2)` agrees with `H22(mu1,N,I)` at roundoff.
4. `N` transforms covariantly under representation changes `I/II/III`
   up to the standard mode-sign freedom.

This means that `mu2` should be treated as a derived convenience object, not as
the primitive second-level tensor.

## Consequence for the paper

If the paper is organized around tensorial lifting, gauge fixing, and
representation/reduction transport, then the natural quartic ingredients are:

- first-level transport object: `mu1`,
- intrinsic second-level transport object: `N`.

The `H22` channel is then the first place where the intrinsic second-level
object enters, while the sextic side already uses first-level geometry through
its Coriolis-coupled constructions.

This is the cleanest current bridge between the code structure and the intended
theoretical narrative.
