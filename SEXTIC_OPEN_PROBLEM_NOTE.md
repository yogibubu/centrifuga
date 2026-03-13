# Sextic Open Problem Note

This note records what is still missing on the sextic side.

## What is already done

The current code now supports the following statements.

- The full currently exposed sextic geometry sector is generated from the
  first-level tensor `mu1 = d(I^-1)/dQ`, together with Coriolis coupling,
  rotational constants, and frequency kernels.
- The explicit anharmonic sextic contribution currently visible in the code
  enters through the cubic force constants `phi3`.

So, at the level currently implemented, the sextics admit the structural split

```text
Phi^(6) = Phi_geometry(mu1, zeta, rot, omega) + Phi_cubic(phi3).
```

## What is still missing

What is *not* yet known is whether this is the final primitive formulation of
the sextic theory.

The open question is:

```text
Does the complete sextic theory close already on

    mu1 + Coriolis + phi3

or does a genuinely new intrinsic tensorial object appear at sextic level?
```

This is the sextic analogue of the quartic `H22` question that led to the
intrinsic second-level tensor `N`.

## Concrete unresolved alternatives

At present there are two live possibilities.

### Option 1

The sextic theory is already structurally closed by:

- first-level geometry `mu1`,
- Coriolis coupling,
- explicit anharmonic data `phi3`.

In this case there is no new sextic intrinsic tensor analogous to quartic `N`.

### Option 2

The current formulas expose only a reduced or partially contracted face of the
sextic theory, and a new intrinsic tensorial object appears once the sextic
structure is pushed beyond the currently implemented sector.

In this case the present `mu1`-generated geometry sector is real but not yet
the whole conceptual story.

## What would settle it

The next real progress on sextics would be one of the following:

1. prove that the full sextic geometry sector is generated from `mu1` and
   Coriolis data, with `phi3` carrying the explicit anharmonic part;
2. identify the first sextic contribution that cannot be reduced to that
   language;
3. isolate the corresponding new intrinsic tensor if such a contribution exists.

## Current safe statement

The strongest safe statement supported by the code is:

```text
For the full sextic geometry sector currently exposed in the code, no new
intrinsic tensor beyond mu1 is needed. What remains open is whether such a new
intrinsic tensor appears in the complete sextic theory beyond the presently
implemented formulas.
```
