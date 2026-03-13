# VPT4 / CeDiTT Tensorial Bridge Note

This note records a strategic point for the next phase of work.

## Scope

CeDiTT and the full VPT4 workflow should remain conceptually separate.

- CeDiTT concerns the standard `H12H12` side, representation transforms, and
  the quartic/sextic transform machinery built around that setting.
- VPT4 concerns the full channel-resolved quartic problem:
  `H12H12`, `H22`, `H12H30`, `H30H30`.

They should not be conflated at the level of final formulas or claims.

## Bridge

There is nevertheless a plausible common layer beneath both projects:
the tensorial geometry of the harmonic normal modes.

The key working hypothesis is that the explicit derivative objects

- `dI/dQ`,
- `d2I/dQ2`,
- equivalently `d(I^-1)/dQ`,
- `d2(I^-1)/dQ2`,

should not be regarded as the final conceptual language, but as coordinate
descriptions of more natural tensorial objects built from the mode geometry.

## Why this matters now

For VPT4, the next natural target is `H22`. The immediate question is:

> what tensorial object replaces the explicit `d2I/dQ2` description?

This is not only a quartic-channel question. The same second-derivative level
also enters the sextic problem, even if in the current benchmark-style sextic
code that dependence is not exposed in a fully explicit tensorial form.

Therefore the target should not be:

- "rewrite `H22` without `d2I/dQ2`".

The target should be:

- "identify a canonical tensorial object whose present coordinate expression is
  `d2I/dQ2` or `d2(I^-1)/dQ2`, and use it in both quartic and sextic work."

## Relation to CeDiTT

This does not mean that CeDiTT and VPT4 are the same theory.

The connection is narrower and more structural:

- CeDiTT already shows that the harmonic quartic problem naturally lifts from
  `R^3` to pair-pair space `Sym^2(R^3)`.
- VPT4 may be seeing a richer version of the same underlying tensorial layer,
  now through the `H22` channel and the sextic machinery.

So the bridge is not

- "CeDiTT formula = VPT4 formula",

but rather

- "both projects may be probing different projections of the same harmonic
  tensorial geometry."

## Practical consequence

The next development path should be:

1. isolate the current `H22` formula cleanly;
2. identify the minimal tensorial object that carries the information now
   written as `d2I/dQ2` or `d2(I^-1)/dQ2`;
3. determine its natural tensor space;
4. reformulate `H22` in terms of that object;
5. reuse the same object in the sextic machinery.

This keeps CeDiTT closed while allowing its tensorial intuition to inform the
next VPT4 step.
