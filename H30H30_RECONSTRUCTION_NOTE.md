# H30H30 Reconstruction Note

## Current Status

The current numerical implementation of `H30H30` in
[quartic_channels.py](/Users/vincenzobarone/centrifugal/quartic_channels.py)
is still a placeholder:

```text
phi3[i,j,j] * mu1[i] * mu1[j] / (omega_i + omega_j + 1)
```

This is useful only as a diagnostic stub. It is not structurally compatible
with the symbolic BCH-derived channel.

## What Is Already Established

### 1. One-mode symbolic baseline

The symbolic extractor
[scripts/extract_h30h30_symbolic_denominators.py](/Users/vincenzobarone/centrifugal/scripts/extract_h30h30_symbolic_denominators.py)
already shows that, in the one-mode case, the genuine `H30,H30` channel scales
as

```text
tau_mu^(30,30) ~ Phi_iii^2 / omega_i^7
```

for all six quartic tensor components.

This is the first hard constraint on any reconstruction:

- the true channel is an `omega^-7` object already at one mode;
- the current placeholder kernel `omega_i + omega_j + 1` is therefore in the
  wrong structural class.

### 2. Resonance sensitivity belongs here first

The notes
[PAPER2_RESONANCE_STRATEGY.md](/Users/vincenzobarone/centrifugal/PAPER2_RESONANCE_STRATEGY.md)
and
[H2O_quartic_VPT4_diagnostic.md](/Users/vincenzobarone/centrifugal/H2O_quartic_VPT4_diagnostic.md)
already make the correct physical point:

- `H30H30` is the first quartic channel that genuinely requires resonance
  analysis;
- `H2O` is the stress test;
- the dominant scaffolds are
  - `diag_1_iii_iii`
  - `diag_1_iii_iij_0`

So reconstruction should start from those families first, not from a fully
generic all-scaffold fit.

### 3. The current symbolic probe is fragile under pruning

The updated denominator extractor now supports

- `--channel-aware`
- `--max-vib-word`
- `--max-j-word`

but the channel is much more fragile than `H12H30`:

- aggressive pruning can kill even the one-mode signal;
- two-mode generic scans quickly become too heavy to be useful as blind probes.

This suggests that a successful BCH-first recovery of `H30H30` will have to be
family-targeted, not just globally pruned.

## Practical Consequences

The immediate next steps should be:

1. keep the current placeholder clearly marked as diagnostic-only;
2. use the one-mode `omega^-7` result as the normalization baseline;
3. reconstruct first the diagonal family corresponding to `diag_1_iii_iii`;
4. then add the semi-diagonal family corresponding to `diag_1_iii_iij_0`;
5. only after these two are under control, reopen the resonance machinery on
   the recovered denominators.

## What Should Not Be Done Next

The following would likely waste time right now:

- broad two-mode symbolic scans without family targeting;
- fitting the current placeholder to working references;
- introducing resonance regularization before the denominator families are
  restored.

## Working Interpretation

At the current stage, the cleanest reading of `H30H30` is:

- the solver placeholder is intentionally not physical;
- the one-mode symbolic result already fixes the asymptotic structure;
- the reconstruction should proceed scaffold-first, beginning with the two
  diagonal families known to dominate `H2O`.
