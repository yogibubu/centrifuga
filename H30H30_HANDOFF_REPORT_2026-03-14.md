# H30H30 Handoff Report

Date: 2026-03-14

## Scope

This report captures the current restart point for the `H30H30` reconstruction
work in the `paper2`/VPT4 branch.

It is meant to allow a new session to resume without re-deriving the recent
decisions, diagnostics, and constraints.

## Current State

### Operative channel definition

In [quartic_channels.py](/Users/vincenzobarone/centrifugal/quartic_channels.py),
the current provisional operative channel is

```text
total = trusted_total = D^(0)_{iii,iii} + D^(1)_{iii,iii}
```

with

- `diag_1_iii_iii`
- `diag_1_iii_iii_correction_candidate`

always included in `total`.

The wider diagnostic sums remain separate:

- `extended_total = trusted_total + D^(1;0)_{iii,iij} + residual`
- `resonance_preview_total = trusted_total + regularized signed iii,iij preview`

### Important warning

This definition follows the latest user instruction ("`D^(1)_{iii,iii}` must be
applied always"), but it is numerically non-uniform across the benchmark set:

- it improves `H2O`
- slightly improves `H2S`
- strongly worsens `H2CO`
- strongly worsens `H2CS`

So the code now follows the requested policy, but this policy is not yet a
validated physical closure.

## Key Files

### Core implementation

- [quartic_channels.py](/Users/vincenzobarone/centrifugal/quartic_channels.py)
- [h30h30_resonance.py](/Users/vincenzobarone/centrifugal/h30h30_resonance.py)
- [channel_contributions.py](/Users/vincenzobarone/centrifugal/channel_contributions.py)
- [harmonic_convention.py](/Users/vincenzobarone/centrifugal/harmonic_convention.py)

### Notes and handoff material

- [H30H30_RECONSTRUCTION_NOTE.md](/Users/vincenzobarone/centrifugal/H30H30_RECONSTRUCTION_NOTE.md)
- [H30H30_HANDOFF_REPORT_2026-03-14.md](/Users/vincenzobarone/centrifugal/H30H30_HANDOFF_REPORT_2026-03-14.md)
- [paper2.tex](/Users/vincenzobarone/centrifugal/paper2.tex)
- [tau_channel_equations.tex](/Users/vincenzobarone/centrifugal/tau_channel_equations.tex)
- [PAPER2_RESONANCE_STRATEGY.md](/Users/vincenzobarone/centrifugal/PAPER2_RESONANCE_STRATEGY.md)
- [H2O_quartic_VPT4_diagnostic.md](/Users/vincenzobarone/centrifugal/H2O_quartic_VPT4_diagnostic.md)

### Diagnostics and checks

- [scripts/h30h30_scaffold_diagnostic.py](/Users/vincenzobarone/centrifugal/scripts/h30h30_scaffold_diagnostic.py)
- [scripts/h30h30_vci_sanity_check.py](/Users/vincenzobarone/centrifugal/scripts/h30h30_vci_sanity_check.py)
- [scripts/extract_h30h30_symbolic_denominators.py](/Users/vincenzobarone/centrifugal/scripts/extract_h30h30_symbolic_denominators.py)
- [test_h30h30_structure.py](/Users/vincenzobarone/centrifugal/test_h30h30_structure.py)
- [test_h30h30_resonance.py](/Users/vincenzobarone/centrifugal/test_h30h30_resonance.py)

### External order-of-magnitude reference

- `/Users/vincenzobarone/Desktop/centrifugal_VCI.pdf`

This PDF is used only as an external order-of-magnitude sanity check via
`PFIT/RVCI(E) - VPT`, not as a channel-by-channel benchmark.

## Scaffold Inventory

### Diagonal families

- `diag_1_iii_iii`
  - one-mode exact structural baseline
  - scales as `Phi_iii^2 / omega_i^7`
- `diag_1_iii_iii_correction_candidate`
  - appendix-derived second diagonal family
  - now always included in `total`

### Semi-diagonal and signed candidates

- `diag_1_iii_iij_0`
  - included only in `extended_total`
- `diag_0_iii_iij_1_candidate`
  - diagnostic only
- `diag_0_iii_iij_2_resonance_candidate`
  - diagnostic only
  - first candidate with signed factor `2*omega_i - omega_j`
- `diag_0_iii_iij_2_regularized_preview`
  - diagnostic only
  - smooth Martin/2x2 regularized preview of the signed candidate

### iij,iij families

- `diag_0_iij_iij_1_candidate`
- `diag_0_iij_iij_2_candidate`

Both are diagnostic only.

## Resonance Status

### What is genuinely Martin-ready

The first scaffold family that is actually suitable for a Martin test is

- `diag_0_iii_iij_2_resonance_candidate`

because it contains the signed factor

```text
2*omega_i - omega_j
```

The factor-level diagnostic is in
[h30h30_resonance.py](/Users/vincenzobarone/centrifugal/h30h30_resonance.py)
and printed by
[channel_contributions.py](/Users/vincenzobarone/centrifugal/channel_contributions.py).

### Concrete H2O result

For `H2O`, the diagnostic now shows a near-resonant signed factor:

```text
2*omega_1 - omega_2 ≈ -9 cm^-1
```

This is the first real resonance-like denominator recovered in the current
scaffold-first reconstruction.

### Effect of the 2x2 preview

The regularized preview reduces the signed `iii,iij` candidate strongly:

- `H2O`: norm `2.35e6 -> 8.50e4`
- `H2S`: norm `5.31e3 -> 2.37e3`

But even after regularization, the channel remains dominated by the diagonal
core.

## External Sanity Check

Use:

```bash
python3 scripts/h30h30_vci_sanity_check.py
```

The current `total = D^(0)_{iii,iii} + D^(1)_{iii,iii}` gives

- `H2O`: ratio `||total|| / ||PFIT-RVCI - VPT|| ≈ 1.52e3`
- `H2S`: `≈ 1.18e4`
- `H2CO`: `≈ 1.00e4`
- `H2CS`: `≈ 4.28e4`

So the channel is still far too large in absolute scale, even though the
`H2O` ratio is better than before.

## Most Important Numerical Conclusions

1. `H30H30` is still not closed.
2. The first real resonance-like denominator has now been recovered.
3. The Martin/2x2 layer is now connected to a physically meaningful signed
   family.
4. The dominant unresolved problem is still the diagonal sector.
5. Including `D^(1)_{iii,iii}` always helps `H2O`, but badly overcorrects
   `H2CO/H2CS`.

## Recommended Next Step

If work resumes later, the most useful next step is:

1. keep the current code state as the official restart point;
2. analyze the diagonal pair
   - `D^(0)_{iii,iii}`
   - `D^(1)_{iii,iii}`
   species by species;
3. understand why the second diagonal family helps `H2O/H2S` but explodes
   `H2CO/H2CS`;
4. do not add more scaffolds before that diagonal issue is understood.

## Useful Commands

```bash
python3 -m pytest -q test_h30h30_structure.py test_h30h30_resonance.py
python3 scripts/h30h30_scaffold_diagnostic.py --species h2o h2s h2co h2cs
python3 scripts/h30h30_vci_sanity_check.py
python3 channel_contributions.py --species H2O h2o --show-h30h30-diagnostic --h30h30-top-terms 10
```
