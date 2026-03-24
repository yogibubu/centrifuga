# CeDiTT App Next Checks

This note tracks the next validation steps for the GUI/backend alignment
after the current handedness and tensor-workflow updates.

## Pending real-molecule checks

These cases require actual Gaussian outputs that are not yet available in
the repository.

### 1. Linear molecule

Suggested targets:
- HCN
- CO2
- OCS

Things to verify:
- harmonic model builds correctly from `fchk`/`log`
- point group and rotor classification are reported correctly
- normal-mode symmetry assignment is sensible
- symmetry-adapted quartic projection is consistent with the linear limit
- symmetry-adapted sextic projection is consistent with the linear limit
- GUI reporting of the `l`-type diagnostic is clear and not overstated

### 2. Symmetric top

Suggested targets:
- NH3
- CH3CN
- CH3Cl

Things to verify:
- harmonic model builds correctly from `fchk`/`log`
- point group and rotor classification are reported correctly
- normal-mode symmetry assignment is sensible
- quartic symmetry-adapted projection is stable in the prolate/oblate limit
- sextic symmetry-adapted projection is stable in the prolate/oblate limit
- GUI wording remains consistent with the tensor workflow in CeDiTT4

## Structural points already settled

- Quartic `r <-> l` is now handled operationally in the report as a
  fixed-representation axis swap.
- Sextic `r <-> l` is not treated as a validated internal transport on the
  current five-dimensional physical subspace.
- The latter point is supported by:
  - `test_sextic_handedness_subspace.py`
  - Appendix A/C wording in `ceditt4/CeDiTT4.tex`
  - GUI notes in `ceditt_gui.py`

## Immediate follow-up once files are available

1. Run the GUI end-to-end on one linear molecule.
2. Run the GUI end-to-end on one symmetric top.
3. Save representative screenshots or numeric reports.
4. Commit the GUI/paper alignment block only after those two checks.
