# Four-Paper Roadmap

## Overview

The overall program is organized around a common tensorial language for
rovibrational effective observables. The first branch concerns centrifugal
distortion constants, the second branch concerns vibration-rotation
corrections to the rotational constants. The same strategy is used throughout:

- identify the minimal tensor sector;
- separate standard linear structure from its first controlled breakdown;
- derive symmetry-adapted spectroscopic projections for asymmetric, symmetric,
  spherical, and linear limits;
- keep the implementation operational and spectroscopically usable.

## Paper 1

### Scope

- standard quartic and standard sextic centrifugal-distortion constants;
- tensor lift / transport / reprojection;
- first controlled breakdown of the linear transport picture;
- symmetry-adapted projections for special rotor limits;
- operational implementation in CeDiTT.

### Central objects

- `mu1 = d(I^-1)/dQ`;
- standard quartic tensor `tau`;
- standard sextic geometry functional;
- first breakdown channel `H22`;
- sextic linear response in `tau(H22)`.

### Output

- complete and autonomous theory of the standard linear transport sector;
- practical transformation tool for spectroscopic constants;
- low-cost diagnostics of the onset of breakdown.

## Paper 2

### Scope

- higher-order quartic centrifugal-distortion theory;
- tensor hierarchy beyond the standard linear sector;
- organization of the already available higher-order quartic equations.

### Central objects

- `mu2 = B(mu1) - N`;
- intrinsic second-level tensor `N`;
- channel-resolved quartic hierarchy beyond `H12H12`;
- identification of closures vs genuinely new tensor levels.

### Output

- structured higher-order quartic theory;
- explicit tensor hierarchy beyond the standard CeDiTT sector;
- conceptual basis for later higher-order spectroscopic observables.

## Paper 3

### Scope

- higher-order centrifugal-distortion observables required by demanding fits;
- octic constants and related higher-order distortion quantities;
- symbolic-algebra derivation of the corresponding formulas.

### Central objects

- higher-order projections of the quartic/sextic tensor hierarchy;
- symmetry-adapted projections for linear and symmetric-top limits;
- relation between higher-order perturbative tensors and fitted octic
  constants.

### Output

- spectroscopically usable higher-order distortion observables;
- bridge between higher-order perturbative structure and fitted octic terms.

## Paper 4

### Scope

- vibration-rotation corrections to rotational constants in the same tensorial
  language;
- reanalysis of `alpha`;
- derivation of `gamma`;
- symbolic organization of higher-order corrections beyond `alpha`.

### Working hypothesis

The `alpha -> gamma -> ...` branch stands to the rotational constants in a way
structurally analogous to the

- quartic -> sextic -> octic -> ...

branch for centrifugal distortion.

This does not imply equality of formulas. It means that the same tensorial
program should be applied:

- first-level geometry carried by `mu1` / `c1`;
- Coriolis structure `zeta`;
- semi-diagonal and then full cubic-force information;
- symmetry-adapted projection on the final spectroscopic observables;
- explicit distinction between standard sector and higher-level breakdown.

### Alpha / Gamma formalism

The `alpha/gamma` branch should be formulated in the same spirit as CeDiTT:

1. `alpha` is rewritten in the first-level tensor language, eliminating
   explicit `dI/dQ` from the working formulas.
2. Intermediate resonance-looking terms are separated from the final
   nonresonant observable sum.
3. Degenerate or nearly degenerate mode subspaces are handled by symmetry
   projectors and canonicalized tensorial features, not by ad hoc branch logic.
4. The same symmetry machinery is used for asymmetric tops, symmetric tops,
   spherical tops when relevant, and linear molecules.
5. `gamma` is then derived from zero as the next observable layer in the same
   perturbative hierarchy.

### Central objects

- `mu1`, `c1`, `zeta`;
- semi-diagonal cubic sector as the standard anharmonic input for `alpha`;
- symmetry projector on degenerate mode subspaces;
- canonicalized two-mode tensor features for degenerate pairs;
- next higher-level observable combinations defining `gamma`.

### Output

- controlled internal theory of `alpha`;
- explicit route to `gamma`;
- practical mode-resolved and subset-resolved vibration-rotation corrections;
- stronger interpretability than black-box variational workflows.

## Immediate work sequence

1. Submit Paper 1.
2. Resume Paper 2 and stabilize the higher-order quartic hierarchy.
3. Build the symbolic-algebra machinery for Paper 3.
4. In parallel, continue Paper 4 by:
   - refining `alpha` mode-by-mode validation;
   - finalizing the degeneracy handling;
   - preparing the `gamma` derivation in the same tensorial language.
