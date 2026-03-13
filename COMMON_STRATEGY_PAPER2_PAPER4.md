# Common Strategy for Paper 2 and Paper 4

This note fixes a single derivational strategy to be used in parallel for:

- Paper 2: higher-order quartic centrifugal distortion;
- Paper 4: `alpha/gamma` vibration-rotation corrections.

The purpose is to avoid developing the two branches with different logical
styles. The same sequence should be followed in both cases.

## 1. Start from the observable, not from the taxonomy

The first step is always:

1. define the physical observable to be expanded;
2. write its direct perturbative/state expansion;
3. identify the coefficient that corresponds to the spectroscopic quantity of
   interest.

For Paper 2 this means:

- start from the quartic effective observable and the next perturbative layer
  beyond the standard sector.

For Paper 4 this means:

- start from the vibrational-state dependence of the rotational constants and
  identify the coefficients giving `alpha` and then `gamma`.

This is the part that should be strictly parallel.

## 2. Only after that identify the raw derivative hierarchy

Once the observable is written directly, one identifies which raw derivative
level appears first in the naive expansion.

For example:

- on the quartic higher-order branch, the direct expansion points to the
  second geometric response `mu2`;
- on the vibration-rotation branch, the direct state expansion shows that
  `alpha` and then `gamma` start from progressively higher raw responses of
  `mu(Q)`.

But this is only the raw perturbative reading, not yet the primitive tensor
content of the final observable.

## 3. Then try to eliminate higher derivatives down to the lowest tensor level

This is the CeDiTT step.

After the raw derivative level is identified, one asks:

- can the assembled observable be rewritten entirely in terms of first-level
  objects (`mu1/c1`, `zeta`, low-order force constants)?
- or does some genuinely higher-level tensor survive?

This step must be done in the same way in both branches.

## 4. Define the standard sector first

The next step is to isolate the largest standard sector:

- the maximal part of the observable that can be written using the lowest
  reusable tensor level;
- together with the standard explicit anharmonic extension.

This gives, in both branches, a decomposition of the form

`observable = standard_sector + remainder`.

For Paper 2:

- standard quartic baseline plus the first beyond-standard correction.

For Paper 4:

- `gamma_std = gamma_geom_std + gamma_anh_std`.

## 5. Only then define the first controlled breakdown

The first controlled breakdown is **not** the first term seen in the raw
expansion. It is the first part of the final assembled observable that cannot
be reduced to the standard tensor sector.

This is exactly what was done in CeDiTT for:

- `H22`,
- and the sextic response in `tau(H22)`.

The same logic must be used for:

- the first quartic higher-order correction in Paper 2;
- the first irreducible `gamma` correction in Paper 4.

## 6. Symbolic algebra comes after the structural split

Only after the split

- direct observable,
- raw derivative hierarchy,
- standard sector,
- first controlled breakdown

is clear, should one generate the heavy symbolic formulas.

This avoids two common failures:

- deriving giant formulas before knowing which part is standard and which part
  is new;
- producing channel taxonomies before the observable itself is under control.

## 7. Implementation comes last

The coding order should also be parallel:

1. derive the direct observable coefficient;
2. identify the standard reduction;
3. identify the first breakdown term;
4. only then expose the corresponding pieces in the backend/API.

This means that backend structure should mirror the theory:

- `std`
- `std_anh`
- `break_first`

rather than reflecting ad hoc implementation history.

## 8. Practical consequence

From now on:

- Paper 2 should be developed by going back to the direct perturbative
  structure of the higher-order quartic observable, then reducing it to
  `mu1` closure plus first irreducible correction.
- Paper 4 should be developed by going back to the direct vibrational-state
  expansion of `B_v`, then reducing `alpha` and `gamma` to the same lowest
  tensor sector plus first irreducible correction.

This is the common method. Anything else should be treated as secondary
supporting material, not as the main derivational path.
