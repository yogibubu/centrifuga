# CeDiTT1.0 GUI Guide

## Start
Open:
- `CeDiTT1.0.app`

If macOS blocks the first launch, use right click -> `Open`.

## General Use
The GUI has two main tabs:
- `Quartic`
- `Sextic`

Both tabs are vertically scrollable.
The result panels are also scrollable, so long reports remain readable.

## XYZ Reference
The optional `XYZ` field is not only cosmetic.

If you provide an `XYZ`, the app derives from geometry:
- point group
- rotational symmetry number `sigma`
- rotor class
- rotational constants `A, B, C` in MHz

These `A, B, C` values are used as the spectroscopic reference order.

## Manual Quartic/Sextic Transforms
For manual transforms you provide:
- `representation`: `I`, `II`, or `III`
- `reduction`: `A` or `S`
- centrifugal distortion constants

If an `XYZ` is also provided:
- the app harmonizes the `A,B,C` order with the geometry-derived `XYZ` order
- the centrifugal constants are still interpreted in the selected representation and reduction

This avoids using manual constants with an `A,B,C` ordering inconsistent with the geometry.

## Harmonic Routes
For harmonic routes you can use:
- `.fchk`
- or `XYZ + Hessian`

Optional extra inputs depend on the tool:
- cubic `2-index` matrix
- Gaussian anharmonic `.log`

When an `XYZ` is available together with harmonic inputs:
- the harmonic model keeps its own internal orientation
- the GUI compares `ABC_model` with `ABC_xyz`
- the report prints the mismatch as `max|ΔABC|`

So the backend remains source-independent, while the spectroscopic reporting stays anchored to the external `XYZ` ordering.

## Quartic Tab
Available tasks:
- manual quartic representation transform
- load standard harmonic quartics from geometry + Hessian or `.fchk`
- validated quartic `H22`
- Gaussian `alpha` parser with mode exclusion
- internal `alpha` route from harmonic model + semi-diagonal cubic data

Units:
- `A, B, C`: MHz
- quartic constants: kHz

## Sextic Tab
Available tasks:
- manual sextic transform
- sextic `H22`-linear diagnostic candidate
- harmonic/cubic sextic hierarchy

Units:
- `A, B, C`: MHz
- sextic constants: as required by the GUI fields and printed explicitly in the reports

## Practical Notes
- For exact symmetric-top and linear limits, prefer the harmonic routes instead of manual asymmetric-top transforms.
- If the input `XYZ` uses atomic numbers instead of element symbols, the app now accepts that format.
- Long reports should now remain fully visible thanks to the built-in scrolling.
