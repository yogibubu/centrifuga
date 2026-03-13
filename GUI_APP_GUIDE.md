# CentrifugalTransform.app - Quick Guide

## Start
Double-click:
- `CentrifugalTransform.app`

If macOS blocks it (Gatekeeper), right-click -> Open.

## Inputs
Always provide rotational constants:
- `A, B, C` in MHz

Then choose tab:
- `Quartic` for quartic constants
- `Sextic` for sextic constants

## Quartic tab
1. Select input representation: `I`, `II`, `III`
2. Select reduction: `A` or `S`
3. Choose method:
   - `Reference (Yamada / Yamada-like S)`
   - `Tensor (quartic)`
4. Enter 5 constants (kHz)
5. Click `Compute quartic transforms`

Outputs:
- transformed constants for the other two representations
- stability diagnostics (condition numbers, singular values)

## Sextic tab
1. Select input representation
2. Select input reduction
3. Select output reduction
4. Enter 7 constants (kHz)
5. Click `Compute sextic transforms`

Outputs:
- transformed sextic constants for the other two representations
- round-trip error check

## Units
- Rotational constants: MHz
- Quartic/sextic distortion constants: kHz

## Notes
- Large condition numbers imply numerically sensitive transforms.
- For A reduction, `deltaK` sign is converted to the standard Watson convention in output.
