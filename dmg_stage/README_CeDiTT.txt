CeDiTT 3.0
Centrifugal Distortion Tensor Transformation

What this package contains
- `CeDiTT.app`: standalone macOS application
- this README

What the app does
- quartic tensor transport across representations `I`, `II`, `III`
- sextic tensor transport on the validated physical subspace
- direct harmonic calculation of quartic and sextic constants for:
  - asymmetric tops
  - symmetric tops
  - linear molecules
- `H22` and `tau(H22)` diagnostics
- `l`-type doubling diagnostics for linear molecules

Installation on another Mac
1. Copy the whole folder to the target Mac.
2. Drag `CeDiTT.app` into `Applications`, or run it directly from the folder.
3. On first launch, right-click `CeDiTT.app` and choose `Open`.

Requirements on the target Mac
- macOS 11 or later
- Apple Silicon (`arm64`)

Important note
- This build is standalone and does not depend on your local Python path or on files in `/Users/vincenzobarone`.
- It is not notarized. macOS may therefore warn on first launch.
- This build was produced on Apple Silicon and should be treated as Apple-Silicon-only unless rebuilt from a universal2 Python environment.

Units
- rotational constants `A`, `B`, `C`: MHz
- quartic constants: kHz in the transform panels
- sextic constants: kHz in the transform panels
- harmonic symmetry-adapted sextic and `l`-type reports: Hz where indicated in the report

Prepared from the CeDiTT 3.0 source tree.
