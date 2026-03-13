# Centrifugal Distortion Workflow

This repository contains the current production and benchmark workflow for
quartic and sextic centrifugal distortion constants derived from geometry,
Cartesian Hessians, and optional Gaussian/GDV anharmonic data.

## Scope

This repository currently contains material for two distinct but related lines
of work.

### Work 1: H12H12 structure and representation transforms

This is the CeDiTT/app side:

- standard VPT2 quartic `H12H12`
- quartic representation transforms
- sextic representation transforms of the analogous tensorial type
- tensor/pseudoinverse transformation algorithms

This is the natural home for questions such as:

- spatial-vs-gauge structure of the standard quartic tensor
- invariant content under representation changes
- app-oriented transformation workflows

### Work 2: full quartic VPT4 workflow

This is the production/benchmark side:

- geometry + Hessian harmonic backend
- channel-resolved quartic VPT4 machinery
- Gaussian-style sextic benchmarking
- complete numerical workflow for the full code

Important: statements in this repository about spatial/gauge structure should be
read in that context. A structural claim that is meaningful for the standard
`H12H12` block does not automatically extend to the full VPT4 quartic tensor.

- `distortion_workflow.py`
  - main entrypoint for quartic and sextic calculations
- `rovib_distortion.py`
  - harmonic backend from geometry + Hessian
  - normal modes, inertia derivatives, Coriolis tensors
- `quartic_channels.py`
  - direct channel-resolved VPT4 quartic machinery
- `gaussian_vpt_parser.py`
  - Gaussian/GDV harmonic, anharmonic, quartic, and sextic parsers
- `compare_gaussian_sextic.py`
  - explicit Gaussian-style sextic benchmark replication

## Production path

The production backend is:

1. geometry + Cartesian Hessian
2. normal modes built internally
3. harmonic tensor objects built from the same mode convention
4. quartic and sextic properties derived from that common backend

Inputs can be provided either as:

- `--xyz` + `--hessian`
- `--input-fchk`

The `.fchk` path is only a convenience bridge to the same geometry+Hessian
backend. It is not a separate theoretical route.

## Representations

The workflow supports the three standard principal-axis representations:

- `I`: `(x,y,z) = (a,b,c)`
- `II`: `(x,y,z) = (b,c,a)`
- `III`: `(x,y,z) = (c,a,b)`

This representation is applied consistently to:

- equilibrium coordinates
- Cartesian Hessian
- normal modes
- Coriolis tensors
- quartic and sextic outputs

## Order-2 quartics

The validated quartic order-2 route is the Gaussian/QCent-compatible route:

1. harmonic model from geometry + Hessian
2. `Tau(ijkl)` and `TauPrime` in the chosen representation
3. Gaussian-consistent projection
   - `TauPrime -> T = TauPrime / 4`
   - `T ->` Watson `A` and `S` quartic constants

This route has been benchmarked against Gaussian on `h2o`.

## Quartic tensor decomposition

The quartic module now exposes an explicit pair-pair decomposition

- `tau = M + g1*G1 + g2*G2`

with

- `M = (Maa, Mbb, Mcc, 0, 0, 0)`
- `G1 = (3, 3, 3, 1, 1, 1)`
- `G2 = (2A, 2B, 2C, A+B, A+C, B+C)`

in the compressed basis

- `(tau_aaaa, tau_bbbb, tau_cccc, tau_aabb, tau_aacc, tau_bbcc)`

Available helpers in [`distortion_workflow.py`](./distortion_workflow.py):

- `build_quartic_tensor(...)`
- `extract_gauge_coefficients(...)`
- `decompose_quartic_tensor(...)`
- `watson_constants_to_quartic_tensor(...)`
- `quartic_hamiltonian_diagnostic(...)`

Important distinction:

- these routines implement the gauge-space formalism exactly
- on a synthetic tensor built from `M`, `g1`, and `g2`, extraction and
  reconstruction close at roundoff
- the current harmonic `h2o` quartic tensor from the validated QCent route does
  not fall exactly into this restricted subspace, so the decomposition is
  reported together with a residual

For this reason the workflow also reports a second, more general decomposition

- `tau = M_pairpair + g1*G1 + g2*G2`

where `M_pairpair` is a full six-component pair-pair spatial tensor. This
generalized decomposition reproduces the compressed quartic tensor exactly and
is the right diagnostic when the restricted diagonal spatial model leaves a
non-zero residual.

This decomposition machinery is mainly a diagnostic bridge between Work 1 and
Work 2. The restricted `M + g1 G1 + g2 G2` ansatz is most naturally interpreted
on the standard `H12H12` side, not as a general structural theorem for the full
VPT4 quartic tensor.

So the code now supports both:

- direct construction: `M, g1, g2 -> tau`
- inverse mapping: Watson constants `-> tau` by pseudoinverse, then `tau -> (M, g1, g2)`

## Order-4 quartics

Order-4 quartics are reported channel by channel:

- `H12H12`
- `H22`
- `H12H30`
- `H30H30`
- `total`

The final quartic constants are obtained through the compressed quartic tensor
using pair-pair closure. In this formulation, the Gaussian-consistent reduced
matrix is obtained directly from the compressed tau basis:

- `TauPrime_xx = tau_xxxx`
- `TauPrime_xy = tau_xxyy / 2`
- `TauPrime_xz = tau_xxzz / 2`
- `TauPrime_yz = tau_yyzz / 2`

No explicit reconstruction of a general Wilson four-index tensor is required.

## Coriolis backend

The harmonic model now stores a canonical Coriolis/tensor backend:

- pairwise Coriolis tensor `zeta_alpha(i,j)`
- Coriolis vectors `G_k`
- symmetric Coriolis tensor `M`

For the standard quartic `H12H12` block, this `3x3` Coriolis tensor is not just
an auxiliary invariant object: after the appropriate physical prefactor and
unit/convention matching, it is linearly equivalent to the reduced quartic
tensor `Tau'`. In matrix form,

- `Tau' = 0.5 * M - 0.5 * Tr(M) * I`
- `M = 2 * Tau' - Tr(Tau') * I`

So `M` and `Tau'` have the same principal axes, and their spectral invariants
are in one-to-one correspondence. This is the clean CeDiTT-side bridge between
the Coriolis physics and the quartic spectroscopic constants.

In the current workflow the stored `coriolis_m_tensor_au` is still reported in
its raw Coriolis-side units, so it should be read as the unscaled tensor form
rather than as a direct numerical copy of `Tau'`.

Important limitation:

- `M` alone does not determine the full quartic Watson set
- the full quartic set still depends on gauge coordinates

For this reason the workflow prints both:

- spatial Coriolis invariants
- quartic gauge parameters

and keeps the validated quartic constants on the QCent/TauPrime route.

This statement applies specifically to the standard `H12H12` quartic block. For
the broader VPT4 workflow one should still distinguish carefully between:

- the CeDiTT/H12H12 tensor relation `M <-> Tau'`,
- the full pair-pair quartic tensor used in the channel-resolved VPT4 code.

For completeness, the workflow also reports the pair-pair metric in
`Sym^2(R^3)`, which is the more natural harmonic quartic object on the full
pair-pair side.

## Sextics

The sextic implementation follows the Gaussian `SEXTIC` machinery.

Current status:

- `Tau`
- `C_i^{ab}`
- `C_i^{abc}`
- final sextic `Phi`

have been benchmarked against Gaussian on `h2o`.

The sextic machinery is already strongly Coriolis-driven through `zeta` and
`C_i^{abc}`, but `C_i^{ab}` still depends explicitly on `dIdQ`.

So at present the sextic formalism is:

- not purely `zeta`-only
- but already expressed in a mixed Coriolis + inertia-derivative language
- in CeDiTT, exposed operationally as a canonical `5+2` decomposition:
  - `5` physical coordinates `sigma` on the validated invariant S-subspace
  - `2` residual coordinates in the orthogonal complement
- structurally, this same physical sector can be viewed as:
  - harmonic `1+4`: one isotropic scalar plus one even rank-6 harmonic component
  - representation-adapted `1+1+1+2` under cyclic axis relabelings

The sextic GUI now reports this decomposition as

\[
H_S = B_{\mathrm{rep}}\,\sigma + r,
\]

where `sigma` are the physical sextic coordinates and `r` is the residual
vector. For physically consistent sextic constants the residual is numerically
zero (up to roundoff). The same 5D physical sector may be interpreted
structurally either as `H_0 ⊕ H_6^even` (`1+4`) or, for representation
transforms, as `1+1+1+2`.

## Cubic force constants from Gaussian

When cubic force constants are read from a Gaussian/GDV anharmonic log, two
different situations must be distinguished.

For the dedicated Gaussian benchmark script:

- the harmonic model is aligned to Gaussian's own phase convention
- therefore the cubic constants are only reordered by frequency
- no additional cubic phase refit is applied

For the general production workflow:

- the internal harmonic model keeps its own phase convention
- the Gaussian cubic constants are reordered by frequency
- and then rephased to the internal mode phases using the `dIdQ` benchmark when
  that information is available in the Gaussian log

This distinction is essential. The same cubic-phase rule cannot be used for
both the Gaussian-aligned benchmark script and the production workflow.

## Typical usage

Order 2 from `.fchk`:

```bash
python distortion_workflow.py \
  --input-fchk h2o.fchk \
  --representation III \
  --order 2 \
  --gaussian-log h2o.log
```

Order 4 quartics with Gaussian cubic data:

```bash
python distortion_workflow.py \
  --input-fchk h2o.fchk \
  --representation III \
  --order 4 \
  --gaussian-log h2o.log
```

Direct benchmark script for sextics:

```bash
python compare_gaussian_sextic.py --fchk h2o.fchk h2o.log
```

## Current boundaries

What is production-ready:

- harmonic model from geometry + Hessian
- validated order-2 quartics
- validated sextic benchmark machinery
- channel-resolved order-4 quartic reporting
- Gaussian-consistent quartic projection from compressed tau

What is not yet rewritten in purely Coriolis form:

- the full `C_i^{ab}` / `dIdQ` block
- the complete replacement of all `dI/dQ` and `d2I/dQ2` objects by Coriolis-only objects

At present, the Coriolis backend is primary for tensorial diagnostics and sextic
structure, while inertia derivatives remain available where the current VPT4
channel implementation still needs them.
