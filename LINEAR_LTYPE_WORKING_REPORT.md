# Linear l-Type Working Report

## Status after manuscript alignment

The final manuscript sources are now the reference for the paper-aligned
linear sector:

- [CeDiTT4.tex](/Users/vincenzobarone/Downloads/CeDiTT4.tex)
- [ceditt4_sextic_partition_appendix.tex](/Users/vincenzobarone/Downloads/ceditt4_sextic_partition_appendix.tex)

These sources fix the boundary unambiguously:

- the manuscript includes linear molecules only in the pure rotational
  sector;
- the paper-aligned linear constants are the symmetry-adapted scalars
  `D` and `H`;
- degenerate-bending `l`-type interactions require additional tensorial
  objects and are outside the scope of the present paper.

Accordingly, the current implementation should be read as two separate
layers:

- `linear pure rotational`
  - paper-aligned
  - built from the transverse-subspace projection
  - returns the special-limit scalars `D` and `H`
- `LINEAR_LTYPE`
  - experimental beyond-paper extension
  - built on top of the paper-aligned `D/H` layer
  - used for diagnostics, operator bookkeeping, and Gaussian benchmark
    comparison on degenerate bending pairs

Everything below in this note refers to that second layer.

## Scope fixed

The current working scope is intentionally minimal:

- one near-degenerate bending pair at a time,
- one real two-state basis for the pair,
- one splitting operator in that pair subspace,
- three reported coefficients in the primary circular convention:
  `q_l`, `q_l^J`, `q_l^H`.
- an additional leading-order harmonic estimate
  `q_l^(0) = 2 B_{\mathrm{lin}}/\omega_t`
- and a Watson-like corrected estimate
  `q_l^(W) = (2 B_{\mathrm{lin}}/\omega_t)\left[1 + \sum_s f_{st}^2/(\omega_s^2-\omega_t^2)\right]`
- explicit spectroscopic estimates:
  - `q_e^(0)`
  - `q_e^(W)`
  - `q_v` (reserved for the future state-specific correction)

This is **not** yet a full effective Hamiltonian for all linear-molecule
l-type interactions. It is an experimental minimal pairwise effective
Hamiltonian layered on top of the paper-aligned linear pure-rotational
CeDiTT workflow.

## Operator basis adopted

For each near-degenerate pair `(i,j)` we use a real doublet basis

- `|u_ij>`
- `|v_ij>`

and define the pair splitting operator

- `O_t,ij = |u_ij><u_ij| - |v_ij><v_ij|`

This remains the equivalent real-doublet form. The primary convention
used by the current backend is now the circular doublet basis

- `|+>`
- `|->`

through the exchange-type operator

- `X_l,ij = |+><-| + |-><+|`.

The backend now also exposes the complete 2x2 pair basis of the doublet
subspace:

- in the real basis: `I_t, O_t, X_t, Y_t`,
- in the circular basis: `I_l, Z_l, X_l, Y_l`.

This does not yet define the final spectroscopic Hamiltonian, but it now
makes the operator carrier space explicit.

## Literature convention selected

The selected primary convention is the circular vibrational-angular-momentum
basis of the degenerate bending doublet:

- primary basis: `I_l, Z_l, X_l, Y_l`,
- active minimal channel: `X_l`.

This choice is preferred because:

- it is the natural basis for degenerate bendings in linear molecules,
- it is closest to standard `l`-type spectroscopic notation,
- and it already coincides with the solver-facing tensor blocks used by
  the current implementation.

## Quantities now reported by the backend

Implemented in [distortion_workflow.py](/Users/vincenzobarone/centrifugal/distortion_workflow.py):

- `B_linear_cm`
- explicit branch decomposition:
  - `pure_rotational_branch`
    - `scope_status = paper_aligned`
  - `pairwise_ltype_branch`
    - `scope_status = experimental_beyond_paper`
    - including explicit quartic/sextic rotational feeds into the active
      pairwise channel
  - `literature_convention`
- global pair-basis metadata:
  - `full_operator_basis_circular = {I_l, Z_l, X_l, Y_l}` as primary basis
  - `full_operator_basis_real = {I_t, O_t, X_t, Y_t}` as equivalent basis
- for each pair:
  - `pair_label`
  - `pair_irrep`
  - `freq_cm`
  - `zeta_parallel`
  - `q_t_abs_hz`
  - `q_tJ_diagnostic_hz`
  - `q_tH_diagnostic_hz`
  - `conventional_pair_mapping`
  - `solver_facing_circular_basis`
  - `solver_facing_blocks_hz`
  - explicit coefficient dictionaries on the full pair basis
  - explicit 2x2 pair Hamiltonian matrices in real and circular bases
  - `operator_basis = circular_doublet`
  - `operator_label = X_l`
  - `operator_basis_alt = real_doublet`
  - `operator_label_alt = O_t`

The present coefficients are

- `q_l^J = |zeta_parallel| |D|`
- `q_l^H = |zeta_parallel| |H|`
- `q_l = sqrt[(q_l^J)^2 + (q_l^H)^2]`

where `D` and `H` are the scalar quartic and sextic coefficients already
defined in the linear pure-rotational limit.

For backward compatibility the backend still reports the legacy aliases

- `q_t`
- `q_t^J`
- `q_t^H`

but these now simply mirror the primary circular-basis constants.

The backend now also exposes an explicit conventional-model ladder for
the same pair:

- `conventional_linear_model_hz`
  - partial internal mapping when only the minimal CeDiTT branch is
    available,
  - reconstructed conventional mapping when Gaussian source blocks are
    available,
- `conventional_linear_model_exact_hz`
  - exact RotL2x constants when the printed Gaussian linear block is
    available.

In addition, the backend now reports the leading-order harmonic
literature estimate

- `q_l^(0) = 2 B_{\mathrm{lin}}/\omega_t`

which is the natural first comparison point against the standard
linear-molecule perturbative formula.

It also reports a Watson-like corrected estimate

- `q_l^(W) = (2 B_{\mathrm{lin}}/\omega_t)\left[1 + \sum_s f_{st}^2/(\omega_s^2-\omega_t^2)\right]`

built from the available harmonic frequencies and Coriolis couplings.

These are now also exported explicitly as the spectroscopic linear
estimates

- `q_e^(0)`
- `q_e^(W)`

while `q_v` is kept reserved until the vibrational-state correction is
derived explicitly.

In addition, the backend now builds the minimal closed effective model

- `H_eff^(lin) = q_e X_l + q_J^(pair) J^2 X_l + q_H^(pair) (J^2)^2 X_l`

where:

- `q_e` is currently either the internal non-resonant source estimate
  or the exact Gaussian RotL2x value when available,
- `q_J^(pair)` and `q_H^(pair)` are imported from the tensorial
  pairwise branch.

For real linear benchmarks such as `C2H2`, the current backend already
returns stable pair labels in the symmetry-adapted form

- `Pi_g(1)`
- `Pi_u(1)`

for the first and second degenerate bending pairs.

## Interpretation

The reported object is the pairwise effective operator in the primary
circular convention

- `H_diag^(l)(ij) = q_l X_l + q_l^J J^2 X_l + q_l^H (J^2)^2 X_l`

This should be read as the current minimal experimental pairwise
l-type Hamiltonian activated by a near-degenerate bending pair, not as
a completed derivation of all l-doubling terms.
In the equivalent real basis this same model is reported as

- `H_diag^(l)(ij) = q_l O_t + q_l^J J^2 O_t + q_l^H (J^2)^2 O_t`.

The backend now also makes this statement explicit as:

- a branch split between
  - pure rotational scalars `D` / `H`,
  - and the pairwise `l`-type layer,
- an explicit identification of how these scalar branches feed the
  pairwise active channel through `J^2 X_l` and `(J^2)^2 X_l`,
- coefficient dictionaries on the complete pair basis,
- and explicit `2x2` Hamiltonian matrices in both bases.
- a conventional-ready pair mapping in which the active minimal channel
  is `X_l` and the inactive channels are `I_l, Z_l, Y_l`.
- ordered solver-facing vectors and block matrices on the circular basis
  `{I_l, Z_l, X_l, Y_l}` for the `1`, `J^2`, `(J^2)^2` sectors.
- a closed minimal effective linear model in which the `J0` block is
  driven by `q_e^(W)` and the `J^2/J^4` blocks are driven by the
  tensorial pairwise branch.

This is the form that should be used for the future conventional
constant mapping.

At the current stage, the closure hierarchy is therefore:

- minimal carrier model:
  - `q_l`, `q_l^J`, `q_l^H`
- internal effective model:
  - `q_e^(src)`, `q_J^(pair)`, `q_H^(pair)`
- conventional-model candidate:
  - `q_e`, `q_J`, `q_K`
  - partial if no Gaussian source reconstruction is available
  - reconstructed if Gaussian source blocks are available
  - exact if the RotL2x block is printed in the Gaussian log

## What is already coherent

- the pure rotational linear-limit sector is already projected onto scalar
  `D` and `H`;
- that `D/H` layer is the part aligned with the final manuscript;
- near-degenerate pairs are already detected from the same mode metadata used
  by the `alpha` branch;
- the GUI already has a compatible reporting structure for pairwise l-type
  diagnostics.
- the leading-order harmonic estimate `q_l^(0)` is now available for a
  first direct comparison with standard literature `q` constants.

## First external benchmark check

Using the present `C2H2` implementation, the backend returns for each
degenerate pair a primary circular-basis pairwise constant of about

- `q_l = 0.0362 MHz`

whereas NIST tabulates experimental `q_ν` constants for `d1-acetylene`
(`HCCD`) of the order

- `ν4: 132.993 MHz`
- `ν5: 105.702 MHz`

This is a very useful diagnostic result:

- the present branch is already internally coherent as a tensor-driven
  pairwise linear model,
- the leading-order estimate `q_l^(0)` is already in the right
  spectroscopic range for `C2H2`,
- and the Watson-like correction is negligible for the current
  harmonic model,
- but the current tensorial `q_l` is **not yet** the final spectroscopic `q_ν`
  constant of the literature.

In other words, the operator carrier and the convention are now fixed,
but the final benchmark-level mapping to the observed `l`-type
constants is still missing.

## Current benchmark-level results

For the available real linear benchmark `C2H2`, the present
implementation now gives:

- pure rotational quartic scalar:
  - `D = 0.0361984 MHz`
- pair `Pi_g(1)`:
  - `omega_t = 535.0236 cm^-1`
  - `q_l = 0.0361984 MHz`
  - `q_e^(0) = 131.860363 MHz`
  - `q_e^(W) = 131.860375 MHz`
  - `q_J^(pair) = 0.0361984 MHz`
  - `q_H^(pair) = 0`
- pair `Pi_u(1)`:
  - `omega_t = 775.2714 cm^-1`
  - `q_l = 0.0361984 MHz`
  - `q_e^(0) = 90.998338 MHz`
  - `q_e^(W) = 90.998346 MHz`
  - `q_J^(pair) = 0.0361984 MHz`
  - `q_H^(pair) = 0`

These are the first results that should now be regarded as stable:

- `q_e^(0)` and `q_e^(W)` provide the spectroscopic-scale `J0` term,
- the tensorial pairwise branch provides the `J^2` and `J^4` feeds,
- and the minimal effective linear model keeps these two layers
  explicitly separated.

For `HCCD`, where both a local benchmark calculation and NIST `q_ν`
values are available, the current branch gives:

- pure rotational quartic scalar:
  - `D = 0.0251590 MHz`
- pair `Pi(1)`:
  - `omega_t = 467.0436 cm^-1`
  - `q_e^(0) = 127.107822 MHz`
  - `q_e^(W) = 127.107836 MHz`
  - `q_v~2B/nu_anh = 100.095585 MHz` using `nu_anh = 593.082 cm^-1`
- pair `Pi(2)`:
  - `omega_t = 697.9785 cm^-1`
  - `q_e^(0) = 85.052605 MHz`
  - `q_e^(W) = 85.052614 MHz`
  - `q_v~2B/nu_anh = 84.046604 MHz` using `nu_anh = 706.333 cm^-1`

against the experimental HCCD values previously collected from NIST:

- `nu4: q_ν = 132.993 MHz`
- `nu5: q_ν = 105.702 MHz`

So the present closure is now precise enough to say:

- the branch already predicts the correct spectroscopic scale,
- `q_e^(W)` is the correct equilibrium-level `J0` driver,
- a first state-specific estimate can be built from the anharmonic
  fundamentals,
- but the final benchmark-level mapping to observed `q_ν` values is not
  yet quantitatively closed for all degenerate modes.

## What is still missing before the block can be called definitive

1. A derivation of the full operator set in the degenerate bending subspace,
   not only the `O_t` diagnostic splitting operator.
2. A proper matching between tensor objects and the full l-type Hamiltonian,
   rather than the current scale-based diagnostic construction.
3. Validation on a real linear benchmark with known l-type constants.

## Files touched in the current step

- [distortion_workflow.py](/Users/vincenzobarone/centrifugal/distortion_workflow.py)
- [test_linear_ltype_terms.py](/Users/vincenzobarone/centrifugal/test_linear_ltype_terms.py)
- [gaussian_vpt_parser.py](/Users/vincenzobarone/centrifugal/gaussian_vpt_parser.py)
- [ceditt_gui.py](/Users/vincenzobarone/centrifugal/ceditt_gui.py)

## Recommended next step

The full next-step roadmap is now fixed in
[LINEAR_CLOSURE_ROADMAP.md](/Users/vincenzobarone/centrifugal/LINEAR_CLOSURE_ROADMAP.md).
The immediate target remains the same:

- match the current pairwise operators to the conventional
  linear-molecule `l`-type Hamiltonian constants of a real benchmark.
