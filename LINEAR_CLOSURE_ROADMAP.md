# Linear Closure Roadmap

## Current status

The current CeDiTT linear-molecule branch is already coherent for the
pure rotational sector:

- exact `linear` rotor classification,
- direct projection of quartic and sextic pure-rotational scalars
  `D` and `H`,
- mode-symmetry assignment for `C_{\infty v}` / `D_{\infty h}`,
- pairwise near-degenerate bending detection,
- minimal pairwise `l`-type effective layer
  `q_t`, `q_t^J`, `q_t^H`,
- stable pair labels for real cases such as `Pi_g(1)` and `Pi_u(1)`.

This means that the linear branch is already usable in the app for:

- pure rotational quartic/sextic diagnostics,
- symmetry-aware mode labeling,
- first pairwise `l`-type scale analysis.

## What is still missing before linears are truly on the same level as asymmetric/symmetric tops

### 1. Full linear effective-operator set

The present layer reports only the minimal pairwise splitting model

- `O_t` in the real-doublet basis,
- `X_l` in the circular-doublet basis,
- plus the rotational prefactors `q_t`, `q_t^J`, `q_t^H`.

This is not yet the full linear-molecule effective Hamiltonian in the
degenerate bending subspace. The missing step is to derive the complete
operator basis used in conventional linear-molecule spectroscopy and to
map the tensor workflow onto it.

### 2. Sextic closure with degenerate bendings

For asymmetric and symmetric tops, the standard sextic hierarchy can be
discussed directly in terms of:

- geometric contribution,
- two-index cubic contribution,
- three-index cubic contribution.

For linear molecules this decomposition is not yet closed at the same
level because degenerate bendings introduce additional `l`-type operator
structure. The next real theoretical step is therefore:

- identify which parts of the standard sextic tensor survive as pure
  rotational scalars,
- identify which parts feed the degenerate-bending `l`-type sector,
- separate these two branches explicitly.

### 3. Benchmark-level conventional mapping

The current `q_t`, `q_t^J`, `q_t^H` layer is already operational, but it
still needs a clean comparison against a conventional linear-molecule
benchmark with literature `l`-type constants.

This is the final step needed to claim full closure:

- derive the conventional constant mapping,
- test it on a real benchmark,
- expose the mapped constants in the GUI and the reporting layer.

## Recommended work order

1. Freeze the current pairwise layer as the stable minimal baseline.
2. Derive the full operator basis in the degenerate-bending subspace.
3. Separate the pure rotational sextic scalar branch from the `l`-type
   sextic branch.
4. Map the resulting coefficients onto conventional linear-molecule
   constants and validate them on a real benchmark.

## Practical conclusion

The linear branch is no longer missing basic functionality. The remaining
work is now focused and theory-facing:

- full `l`-type operator closure,
- sextic closure with degenerate bendings,
- conventional benchmark mapping.
