# Two-Paper Positioning Note

This note fixes the present manuscript split and the refocusing of the material
already written in the repository.

## General decision

The existing drafts should not be abandoned, but refocused into two distinct
papers with different scopes.

The split is not:

- paper 1 = old CeDiTT as originally conceived,
- paper 2 = everything else.

The split is instead:

- paper 1 = quartic standard + sextic standard, written in a unified tensorial
  language;
- paper 2 = the genuinely new quartic contributions beyond the standard level,
  starting with `H22` and then the heavier VPT4 channels.

## Paper 1

### Scope

Paper 1 should treat only:

- standard quartics,
- standard sextics,
- representation/reduction transport,
- pseudoinverse lifting and gauge fixing.

### Central claim

In this standard sector, one can eliminate the second derivatives of the
inertia tensor as primitive objects.

The standard sextic structure can be expressed using the same basic tensorial
ingredients as the standard quartic structure, plus the cubic force constants.

So the basic language of paper 1 is:

- `mu1 = d(I^-1)/dQ`,
- Coriolis couplings,
- cubic constants `phi3`,
- pseudoinverse lifting to the higher-rank tensor representation,
- gauge fixing through the Moore-Penrose inverse.

### Consequence

This means that the refocused CeDiTT work is not a paper about a special
quartic tensor alone.

It becomes a paper about a unified tensorial framework for:

- standard quartic constants,
- standard sextic constants,
- and transport between representations/reductions.

### What does not belong here

Paper 1 should not be burdened with:

- `H22` as a new intrinsic quartic object,
- `H21H30`,
- `H30H30`,
- full VPT4 quartic closure.

`H22` may appear only as the boundary marker: the first place where the
standard tensorial language is no longer sufficient by itself.

## Paper 2

### Scope

Paper 2 begins where the standard sector ends.

Its first genuine step is:

- `H22`.

### Central claim

`H22` is the first quartic contribution that forces the appearance of a new
intrinsic tensorial object beyond the first-level tensor `mu1`.

In current notation:

- the bilinear part is generated from `mu1`,
- the genuinely new content is the intrinsic tensor `N`.

So paper 2 starts from the statement that `H22` is the first place where a new
object is required.

### Further scope

After `H22`, paper 2 can naturally continue to:

- `H21H30`,
- `H30H30`,
- and the full VPT4 quartic structure.

## Practical refocusing of existing drafts

The current written material should therefore be redistributed as follows.

### Material to retain and reshape for paper 1

- pseudoinverse lifting / gauge-fixing logic,
- representation and reduction transport,
- standard quartic tensor language,
- standard sextic tensor language,
- invariant-subspace / transport arguments that belong to the standard sector.

### Material to move to paper 2

- `H22` as a new tensorial level,
- any discussion where `d2I` or `d2(I^-1)` appears as genuinely new content,
- full VPT4 quartic channels,
- heavier anharmonic quartic contributions.

## One-line summary

Paper 1 says:

> standard quartics and standard sextics can be written in one common tensorial
> language, based on the same primitive objects plus cubic force constants.

Paper 2 says:

> `H22` is the first quartic contribution that forces a genuinely new tensorial
> object, and from there the full VPT4 quartic problem begins.

## Perturbative partition convention

In both works, the standard perturbative partition should be stated explicitly:

- `H03`, `H12`, and `H30` are first-order terms;
- `H40`, `H22`, `H13`, `H31`, and `H04` are second-order terms.

So the labels `Hnm` identify the vibrational/rotational bidegree of each
block, but do not by themselves determine the perturbative order. The latter
depends on the chosen Hamiltonian partition.

This is the standard Watson/VPT2-style partition used in the present work, but
it is not universal. In alternative formulations, including state-summation
schemes that are also often labeled as VPT2, the perturbative bookkeeping may
be organized differently, and blocks such as `H03` and `H04` may effectively
enter at the same order.

## Channel decomposition

The decomposition into channels such as

- `H12H12`,
- `H22`,
- `H12H30`,
- `H30H30`,
- `H04`,

should be understood as exact only after fixing:

- the perturbative partition,
- the truncation order,
- and the adopted contact-transformation/effective-Hamiltonian construction.

Within that fixed framework, the total contribution at the chosen order is the
sum of the retained channels. This is not, however, a partition-independent
statement.

It is also important to remember that these channel coefficients are not read
off naively from the symbolic labels alone. Once the exponential contact
transformation is expanded, the Baker--Campbell--Hausdorff series produces the
usual combinatorial prefactors and factorial denominators. So the exact
weighting of each channel is the one induced by the chosen perturbative
construction, not just by the formal block labels.

For this reason, whenever the channel decomposition is used explicitly in the
manuscripts, the channels should be presented together with their actual BCH /
contact-transformation prefactors, not only by symbolic labels. The derivation
of those coefficients belongs in the technical appendices, and the text should
include explicit references to the Van Vleck contact transformation and to the
Baker--Campbell--Hausdorff expansion on which the channel weighting rests.
