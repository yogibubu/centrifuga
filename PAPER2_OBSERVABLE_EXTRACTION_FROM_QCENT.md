# Paper 2 Observable Extraction from the Qcent VPT4 Equations

This note aligns Paper 2 with the common Paper 2 / Paper 4 strategy, but
respecting the fact that for Paper 2 the final equations already exist.

The source used here is:

- `/Users/vincenzobarone/Desktop/Qcent_VPT4.pdf`

whose central observable is the fourth-order quartic rotational tensor
`tau^(4)`.

## 1. Observable-first statement

For Paper 2 the relevant observable is already explicit:

`tau^(4) = tau(H12 H12) + tau(H22) + tau(H12 H30) + tau(H30 H30)`

with the isolated `H40` channel absent.

So, unlike the `gamma` branch, Paper 2 does not need a fresh observable
extraction from scratch. The observable has already been identified and the
final channel decomposition is already available.

The task is different:

- reinterpret the existing final equations in the common CeDiTT language;
- identify the standard sector inside the already derived formula;
- isolate the first controlled breakdown inside that same formula.

## 2. Direct parallel with Paper 4

The common method remains the same in spirit:

1. start from the physical observable;
2. identify its direct perturbative pieces;
3. reduce those pieces to the lowest tensorial sector possible;
4. isolate the first irreducible remainder.

For Paper 4 this means:

- extract `M_i` and `M_ij` from `mu^(eff)(v)`.

For Paper 2 this means:

- start from the already available channel decomposition of `tau^(4)` and ask
  which part belongs to the lowest CeDiTT tensor sector and which part does
  not.

## 3. Standard sector in the Qcent equations

The first key structural reading of the Qcent equations is:

- `tau(H12 H12)` is the standard first-level quartic sector;
- `tau(H12 H30)` is the first explicit anharmonic extension built on the same
  lower-level rovibrational structure;
- `tau(H30 H30)` is a higher anharmonic channel that may still be represented
  in a closed scaffold basis, but is no longer part of the minimal linear
  CeDiTT transport sector;
- `tau(H22)` is the first channel that points directly to the second geometric
  response and therefore to the split
  `mu2 = B(mu1) - N`.

This immediately suggests the decomposition

`tau^(4) = tau_std + delta_tau_anh + delta_tau_break + ...`

with:

- `tau_std = tau(H12 H12)`,
- `delta_tau_anh` collecting the standard explicit anharmonic channels as far
  as they remain reducible to lower-level closures,
- `delta_tau_break` the first term requiring genuinely new tensor content.

## 4. Why H22 remains the key breakdown channel

The Qcent equations confirm that `H22` is not merely “one more channel”.

It is special because:

- it is the channel in which the second inverse-inertia response appears
  directly;
- that response admits the split
  `mu2 = B(mu1) - N`;
- only the `N` part represents genuinely new intrinsic tensor content.

Therefore, even in the VPT4 equations already derived, the first controlled
breakdown should still be read as:

`delta_tau_break = delta_tau[B(mu1)] + delta_tau[N]`

with only the `N` contribution counted as irreducibly new.

This keeps Paper 2 fully parallel to the CeDiTT philosophy rather than turning
it into a mere list of channels.

## 5. Role of the other channels

The Qcent decomposition also clarifies the status of the remaining terms.

### H12 H12

- baseline first-level closure;
- fully inside the standard transport sector.

### H12 H30

- mixed anharmonic channel;
- not part of the purely linear CeDiTT baseline;
- but still naturally interpreted as an explicit anharmonic extension of the
  first-level structure rather than as a new geometric tensor level.

### H30 H30

- purely cubic-cubic anharmonic channel;
- structurally more complex than `H12 H30`;
- can be represented in a closed scaffold basis, but should still be kept
  distinct from the first controlled geometric breakdown.

This distinction matters: “anharmonic complexity” and “new geometric tensor
level” are not the same thing.

## 6. Recommended Paper 2 master formula

The best aligned summary for Paper 2 is therefore:

`tau^(4) = tau_std + tau_anh_std + delta_tau_break + tau_anh_high + ...`

where:

- `tau_std = tau(H12 H12)`;
- `tau_anh_std` collects the lowest explicit anharmonic channels that still
  dress the first-level structure;
- `delta_tau_break` is the first genuinely new geometric tensor contribution,
  carried by the `N` part of `H22`;
- `tau_anh_high` collects higher anharmonic channels such as the intrinsic
  cubic-cubic contribution that are important numerically but conceptually
  distinct from the first geometric breakdown.

This is the direct Paper 2 analogue of the Paper 4 working formula

`gamma = gamma_std + delta_gamma_break + ...`

with the difference that the channels are already available explicitly in the
quartic case.

## 7. Immediate consequence for writing Paper 2

Paper 2 should therefore not be presented as:

- “here are the VPT4 channels”.

It should be presented as:

1. the quartic observable `tau^(4)` is already known in final channel form;
2. the channel decomposition is reorganized into:
   - standard sector,
   - standard explicit anharmonic extension,
   - first controlled geometric breakdown,
   - higher anharmonic channels;
3. the key theoretical point is the split
   `mu2 = B(mu1) - N`
   inside `H22`;
4. this gives a true tensor hierarchy, not just a perturbative bookkeeping.

That is the right way to keep Paper 2 parallel to Paper 4 while respecting the
fact that the quartic VPT4 equations are already in hand.
