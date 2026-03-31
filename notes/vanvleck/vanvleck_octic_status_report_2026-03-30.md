# Van Vleck Octic Status Report (2026-03-30)

## Notation

- `S03, S05, S07`: Watson--Aliev labels by total rotational degree in the source-level counting.
- `S^(1), S^(2), S^(3), ...`: perturbative BCH generators returned by the new Van Vleck engine.
- Working bridge:
  - `S^(1) <-> S03`
  - `S^(2) <-> S05`
  - `S^(3) <-> S07`

## What Is Already Closed

1. A generic BCH/Van Vleck engine exists in
   `/Users/vincenzobarone/centrifugal/vanvleck_bch_engine.py`.
2. Generic tensorial `CeDiTT4` block builders exist in
   `/Users/vincenzobarone/centrifugal/vanvleck_hamiltonian_builders.py`.
3. A perturbative input composer exists in
   `/Users/vincenzobarone/centrifugal/vanvleck_perturbative_input.py`.
4. A generic Van Vleck driver exists in
   `/Users/vincenzobarone/centrifugal/derive_watson_generic_vanvleck.py`.
5. The new stack reproduces the old quartic engine when run with the same reduction policy.
6. The octic hierarchy is encoded exactly in
   `/Users/vincenzobarone/centrifugal/vanvleck_octic_hierarchy.py`.

## First Exact Octic Results

1. In the low-order two-mode diagonal scan, `H12` alone produces an ordered `J^8` block at perturbative order 4.
2. That ordered `J^8` block vanishes after commuting projection.
3. `H22` alone produces no `J^8` block.
4. Therefore the first physically surviving low-order octic block is not `H12` alone and not `H22` alone.

## Immediate Next Task

Extract the BCH generators explicitly:

- `S^(1)`
- `S^(2)`
- `S^(3)`

grouped by source origin and rotational/vibrational degree, then compare their structure with the Watson--Aliev source labels `S03/S05/S07`.
