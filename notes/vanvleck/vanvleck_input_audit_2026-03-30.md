# Van Vleck Input Audit

## Verdict

The current BCH engine is not the limiting issue. The Hamiltonian fed into it is still structurally incomplete for the physical octic problem.

## What Watson--Aliev requires

The printed octic hierarchy in `vanvleck_octic_hierarchy.py` is

\[
\tilde H_{08}
=
H_{08}
- \frac{i}{6}[S03,[S03,[S03,H02]]]
- \frac{1}{2}[S03,[S03,H04]]
+ i[S03,H06]
+ i[S05,H04]
- [S05,[S03,H02]]
+ i[S07,H02].
\]

So the hierarchy contains:
- an explicit bare `H08` block
- completion terms built from `H02`, `H04`, `H06`, `S03`, `S05`, `S07`

## What the current scans actually use

Low-order scans in `derive_watson_octic_from_loworder_vanvleck.py` use only:
- `H12`
- `H22`
- `H30`
- `H40`

Targeted scans in `run_targeted_octic_sectors.py` extend this to:
- `H32`
- `H50`
- `H60`

but still do **not** include:
- `H08` bare

Moreover, these scans use coefficient-collapsed generic symbolic blocks, not the physical Watson--Aliev content of the octic source.

## Consequence

The statement

\[
J^8 = 0
\]

for all scans up to perturbative order 4 only proves that the **current toy/collapsed input family** does not generate a surviving octic term.

It does **not** prove that the full Watson--Aliev Hamiltonian would vanish under BCH.

## Immediate next step

Do not run more scans on the same input family.

Instead:
1. build the explicit bare `H08` source block
2. audit the physical content of `H04` and `H06`
3. rerun BCH with the real hierarchy inputs, not only collapsed class blocks
