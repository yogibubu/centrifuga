#!/usr/bin/env python3
"""Dimension match between the S07 nonorthorhombic sector and H08 complement.

For Eq. (99), the role of S07 is to contribute through i[S07, H02].
Since H02 is orthorhombic and quadratic, the orthorhombic parity class is
preserved by multiplication with its coefficients, while the operator degree is
raised by one commutator level in the source hierarchy from degree 7 to the
degree-8 target sector.

The key finite-dimensional fact is:

    nonorthorhombic degree-7 monomials: 30
    nonorthorhombic degree-8 monomials: 30

So the S07 ansatz has exactly the right dimension to eliminate the full
nonorthorhombic complement of the octic target.
"""

from __future__ import annotations

from scripts.orthorhombic_parity_decomposition import paper1_sector_count_summary


def s07_h08_dimension_match() -> dict[str, object]:
    out = paper1_sector_count_summary()
    s07_nonorth = out["S07"]["nonorthorhombic_count"]
    h08_nonorth = out["H08"]["nonorthorhombic_count"]
    return {
        "S07_total": out["S07"]["total_count"],
        "S07_orthorhombic": out["S07"]["orthorhombic_count"],
        "S07_nonorthorhombic": s07_nonorth,
        "H08_total": out["H08"]["total_count"],
        "H08_orthorhombic": out["H08"]["orthorhombic_count"],
        "H08_nonorthorhombic": h08_nonorth,
        "dimension_match": s07_nonorth == h08_nonorth == 30,
        "consequence": (
            "The nonorthorhombic S07 ansatz has exactly the right dimension to "
            "eliminate the full nonorthorhombic complement of the octic target."
        ),
    }


def main() -> None:
    print(s07_h08_dimension_match())


if __name__ == "__main__":
    main()
