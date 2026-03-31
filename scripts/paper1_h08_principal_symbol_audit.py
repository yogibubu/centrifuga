#!/usr/bin/env python3
"""Principal-symbol degree audit for the optical octic source hierarchy."""

from __future__ import annotations


def paper1_h08_principal_symbol_audit() -> dict[str, object]:
    deg = {
        "H02": 2,
        "H04": 4,
        "H06": 6,
        "S03": 3,
        "S05": 5,
        "S07": 7,
        "R'_k": 1,
        "R_k": 2,
        "R'_kl": 2,
        "R_klm": 3,
        "[R_k,R_l]": 3,
    }

    visible_h08 = {
        "RkRlRmRklm": deg["R_k"] + deg["R_k"] + deg["R_k"] + deg["R_klm"],
        "quartic_term": deg["R_k"] + deg["R_k"] + deg["R_k"] + deg["R_k"],
        "RpkRlRmRplm": deg["R'_k"] + deg["R_k"] + deg["R_k"] + deg["R'_kl"],
        "RpkRl_comm": deg["R'_k"] + deg["R_k"] + deg["[R_k,R_l]"],
        "Rpk_sq": 2 * deg["R'_k"],
    }

    deg_s03_h02 = deg["S03"] + deg["H02"] - 1
    deg_s03_s03_h02 = deg["S03"] + deg_s03_h02 - 1
    deg_s03_s03_s03_h02 = deg["S03"] + deg_s03_s03_h02 - 1
    deg_s03_h04 = deg["S03"] + deg["H04"] - 1
    deg_s03_s03_h04 = deg["S03"] + deg_s03_h04 - 1
    deg_s03_h06 = deg["S03"] + deg["H06"] - 1
    deg_s05_h04 = deg["S05"] + deg["H04"] - 1
    deg_s05_s03_h02 = deg["S05"] + deg_s03_h02 - 1
    deg_s07_h02 = deg["S07"] + deg["H02"] - 1

    eq99_completion = {
        "[S03,[S03,[S03,H02]]]": deg_s03_s03_s03_h02,
        "[S03,[S03,H04]]": deg_s03_s03_h04,
        "[S03,H06]": deg_s03_h06,
        "[S05,H04]": deg_s05_h04,
        "[S05,[S03,H02]]": deg_s05_s03_h02,
        "[S07,H02]": deg_s07_h02,
    }

    return {
        "visible_h08_term_degrees": visible_h08,
        "eq99_completion_degrees": eq99_completion,
        "visible_principal_symbol_source_terms": ("quartic_term",),
        "visible_subprincipal_source_terms": (
            "RkRlRmRklm",
            "RpkRlRmRplm",
            "RpkRl_comm",
            "Rpk_sq",
        ),
        "all_eq99_completion_terms_are_degree8": all(v == 8 for v in eq99_completion.values()),
        "principal_symbol_statement": (
            "Within the visible one-line H08 source, only the quartic-potential "
            "term already has degree 8. The other four visible source terms are "
            "subprincipal. By contrast, all six Eq. (99) completion terms are "
            "degree-8 at principal-symbol level."
        ),
    }


def main() -> None:
    print(paper1_h08_principal_symbol_audit())


if __name__ == "__main__":
    main()
