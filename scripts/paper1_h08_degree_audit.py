#!/usr/bin/env python3
"""Operator-degree audit for the visible one-line H08 source.

The printed one-line Table V expression is *not* the final reduced orthorhombic
octic Hamiltonian. Watson--Aliev state explicitly after Eq. (99) that a formula
for the coefficients in the final reduced block ``H~08`` including elimination
of lower-order nonorthorhombic terms "has not been obtained yet".

Therefore this audit is used only to show that the visible source line mixes
different rotational degrees and must be interpreted as a pre-reduction source
expression, not as the final pure degree-eight commuting operator.
"""

from __future__ import annotations


def paper1_h08_operator_degrees() -> dict[str, object]:
    deg = {
        "R'_k": 1,
        "R_k": 2,
        "R_kl": 2,
        "R'_kl": 2,
        "R_klm": 3,
        "[R_k,R_l]": 3,
    }
    term_degrees = {
        "term1_RkRlRmRklm": deg["R_k"] + deg["R_k"] + deg["R_k"] + deg["R_klm"],
        "term2_RkRlRmRn_k4": deg["R_k"] + deg["R_k"] + deg["R_k"] + deg["R_k"],
        "term3_RpkRlRmRplm": deg["R'_k"] + deg["R_k"] + deg["R_k"] + deg["R'_kl"],
        "term4_RpkRl_comm": deg["R'_k"] + deg["R_k"] + deg["[R_k,R_l]"],
        "term5_Rpk_sq": 2 * deg["R'_k"],
    }
    return {
        "primitive_operator_degrees": deg,
        "visible_h08_term_degrees": term_degrees,
        "octic_target_degree_if_pure_rotational": 8,
        "interpretation": (
            "The visible one-line H08 expression is a pre-reduction source line; "
            "the final reduced coefficients belong to tilde H08 after Eq. (99), "
            "and Watson--Aliev explicitly state that the coefficient formula "
            "including lower-order nonorthorhombic elimination had not been "
            "obtained."
        ),
        "not_a_contradiction": True,
    }


def main() -> None:
    print(paper1_h08_operator_degrees())


if __name__ == "__main__":
    main()
