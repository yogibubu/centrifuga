#!/usr/bin/env python3
"""Consistency constraints on the visible one-index prime object R'_k.

Input:
    1. printed statement: H08 depends linearly on quartic potential constants
       and quadratically on cubic potential constants;
    2. visible bare H08 line;
    3. already established dependencies
         R_k <- mu1
         R_l <- mu1
         R_m <- mu1
         R_klm <- mu1
         R'_lm <- (mu1, phi4, zeta, omega, B)

Conclusion:
    the visible one-index object R'_k cannot carry phi4 or phi3, otherwise
    the visible bare H08 line would generate forbidden phi4^2, phi3, or
    phi3*phi4 sectors. Therefore R'_k belongs to the mu1-only rotational
    algebra (up to harmonic scalar weights).
"""

from __future__ import annotations


def paper1_octic_rprimek_consistency_verdict() -> dict[str, object]:
    return {
        "printed_h08_potential_constraint": ("phi4_linear", "phi3_quadratic"),
        "visible_terms_using_R'_k": (
            "R'_k R_l R_m R'_lm",
            "R'_k R_l [R_k,R_l]",
            "(R'_k)^2",
        ),
        "established_inputs": {
            "R_k": ("mu1",),
            "R_l": ("mu1",),
            "R_m": ("mu1",),
            "R_klm": ("mu1",),
            "R'_lm": ("mu1", "phi4", "zeta", "omega", "B"),
        },
        "forbidden_if_R'_k_contains_phi4": (
            "phi4^2 from (R'_k)^2",
            "phi4^2 from R'_k R_l R_m R'_lm when R'_lm carries phi4",
        ),
        "forbidden_if_R'_k_contains_phi3": (
            "phi3-linear from (R'_k)^2 or R'_k R_l [R_k,R_l]",
            "phi3*phi4 from R'_k R_l R_m R'_lm when R'_lm carries phi4",
        ),
        "only_consistent_potential_support_for_R'_k": tuple(),
        "consistent_rotational_support_for_R'_k": ("mu1",),
        "consistent_harmonic_support_for_R'_k": ("omega", "B", "zeta"),
    }


def main() -> None:
    print(paper1_octic_rprimek_consistency_verdict())


if __name__ == "__main__":
    main()
