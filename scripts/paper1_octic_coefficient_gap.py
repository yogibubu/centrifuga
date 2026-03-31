#!/usr/bin/env python3
"""Exact status of coefficient availability for H06 versus H08 in Paper 1.

Watson--Aliev provide:
1. explicit orthorhombic sextic coefficients for H~6 via Eqs. (96)-(97);
2. only a source expression plus Eq. (99) completion for H~8;
3. an explicit statement that the coefficient formula for H~8 including
   lower-order nonorthorhombic elimination had not been obtained.

This resolves the apparent degree mismatch of the visible H08 source line:
the printed one-line Table V formula is not yet the final reduced commuting
octic Hamiltonian coefficient form.
"""

from __future__ import annotations


def paper1_h06_h08_coefficient_status() -> dict[str, object]:
    return {
        "H06_status": {
            "final_orthorhombic_coefficients_available": True,
            "equations": ("96", "97"),
            "objects": ("W_xxx", "W_xxy", "W_xyy", "(bx)'_y"),
        },
        "H08_status": {
            "final_orthorhombic_coefficients_available": False,
            "equation": "99",
            "available_objects": ("H08_source", "S03", "S05", "S07", "H02", "H04", "H06"),
            "paper1_statement": (
                "It would be desirable to derive a formula for the coefficients in "
                "H~8 that includes the effects of the elimination of the lower-order "
                "nonorthorhombic terms ... This formula ... has not been obtained yet."
            ),
        },
        "consequence": (
            "The visible one-line H08 expression is a pre-reduction source line, not "
            "the final reduced pure degree-eight coefficient form."
        ),
    }


def main() -> None:
    print(paper1_h06_h08_coefficient_status())


if __name__ == "__main__":
    main()
