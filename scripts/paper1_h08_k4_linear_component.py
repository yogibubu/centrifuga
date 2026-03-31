#!/usr/bin/env python3
"""Linear c080 component of the visible quartic H08 source."""

from __future__ import annotations

import sympy as sp


def paper1_h08_k4_linear_component() -> dict[str, object]:
    k4 = sp.Symbol("k'_{klmn}")
    Ckyy = sp.Symbol("C_k^{yy}")
    Clyy = sp.Symbol("C_l^{yy}")
    Cmyy = sp.Symbol("C_m^{yy}")
    Cnyy = sp.Symbol("C_n^{yy}")

    c080 = sp.Rational(1, 24) * k4 * Ckyy * Clyy * Cmyy * Cnyy

    return {
        "component": "c_080",
        "source_block": "H08_k4",
        "formula": c080,
        "statement": (
            "The linear optical component c080 of the visible quartic H08 source "
            "is obtained by selecting the Jy^8 monomial in the commuting degree-8 "
            "product, hence only the yy components of the four first-order "
            "rotational-derivative tensors survive."
        ),
    }


def main() -> None:
    print("c080(H08_k4) =", paper1_h08_k4_linear_component()["formula"])


if __name__ == "__main__":
    main()
