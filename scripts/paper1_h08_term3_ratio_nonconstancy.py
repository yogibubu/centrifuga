#!/usr/bin/env python3
"""Generic non-constancy of C_k^2 / r_k in the known linear branch."""

from __future__ import annotations

import sympy as sp


def paper1_h08_term3_ratio_nonconstancy() -> dict[str, sp.Expr]:
    B, D, w, C, cubic = sp.symbols("B D omega C cubic", nonzero=True)

    r = sp.expand(4 * w * C * (D / B - B**2 / w**2) + cubic)
    ratio = sp.simplify(C**2 / r)
    dratio_dC = sp.simplify(sp.diff(ratio, C))
    dratio_dw = sp.simplify(sp.diff(ratio, w))

    ratio_no_cubic = sp.simplify(ratio.subs({cubic: 0}))
    dratio_no_cubic_dC = sp.simplify(sp.diff(ratio_no_cubic, C))
    dratio_no_cubic_dw = sp.simplify(sp.diff(ratio_no_cubic, w))

    return {
        "r_k": r,
        "ratio": ratio,
        "dratio_dC": dratio_dC,
        "dratio_dw": dratio_dw,
        "ratio_no_cubic": ratio_no_cubic,
        "dratio_no_cubic_dC": dratio_no_cubic_dC,
        "dratio_no_cubic_dw": dratio_no_cubic_dw,
        "statement": (
            "For the known linear branch r_k = 4 omega_k C_k (D_J/B - B^2/omega_k^2) + cubic_k, "
            "the ratio C_k^2 / r_k is generically not mode-independent. This remains true even "
            "if the cubic correction is set to zero."
        ),
    }


def main() -> None:
    out = paper1_h08_term3_ratio_nonconstancy()
    for key, val in out.items():
        print(key, "=", val)


if __name__ == "__main__":
    main()
