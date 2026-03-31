#!/usr/bin/env python3
"""Exact linear K=0 projection of the visible quartic H08 source."""

from __future__ import annotations

import sympy as sp

try:
    from scripts.octic_linear_k0_quantum_projection import project_x_axis_linear_polynomial
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.octic_linear_k0_quantum_projection import project_x_axis_linear_polynomial


def paper1_h08_k4_linear_projection() -> dict[str, object]:
    k4 = sp.Symbol("k'_{klmn}")

    Ckyy, Ck_yz, Ckzz = sp.symbols("C_k^{yy} C_k^{yz} C_k^{zz}")
    Clyy, Cl_yz, Clzz = sp.symbols("C_l^{yy} C_l^{yz} C_l^{zz}")
    Cmyy, Cm_yz, Cmzz = sp.symbols("C_m^{yy} C_m^{yz} C_m^{zz}")
    Cnyy, Cn_yz, Cnzz = sp.symbols("C_n^{yy} C_n^{yz} C_n^{zz}")

    Y, Z = sp.symbols("Y Z")
    qk = Ckyy * Y**2 + 2 * Ck_yz * Y * Z + Ckzz * Z**2
    ql = Clyy * Y**2 + 2 * Cl_yz * Y * Z + Clzz * Z**2
    qm = Cmyy * Y**2 + 2 * Cm_yz * Y * Z + Cmzz * Z**2
    qn = Cnyy * Y**2 + 2 * Cn_yz * Y * Z + Cnzz * Z**2

    poly = sp.expand(sp.Rational(1, 24) * k4 * qk * ql * qm * qn)
    p = sp.Poly(poly, Y, Z)

    plane_coeffs = {
        (0, 8, 0): sp.expand(p.coeff_monomial(Y**8)),
        (0, 6, 2): sp.expand(p.coeff_monomial(Y**6 * Z**2)),
        (0, 4, 4): sp.expand(p.coeff_monomial(Y**4 * Z**4)),
        (0, 2, 6): sp.expand(p.coeff_monomial(Y**2 * Z**6)),
        (0, 0, 8): sp.expand(p.coeff_monomial(Z**8)),
    }

    x_poly = project_x_axis_linear_polynomial(plane_coeffs)
    return {
        "source_block": "H08_k4",
        "plane_coeffs": plane_coeffs,
        "linear_polynomial": x_poly,
        "X4": sp.expand(x_poly["X4"]),
        "statement": (
            "The visible quartic H08 source contributes to the final linear "
            "scalar through the exact K=0 projection of its five plane "
            "coefficients, not through the raw c080 coefficient alone."
        ),
    }


def main() -> None:
    out = paper1_h08_k4_linear_projection()
    print("X4(H08_k4) =", out["X4"])


if __name__ == "__main__":
    main()
