#!/usr/bin/env python3
"""Final orthorhombic degree-8 coefficients from the six direct octic channels."""

from __future__ import annotations

import sympy as sp

try:
    from scripts.paper1_h08_orth_projector import degree8_orth_basis
    from scripts.paper1_octic_s03s03s03h02_principal_tensor import (
        paper1_octic_s03s03s03h02_principal_tensor,
    )
    from scripts.paper1_octic_s03s03h04_principal_tensor import (
        paper1_octic_s03s03h04_principal_tensor,
    )
    from scripts.paper1_octic_s03h06_principal_tensor import paper1_octic_s03h06_principal_tensor
    from scripts.paper1_octic_s05h04_principal_tensor import paper1_octic_s05h04_principal_tensor
    from scripts.paper1_octic_s05s03h02_principal_tensor import (
        paper1_octic_s05s03h02_principal_tensor,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.paper1_h08_orth_projector import degree8_orth_basis
    from scripts.paper1_octic_s03s03s03h02_principal_tensor import (
        paper1_octic_s03s03s03h02_principal_tensor,
    )
    from scripts.paper1_octic_s03s03h04_principal_tensor import (
        paper1_octic_s03s03h04_principal_tensor,
    )
    from scripts.paper1_octic_s03h06_principal_tensor import paper1_octic_s03h06_principal_tensor
    from scripts.paper1_octic_s05h04_principal_tensor import paper1_octic_s05h04_principal_tensor
    from scripts.paper1_octic_s05s03h02_principal_tensor import (
        paper1_octic_s05s03h02_principal_tensor,
    )


def paper1_octic_final_orth_coefficients() -> dict[str, object]:
    orth_basis = degree8_orth_basis()

    # Visible quartic H08 source left as symbolic orth coefficients until the
    # fully expanded cartesian tensor is written out component by component.
    h08_k4 = {
        mon: sp.Symbol(f"h08k4_{mon[0]}{mon[1]}{mon[2]}")
        for mon in orth_basis
    }

    s03s03s03h02 = paper1_octic_s03s03s03h02_principal_tensor()["orth_coeffs_S111"]
    s03s03h04 = paper1_octic_s03s03h04_principal_tensor()["orth_coeffs_S111"]
    s03h06 = paper1_octic_s03h06_principal_tensor()["orth_coeffs_S111"]
    s05h04 = paper1_octic_s05h04_principal_tensor()["orth_coeffs"]
    s05s03h02 = paper1_octic_s05s03h02_principal_tensor()["orth_coeffs_S111"]

    total = {}
    for mon in orth_basis:
        total[mon] = sp.expand(
            h08_k4[mon]
            - sp.I / 6 * s03s03s03h02[mon]
            - sp.Rational(1, 2) * s03s03h04[mon]
            + sp.I * s03h06[mon]
            + sp.I * s05h04[mon]
            - s05s03h02[mon]
        )

    return {
        "orth_basis": orth_basis,
        "h08_k4_symbols": h08_k4,
        "term_prefactors": {
            "H08_k4": sp.Integer(1),
            "S03S03S03H02": -sp.I / 6,
            "S03S03H04": -sp.Rational(1, 2),
            "S03H06": sp.I,
            "S05H04": sp.I,
            "S05S03H02": -sp.Integer(1),
        },
        "total_orth_coeffs": total,
        "linear_constant_L": sp.expand(total[(0, 8, 0)]),
        "linear_constant_principal_statement": (
            "At principal-symbol level the linear optical constant is carried by "
            "the c080 component of the visible quartic H08 source."
        ),
    }


def main() -> None:
    out = paper1_octic_final_orth_coefficients()
    print("L =", out["linear_constant_L"])


if __name__ == "__main__":
    main()
