#!/usr/bin/env python3
"""Final linear X^4 coefficient from the six direct octic channels."""

from __future__ import annotations

import sympy as sp

try:
    from scripts.paper1_h08_k4_linear_projection import paper1_h08_k4_linear_projection
    from scripts.paper1_octic_channel_linear_projection import project_all_channels_to_linear_polynomial
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.paper1_h08_k4_linear_projection import paper1_h08_k4_linear_projection
    from scripts.paper1_octic_channel_linear_projection import project_all_channels_to_linear_polynomial


def paper1_octic_final_linear_x4() -> dict[str, object]:
    h08 = paper1_h08_k4_linear_projection()["linear_polynomial"]
    ch = project_all_channels_to_linear_polynomial()

    total = {
        key: sp.expand(
            h08[key]
            - sp.I / 6 * ch["S03S03S03H02"][key]
            - sp.Rational(1, 2) * ch["S03S03H04"][key]
            + sp.I * ch["S03H06"][key]
            + sp.I * ch["S05H04"][key]
            - ch["S05S03H02"][key]
        )
        for key in ("X0", "X1", "X2", "X3", "X4")
    }

    return {
        "term_prefactors": {
            "H08_k4": sp.Integer(1),
            "S03S03S03H02": -sp.I / 6,
            "S03S03H04": -sp.Rational(1, 2),
            "S03H06": sp.I,
            "S05H04": sp.I,
            "S05S03H02": -sp.Integer(1),
        },
        "H08_k4": h08,
        "channels": ch,
        "total_linear_polynomial": total,
        "X4_total": sp.expand(total["X4"]),
        "statement": (
            "The final linear octic contribution is obtained by summing the "
            "visible quartic H08 source and the five direct commutator channels "
            "after exact K=0 projection, with the Eq. (99) prefactors."
        ),
    }


def main() -> None:
    out = paper1_octic_final_linear_x4()
    print("X4_total =", out["X4_total"])


if __name__ == "__main__":
    main()
