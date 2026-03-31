#!/usr/bin/env python3
"""Project explicit octic channel coefficients onto the linear K=0 polynomial."""

from __future__ import annotations

import sympy as sp

try:
    from scripts.octic_linear_k0_quantum_projection import project_x_axis_linear_polynomial
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
    from scripts.octic_linear_k0_quantum_projection import project_x_axis_linear_polynomial
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


def _project_dict(d: dict[tuple[int, int, int], sp.Expr]) -> dict[str, sp.Expr]:
    plane = {
        mon: coeff
        for mon, coeff in d.items()
        if mon in {(0, 8, 0), (0, 6, 2), (0, 4, 4), (0, 2, 6), (0, 0, 8)}
    }
    return project_x_axis_linear_polynomial(plane)


def project_all_channels_to_linear_polynomial() -> dict[str, dict[str, sp.Expr]]:
    return {
        "S03S03S03H02": _project_dict(paper1_octic_s03s03s03h02_principal_tensor()["orth_coeffs_S111"]),
        "S03S03H04": _project_dict(paper1_octic_s03s03h04_principal_tensor()["orth_coeffs_S111"]),
        "S03H06": _project_dict(paper1_octic_s03h06_principal_tensor()["orth_coeffs_S111"]),
        "S05H04": _project_dict(paper1_octic_s05h04_principal_tensor()["orth_coeffs"]),
        "S05S03H02": _project_dict(paper1_octic_s05s03h02_principal_tensor()["orth_coeffs_S111"]),
    }


def main() -> None:
    out = project_all_channels_to_linear_polynomial()
    for name, coeffs in out.items():
        print(name, coeffs)


if __name__ == "__main__":
    main()
