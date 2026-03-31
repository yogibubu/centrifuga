#!/usr/bin/env python3
"""CeDiTT4-style axial decomposition of the six direct octic channels."""

from __future__ import annotations

import sympy as sp

from scripts.octic_tensorial_projection import (
    axial_octic_from_cartesian_matrix,
    axial_octic_labels,
    linear_from_axial_vector,
)
from scripts.paper1_h08_orth_projector import degree8_orth_basis
from scripts.paper1_octic_final_orth_coefficients import paper1_octic_final_orth_coefficients


def _orth_to_cartesian_vector(coeffs: dict[tuple[int, int, int], sp.Expr]) -> sp.Matrix:
    basis = degree8_orth_basis()
    return sp.Matrix([coeffs[mon] for mon in basis])


def paper1_octic_axial_decomposition() -> dict[str, object]:
    out = paper1_octic_final_orth_coefficients()
    pref = out["term_prefactors"]
    P = axial_octic_from_cartesian_matrix("a")
    xlabels = axial_octic_labels()
    terms = {}

    source_terms = {
        "H08_k4": out["h08_k4_symbols"],
        "S03S03S03H02": {
            mon: pref["S03S03S03H02"]
            * __import__(
                "scripts.paper1_octic_s03s03s03h02_principal_tensor",
                fromlist=["paper1_octic_s03s03s03h02_principal_tensor"],
            ).paper1_octic_s03s03s03h02_principal_tensor()["orth_coeffs_S111"][mon]
            for mon in degree8_orth_basis()
        },
        "S03S03H04": {
            mon: pref["S03S03H04"]
            * __import__(
                "scripts.paper1_octic_s03s03h04_principal_tensor",
                fromlist=["paper1_octic_s03s03h04_principal_tensor"],
            ).paper1_octic_s03s03h04_principal_tensor()["orth_coeffs_S111"][mon]
            for mon in degree8_orth_basis()
        },
        "S03H06": {
            mon: pref["S03H06"]
            * __import__(
                "scripts.paper1_octic_s03h06_principal_tensor",
                fromlist=["paper1_octic_s03h06_principal_tensor"],
            ).paper1_octic_s03h06_principal_tensor()["orth_coeffs_S111"][mon]
            for mon in degree8_orth_basis()
        },
        "S05H04": {
            mon: pref["S05H04"]
            * __import__(
                "scripts.paper1_octic_s05h04_principal_tensor",
                fromlist=["paper1_octic_s05h04_principal_tensor"],
            ).paper1_octic_s05h04_principal_tensor()["orth_coeffs"][mon]
            for mon in degree8_orth_basis()
        },
        "S05S03H02": {
            mon: pref["S05S03H02"]
            * __import__(
                "scripts.paper1_octic_s05s03h02_principal_tensor",
                fromlist=["paper1_octic_s05s03h02_principal_tensor"],
            ).paper1_octic_s05s03h02_principal_tensor()["orth_coeffs_S111"][mon]
            for mon in degree8_orth_basis()
        },
    }

    for name, coeffs in source_terms.items():
        xvec = sp.simplify(P * _orth_to_cartesian_vector(coeffs))
        axial = {xlabel: sp.expand(xvec[i]) for i, xlabel in enumerate(xlabels)}
        terms[name] = {
            "axial": axial,
            "L": sp.expand((linear_from_axial_vector() * xvec)[0]),
        }

    total_x = {
        xlabel: sp.expand(sum(terms[name]["axial"][xlabel] for name in terms))
        for xlabel in xlabels
    }
    return {
        "termwise": terms,
        "total_axial": total_x,
        "total_L": sp.expand(total_x["X0"]),
        "statement": (
            "All six direct octic channels are now organized directly at the "
            "CeDiTT4 axial level X0..X4; the linear limit is obtained only at "
            "the end by taking L = X0."
        ),
    }


if __name__ == "__main__":
    out = paper1_octic_axial_decomposition()
    print(out["total_axial"])
