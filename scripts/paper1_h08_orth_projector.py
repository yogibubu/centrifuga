#!/usr/bin/env python3
"""Explicit orth/nonorth projectors for the degree-8 octic source space."""

from __future__ import annotations

import sympy as sp

from scripts.orthorhombic_parity_decomposition import (
    homogeneous_degree_monomials,
    is_orthorhombic_monomial,
)


def degree8_full_basis() -> tuple[tuple[int, int, int], ...]:
    return tuple(homogeneous_degree_monomials(8))


def degree8_orth_basis() -> tuple[tuple[int, int, int], ...]:
    return tuple(m for m in degree8_full_basis() if is_orthorhombic_monomial(*m))


def degree8_nonorth_basis() -> tuple[tuple[int, int, int], ...]:
    return tuple(m for m in degree8_full_basis() if not is_orthorhombic_monomial(*m))


def degree8_selection_projectors() -> dict[str, object]:
    full = degree8_full_basis()
    orth = degree8_orth_basis()
    nonorth = degree8_nonorth_basis()
    idx = {m: i for i, m in enumerate(full)}

    P_orth = sp.zeros(len(orth), len(full))
    for i, m in enumerate(orth):
        P_orth[i, idx[m]] = 1

    P_nonorth = sp.zeros(len(nonorth), len(full))
    for i, m in enumerate(nonorth):
        P_nonorth[i, idx[m]] = 1

    return {
        "full_basis": full,
        "orth_basis": orth,
        "nonorth_basis": nonorth,
        "P_orth": P_orth,
        "P_nonorth": P_nonorth,
        "shapes": {
            "P_orth": P_orth.shape,
            "P_nonorth": P_nonorth.shape,
        },
    }


def principal_symbol_reduction_summary() -> dict[str, object]:
    proj = degree8_selection_projectors()
    return {
        "full_dimension": len(proj["full_basis"]),
        "orth_dimension": len(proj["orth_basis"]),
        "nonorth_dimension": len(proj["nonorth_basis"]),
        "projector_shapes": proj["shapes"],
        "principal_symbol_statement": (
            "At principal-symbol level, elimination by i[S07,H02] acts only on "
            "the 30-dimensional nonorthorhombic complement. The 15 orthorhombic "
            "degree-8 coefficients are therefore the direct orth projection of "
            "the source once the nonorthorhombic component has been canceled."
        ),
    }


def main() -> None:
    print(principal_symbol_reduction_summary())


if __name__ == "__main__":
    main()
