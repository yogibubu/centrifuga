#!/usr/bin/env python3
"""Exact support verdict for the principal optical channel [S03,[S03,[S03,H02]]]."""

from __future__ import annotations


def paper1_octic_s03s03s03h02_channel() -> dict[str, object]:
    return {
        "source_block": "[S03,[S03,[S03,H02]]]",
        "principal_degree": 8,
        "survives_directly_in_orth_coefficients": True,
        "acts_as_elimination_only": False,
        "rotational_derivative_content": tuple(),
        "potential_content": ("phi3", "phi3", "phi3"),
        "harmonic_content": ("B", "omega"),
        "channel_statement": (
            "At principal-symbol level the triple commutator "
            "[S03,[S03,[S03,H02]]] is a direct optical source term. Since H02 is "
            "purely harmonic rotational and S03 is the cubic pure-vibrational "
            "generator, this block belongs to the purely vibrational phi3^3 "
            "channel. It contributes directly to the orthorhombic octic "
            "coefficients, hence to the non-linear optical constants and to the "
            "linear limit L, without introducing additional rotational-derivative "
            "tensors."
        ),
    }


def main() -> None:
    print(paper1_octic_s03s03s03h02_channel())


if __name__ == "__main__":
    main()
