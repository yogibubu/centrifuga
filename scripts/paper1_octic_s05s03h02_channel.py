#!/usr/bin/env python3
"""Exact support verdict for the principal optical channel [S05,[S03,H02]]."""

from __future__ import annotations


def paper1_octic_s05s03h02_channel() -> dict[str, object]:
    return {
        "source_block": "[S05,[S03,H02]]",
        "principal_degree": 8,
        "survives_directly_in_orth_coefficients": True,
        "acts_as_elimination_only": False,
        "rotational_derivative_content": tuple(),
        "potential_content": ("phi3", "phi3", "phi4"),
        "harmonic_content": ("B", "omega"),
        "channel_statement": (
            "At principal-symbol level the nested commutator [S05,[S03,H02]] is "
            "a direct optical source term. Since H02 is purely harmonic "
            "rotational, while S03 and S05 are pure-vibrational odd generators, "
            "this block belongs to the purely vibrational phi3^2*phi4 channel. "
            "It contributes directly to the orthorhombic octic coefficients, "
            "hence to the non-linear optical constants and to the linear limit L, "
            "without introducing additional rotational-derivative tensors."
        ),
    }


def main() -> None:
    print(paper1_octic_s05s03h02_channel())


if __name__ == "__main__":
    main()
