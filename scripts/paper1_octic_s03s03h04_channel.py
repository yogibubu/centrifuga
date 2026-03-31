#!/usr/bin/env python3
"""Exact support verdict for the principal optical channel [S03,[S03,H04]]."""

from __future__ import annotations


def paper1_octic_s03s03h04_channel() -> dict[str, object]:
    return {
        "source_block": "[S03,[S03,H04]]",
        "principal_degree": 8,
        "survives_directly_in_orth_coefficients": True,
        "acts_as_elimination_only": False,
        "rotational_derivative_content": ("mu1",),
        "potential_content": ("phi3", "phi3"),
        "harmonic_content": ("omega",),
        "channel_statement": (
            "At principal-symbol level the double commutator [S03,[S03,H04]] is "
            "a direct optical source term. Since H04 carries the quartic "
            "rotational mu1 content and each S03 contributes one cubic "
            "vibrational generator, this block belongs to the mu1*phi3^2 "
            "channel and contributes directly to the orthorhombic octic "
            "coefficients, hence also to the non-linear optical constants and "
            "to the linear limit L."
        ),
    }


def main() -> None:
    print(paper1_octic_s03s03h04_channel())


if __name__ == "__main__":
    main()
