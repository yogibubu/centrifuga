#!/usr/bin/env python3
"""Exact support verdict for the principal optical channel [S03, H06]."""

from __future__ import annotations


def paper1_octic_s03h06_channel() -> dict[str, object]:
    return {
        "source_block": "[S03,H06]",
        "principal_degree": 8,
        "survives_directly_in_orth_coefficients": True,
        "acts_as_elimination_only": False,
        "rotational_derivative_content": ("mu1",),
        "potential_content": ("phi3", "phi3"),
        "harmonic_content": ("omega", "B"),
        "channel_statement": (
            "At principal-symbol level the block [S03,H06] is a direct optical "
            "source term. Because S03 is purely vibrational and H06 already "
            "carries the sextic mu1-phi3 content, this block belongs to the "
            "mu1*phi3^2 channel and contributes directly to the orthorhombic "
            "octic coefficients, hence also to the non-linear optical constants "
            "and to the linear limit L."
        ),
    }


def main() -> None:
    print(paper1_octic_s03h06_channel())


if __name__ == "__main__":
    main()
