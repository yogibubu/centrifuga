#!/usr/bin/env python3
"""Exact support verdict for the principal optical channel [S05, H04]."""

from __future__ import annotations


def paper1_octic_s05h04_channel() -> dict[str, object]:
    return {
        "source_block": "[S05,H04]",
        "principal_degree": 8,
        "survives_directly_in_orth_coefficients": True,
        "acts_as_elimination_only": False,
        "rotational_derivative_content": ("mu1",),
        "potential_content": ("phi3", "phi4"),
        "harmonic_content": ("omega",),
        "channel_statement": (
            "At principal-symbol level the block [S05,H04] is a direct optical "
            "source term. Since H04 carries the quartic rotational mu1 content "
            "and S05 is the quintic pure-vibrational generator built from the "
            "cubic and quartic potentials, this block belongs to the mixed "
            "mu1*phi3*phi4 channel and contributes directly to the orthorhombic "
            "octic coefficients, hence also to the non-linear optical constants "
            "and to the linear limit L."
        ),
    }


def main() -> None:
    print(paper1_octic_s05h04_channel())


if __name__ == "__main__":
    main()
