#!/usr/bin/env python3
"""Exact requirements for extending the current octic derivation to the degenerate linear branch."""

from __future__ import annotations

from scripts.linear_pairwise_octic_projection import linear_pairwise_octic_projection


def paper1_octic_degenerate_extension_requirements() -> dict[str, object]:
    pairwise = linear_pairwise_octic_projection()
    return {
        "current_completed_branch": "rotational_scalar_X0_only",
        "required_extension": "retain vibrational doublet carrier through contact transformation",
        "target_pairwise_basis": pairwise["basis"],
        "target_pairwise_labels": pairwise["labels"],
        "target_observable": "q_H",
        "statement": (
            "To complete the exact linear octic derivation from our equations, the "
            "contact-transformed H08 block must be carried not only on the scalar "
            "rotational basis X0, but also on the tensor-product basis {X_l, J^2 X_l, "
            "(J^2)^2 X_l} of the degenerate vibrational doublet. The missing octic "
            "observable is q_H."
        ),
    }


if __name__ == "__main__":
    print(paper1_octic_degenerate_extension_requirements())
