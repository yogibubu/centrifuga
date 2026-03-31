#!/usr/bin/env python3
"""Audit what the current derived octic program does and does not cover in the linear limit."""

from __future__ import annotations

from scripts.linear_degenerate_octic_basis import linear_degenerate_octic_basis


def paper1_octic_degenerate_scope_audit() -> dict[str, object]:
    deg = linear_degenerate_octic_basis()
    return {
        "current_derived_branch": "pure_rotational_scalar_only",
        "missing_branch": "degenerate_pairwise_linear_branch",
        "missing_basis": deg["basis"],
        "missing_labels": deg["labels"],
        "statement": (
            "The current paper1 octic derivation closes only the pure rotational "
            "linear scalar branch. To derive the degenerate linear observables from "
            "our equations, the contact-transformed octic block must be retained on "
            "the pairwise operator basis {X_l, J^2 X_l, (J^2)^2 X_l}. In particular, "
            "q_H cannot be recovered from the scalar-only branch."
        ),
    }


if __name__ == "__main__":
    print(paper1_octic_degenerate_scope_audit())
