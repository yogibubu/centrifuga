#!/usr/bin/env python3

from scripts.paper1_h08_k4_octic_projected_piece import paper1_h08_k4_octic_projected_piece


def test_k4_octic_projection_has_exact_hierarchy_components() -> None:
    summary = paper1_h08_k4_octic_projected_piece()
    assert set(summary["hierarchy_components"]) == {
        "octic_X0",
        "sextic_X3",
        "quartic_X2",
        "quadratic_X1",
        "constant_X0",
    }
