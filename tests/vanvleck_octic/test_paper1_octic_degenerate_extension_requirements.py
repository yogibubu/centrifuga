#!/usr/bin/env python3

from scripts.paper1_octic_degenerate_extension_requirements import paper1_octic_degenerate_extension_requirements


def test_extension_requirements_target_qh() -> None:
    out = paper1_octic_degenerate_extension_requirements()
    assert out["target_observable"] == "q_H"
    assert out["target_pairwise_labels"] == ("q_e", "q_J", "q_H")
