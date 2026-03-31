#!/usr/bin/env python3

from scripts.linear_pairwise_octic_projection import linear_pairwise_octic_projection


def test_linear_pairwise_octic_projection_labels_qh() -> None:
    out = linear_pairwise_octic_projection()
    assert out["labels"] == ("q_e", "q_J", "q_H")
    assert len(out["basis"]) == 3
