#!/usr/bin/env python3

from scripts.linear_degenerate_octic_basis import linear_degenerate_octic_basis


def test_linear_degenerate_octic_basis_has_three_observables() -> None:
    out = linear_degenerate_octic_basis()
    assert out["labels"] == ("q_e", "q_J", "q_H")
    assert len(out["basis"]) == 3
