#!/usr/bin/env python3
from __future__ import annotations

from scripts.paper1_h08_degree_audit import paper1_h08_operator_degrees


def test_visible_h08_degrees_are_not_uniformly_octic() -> None:
    out = paper1_h08_operator_degrees()
    assert out["visible_h08_term_degrees"]["term1_RkRlRmRklm"] == 9
    assert out["visible_h08_term_degrees"]["term2_RkRlRmRn_k4"] == 8
    assert out["visible_h08_term_degrees"]["term3_RpkRlRmRplm"] == 7
    assert out["visible_h08_term_degrees"]["term4_RpkRl_comm"] == 6
    assert out["visible_h08_term_degrees"]["term5_Rpk_sq"] == 2
    assert out["not_a_contradiction"] is True
    assert "pre-reduction source line" in out["interpretation"]
