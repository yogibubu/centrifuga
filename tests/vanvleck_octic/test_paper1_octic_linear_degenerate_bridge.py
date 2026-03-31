#!/usr/bin/env python3

from scripts.paper1_octic_linear_degenerate_bridge import paper1_octic_linear_degenerate_bridge


def test_c2h2_degenerate_bridge_targets_pairwise_qh() -> None:
    out = paper1_octic_linear_degenerate_bridge(
        species="c2h2",
        gaussian_dir="/Users/vincenzobarone/centrifugal/gaussian",
    )
    assert out["observable_exact_linear_replacement"] == "(J^2)^2 X_l pairwise branch"
    assert len(out["pairwise_qH_values_hz"]) == 2
    assert out["pairwise_qH_values_hz"][0]["q_H_hz"] > 0.0
