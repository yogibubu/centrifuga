#!/usr/bin/env python3

from scripts.paper1_octic_true_linear_parallel_perp_form import paper1_octic_true_linear_parallel_perp_form


def test_parallel_perp_form_exposes_clean_split() -> None:
    out = paper1_octic_true_linear_parallel_perp_form()
    assert "tau_perp" in str(out["tau_perp_definition"])
    assert "L_pure^parallel" in str(out["pure_rotational_definition"])
    assert "L_deg^perp" in str(out["degenerate_block_definition"])
