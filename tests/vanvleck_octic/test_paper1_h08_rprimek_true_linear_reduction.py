#!/usr/bin/env python3

from scripts.paper1_h08_rprimek_true_linear_reduction import paper1_h08_rprimek_true_linear_reduction


def test_rprimek_true_linear_reduction_gives_rk_contraction() -> None:
    summary = paper1_h08_rprimek_true_linear_reduction()
    assert str(summary["r_k_reduction"]) == "Eq(r_k, Σ_l C_l r_kl)"
