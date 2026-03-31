#!/usr/bin/env python3

from scripts.paper1_h08_term5_pure_octic import paper1_h08_term5_pure_octic


def test_term5_is_pure_x0_in_true_linear_limit() -> None:
    summary = paper1_h08_term5_pure_octic()
    assert summary["difference"] == 0
