#!/usr/bin/env python3

from scripts.paper1_octic_true_linear_collect import paper1_octic_true_linear_collect


def test_true_linear_collect_has_three_square_blocks() -> None:
    summary = paper1_octic_true_linear_collect()
    assert summary["geometric_piece"] != 0
    assert summary["mixed_piece"] != 0
    assert summary["cubic_square_piece"] != 0
