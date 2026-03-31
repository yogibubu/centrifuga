import sympy as sp

from scripts.paper1_h08_rprimekl_linear_matching import paper1_h08_rprimekl_linear_matching


def test_rprimekl_diagonal_matches_rnn_exactly() -> None:
    out = paper1_h08_rprimekl_linear_matching()
    assert sp.simplify(out["gap"]) == 0
