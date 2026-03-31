import sympy as sp

from scripts.paper1_h08_linear_matching import paper1_h08_linear_matching


def test_u_eq_r_matches_exact_rsq_piece() -> None:
    out = paper1_h08_linear_matching()
    assert sp.simplify(out["u_eq_r_gap"]) == 0
