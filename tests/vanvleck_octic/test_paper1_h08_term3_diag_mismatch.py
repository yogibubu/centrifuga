import sympy as sp

from scripts.paper1_h08_term3_diag_mismatch import paper1_h08_term3_diag_mismatch


def test_term3_naive_diagonal_is_not_geom_target() -> None:
    out = paper1_h08_term3_diag_mismatch()
    assert sp.simplify(out["gap"]) != 0
