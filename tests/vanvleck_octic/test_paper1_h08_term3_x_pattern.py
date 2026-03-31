import sympy as sp

from scripts.paper1_h08_term3_x_pattern import paper1_h08_term3_x_pattern


def test_one_and_two_mode_ratios_match() -> None:
    out = paper1_h08_term3_x_pattern()
    assert out["one_mode_ratio"] == out["two_mode_ratio"]


def test_ratio_is_exactly_minus_63_over_20() -> None:
    out = paper1_h08_term3_x_pattern()
    assert sp.simplify(out["one_mode_ratio"] + sp.Rational(63, 20)) == 0
