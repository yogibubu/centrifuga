import sympy as sp

from scripts.paper1_h08_term3_h02_factor import paper1_h08_term3_h02_factor


def test_universal_polynomial_is_divisible_by_x() -> None:
    out = paper1_h08_term3_h02_factor()
    assert out["remainder"] == 0


def test_quotient_is_exact() -> None:
    out = paper1_h08_term3_h02_factor()
    X = sp.Symbol("X")
    expected = (
        sp.Rational(31, 2)
        - sp.Rational(97, 16) * X
        - sp.Rational(9, 64) * X**2
        + sp.Rational(5, 64) * X**3
        - sp.Rational(63, 256) * X**4
    )
    assert sp.expand(out["quotient_after_dividing_by_X"] - expected) == 0
