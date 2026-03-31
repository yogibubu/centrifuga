import sympy as sp

from scripts.paper1_h08_term3_universal_polynomial import (
    paper1_h08_term3_universal_polynomial,
)


def test_one_and_two_mode_normalized_polynomials_match() -> None:
    out = paper1_h08_term3_universal_polynomial()
    one = out["one_mode_normalized"]
    two = out["two_mode_normalized"]
    for key in ("X1", "X2", "X3", "X4", "X5"):
        assert sp.simplify(one[key] - two[key]) == 0


def test_universal_polynomial_is_exact() -> None:
    out = paper1_h08_term3_universal_polynomial()
    X = sp.Symbol("X")
    expected = (
        sp.Rational(31, 2) * X
        - sp.Rational(97, 16) * X**2
        - sp.Rational(9, 64) * X**3
        + sp.Rational(5, 64) * X**4
        - sp.Rational(63, 256) * X**5
    )
    assert sp.expand(out["universal_polynomial"] - expected) == 0
