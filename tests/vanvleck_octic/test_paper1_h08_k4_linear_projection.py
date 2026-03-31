import sympy as sp

from scripts.paper1_h08_k4_linear_projection import paper1_h08_k4_linear_projection


def test_h08_k4_linear_projection_reduces_to_old_c080_only_when_yz_and_zz_drop() -> None:
    out = paper1_h08_k4_linear_projection()
    x4 = out["X4"]
    subs = {
        sp.Symbol("C_k^{yz}"): 0,
        sp.Symbol("C_l^{yz}"): 0,
        sp.Symbol("C_m^{yz}"): 0,
        sp.Symbol("C_n^{yz}"): 0,
        sp.Symbol("C_k^{zz}"): 0,
        sp.Symbol("C_l^{zz}"): 0,
        sp.Symbol("C_m^{zz}"): 0,
        sp.Symbol("C_n^{zz}"): 0,
    }
    expected = sp.Rational(35, 128) * sp.Symbol("k'_{klmn}") * sp.Symbol("C_k^{yy}") * sp.Symbol("C_l^{yy}") * sp.Symbol("C_m^{yy}") * sp.Symbol("C_n^{yy}") / 24
    assert sp.simplify(x4.subs(subs) - expected) == 0


def test_h08_k4_linear_projection_is_not_raw_c080_in_general() -> None:
    out = paper1_h08_k4_linear_projection()
    raw = sp.Rational(1, 24) * sp.Symbol("k'_{klmn}") * sp.Symbol("C_k^{yy}") * sp.Symbol("C_l^{yy}") * sp.Symbol("C_m^{yy}") * sp.Symbol("C_n^{yy}")
    assert sp.simplify(out["X4"] - raw) != 0
