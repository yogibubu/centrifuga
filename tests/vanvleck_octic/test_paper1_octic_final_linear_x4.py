import sympy as sp

from scripts.paper1_octic_final_linear_x4 import paper1_octic_final_linear_x4


def test_final_linear_x4_contains_commutator_content() -> None:
    out = paper1_octic_final_linear_x4()
    x4 = out["X4_total"]
    assert x4 != out["H08_k4"]["X4"]


def test_final_linear_x4_reduces_to_h08_only_if_all_commutator_symbols_are_zero() -> None:
    out = paper1_octic_final_linear_x4()
    subs = {
        sp.Symbol("S111"): 0,
        sp.Symbol("s311"): 0,
        sp.Symbol("s131"): 0,
        sp.Symbol("s113"): 0,
        sp.Symbol("q400"): 0,
        sp.Symbol("q040"): 0,
        sp.Symbol("q004"): 0,
        sp.Symbol("q220"): 0,
        sp.Symbol("q202"): 0,
        sp.Symbol("q022"): 0,
        sp.Symbol("w600"): 0,
        sp.Symbol("w060"): 0,
        sp.Symbol("w006"): 0,
        sp.Symbol("w420"): 0,
        sp.Symbol("w402"): 0,
        sp.Symbol("w240"): 0,
        sp.Symbol("w204"): 0,
        sp.Symbol("w042"): 0,
        sp.Symbol("w024"): 0,
        sp.Symbol("w222"): 0,
        sp.Symbol("A"): 0,
        sp.Symbol("B"): 0,
        sp.Symbol("C"): 0,
    }
    assert sp.simplify(out["X4_total"].subs(subs) - out["H08_k4"]["X4"].subs(subs)) == 0
