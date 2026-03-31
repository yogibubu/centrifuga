import sympy as sp

from scripts.paper1_h08_subprincipal_linear_ansatz import paper1_h08_subprincipal_linear_ansatz


def test_h08_subprincipal_linear_ansatz_coefficients_are_exact() -> None:
    out = paper1_h08_subprincipal_linear_ansatz()
    assert sp.simplify(out["term1_x4"] - sp.Symbol("C_k") * sp.Symbol("C_l") * sp.Symbol("C_m") * sp.Symbol("t_klm")) == 0
    assert sp.simplify(out["term2_x4"] - sp.Symbol("C_k") * sp.Symbol("C_l") * sp.Symbol("C_m") * sp.Symbol("C_n") * sp.Symbol("k'_{klmn}") / 24) == 0
    assert {sym.name for sym in out["term3_x4"].free_symbols} == {"C_l", "C_m", "r_lm", "u_k", "omega_k"}
    assert {sym.name for sym in out["term5_x4"].free_symbols} == {"u_k", "omega_k"}
