import sympy as sp

from scripts.paper1_h08_term3_effective_matching import paper1_h08_term3_effective_matching


def test_term3_effective_matching_condition_is_exact() -> None:
    out = paper1_h08_term3_effective_matching()
    lhs = out["term3_eff"].subs({sp.Symbol("S_k[lm]"): out["matching_condition"].rhs})
    assert sp.simplify(lhs - out["target_geom"]) == 0
