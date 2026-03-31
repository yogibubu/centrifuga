import sympy as sp

from scripts.linear_scalar_reduction_freedom import linear_scalar_reduction_freedom


def test_quartic_invariance_fixes_only_a_not_b() -> None:
    out = linear_scalar_reduction_freedom()
    B, D, b, c = sp.symbols("B D b c", nonzero=True)
    assert sp.simplify(out["a_from_quartic_invariance"]) == 0
    assert sp.sstr(out["sextic_after_quartic_fix"]) == sp.sstr(B * b)
    assert sp.sstr(out["octic_after_quartic_fix"]) == sp.sstr(B * c - 2 * D * b)
