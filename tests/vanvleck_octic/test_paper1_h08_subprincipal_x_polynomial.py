from scripts.paper1_h08_subprincipal_x_polynomial import paper1_h08_subprincipal_x_polynomial


def test_term1_and_term4_are_zero_in_x_polynomial() -> None:
    out = paper1_h08_subprincipal_x_polynomial()
    assert all(v == 0 for v in out["term1_X_polynomial"].values())
    assert all(v == 0 for v in out["term4_X_polynomial"].values())


def test_term3_contains_x5_and_x4() -> None:
    out = paper1_h08_subprincipal_x_polynomial()
    assert out["term3_X_polynomial"]["X5"] != 0
    assert out["term3_X_polynomial"]["X4"] != 0
