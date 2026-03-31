from scripts.paper1_octic_linear_L_gap import linear_L_gap_one_mode


def test_linear_L_gap_is_generically_nonzero():
    out = linear_L_gap_one_mode()
    gap = out["gap"]
    assert gap != 0
