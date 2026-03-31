from scripts.paper1_octic_axial_decomposition import paper1_octic_axial_decomposition


def test_termwise_axial_decomposition_has_all_six_channels() -> None:
    out = paper1_octic_axial_decomposition()
    assert tuple(out["termwise"].keys()) == (
        "H08_k4",
        "S03S03S03H02",
        "S03S03H04",
        "S03H06",
        "S05H04",
        "S05S03H02",
    )


def test_total_l_is_x0() -> None:
    out = paper1_octic_axial_decomposition()
    assert out["total_L"] == out["total_axial"]["X0"]
