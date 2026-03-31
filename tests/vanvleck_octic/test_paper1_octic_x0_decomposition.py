from scripts.paper1_octic_x0_decomposition import paper1_octic_x0_decomposition


def test_x0_decomposition_has_all_six_channels() -> None:
    out = paper1_octic_x0_decomposition()
    assert tuple(out["termwise_X0"].keys()) == (
        "H08_k4",
        "S03S03S03H02",
        "S03S03H04",
        "S03H06",
        "S05H04",
        "S05S03H02",
    )


def test_total_x0_is_sum_of_channels() -> None:
    out = paper1_octic_x0_decomposition()
    assert out["total_X0"] == sum(out["termwise_X0"].values())
