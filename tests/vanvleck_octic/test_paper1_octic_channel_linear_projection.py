from scripts.paper1_octic_channel_linear_projection import project_all_channels_to_linear_polynomial


def test_at_least_one_commutator_channel_contributes_nontrivially_to_x4_after_k0_projection() -> None:
    out = project_all_channels_to_linear_polynomial()
    assert any(poly["X4"] != 0 for poly in out.values())
