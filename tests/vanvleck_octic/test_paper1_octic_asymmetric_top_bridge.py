from scripts.paper1_octic_asymmetric_top_bridge import (
    paper1_octic_asymmetric_top_bridge,
)


def test_asymmetric_top_bridge_counts() -> None:
    out = paper1_octic_asymmetric_top_bridge()
    assert out["cartesian_commuting_coefficients"] == 15
    assert out["aliev1983_determinable_combinations"] == 9
    assert out["redundant_directions"] == 6


def test_asymmetric_top_bridge_statement_mentions_both_levels() -> None:
    text = paper1_octic_asymmetric_top_bridge()["statement"]
    assert "15" in text
    assert "9" in text
    assert "6-dimensional redundancy" in text
