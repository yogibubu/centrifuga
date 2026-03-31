from scripts.octic_hierarchy_bridge import octic_hierarchy_bridge


def test_octic_hierarchy_counts() -> None:
    out = octic_hierarchy_bridge()
    assert out["hierarchy"] == (15, 9, 5, 1)
    assert out["drops"] == (6, 4, 4)


def test_octic_hierarchy_statement() -> None:
    text = octic_hierarchy_bridge()["statement"]
    assert "15 -> 9 -> 5 -> 1" in text
