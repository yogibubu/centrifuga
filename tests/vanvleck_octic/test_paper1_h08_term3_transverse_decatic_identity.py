from scripts.paper1_h08_term3_transverse_decatic_identity import (
    paper1_h08_term3_transverse_decatic_identity,
)


def test_visible_term3_is_negative_transverse_decatic() -> None:
    out = paper1_h08_term3_transverse_decatic_identity()
    assert out["difference"] == 0
