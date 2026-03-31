from scripts.paper1_octic_true_linear_limit_audit import audit_true_linear_limit


def test_h08_quartic_projection_reduces_to_known_quartic_term_in_true_linear_limit() -> None:
    out = audit_true_linear_limit()
    assert out["gap_h08_minus_quartic"] == 0


def test_total_x4_reduces_to_same_quartic_term_in_true_linear_limit_at_current_level() -> None:
    out = audit_true_linear_limit()
    assert out["gap_total_minus_quartic"] == 0
