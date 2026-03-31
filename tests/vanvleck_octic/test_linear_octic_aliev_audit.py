from pathlib import Path

from scripts.linear_octic_aliev_audit import build_linear_octic_aliev_audit, format_linear_octic_aliev_audit


def test_aliev_audit_shows_same_Cn_but_offset_dominated_L() -> None:
    out = build_linear_octic_aliev_audit(
        species="c2h2",
        gaussian_dir=Path("/Users/vincenzobarone/centrifugal/gaussian"),
    )
    assert out.same_Cn is True
    assert abs(out.aliev_shared_offset_cm) > 1.0e-18
    assert abs(out.quartic_total_cm) < 1.0e-30
    assert abs(out.geom_total_cm) < 1.0e-20
    assert abs(out.r_square_total_cm) < 1.0e-20
    assert abs(out.ours_pure_total_cm) < 1.0e-35


def test_aliev_audit_report_mentions_offset_dominance() -> None:
    text = format_linear_octic_aliev_audit(
        species="c2h2",
        gaussian_dir=Path("/Users/vincenzobarone/centrifugal/gaussian"),
    )
    assert "same_Cn = True" in text
    assert "aliev_shared_offset" in text
    assert "aliev_geom_total" in text
    assert "aliev_rsquare_total" in text
