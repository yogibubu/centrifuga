from pathlib import Path

from scripts.linear_octic_pairwise_minimal_report import format_linear_octic_pairwise_minimal_report


def test_pairwise_minimal_report_contains_only_minimal_branch_data() -> None:
    text = format_linear_octic_pairwise_minimal_report(
        species="c2h2",
        gaussian_dir=Path("/Users/vincenzobarone/centrifugal/gaussian"),
    )
    assert "tau_perp =" in text
    assert "pair = [0, 1]" in text
    assert "pair = [2, 3]" in text
    assert "Sigma_t =" in text
    assert "q_H^(ours) =" in text
    assert "freq=" not in text
    assert "pairwise octic branch from our equations" not in text
