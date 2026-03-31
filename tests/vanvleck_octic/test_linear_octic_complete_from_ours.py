from pathlib import Path

from scripts.linear_octic_complete_from_ours import (
    build_linear_octic_complete_from_ours,
    format_linear_octic_complete_report,
)


def test_complete_linear_octic_wrapper_builds_c2h2() -> None:
    report = build_linear_octic_complete_from_ours(
        species="c2h2",
        gaussian_dir=Path("/Users/vincenzobarone/centrifugal/gaussian"),
    )
    assert report.pure.parallel_mode_indices_0based == (4, 5, 6)
    assert report.degenerate.pairs[0]["pair"] == (0, 1)
    assert report.pure.total_cm > 0.0
    assert report.degenerate.tau_perp_cm < 0.0


def test_complete_linear_octic_report_exposes_both_branches() -> None:
    text = format_linear_octic_complete_report(
        species="c2h2",
        gaussian_dir=Path("/Users/vincenzobarone/centrifugal/gaussian"),
    )
    assert "pure rotational scalar branch:" in text
    assert "L_parallel^(ours)" in text
    assert "degenerate pairwise branch:" in text
    assert "tau_perp =" in text
    assert "q_H^(ours) =" in text
