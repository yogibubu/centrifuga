from pathlib import Path

from scripts.linear_octic_pure_from_ours import _format_report, build_linear_octic_pure_from_ours


def test_c2h2_linear_octic_pure_from_ours_builds() -> None:
    report = build_linear_octic_pure_from_ours(
        species="c2h2",
        gaussian_dir=Path("/Users/vincenzobarone/centrifugal/gaussian"),
    )
    assert report.parallel_mode_indices_0based == (4, 5, 6)
    assert report.perpendicular_pairs_0based == ((0, 1), (2, 3))
    assert report.B_perp_cm > 0.0
    assert len(report.C_parallel_cm) == 3
    assert report.quartic_block_cm > 0.0
    assert report.term5_block_cm < 0.0
    assert abs(report.term3_block_cm) < 1.0e-50
    assert report.total_cm > 0.0


def test_hcn_linear_octic_pure_from_ours_builds() -> None:
    report = build_linear_octic_pure_from_ours(
        species="hcn",
        gaussian_dir=Path("/Users/vincenzobarone/centrifugal/gaussian"),
    )
    assert report.parallel_mode_indices_0based == (2, 3)
    assert report.perpendicular_pairs_0based == ((0, 1),)
    assert report.B_perp_cm > 0.0
    assert len(report.C_parallel_cm) == 2
    assert report.total_cm > 0.0


def test_formatted_pure_report_mentions_only_our_parallel_branch() -> None:
    report = build_linear_octic_pure_from_ours(
        species="c2h2",
        gaussian_dir=Path("/Users/vincenzobarone/centrifugal/gaussian"),
    )
    text = _format_report(report)
    assert "L_parallel^(ours)" in text
    assert "L4_parallel" in text
    assert "L3_parallel" in text
    assert "L5_parallel" in text
