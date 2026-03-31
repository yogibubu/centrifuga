from pathlib import Path

from scripts.linear_octic_pairwise_from_ours import build_linear_octic_pairwise_from_ours


def test_c2h2_pairwise_from_ours_is_finite_and_pair_resolved() -> None:
    report = build_linear_octic_pairwise_from_ours(
        species="c2h2",
        gaussian_dir=Path("/Users/vincenzobarone/centrifugal/gaussian"),
    )
    assert len(report.pairs) == 2
    assert report.tau_perp_cm < 0.0
    for pair in report.pairs:
        assert pair["sigma_t"] > 0.0
        assert pair["q_H_hz_from_ours"] < 0.0
        assert abs(pair["q_H_hz_from_ours"]) > 1.0e4


def test_hcn_pairwise_from_ours_is_finite_and_pair_resolved() -> None:
    report = build_linear_octic_pairwise_from_ours(
        species="hcn",
        gaussian_dir=Path("/Users/vincenzobarone/centrifugal/gaussian"),
    )
    assert len(report.pairs) == 1
    assert report.tau_perp_cm < 0.0
    assert report.pairs[0]["sigma_t"] > 0.0
    assert report.pairs[0]["q_H_hz_from_ours"] < 0.0
    assert abs(report.pairs[0]["q_H_hz_from_ours"]) > 1.0e4
