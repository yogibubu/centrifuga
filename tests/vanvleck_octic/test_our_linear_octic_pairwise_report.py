#!/usr/bin/env python3

from scripts.our_linear_octic_pairwise_report import build_combined_report


def test_combined_report_mentions_scalar_and_pairwise_observables() -> None:
    text = build_combined_report(
        species="c2h2",
        gaussian_dir="/Users/vincenzobarone/centrifugal/gaussian",
    )
    assert "L_parallel_pure" in text
    assert "q_H =" in text
    assert "Pi_g(1)" in text
