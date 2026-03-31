#!/usr/bin/env python3

from pathlib import Path

from scripts.linear_s111_builder import build_linear_s111_builder_report, _format_report


def test_c2h2_linear_s111_builder_reports_pairwise_replacement() -> None:
    report = build_linear_s111_builder_report(
        species="c2h2",
        gaussian_dir=Path("/Users/vincenzobarone/centrifugal/gaussian"),
    )
    assert report.exact_linear_identifiability == "no_unique_scalar_S111"
    assert report.observable_replacement == "pairwise_degenerate_X_l_branch"
    assert len(report.pairs) == 2
    assert report.pairs[0]["carrier"] == "X_l"


def test_linear_s111_builder_format_mentions_observable_replacement() -> None:
    report = build_linear_s111_builder_report(
        species="c2h2",
        gaussian_dir=Path("/Users/vincenzobarone/centrifugal/gaussian"),
    )
    text = _format_report(report)
    assert "no_unique_scalar_S111" in text
    assert "pairwise_degenerate_X_l_branch" in text
