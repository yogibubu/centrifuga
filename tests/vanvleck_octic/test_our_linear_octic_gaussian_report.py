#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import sympy as sp

from scripts.our_linear_octic_gaussian_report import (
    _format_report,
    build_our_linear_octic_pure_report,
)


def test_c2h2_our_linear_octic_report_builds_from_gaussian() -> None:
    report = build_our_linear_octic_pure_report(
        species="c2h2",
        gaussian_dir=Path("/Users/vincenzobarone/centrifugal/gaussian"),
    )
    assert report.parallel_mode_indices_0based == (4, 5, 6)
    assert report.perpendicular_pairs_0based == ((0, 1), (2, 3))
    assert report.total.is_real is not False
    assert report.aliev_total.is_real is not False
    assert sp.N(report.total).is_finite
    assert sp.N(report.aliev_total).is_finite
    assert sp.N(report.difference_vs_aliev).is_finite
    assert len(report.degenerate_pair_branch) == 2


def test_formatted_report_exposes_degenerate_branch_separately() -> None:
    report = build_our_linear_octic_pure_report(
        species="c2h2",
        gaussian_dir=Path("/Users/vincenzobarone/centrifugal/gaussian"),
    )
    text = _format_report(report)
    assert "L_our_pure_rotational" in text
    assert "L_deg^perp = (18/35) S_perp^2 tau_perp" in text
    assert "observable exact-linear replacement" in text
    assert "q_e =" in text
    assert "L_Aliev" in text
