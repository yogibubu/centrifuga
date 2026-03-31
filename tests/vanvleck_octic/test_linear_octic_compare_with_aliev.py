from pathlib import Path

from scripts.linear_octic_compare_with_aliev import (
    build_linear_octic_compare_with_aliev,
    format_linear_octic_compare_with_aliev,
)


def test_compare_with_aliev_builds_and_exposes_large_mismatch() -> None:
    out = build_linear_octic_compare_with_aliev(
        species="c2h2",
        gaussian_dir=Path("/Users/vincenzobarone/centrifugal/gaussian"),
    )
    assert out.L_aliev_cm < 0.0
    assert out.L_parallel_ours_cm > 0.0
    assert abs(out.delta_L_cm) > 1.0e-18
    assert len(out.pairwise_rows) == 2
    for row in out.pairwise_rows:
        assert row["q_H_ours_hz"] < 0.0
        assert row["q_H_aliev_hz"] > 0.0
        assert abs(row["delta_q_H_hz"]) > 1.0e4


def test_compare_report_mentions_both_scalar_and_pairwise_deltas() -> None:
    text = format_linear_octic_compare_with_aliev(
        species="c2h2",
        gaussian_dir=Path("/Users/vincenzobarone/centrifugal/gaussian"),
    )
    assert "L_parallel^(ours)" in text
    assert "L_Aliev" in text
    assert "Delta_L" in text
    assert "pairwise q_H comparison:" in text
    assert "Delta_q_H" in text
