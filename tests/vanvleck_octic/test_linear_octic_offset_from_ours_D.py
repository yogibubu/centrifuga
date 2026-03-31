from pathlib import Path

from scripts.linear_octic_aliev_audit import build_linear_octic_aliev_audit
from scripts.linear_octic_offset_from_ours_D import build_linear_octic_offset_from_ours_D


def test_offset_from_ours_D_matches_aliev_shared_offset_c2h2() -> None:
    ours = build_linear_octic_offset_from_ours_D(
        species="c2h2",
        gaussian_dir=Path("/Users/vincenzobarone/centrifugal/gaussian"),
    )
    audit = build_linear_octic_aliev_audit(
        species="c2h2",
        gaussian_dir=Path("/Users/vincenzobarone/centrifugal/gaussian"),
    )
    assert abs(ours.offset_cm - audit.aliev_shared_offset_cm) < 1.0e-23


def test_offset_from_ours_D_matches_aliev_shared_offset_hcn() -> None:
    ours = build_linear_octic_offset_from_ours_D(
        species="hcn",
        gaussian_dir=Path("/Users/vincenzobarone/centrifugal/gaussian"),
    )
    audit = build_linear_octic_aliev_audit(
        species="hcn",
        gaussian_dir=Path("/Users/vincenzobarone/centrifugal/gaussian"),
    )
    assert abs(ours.offset_cm - audit.aliev_shared_offset_cm) < 1.0e-23
