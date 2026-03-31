from pathlib import Path

import sympy as sp

from scripts.linear_octic_offset_justification import build_linear_octic_offset_justification


def test_unique_scalar_monomial_is_D3_over_B2() -> None:
    out = build_linear_octic_offset_justification(
        gaussian_dir=Path("/Users/vincenzobarone/centrifugal/gaussian"),
    )
    B, D = sp.symbols("B D", nonzero=True)
    assert sp.simplify(out.unique_scalar_monomial - D**3 / B**2) == 0


def test_lambda_is_minus_eight_from_ours_for_both_species() -> None:
    out = build_linear_octic_offset_justification(
        gaussian_dir=Path("/Users/vincenzobarone/centrifugal/gaussian"),
    )
    assert abs(out.c2h2_lambda_from_ours + 8.0) < 1.0e-12
    assert abs(out.hcn_lambda_from_ours + 8.0) < 1.0e-12
