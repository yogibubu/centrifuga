from pathlib import Path

import sympy as sp

from scripts.linear_scalar_reduction_law import build_linear_scalar_reduction_law


def test_scalar_reduction_law_is_exact() -> None:
    out = build_linear_scalar_reduction_law(
        gaussian_dir=Path("/Users/vincenzobarone/centrifugal/gaussian"),
    )
    B, D, H, L, b = sp.symbols("B D H L b", nonzero=True)
    assert sp.sstr(out.Hprime_formula) == sp.sstr(B * b + H)
    assert sp.simplify(out.b_eliminate_H + H / B) == 0
    assert sp.simplify(out.L_shift_after_H_elimination - 2 * D * H / B) == 0


def test_available_linear_H_tracks_positive_3D2_over_B_not_hidden_shared_counterterm() -> None:
    out = build_linear_scalar_reduction_law(
        gaussian_dir=Path("/Users/vincenzobarone/centrifugal/gaussian"),
    )
    assert abs(out.c2h2_H_phys_hz - out.c2h2_3D2_over_B_hz) / out.c2h2_3D2_over_B_hz < 0.01
    assert abs(out.hcn_H_phys_hz - out.hcn_3D2_over_B_hz) / out.hcn_3D2_over_B_hz < 0.01
