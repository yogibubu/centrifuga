#!/usr/bin/env python3
"""Exact one-dimensional scalar reduction law for the linear J^2 branch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ceditt_gui import _build_harmonic_model_from_inputs  # noqa: E402
from distortion_workflow import compute_order2_quartic  # noqa: E402
from compare_gaussian_sextic import sextic_linear_source_formula_hz  # noqa: E402
from gaussian_vpt_parser import parse_gaussian_anharmonic_force_data  # noqa: E402


@dataclass(frozen=True)
class LinearScalarReductionLaw:
    Hprime_formula: sp.Expr
    Lprime_formula: sp.Expr
    b_eliminate_H: sp.Expr
    L_shift_after_H_elimination: sp.Expr
    c2h2_H_phys_hz: float
    c2h2_3D2_over_B_hz: float
    hcn_H_phys_hz: float
    hcn_3D2_over_B_hz: float


def build_linear_scalar_reduction_law(*, gaussian_dir: Path) -> LinearScalarReductionLaw:
    B, D, H, L, b = sp.symbols("B D H L b", nonzero=True)
    X = sp.symbols("X")

    x = X + b * X**3
    expr = sp.expand(B * x - D * x**2 + H * x**3 + L * x**4)
    Hprime = sp.simplify(expr.coeff(X, 3))
    Lprime = sp.simplify(expr.coeff(X, 4))
    b0 = sp.solve(sp.Eq(Hprime, 0), b)[0]
    L_after = sp.simplify(Lprime.subs(b, b0))

    def _species_vals(species: str, fchk: Path, log: Path) -> tuple[float, float]:
        model, _ = _build_harmonic_model_from_inputs("I", fchk_path=str(fchk))
        anh = parse_gaussian_anharmonic_force_data(str(log))
        q = compute_order2_quartic(model)
        D_hz = float(q["special_quartic_projection"]["quartic_mhz"]["D"] * 1.0e6)
        B_hz = float(((model.abc_mhz[1] + model.abc_mhz[2]) / 2.0) * 1.0e6)
        H_phys_hz = float(sextic_linear_source_formula_hz(model, anh.phi3_reduced_cm)["H"])
        return H_phys_hz, float(3.0 * D_hz * D_hz / B_hz)

    c2h2_H, c2h2_3D2B = _species_vals(
        "c2h2",
        gaussian_dir / "c2h2.fchk",
        gaussian_dir / "c2h2.log",
    )
    hcn_H, hcn_3D2B = _species_vals(
        "hcn",
        ROOT / "hcn.fchk",
        ROOT / "hcn.log",
    )

    return LinearScalarReductionLaw(
        Hprime_formula=Hprime,
        Lprime_formula=Lprime,
        b_eliminate_H=sp.simplify(b0),
        L_shift_after_H_elimination=sp.simplify(L_after - L),
        c2h2_H_phys_hz=c2h2_H,
        c2h2_3D2_over_B_hz=c2h2_3D2B,
        hcn_H_phys_hz=hcn_H,
        hcn_3D2_over_B_hz=hcn_3D2B,
    )


def format_linear_scalar_reduction_law(*, gaussian_dir: Path) -> str:
    out = build_linear_scalar_reduction_law(gaussian_dir=gaussian_dir)
    return "\n".join(
        [
            f"H' = {sp.sstr(out.Hprime_formula)}",
            f"L' = {sp.sstr(out.Lprime_formula)}",
            f"b(H' = 0) = {sp.sstr(out.b_eliminate_H)}",
            f"Delta L after eliminating H = {sp.sstr(out.L_shift_after_H_elimination)}",
            f"c2h2: H_phys = {out.c2h2_H_phys_hz:+.12e} Hz, 3 D^2/B = {out.c2h2_3D2_over_B_hz:+.12e} Hz",
            f"hcn: H_phys = {out.hcn_H_phys_hz:+.12e} Hz, 3 D^2/B = {out.hcn_3D2_over_B_hz:+.12e} Hz",
        ]
    )


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gaussian-dir", default="/Users/vincenzobarone/centrifugal/gaussian")
    args = ap.parse_args()
    print(format_linear_scalar_reduction_law(gaussian_dir=Path(args.gaussian_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
